"""DyGFormer model for temporal link prediction.

Based on: "Towards Better Dynamic Graph Learning: New Architecture and
Unified Library" (Yu et al., NeurIPS 2023).

Architecture (Figure 1 in the paper):
    1. Extract first-hop interactions for source node u and destination
       node v before timestamp t → sequences S_u^t, S_v^t.
    2. Encode neighbors, links (edge features), and time intervals:
       - X_{*,N}: neighbor node encodings (node features or zero)
       - X_{*,E}: link/edge feature encodings
       - X_{*,T}: time interval encodings (trainable cos/sin)
       - X_{*,C}: neighbor co-occurrence encodings (NCoE)
    3. Patching: divide each encoding sequence into non-overlapping patches
       of size P, flatten patch contents → M_{*,*} matrices.
    4. Alignment: project each patched encoding to dimension d via linear
       layers, concatenate → Z_u, Z_v ∈ R^{l × 4d}.
    5. Transformer encoder: stack Z = [Z_u; Z_v] and feed through L layers
       of multi-head self-attention + FFN. This jointly models within-node
       and cross-node temporal dependencies.
    6. Mean pooling: average the output representations for u's patches
       and v's patches separately → h_u, h_v.
    7. Edge predictor: concatenation MLP on [h_u; h_v] → score.

Key innovations vs EAGLE/GLFormer:
    - NCoE: explicitly encodes co-occurrence frequencies of neighbors
      in both src and dst sequences (per-neighbor, not per-pair).
    - Patching: reduces computational cost from O(S²) to O((S/P)²)
      while preserving local temporal proximities within each patch.
    - Joint Transformer: processes src and dst patches together,
      enabling cross-sequence attention (unlike independent encoders).

Feature modes:
    - edge_feat_dim > 0: per-neighbor edge features (btc/usd) used
      in the link encoding channel.
    - node_feat_dim > 0: per-neighbor node features used in the
      neighbor encoding channel.
    - Both can be 0 (non-attributed graph): corresponding channels
      contribute zero vectors, but time and co-occurrence channels
      still provide signal.

Sequence convention:
    Neighbor sequences are in most-recent-first order (index 0 = most
    recent), matching sample_neighbors_batch output. Padded positions
    have all-zero encodings.
"""

import math
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class DyGFormerTimeEncoding(nn.Module):
    """Trainable time encoding following TGAT / DyGFormer convention.

    Encodes time intervals Δt via:
        [cos(w_1 Δt), sin(w_1 Δt), ..., cos(w_d Δt), sin(w_d Δt)] / √d

    where w_1, ..., w_d are learnable frequency parameters initialized
    to linearly spaced values. This differs from EAGLE/GLFormer's fixed
    cosine encoding — DyGFormer uses trainable frequencies with both
    cos and sin components.

    Args:
        dim: Output dimension (must be even; uses dim//2 frequencies).
    """

    def __init__(self, dim: int = 100):
        super().__init__()
        self.dim = dim
        self.w = nn.Linear(1, dim)
        nn.init.xavier_uniform_(self.w.weight)
        nn.init.zeros_(self.w.bias)
        self.scale = 1.0 / math.sqrt(dim)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """Encode timestamps into d-dimensional vectors.

        Args:
            t: Tensor of shape [...] with time interval values.

        Returns:
            Tensor of shape [..., dim].
        """
        # Large temporal deltas can overflow under CUDA autocast fp16 before the
        # cosine is applied, which then turns the whole batch into NaNs.
        if t.is_cuda:
            with torch.autocast(device_type="cuda", enabled=False):
                t_proj = self.w(t.float().unsqueeze(-1))
                return torch.cos(t_proj) * self.scale

        t_proj = self.w(t.float().unsqueeze(-1))
        return torch.cos(t_proj) * self.scale


class NeighborCooccurrenceEncoder(nn.Module):
    """Neighbor Co-occurrence Encoding (NCoE) from the paper.

    For each neighbor in S_u and S_v, counts its occurrences in both
    sequences, producing a 2D vector [count_in_u, count_in_v] per
    neighbor position. A two-layer MLP with ReLU projects each count
    vector to d_C dimensions:

        X_{*,C} = f(C_*[:, 0]) + f(C_*[:, 1])

    where f is a two-layer perceptron (1 → d_C with ReLU).

    Args:
        cooc_dim: Output dimension d_C for co-occurrence features.
    """

    def __init__(self, cooc_dim: int = 50):
        super().__init__()
        self.fc1 = nn.Linear(1, cooc_dim)
        self.fc2 = nn.Linear(cooc_dim, cooc_dim)

    def forward(self, counts: torch.Tensor) -> torch.Tensor:
        """Encode co-occurrence count vectors.

        Args:
            counts: [batch, seq_len, 2] float32 tensor where
                counts[:, :, 0] = count of this neighbor in src sequence,
                counts[:, :, 1] = count of this neighbor in dst sequence.

        Returns:
            [batch, seq_len, cooc_dim] co-occurrence feature vectors.
        """
        c0 = self.fc2(F.relu(self.fc1(counts[:, :, 0:1])))
        c1 = self.fc2(F.relu(self.fc1(counts[:, :, 1:2])))
        return c0 + c1


class DyGFormerTransformerLayer(nn.Module):
    """One Transformer layer: multi-head self-attention + FFN.

    Pre-norm architecture (LayerNorm before attention and FFN),
    following the paper's specification with GELU activation.

    Args:
        d_model: Model dimension (4d in the paper, where d = aligned dim).
        n_heads: Number of attention heads I.
        d_ff_factor: FFN expansion factor (default 4, so FFN hidden = 4 * d_model).
        dropout: Dropout rate.
    """

    def __init__(self, d_model: int, n_heads: int = 2,
                 d_ff_factor: int = 4, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        assert d_model % n_heads == 0
        self.d_k = d_model // n_heads

        self.dropout_p = dropout

        self.norm1 = nn.LayerNorm(d_model)
        self.W_Q = nn.Linear(d_model, d_model)
        self.W_K = nn.Linear(d_model, d_model)
        self.W_V = nn.Linear(d_model, d_model)
        self.W_O = nn.Linear(d_model, d_model)

        self.norm2 = nn.LayerNorm(d_model)
        d_ff = d_model * d_ff_factor
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply one Transformer layer.

        Args:
            x: [batch, seq_len, d_model] input.

        Returns:
            [batch, seq_len, d_model] output.
        """
        h = self.norm1(x)
        B, S, _ = h.shape

        Q = self.W_Q(h).view(B, S, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_K(h).view(B, S, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_V(h).view(B, S, self.n_heads, self.d_k).transpose(1, 2)

        context = F.scaled_dot_product_attention(
            Q, K, V, dropout_p=self.dropout_p if self.training else 0.0
        )

        context = context.transpose(1, 2).contiguous().view(B, S, self.d_model)
        attn_out = self.W_O(context)
        x = x + attn_out

        x = x + self.ffn(self.norm2(x))
        return x


class DyGFormerEncoder(nn.Module):
    """DyGFormer encoder: encode → patch → align → Transformer → pool.

    Processes both source and destination neighbor sequences jointly
    through a shared Transformer, enabling cross-sequence attention.

    Pipeline:
        1. Encode each channel (N, E, T, C) for src and dst sequences.
        2. Patch: reshape [B, S, d_ch] → [B, S/P, P*d_ch] for each channel.
        3. Align: project each patched channel to dimension d via linear.
        4. Concatenate channels: Z_* = [Z_{*,N} || Z_{*,E} || Z_{*,T} || Z_{*,C}]
           giving [B, l, 4d] per node (or 2d/3d if channels are disabled).
        5. Stack: Z = [Z_u; Z_v] → [B, l_u + l_v, 4d].
        6. Transformer: L layers of multi-head self-attention + FFN.
        7. Mean pool: h_u = mean(Z[:l_u]), h_v = mean(Z[l_u:]).

    Args:
        time_dim: Time encoding dimension d_T.
        aligned_dim: Aligned encoding dimension d per channel.
        num_neighbors: Maximum sequence length S (K neighbors sampled).
        patch_size: Patch size P. S must be divisible by P, or sequences
            will be padded to the nearest multiple.
        num_transformer_layers: Number of Transformer layers L.
        num_attention_heads: Number of attention heads I.
        dropout: Dropout rate.
        edge_feat_dim: Per-neighbor edge feature dimension d_E.
        node_feat_dim: Per-neighbor node feature dimension d_N.
        cooc_dim: Co-occurrence encoding dimension d_C.
        output_dim: Output embedding dimension d_out.
    """

    def __init__(self, time_dim: int = 100, aligned_dim: int = 50,
                 num_neighbors: int = 32, patch_size: int = 1,
                 num_transformer_layers: int = 2,
                 num_attention_heads: int = 2, dropout: float = 0.1,
                 edge_feat_dim: int = 2, node_feat_dim: int = 0,
                 cooc_dim: int = 50, output_dim: int = 172):
        super().__init__()
        self.time_dim = time_dim
        self.aligned_dim = aligned_dim
        self.num_neighbors = num_neighbors
        self.patch_size = patch_size
        self.num_transformer_layers = num_transformer_layers
        self.edge_feat_dim = edge_feat_dim
        self.node_feat_dim = node_feat_dim
        self.cooc_dim = cooc_dim
        self.output_dim = output_dim

        self.time_encoder = DyGFormerTimeEncoding(time_dim)
        self.cooc_encoder = NeighborCooccurrenceEncoder(cooc_dim)

        n_channels = 0

        self.align_T = nn.Linear(patch_size * time_dim, aligned_dim)
        n_channels += 1

        self.align_C = nn.Linear(patch_size * cooc_dim, aligned_dim)
        n_channels += 1

        if edge_feat_dim > 0:
            self.align_E = nn.Linear(patch_size * edge_feat_dim, aligned_dim)
            n_channels += 1
        else:
            self.align_E = None

        if node_feat_dim > 0:
            self.align_N = nn.Linear(patch_size * node_feat_dim, aligned_dim)
            n_channels += 1
        else:
            self.align_N = None

        self.n_channels = n_channels
        d_model = n_channels * aligned_dim

        self.transformer_layers = nn.ModuleList([
            DyGFormerTransformerLayer(
                d_model=d_model,
                n_heads=num_attention_heads,
                dropout=dropout,
            )
            for _ in range(num_transformer_layers)
        ])

        self.output_layer = nn.Linear(d_model, output_dim)

    def _patch_and_align(self, x: torch.Tensor,
                         align_layer: nn.Linear) -> torch.Tensor:
        """Reshape sequence into patches and project to aligned dimension.

        Args:
            x: [B, S, d_ch] sequence encodings.
            align_layer: Linear(P * d_ch, aligned_dim).

        Returns:
            [B, S/P, aligned_dim] patched and aligned encodings.
        """
        B, S, d_ch = x.shape
        P = self.patch_size
        n_patches = S // P
        x = x[:, :n_patches * P, :].reshape(B, n_patches, P * d_ch)
        return align_layer(x)

    def forward(self,
                src_delta_times: torch.Tensor,
                src_lengths: torch.Tensor,
                dst_delta_times: torch.Tensor,
                dst_lengths: torch.Tensor,
                src_cooc_counts: torch.Tensor,
                dst_cooc_counts: torch.Tensor,
                src_edge_feats: Optional[torch.Tensor] = None,
                dst_edge_feats: Optional[torch.Tensor] = None,
                src_node_feats: Optional[torch.Tensor] = None,
                dst_node_feats: Optional[torch.Tensor] = None,
                ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode source and destination nodes jointly.

        Args:
            src_delta_times: [B, S] source time deltas.
            src_lengths: [B] valid source neighbor counts.
            dst_delta_times: [B, S] destination time deltas.
            dst_lengths: [B] valid destination neighbor counts.
            src_cooc_counts: [B, S, 2] co-occurrence counts for src neighbors.
            dst_cooc_counts: [B, S, 2] co-occurrence counts for dst neighbors.
            src_edge_feats: [B, S, d_E] source edge features (optional).
            dst_edge_feats: [B, S, d_E] destination edge features (optional).
            src_node_feats: [B, S, d_N] source neighbor node features (optional).
            dst_node_feats: [B, S, d_N] destination neighbor node features (optional).

        Returns:
            Tuple (h_src, h_dst) each [B, output_dim].
        """
        B = src_delta_times.shape[0]
        S = src_delta_times.shape[1]
        P = self.patch_size
        device = src_delta_times.device

        pad_len = (P - S % P) % P
        if pad_len > 0:
            def _pad(t):
                if t is None:
                    return None
                pad_shape = list(t.shape)
                pad_shape[1] = pad_len
                return torch.cat([t, torch.zeros(*pad_shape, device=device, dtype=t.dtype)], dim=1)
            src_delta_times = _pad(src_delta_times)
            dst_delta_times = _pad(dst_delta_times)
            src_cooc_counts = _pad(src_cooc_counts)
            dst_cooc_counts = _pad(dst_cooc_counts)
            src_edge_feats = _pad(src_edge_feats)
            dst_edge_feats = _pad(dst_edge_feats)
            src_node_feats = _pad(src_node_feats)
            dst_node_feats = _pad(dst_node_feats)
            S_padded = S + pad_len
        else:
            S_padded = S

        n_patches = S_padded // P

        src_T = self.time_encoder(src_delta_times)
        dst_T = self.time_encoder(dst_delta_times)
        src_C = self.cooc_encoder(src_cooc_counts)
        dst_C = self.cooc_encoder(dst_cooc_counts)

        src_parts = [
            self._patch_and_align(src_T, self.align_T),
            self._patch_and_align(src_C, self.align_C),
        ]
        dst_parts = [
            self._patch_and_align(dst_T, self.align_T),
            self._patch_and_align(dst_C, self.align_C),
        ]

        if self.align_E is not None:
            if src_edge_feats is not None:
                src_parts.append(self._patch_and_align(src_edge_feats, self.align_E))
                dst_parts.append(self._patch_and_align(dst_edge_feats, self.align_E))
            else:
                zeros = torch.zeros(B, n_patches, self.aligned_dim, device=device)
                src_parts.append(zeros)
                dst_parts.append(zeros)

        if self.align_N is not None:
            if src_node_feats is not None:
                src_parts.append(self._patch_and_align(src_node_feats, self.align_N))
                dst_parts.append(self._patch_and_align(dst_node_feats, self.align_N))
            else:
                zeros = torch.zeros(B, n_patches, self.aligned_dim, device=device)
                src_parts.append(zeros)
                dst_parts.append(zeros)

        Z_src = torch.cat(src_parts, dim=-1)
        Z_dst = torch.cat(dst_parts, dim=-1)

        l_src = n_patches
        l_dst = n_patches
        Z = torch.cat([Z_src, Z_dst], dim=1)

        for layer in self.transformer_layers:
            Z = layer(Z)

        H_src = Z[:, :l_src, :]
        H_dst = Z[:, l_src:l_src + l_dst, :]

        src_patch_lengths = torch.clamp(
            torch.ceil(src_lengths.float() / P), min=1.0
        ).unsqueeze(-1)
        dst_patch_lengths = torch.clamp(
            torch.ceil(dst_lengths.float() / P), min=1.0
        ).unsqueeze(-1)

        h_src = H_src.sum(dim=1) / src_patch_lengths
        h_dst = H_dst.sum(dim=1) / dst_patch_lengths

        h_src = self.output_layer(h_src)
        h_dst = self.output_layer(h_dst)

        return h_src, h_dst


class DyGFormerEdgePredictor(nn.Module):
    """Concatenation MLP for edge scoring (same pattern as GLFormer).

    Scores a pair (src, dst) as:
        z = fc2(ReLU(fc1([h_src; h_dst])))

    Args:
        input_dim: Dimension of each node embedding (output_dim).
        hidden_dim: Predictor hidden dimension.
    """

    def __init__(self, input_dim: int, hidden_dim: int = 100):
        super().__init__()
        self.fc1 = nn.Linear(2 * input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, h_src: torch.Tensor,
                h_dst: torch.Tensor) -> torch.Tensor:
        """Predict link scores.

        Args:
            h_src: [B, input_dim] source embeddings.
            h_dst: [B, input_dim] or [B, C, input_dim] dest embeddings.

        Returns:
            [B] or [B, C] logits.
        """
        if h_dst.dim() == 3:
            B, C, _ = h_dst.shape
            src_exp = h_src.unsqueeze(1).expand(-1, C, -1)
            combined = torch.cat([src_exp, h_dst], dim=-1)
            return self.fc2(F.relu(self.fc1(combined))).squeeze(-1)
        else:
            combined = torch.cat([h_src, h_dst], dim=-1)
            return self.fc2(F.relu(self.fc1(combined))).squeeze(-1)


class DyGFormerTime(nn.Module):
    """Full DyGFormer model for temporal link prediction on stream graphs.

    Combines the DyGFormer encoder (NCoE + patching + Transformer) with
    a concatenation edge predictor.

    Unlike EAGLE and GLFormer which encode src and dst independently,
    DyGFormer processes them jointly through a shared Transformer.
    This enables cross-sequence attention — the model can learn
    correlations between src's and dst's interaction histories.

    The neighbor co-occurrence encoding (NCoE) is always enabled
    (integral part of DyGFormer, not optional like in GLFormer).

    Args:
        time_dim: Time encoding dimension d_T (paper default: 100).
        aligned_dim: Per-channel aligned dimension d (paper default: 50).
        num_neighbors: K most-recent neighbors sampled per node (paper: 32-4096).
        patch_size: Patch size P (paper: 1-128, scales with num_neighbors).
        num_transformer_layers: Number of Transformer layers L (paper default: 2).
        num_attention_heads: Number of attention heads I (paper default: 2).
        dropout: Dropout rate.
        edge_feat_dim: Per-neighbor edge feature dimension d_E (2 for btc+usd).
        node_feat_dim: Per-neighbor node feature dimension d_N (0 for non-attributed).
        cooc_dim: Co-occurrence encoding dimension d_C (paper default: 50).
        output_dim: Output embedding dimension d_out (paper default: 172).
    """

    def __init__(self, time_dim: int = 100, aligned_dim: int = 50,
                 num_neighbors: int = 32, patch_size: int = 1,
                 num_transformer_layers: int = 2,
                 num_attention_heads: int = 2, dropout: float = 0.1,
                 edge_feat_dim: int = 2, node_feat_dim: int = 0,
                 cooc_dim: int = 50, output_dim: int = 172):
        super().__init__()
        self.time_dim = time_dim
        self.aligned_dim = aligned_dim
        self.num_neighbors = num_neighbors
        self.patch_size = patch_size
        self.edge_feat_dim = edge_feat_dim
        self.node_feat_dim = node_feat_dim
        self.cooc_dim = cooc_dim
        self.output_dim = output_dim

        self.encoder = DyGFormerEncoder(
            time_dim=time_dim,
            aligned_dim=aligned_dim,
            num_neighbors=num_neighbors,
            patch_size=patch_size,
            num_transformer_layers=num_transformer_layers,
            num_attention_heads=num_attention_heads,
            dropout=dropout,
            edge_feat_dim=edge_feat_dim,
            node_feat_dim=node_feat_dim,
            cooc_dim=cooc_dim,
            output_dim=output_dim,
        )

        self.edge_predictor = DyGFormerEdgePredictor(
            input_dim=output_dim,
            hidden_dim=output_dim,
        )

    def forward(self,
                src_delta_times: torch.Tensor,
                src_lengths: torch.Tensor,
                dst_delta_times: torch.Tensor,
                dst_lengths: torch.Tensor,
                src_cooc_counts: torch.Tensor,
                dst_cooc_counts: torch.Tensor,
                src_edge_feats: Optional[torch.Tensor] = None,
                dst_edge_feats: Optional[torch.Tensor] = None,
                src_node_feats: Optional[torch.Tensor] = None,
                dst_node_feats: Optional[torch.Tensor] = None,
                ) -> torch.Tensor:
        """Score src->dst edges using joint encoding.

        This is the pairwise mode: each (src[i], dst[i]) pair is scored.

        Args:
            src_delta_times: [B, S] source neighbor delta times.
            src_lengths: [B] source valid neighbor counts.
            dst_delta_times: [B, S] destination delta times.
            dst_lengths: [B] destination valid counts.
            src_cooc_counts: [B, S, 2] co-occurrence counts for src.
            dst_cooc_counts: [B, S, 2] co-occurrence counts for dst.
            src_edge_feats: [B, S, d_E] source edge features (optional).
            dst_edge_feats: [B, S, d_E] destination edge features (optional).
            src_node_feats: [B, S, d_N] source neighbor node feats (optional).
            dst_node_feats: [B, S, d_N] dest neighbor node feats (optional).

        Returns:
            [B] logits.
        """
        h_src, h_dst = self.encoder(
            src_delta_times, src_lengths,
            dst_delta_times, dst_lengths,
            src_cooc_counts, dst_cooc_counts,
            src_edge_feats, dst_edge_feats,
            src_node_feats, dst_node_feats,
        )
        return self.edge_predictor(h_src, h_dst)

    def encode_pair(self,
                    src_delta_times: torch.Tensor,
                    src_lengths: torch.Tensor,
                    dst_delta_times: torch.Tensor,
                    dst_lengths: torch.Tensor,
                    src_cooc_counts: torch.Tensor,
                    dst_cooc_counts: torch.Tensor,
                    src_edge_feats: Optional[torch.Tensor] = None,
                    dst_edge_feats: Optional[torch.Tensor] = None,
                    src_node_feats: Optional[torch.Tensor] = None,
                    dst_node_feats: Optional[torch.Tensor] = None,
                    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode src and dst jointly, returning both embeddings.

        Useful for ranking mode where src is fixed and dst varies.
        Note: DyGFormer encodes src and dst jointly, so each (src, dst)
        combination requires a full encoder forward pass.

        Returns:
            Tuple (h_src, h_dst) each [B, output_dim].
        """
        return self.encoder(
            src_delta_times, src_lengths,
            dst_delta_times, dst_lengths,
            src_cooc_counts, dst_cooc_counts,
            src_edge_feats, dst_edge_feats,
            src_node_feats, dst_node_feats,
        )

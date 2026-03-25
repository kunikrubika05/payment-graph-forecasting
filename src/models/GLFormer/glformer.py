"""GLFormer model for temporal link prediction.

Based on: "Global-Lens Transformers: Adaptive Token Mixing for Dynamic
Link Prediction" (Zou et al., AAAI 2026).

Architecture:
    1. GLFormerTimeEncoding: fixed cosine time encoding (log-spaced frequencies),
       same as EAGLE.
    2. AdaptiveTokenMixer: causal local aggregation for one GLFormer layer.
       For each position i (most-recent = 0), aggregates from positions
       {i, i+1, ..., i+s} (older neighbors), using scores that combine
       learnable order weights and time-based decay (softmax of -delta_diff).
    3. GLFormerBlock: AdaptiveTokenMixer + channel FFN with LayerNorm residuals.
    4. GLFormerEncoder: token initialization -> L hierarchical GLFormerBlocks
       with expanding receptive fields -> LayerNorm -> mean pool -> optional
       query-node feature projection.
    5. NeighborCooccurrenceEncoder: projects pre-computed shared-neighbor counts
       (log1p-scaled) to a hidden-dim vector. Optional.
    6. GLFormerEdgePredictor: two-layer MLP on concatenated [h_src; h_dst] (and
       optionally co-occurrence features). Unlike EAGLE's additive predictor,
       concatenation can capture src-dst interactions (common neighbors etc.).
    7. GLFormerTime: full model combining encoder, optional co-occurrence
       encoder, and edge predictor.

Sequence convention:
    Neighbor sequences are kept in most-recent-first order (index 0 = most
    recent, index K-1 = oldest) — matching the output of sample_neighbors_batch.
    The AdaptiveTokenMixer aggregates from OLDER positions (larger index),
    which is equivalent to looking into the past for each token.

Feature modes:
    - edge_feat_dim > 0: per-neighbor edge features (e.g. btc/usd of each
      neighboring transaction) concatenated to the time encoding before
      the input projection. This is the primary feature mode for our dataset.
    - node_feat_dim > 0: query-node feature vector added to the pooled
      embedding (same as EAGLE's node feature mode).
    - use_cooccurrence: shared-neighbor count between src and dst is encoded
      and appended to the predictor input. Requires pre-computing counts
      outside the model (see glformer_train.py).

Difference from EAGLE:
    - Token mixer: EAGLE uses non-causal MLP-Mixer (parallel, no time awareness).
      GLFormer uses causal Adaptive Token Mixer (order + time-decay weights,
      hierarchical receptive field expansion across layers).
    - Predictor: EAGLE uses additive MLP (fc_src(h) + fc_dst(h)).
      GLFormer uses concatenation MLP (K2(ReLU(K1([h_src; h_dst])))).
      Concatenation can model src-dst interactions that additive cannot.
    - Layer strides: each GLFormer layer l uses offsets from s_{l-1} to s_l,
      so the effective temporal receptive field grows with depth.
"""

import math
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class GLFormerTimeEncoding(nn.Module):
    """Fixed cosine time encoding with log-spaced frequencies.

    Identical to EAGLE's EAGLETimeEncoding. Uses frequencies
    omega_i = 1 / 10^(i * 9 / (dim-1)), spanning 9 orders of magnitude.

    Args:
        dim: Number of frequency components (output dimension).
    """

    def __init__(self, dim: int = 100):
        super().__init__()
        self.dim = dim
        omega = torch.from_numpy(
            1.0 / 10 ** np.linspace(0, 9, dim, dtype=np.float32)
        )
        self.register_buffer("omega", omega)

    @torch.no_grad()
    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """Encode scalar timestamps into d-dimensional cosine vectors.

        Args:
            t: Tensor of shape [...] with delta-time values.

        Returns:
            Tensor of shape [..., dim].
        """
        return torch.cos(t.unsqueeze(-1) * self.omega)


class GLFormerFeedForward(nn.Module):
    """Two-layer MLP with GELU activation and dropout (channel mixer)."""

    def __init__(self, dim: int, expansion_factor: float = 4.0,
                 dropout: float = 0.1):
        super().__init__()
        hidden = int(dim * expansion_factor)
        self.net = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class AdaptiveTokenMixer(nn.Module):
    """Causal adaptive token mixer for one GLFormer layer.

    For each position i in the neighbor sequence (most-recent-first, so
    index 0 is the most recent neighbor), aggregates from positions
    {i, i+1, ..., i+s_curr} (older positions = larger index), using a
    convex combination of two weights:

        alpha_p^i = beta * w_p + (1 - beta) * theta_p^i

    where:
        w_p    — learnable scalar per offset p (captures interaction order)
        theta_p^i = softmax( -(delta[i+p] - delta[i]) ) over p in offsets
                  — temporal decay: closer past gets more weight
        beta   — learnable scalar (sigmoid) that balances the two signals

    Positions where i+p >= lengths[b] are masked (causal constraint:
    we can only look at neighbors that exist in the actual sequence).

    Args:
        s_prev: Start of offset range (inclusive). First layer: 0.
        s_curr: End of offset range (inclusive). E.g. 2 for first layer.
        hidden_dim: Token embedding dimension.
        dropout: Dropout applied to the output.
    """

    def __init__(self, s_prev: int, s_curr: int,
                 hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.offsets = list(range(s_prev, s_curr + 1))
        self.kernel_size = len(self.offsets)
        self.w = nn.Parameter(torch.zeros(self.kernel_size))
        self.beta = nn.Parameter(torch.zeros(1))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, delta_times: torch.Tensor,
                valid_mask: torch.Tensor) -> torch.Tensor:
        """Apply adaptive causal token mixing.

        Args:
            x: [batch, N, hidden_dim] token embeddings, most-recent-first.
            delta_times: [batch, N] time deltas (query_time - neighbor_time),
                most-recent-first (index 0 has smallest delta).
            valid_mask: [batch, N] bool, True for real (non-padded) positions.

        Returns:
            [batch, N, hidden_dim] mixed token embeddings.
        """
        B, N, d = x.shape
        device = x.device

        shifted_x_list = []
        time_diff_list = []
        causal_valid_list = []

        for p in self.offsets:
            if p == 0:
                shifted_x_list.append(x)
                time_diff_list.append(torch.zeros(B, N, device=device))
                causal_valid_list.append(valid_mask)
            else:
                # Shift left by p along the sequence dim:
                # new[b, i, :] = x[b, i+p, :] for i+p < N, else 0
                sx = torch.zeros_like(x)
                if p < N:
                    sx[:, :N - p, :] = x[:, p:, :]

                # time_diff[b, i] = delta[i+p] - delta[i] >= 0
                # (i+p is older → larger delta time)
                td = torch.zeros(B, N, device=device)
                if p < N:
                    td[:, :N - p] = (
                        delta_times[:, p:] - delta_times[:, :N - p]
                    ).clamp(min=0.0)

                # Valid if both position i and position i+p are valid
                cv = torch.zeros(B, N, dtype=torch.bool, device=device)
                if p < N:
                    cv[:, :N - p] = valid_mask[:, p:] & valid_mask[:, :N - p]

                shifted_x_list.append(sx)
                time_diff_list.append(td)
                causal_valid_list.append(cv)

        # [B, N, K, d], [B, N, K], [B, N, K]
        shifted_x = torch.stack(shifted_x_list, dim=2)
        time_diff = torch.stack(time_diff_list, dim=2)
        causal_valid = torch.stack(causal_valid_list, dim=2)

        # Time-based weights: softmax(-time_diff) over K, masked
        td_masked = time_diff.masked_fill(~causal_valid, float("inf"))
        theta = F.softmax(-td_masked, dim=2)           # [B, N, K]

        # Order weights: softmax over K
        w = F.softmax(self.w, dim=0).view(1, 1, -1)    # [1, 1, K]

        # Fused weights
        beta = torch.sigmoid(self.beta)
        alpha = beta * w + (1.0 - beta) * theta        # [B, N, K]

        # Zero-out invalid, renormalize
        alpha = alpha.masked_fill(~causal_valid, 0.0)
        alpha = alpha / alpha.sum(dim=2, keepdim=True).clamp(min=1e-8)

        # Weighted sum → [B, N, d]
        out = (shifted_x * alpha.unsqueeze(-1)).sum(dim=2)

        # Zero out fully-padded positions
        out = out * valid_mask.float().unsqueeze(-1)

        return self.dropout(out)


class GLFormerBlock(nn.Module):
    """One GLFormer layer: AdaptiveTokenMixer + channel FFN with residuals.

    Follows the Transformer residual structure:
        H' = H + TokenMixer(LayerNorm(H))
        H'' = H' + FFN(LayerNorm(H'))

    Args:
        s_prev: Start of offset range for the token mixer.
        s_curr: End of offset range for the token mixer.
        hidden_dim: Embedding dimension.
        channel_expansion: Expansion factor for the channel FFN.
        dropout: Dropout rate.
    """

    def __init__(self, s_prev: int, s_curr: int,
                 hidden_dim: int, channel_expansion: float = 4.0,
                 dropout: float = 0.1):
        super().__init__()
        self.token_norm = nn.LayerNorm(hidden_dim)
        self.token_mixer = AdaptiveTokenMixer(
            s_prev, s_curr, hidden_dim, dropout
        )
        self.channel_norm = nn.LayerNorm(hidden_dim)
        self.channel_ff = GLFormerFeedForward(
            hidden_dim, channel_expansion, dropout
        )

    def forward(self, x: torch.Tensor, delta_times: torch.Tensor,
                valid_mask: torch.Tensor) -> torch.Tensor:
        """Apply one GLFormer block.

        Args:
            x: [batch, N, hidden_dim] token embeddings.
            delta_times: [batch, N] time deltas (most-recent-first).
            valid_mask: [batch, N] bool validity mask.

        Returns:
            [batch, N, hidden_dim] updated token embeddings.
        """
        x = x + self.token_mixer(self.token_norm(x), delta_times, valid_mask)
        x = x + self.channel_ff(self.channel_norm(x))
        return x


class NeighborCooccurrenceEncoder(nn.Module):
    """Encodes shared-neighbor counts between src and dst nodes.

    Takes pre-computed intersection sizes (number of common temporal
    neighbors) and projects them to a hidden vector via log1p scaling
    followed by a linear layer.

    Args:
        cooc_dim: Output dimension (appended to predictor input).
    """

    def __init__(self, cooc_dim: int = 16):
        super().__init__()
        self.fc = nn.Linear(1, cooc_dim)

    def forward(self, counts: torch.Tensor) -> torch.Tensor:
        """Encode co-occurrence counts.

        Args:
            counts: [batch] float32 tensor of intersection sizes.

        Returns:
            [batch, cooc_dim] co-occurrence feature vectors.
        """
        return self.fc(torch.log1p(counts.unsqueeze(-1)))


class GLFormerEncoder(nn.Module):
    """GLFormer temporal encoder: token init -> hierarchical blocks -> pool.

    Initializes each neighbor token as Linear([time_enc, edge_feat]) then
    passes them through L GLFormerBlocks with expanding receptive fields.
    After the final block, applies LayerNorm and mean-pools over valid
    positions to produce a per-node embedding.

    Layer l uses offset range [s_{l-1}, s_l] where s_0=0 and the s values
    are computed as 2^1, 2^2, ..., 2^L (doubling per layer, matching the
    paper's example of s=[2,4,8] for L=3).

    Args:
        time_dim: Cosine time encoding dimension.
        hidden_dim: Hidden embedding dimension throughout.
        num_neighbors: K neighbors sampled per node.
        num_glformer_layers: Number of stacked GLFormerBlocks (1-3).
        channel_expansion: FFN expansion factor in each block.
        dropout: Dropout rate.
        edge_feat_dim: Dimension of per-neighbor edge features.
            If > 0, concatenated to time encoding before input projection.
        node_feat_dim: Dimension of query-node own features.
            If > 0, a linear projection is added to the pooled embedding.
    """

    def __init__(self, time_dim: int = 100, hidden_dim: int = 100,
                 num_neighbors: int = 20, num_glformer_layers: int = 2,
                 channel_expansion: float = 4.0, dropout: float = 0.1,
                 edge_feat_dim: int = 2, node_feat_dim: int = 0):
        super().__init__()
        self.num_neighbors = num_neighbors
        self.hidden_dim = hidden_dim
        self.edge_feat_dim = edge_feat_dim
        self.node_feat_dim = node_feat_dim
        self.num_glformer_layers = num_glformer_layers

        self.time_encoder = GLFormerTimeEncoding(time_dim)
        self.feat_encoder = nn.Linear(time_dim + edge_feat_dim, hidden_dim)

        # Build hierarchical blocks with expanding offset ranges.
        # s_0 = 0, s_l = 2^l for l = 1..L.
        s_values = [0] + [2 ** l for l in range(1, num_glformer_layers + 1)]
        self.blocks = nn.ModuleList([
            GLFormerBlock(
                s_prev=s_values[l],
                s_curr=s_values[l + 1],
                hidden_dim=hidden_dim,
                channel_expansion=channel_expansion,
                dropout=dropout,
            )
            for l in range(num_glformer_layers)
        ])

        self.layer_norm = nn.LayerNorm(hidden_dim)

        if node_feat_dim > 0:
            self.node_fc = nn.Linear(node_feat_dim, hidden_dim)
        else:
            self.node_fc = None

    def forward(self, delta_times: torch.Tensor, lengths: torch.Tensor,
                edge_feats: Optional[torch.Tensor] = None,
                node_feats: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Encode a batch of nodes from their temporal neighbor sequences.

        Args:
            delta_times: [batch, K] time deltas (query_time - neighbor_time),
                most-recent-first. Padded positions should have delta=0.
            lengths: [batch] number of valid (non-padded) neighbors per node.
            edge_feats: [batch, K, edge_feat_dim] per-neighbor edge features.
                Required when edge_feat_dim > 0.
            node_feats: [batch, node_feat_dim] query-node feature vectors.
                Required when node_feat_dim > 0.

        Returns:
            [batch, hidden_dim] node embeddings.
        """
        B, K = delta_times.shape
        device = delta_times.device

        # Build validity mask: [B, K]
        pos = torch.arange(K, device=device).unsqueeze(0)
        valid_mask = pos < lengths.unsqueeze(1)         # [B, K]

        # Token initialization
        time_enc = self.time_encoder(delta_times)       # [B, K, time_dim]

        if self.edge_feat_dim > 0 and edge_feats is not None:
            inp = torch.cat([time_enc, edge_feats], dim=-1)
        else:
            inp = time_enc

        x = self.feat_encoder(inp)                      # [B, K, hidden_dim]

        # Zero out padded token positions before processing
        x = x * valid_mask.float().unsqueeze(-1)

        # Hierarchical GLFormer blocks
        for block in self.blocks:
            x = block(x, delta_times, valid_mask)

        # LayerNorm + masked mean pool → [B, hidden_dim]
        x = self.layer_norm(x)
        lengths_clamped = lengths.float().clamp(min=1.0).unsqueeze(-1)
        x = (x * valid_mask.float().unsqueeze(-1)).sum(dim=1) / lengths_clamped

        # Optional query-node feature projection
        if self.node_fc is not None and node_feats is not None:
            x = x + self.node_fc(node_feats)

        return x


class GLFormerEdgePredictor(nn.Module):
    """Concatenation MLP for edge scoring.

    Scores a pair (src, dst) as:
        z = K2( ReLU( K1([h_src; h_dst; cooc_feat?]) ) )

    Concatenation (vs. EAGLE's additive approach) allows the predictor
    to model non-linear interactions between src and dst embeddings, which
    is important for capturing shared structural context.

    Args:
        input_dim: Dimension of each node embedding (hidden_dim).
        hidden_dim: Predictor hidden dimension.
        cooc_dim: Dimension of co-occurrence features. 0 if not used.
    """

    def __init__(self, input_dim: int, hidden_dim: int, cooc_dim: int = 0):
        super().__init__()
        self.cooc_dim = cooc_dim
        concat_dim = 2 * input_dim + cooc_dim
        self.fc1 = nn.Linear(concat_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, h_src: torch.Tensor, h_dst: torch.Tensor,
                cooc_feat: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Predict link scores.

        Args:
            h_src: [batch, input_dim] source embeddings.
            h_dst: [batch, input_dim] or [batch, C, input_dim] dest embeddings.
            cooc_feat: [batch, cooc_dim] or [batch, C, cooc_dim] or None.
                When cooc_dim > 0 but cooc_feat is None, zeros are used.

        Returns:
            [batch] or [batch, C] logits.
        """
        if h_dst.dim() == 3:
            B, C, _ = h_dst.shape
            src_exp = h_src.unsqueeze(1).expand(-1, C, -1)  # [B, C, d]
            if self.cooc_dim > 0:
                if cooc_feat is None:
                    cooc_feat = torch.zeros(
                        B, C, self.cooc_dim, device=h_src.device
                    )
                parts = [src_exp, h_dst, cooc_feat]
            else:
                parts = [src_exp, h_dst]
            combined = torch.cat(parts, dim=-1)             # [B, C, 2d+cooc]
            return self.fc2(F.relu(self.fc1(combined))).squeeze(-1)
        else:
            B = h_src.shape[0]
            if self.cooc_dim > 0:
                if cooc_feat is None:
                    cooc_feat = torch.zeros(
                        B, self.cooc_dim, device=h_src.device
                    )
                combined = torch.cat([h_src, h_dst, cooc_feat], dim=-1)
            else:
                combined = torch.cat([h_src, h_dst], dim=-1)
            return self.fc2(F.relu(self.fc1(combined))).squeeze(-1)


class GLFormerTime(nn.Module):
    """Full GLFormer model for temporal link prediction on stream graphs.

    Encodes temporal neighbor patterns via hierarchical Adaptive Token
    Mixing and scores edges with a concatenation MLP predictor.

    Compared to EAGLE-Time:
        - AdaptiveTokenMixer replaces MLP-Mixer: causal, time-aware, and
          hierarchical (each layer expands the temporal receptive field).
        - Concatenation predictor replaces additive: captures src-dst
          interactions necessary for finding common neighbor patterns.
        - Optional co-occurrence features: pre-computed shared-neighbor
          counts can be added to the predictor input.

    Args:
        time_dim: Time encoding dimension.
        hidden_dim: Hidden dimension for all modules.
        num_neighbors: K most-recent neighbors sampled per node.
        num_glformer_layers: Number of stacked GLFormerBlocks (1-3).
            Layer l covers offsets [2^(l-1)*2, 2^l*2], so:
                L=1: offsets 0-2
                L=2: offsets 0-2 (layer 1) + 2-4 (layer 2)
                L=3: additionally 4-8 (layer 3)
        channel_expansion: Expansion factor for channel FFN in each block.
        dropout: Dropout rate.
        edge_feat_dim: Per-neighbor edge feature dimension (e.g. 2 for btc+usd).
        node_feat_dim: Query-node feature dimension (added after pooling).
        use_cooccurrence: If True, enables co-occurrence feature encoding.
            Pre-computed counts must be passed to forward() and encode_nodes().
        cooc_dim: Dimension of co-occurrence encoding. Used when
            use_cooccurrence=True.
    """

    def __init__(self, time_dim: int = 100, hidden_dim: int = 100,
                 num_neighbors: int = 20, num_glformer_layers: int = 2,
                 channel_expansion: float = 4.0, dropout: float = 0.1,
                 edge_feat_dim: int = 2, node_feat_dim: int = 0,
                 use_cooccurrence: bool = False, cooc_dim: int = 16):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_neighbors = num_neighbors
        self.edge_feat_dim = edge_feat_dim
        self.node_feat_dim = node_feat_dim
        self.use_cooccurrence = use_cooccurrence
        self.cooc_dim = cooc_dim if use_cooccurrence else 0

        self.encoder = GLFormerEncoder(
            time_dim=time_dim,
            hidden_dim=hidden_dim,
            num_neighbors=num_neighbors,
            num_glformer_layers=num_glformer_layers,
            channel_expansion=channel_expansion,
            dropout=dropout,
            edge_feat_dim=edge_feat_dim,
            node_feat_dim=node_feat_dim,
        )

        if use_cooccurrence:
            self.cooc_encoder = NeighborCooccurrenceEncoder(cooc_dim)
        else:
            self.cooc_encoder = None

        self.edge_predictor = GLFormerEdgePredictor(
            input_dim=hidden_dim,
            hidden_dim=hidden_dim,
            cooc_dim=self.cooc_dim,
        )

    def encode_nodes(self, delta_times: torch.Tensor,
                     lengths: torch.Tensor,
                     edge_feats: Optional[torch.Tensor] = None,
                     node_feats: Optional[torch.Tensor] = None
                     ) -> torch.Tensor:
        """Encode a batch of nodes from their temporal neighbor sequences.

        Args:
            delta_times: [batch, K] time deltas, most-recent-first.
            lengths: [batch] valid neighbor counts.
            edge_feats: [batch, K, edge_feat_dim]. Pass when edge_feat_dim > 0.
            node_feats: [batch, node_feat_dim]. Pass when node_feat_dim > 0.

        Returns:
            [batch, hidden_dim] node embeddings.
        """
        return self.encoder(delta_times, lengths, edge_feats, node_feats)

    def forward(self,
                src_delta_times: torch.Tensor,
                src_lengths: torch.Tensor,
                dst_delta_times: torch.Tensor,
                dst_lengths: torch.Tensor,
                src_edge_feats: Optional[torch.Tensor] = None,
                dst_edge_feats: Optional[torch.Tensor] = None,
                src_node_feats: Optional[torch.Tensor] = None,
                dst_node_feats: Optional[torch.Tensor] = None,
                cooc_counts: Optional[torch.Tensor] = None,
                ) -> torch.Tensor:
        """Score src->dst edges.

        Supports two modes:
            - Pairwise: dst_delta_times is [B, K] → returns [B] logits.
            - Ranking: dst_delta_times is [B, C, K] → returns [B, C] logits,
              where C is the number of destination candidates.

        Args:
            src_delta_times: [B, K] source neighbor delta times.
            src_lengths: [B] source valid neighbor counts.
            dst_delta_times: [B, K] or [B, C, K] destination delta times.
            dst_lengths: [B] or [B, C] destination valid counts.
            src_edge_feats: [B, K, edge_feat_dim] source neighbor edge feats.
            dst_edge_feats: [B, K, ef] or [B, C, K, ef] dest edge feats.
            src_node_feats: [B, node_feat_dim] source node features.
            dst_node_feats: [B, nf] or [B, C, nf] destination node features.
            cooc_counts: [B] or [B, C] pre-computed shared-neighbor counts.
                Required when use_cooccurrence=True.

        Returns:
            [B] or [B, C] logits.
        """
        h_src = self.encode_nodes(
            src_delta_times, src_lengths, src_edge_feats, src_node_feats
        )

        if dst_delta_times.dim() == 3:
            B, C, K = dst_delta_times.shape
            flat_dt = dst_delta_times.reshape(B * C, K)
            flat_len = dst_lengths.reshape(B * C)

            flat_ef = None
            if dst_edge_feats is not None:
                flat_ef = dst_edge_feats.reshape(B * C, K, dst_edge_feats.shape[-1])

            flat_nf = None
            if dst_node_feats is not None:
                flat_nf = dst_node_feats.reshape(B * C, dst_node_feats.shape[-1])

            h_dst = self.encode_nodes(flat_dt, flat_len, flat_ef, flat_nf)
            h_dst = h_dst.reshape(B, C, self.hidden_dim)

            cooc_feat = None
            if self.cooc_encoder is not None and cooc_counts is not None:
                flat_cooc = cooc_counts.reshape(B * C)
                cooc_feat = self.cooc_encoder(flat_cooc).reshape(B, C, self.cooc_dim)
        else:
            h_dst = self.encode_nodes(
                dst_delta_times, dst_lengths, dst_edge_feats, dst_node_feats
            )
            cooc_feat = None
            if self.cooc_encoder is not None and cooc_counts is not None:
                cooc_feat = self.cooc_encoder(cooc_counts)

        return self.edge_predictor(h_src, h_dst, cooc_feat)

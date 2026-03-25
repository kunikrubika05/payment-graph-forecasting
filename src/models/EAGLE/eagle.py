"""EAGLE-Time model for temporal link prediction.

Based on: "EAGLE: Expressive dynamics-Aware Graph LEarning"
(Yue et al., 2024)

Architecture:
    1. EAGLETimeEncoder: MLP-Mixer on temporal neighbor delta times,
       optionally enriched with per-neighbor edge features and per-node
       node features.
    2. EAGLEEdgePredictor: Additive MLP for link scoring.
    3. EAGLETime: Full model combining encoder and predictor.

Feature modes:
    - Time-only (edge_feat_dim=0, node_feat_dim=0): encodes only delta
      times. Fast training, focuses on temporal interaction patterns.
    - With edge features (edge_feat_dim>0): concatenates edge feature
      vector to the time encoding of each neighbor interaction, giving
      the MLP-Mixer richer per-token input.
    - With node features (node_feat_dim>0): adds a linear projection of
      the query node's own feature vector to the pooled embedding.
    - Full (both non-zero): all three signals combined.
"""

import math
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class EAGLETimeEncoding(nn.Module):
    """Fixed cosine time encoding with EAGLE's log-spaced frequencies.

    Uses frequencies omega_i = 1 / 10^(i * 9 / (dim-1)) for i in 0..dim-1.
    Spans 9 orders of magnitude (1 to 1e-9), capturing both short-term
    and long-term temporal patterns.
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
        """Encode timestamps into d-dimensional cosine vectors.

        Args:
            t: Tensor of shape [...] with timestamp/delta values.

        Returns:
            Tensor of shape [..., dim] with cosine time encodings.
        """
        return torch.cos(t.unsqueeze(-1) * self.omega)


class EAGLEFeedForward(nn.Module):
    """Two-layer MLP with GELU activation and dropout."""

    def __init__(self, dim: int, expansion_factor: float = 2.0,
                 dropout: float = 0.1, single_layer: bool = False):
        super().__init__()
        self.single_layer = single_layer

        if single_layer:
            self.net = nn.Sequential(
                nn.Linear(dim, dim),
                nn.GELU(),
                nn.Dropout(dropout),
            )
        else:
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


class EAGLEMixerBlock(nn.Module):
    """MLP-Mixer block: token-mixing + channel-mixing with residuals.

    Token expansion is typically 0.5 (compressive) to avoid overfitting
    on the small number of temporal neighbors. Channel expansion is 4
    (standard) for rich feature mixing.
    """

    def __init__(self, num_tokens: int, hidden_dim: int,
                 token_expansion: float = 0.5,
                 channel_expansion: float = 4.0,
                 dropout: float = 0.1):
        super().__init__()
        self.token_norm = nn.LayerNorm(hidden_dim)
        self.token_ff = EAGLEFeedForward(num_tokens, token_expansion, dropout)
        self.channel_norm = nn.LayerNorm(hidden_dim)
        self.channel_ff = EAGLEFeedForward(hidden_dim, channel_expansion, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply token and channel mixing.

        Args:
            x: [batch, num_tokens, hidden_dim]

        Returns:
            [batch, num_tokens, hidden_dim]
        """
        residual = x
        x = self.token_norm(x).permute(0, 2, 1)
        x = self.token_ff(x).permute(0, 2, 1)
        x = x + residual

        residual = x
        x = self.channel_norm(x)
        x = self.channel_ff(x)
        x = x + residual

        return x


class EAGLETimeEncoder(nn.Module):
    """EAGLE encoder: TimeEncode -> Linear -> Mixer -> pool -> head.

    Encodes temporal neighbor patterns from delta times plus optional
    per-neighbor edge features and per-query-node node features.

    Args:
        time_dim: Dimension of cosine time encoding.
        hidden_dim: Hidden dimension used throughout.
        num_neighbors: Number of temporal neighbors per node.
        num_mixer_layers: Number of MLP-Mixer blocks.
        token_expansion: Expansion factor for token-mixing MLP.
        channel_expansion: Expansion factor for channel-mixing MLP.
        dropout: Dropout rate.
        edge_feat_dim: Dimension of per-neighbor edge features.
            If > 0, edge feature vector is concatenated to the time
            encoding before the input projection.
        node_feat_dim: Dimension of query-node features.
            If > 0, a linear projection of the node's own feature
            vector is added to the pooled embedding.
    """

    def __init__(self, time_dim: int = 100, hidden_dim: int = 100,
                 num_neighbors: int = 20, num_mixer_layers: int = 1,
                 token_expansion: float = 0.5,
                 channel_expansion: float = 4.0,
                 dropout: float = 0.1,
                 edge_feat_dim: int = 0,
                 node_feat_dim: int = 0):
        super().__init__()
        self.num_neighbors = num_neighbors
        self.hidden_dim = hidden_dim
        self.edge_feat_dim = edge_feat_dim
        self.node_feat_dim = node_feat_dim

        self.time_encoder = EAGLETimeEncoding(time_dim)
        self.feat_encoder = nn.Linear(time_dim + edge_feat_dim, hidden_dim)

        self.mixer_blocks = nn.ModuleList([
            EAGLEMixerBlock(
                num_tokens=num_neighbors,
                hidden_dim=hidden_dim,
                token_expansion=token_expansion,
                channel_expansion=channel_expansion,
                dropout=dropout,
            )
            for _ in range(num_mixer_layers)
        ])

        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.mlp_head = nn.Linear(hidden_dim, hidden_dim)

        if node_feat_dim > 0:
            self.node_fc = nn.Linear(node_feat_dim, hidden_dim)
        else:
            self.node_fc = None

    def forward(self, delta_times: torch.Tensor,
                lengths: torch.Tensor,
                edge_feats: Optional[torch.Tensor] = None,
                node_feats: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Encode temporal patterns for a batch of nodes.

        Args:
            delta_times: [batch, num_neighbors] relative timestamps
                         (query_time - neighbor_time), padded with 0.
            lengths: [batch] actual number of valid neighbors per node.
            edge_feats: [batch, num_neighbors, edge_feat_dim] features of
                        each neighboring edge, padded with 0. Required when
                        edge_feat_dim > 0.
            node_feats: [batch, node_feat_dim] feature vector of each query
                        node. Required when node_feat_dim > 0.

        Returns:
            [batch, hidden_dim] node encodings.
        """
        time_enc = self.time_encoder(delta_times)

        if self.edge_feat_dim > 0 and edge_feats is not None:
            inp = torch.cat([time_enc, edge_feats], dim=-1)
        else:
            inp = time_enc

        x = self.feat_encoder(inp)

        mask = torch.arange(self.num_neighbors, device=delta_times.device)
        mask = mask.unsqueeze(0) >= lengths.unsqueeze(1)
        x = x.masked_fill(mask.unsqueeze(-1), 0.0)

        for block in self.mixer_blocks:
            x = block(x)

        x = self.layer_norm(x)
        x = x.mean(dim=1)
        x = self.mlp_head(x)

        if self.node_fc is not None and node_feats is not None:
            x = x + self.node_fc(node_feats)

        return x


class EAGLEEdgePredictor(nn.Module):
    """Additive edge predictor: score = out(relu(fc_src(h_src) + fc_dst(h_dst))).

    Supports both pairwise scoring [batch] and ranking [batch, num_candidates].
    """

    def __init__(self, input_dim: int, hidden_dim: int = 100):
        super().__init__()
        self.src_fc = nn.Linear(input_dim, hidden_dim)
        self.dst_fc = nn.Linear(input_dim, hidden_dim)
        self.out_fc = nn.Linear(hidden_dim, 1)

    def forward(self, h_src: torch.Tensor,
                h_dst: torch.Tensor) -> torch.Tensor:
        """Predict link scores.

        Args:
            h_src: [batch, input_dim] source node embeddings.
            h_dst: [batch, input_dim] or [batch, C, input_dim] dest embeddings.

        Returns:
            [batch] or [batch, C] logits.
        """
        src_proj = self.src_fc(h_src)

        if h_dst.dim() == 3:
            dst_proj = self.dst_fc(h_dst)
            combined = F.relu(src_proj.unsqueeze(1) + dst_proj)
            return self.out_fc(combined).squeeze(-1)
        else:
            dst_proj = self.dst_fc(h_dst)
            combined = F.relu(src_proj + dst_proj)
            return self.out_fc(combined).squeeze(-1)


class EAGLETime(nn.Module):
    """Full EAGLE-Time model for temporal link prediction.

    Encodes interaction timing patterns via MLP-Mixer and scores edges
    with an additive predictor. Supports three feature modes:

        - Time-only (edge_feat_dim=0, node_feat_dim=0): uses only delta
          times. Fast, ~120K params at default settings.
        - With edge features (edge_feat_dim>0): concatenates the feature
          vector of each neighboring edge to its time encoding.
        - With node features (node_feat_dim>0): adds a linear projection
          of the query node's own features to the pooled embedding.
    """

    def __init__(self, time_dim: int = 100, hidden_dim: int = 100,
                 num_neighbors: int = 20, num_mixer_layers: int = 1,
                 token_expansion: float = 0.5,
                 channel_expansion: float = 4.0,
                 dropout: float = 0.1,
                 edge_feat_dim: int = 0,
                 node_feat_dim: int = 0):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_neighbors = num_neighbors
        self.edge_feat_dim = edge_feat_dim
        self.node_feat_dim = node_feat_dim

        self.encoder = EAGLETimeEncoder(
            time_dim=time_dim,
            hidden_dim=hidden_dim,
            num_neighbors=num_neighbors,
            num_mixer_layers=num_mixer_layers,
            token_expansion=token_expansion,
            channel_expansion=channel_expansion,
            dropout=dropout,
            edge_feat_dim=edge_feat_dim,
            node_feat_dim=node_feat_dim,
        )

        self.edge_predictor = EAGLEEdgePredictor(hidden_dim, hidden_dim)

    def encode_nodes(self, delta_times: torch.Tensor,
                     lengths: torch.Tensor,
                     edge_feats: Optional[torch.Tensor] = None,
                     node_feats: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Encode a batch of nodes from their temporal neighbor patterns.

        Args:
            delta_times: [batch, K] relative timestamps to recent neighbors.
            lengths: [batch] valid neighbor counts.
            edge_feats: [batch, K, edge_feat_dim] features of neighboring
                        edges. Pass when edge_feat_dim > 0.
            node_feats: [batch, node_feat_dim] query node features.
                        Pass when node_feat_dim > 0.

        Returns:
            [batch, hidden_dim] node embeddings.
        """
        return self.encoder(delta_times, lengths, edge_feats, node_feats)

    def forward(self, src_delta_times: torch.Tensor,
                src_lengths: torch.Tensor,
                dst_delta_times: torch.Tensor,
                dst_lengths: torch.Tensor,
                src_edge_feats: Optional[torch.Tensor] = None,
                dst_edge_feats: Optional[torch.Tensor] = None,
                src_node_feats: Optional[torch.Tensor] = None,
                dst_node_feats: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Score src->dst edges.

        Args:
            src_delta_times: [batch, K] source neighbor delta times.
            src_lengths: [batch] source valid neighbor counts.
            dst_delta_times: [batch, K] or [batch, C, K] destination delta times.
            dst_lengths: [batch] or [batch, C] destination valid counts.
            src_edge_feats: [batch, K, edge_feat_dim] source neighbor edge feats.
            dst_edge_feats: [batch, K, edge_feat_dim] or [batch, C, K, edge_feat_dim].
            src_node_feats: [batch, node_feat_dim] source node features.
            dst_node_feats: [batch, node_feat_dim] or [batch, C, node_feat_dim].

        Returns:
            [batch] or [batch, C] logits.
        """
        h_src = self.encode_nodes(
            src_delta_times, src_lengths, src_edge_feats, src_node_feats
        )

        if dst_delta_times.dim() == 3:
            batch_size, num_candidates = dst_delta_times.shape[:2]
            flat_dt = dst_delta_times.reshape(-1, dst_delta_times.shape[-1])
            flat_len = dst_lengths.reshape(-1)

            flat_ef = None
            if dst_edge_feats is not None:
                flat_ef = dst_edge_feats.reshape(
                    -1, dst_edge_feats.shape[-2], dst_edge_feats.shape[-1]
                )

            flat_nf = None
            if dst_node_feats is not None:
                flat_nf = dst_node_feats.reshape(-1, dst_node_feats.shape[-1])

            h_dst_flat = self.encode_nodes(flat_dt, flat_len, flat_ef, flat_nf)
            h_dst = h_dst_flat.reshape(batch_size, num_candidates, -1)
        else:
            h_dst = self.encode_nodes(
                dst_delta_times, dst_lengths, dst_edge_feats, dst_node_feats
            )

        return self.edge_predictor(h_src, h_dst)

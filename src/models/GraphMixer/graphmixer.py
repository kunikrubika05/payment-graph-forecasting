"""GraphMixer model adapted for temporal link prediction on stream graphs.

Based on: "Do We Really Need Complicated Model Architectures For Temporal Networks?"
(Cong et al., ICLR 2023, arXiv:2302.11636)

This module contains the internal building blocks (unchanged from the original)
and a new top-level class GraphMixerTime that provides a stream-graph-compatible
API matching the EAGLE / GLFormer convention.

Architecture:
    LinkEncoder: MLP-Mixer over K most recent temporal edges.
        FeatEncoder maps (delta_time, edge_feats) → hidden_dim token.
        MixerBlocks mix tokens across both the sequence and channel dims.
        Masked mean-pooling produces a single hidden_dim node embedding.
    NodeEncoder (optional, node_feat_dim > 0):
        Projects own node features + mean of neighbor node features → hidden_dim.
    LinkClassifier (additive predictor):
        score = out(ReLU(fc_src(h_src) + fc_dst(h_dst)))

Top-level API (compatible with EAGLE / GLFormer evaluation code):
    encode_nodes(delta_times, lengths, edge_feats, node_feats, neighbor_node_feats)
        → [B, hidden_dim] or [B, 2*hidden_dim] when node_feat_dim > 0
    edge_predictor(h_src, h_dst) → [B] logits
    forward(src_delta_times, src_lengths, dst_delta_times, dst_lengths,
            src_edge_feats, dst_edge_feats, src_node_feats, dst_node_feats,
            src_neighbor_node_feats, dst_neighbor_node_feats) → [B] logits
"""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class FixedTimeEncoding(nn.Module):
    """Fixed (non-trainable) cosine time encoding with exponential frequencies.

    Encodes scalar timestamps as d-dimensional vectors:
        enc_i(t) = cos(t * omega_i),  omega_i = alpha^{-i/beta}
    with alpha = beta = sqrt(d).
    """

    def __init__(self, dim: int = 100):
        super().__init__()
        self.dim = dim
        alpha = math.sqrt(dim)
        beta = math.sqrt(dim)
        omega = torch.tensor(
            [alpha ** (-i / beta) for i in range(dim)], dtype=torch.float32
        )
        self.register_buffer("omega", omega)

    def forward(self, timestamps: torch.Tensor) -> torch.Tensor:
        """Encode timestamps.

        Args:
            timestamps: Tensor of any shape [...].

        Returns:
            [..., dim] cosine time encodings.
        """
        return torch.cos(timestamps.unsqueeze(-1).float() * self.omega)


class FeatEncoder(nn.Module):
    """Encodes a temporal edge as (time_encoding, edge_features) → hidden_dim."""

    def __init__(self, edge_feat_dim: int, time_dim: int = 100, out_dim: int = 100):
        super().__init__()
        self.time_enc = FixedTimeEncoding(time_dim)
        self.linear = nn.Linear(time_dim + edge_feat_dim, out_dim)

    def forward(self, edge_feats: torch.Tensor, edge_ts: torch.Tensor) -> torch.Tensor:
        """Encode edge features with temporal information.

        Args:
            edge_feats: [B, edge_feat_dim] raw edge features.
            edge_ts: [B] relative timestamps (query_time - edge_time).

        Returns:
            [B, out_dim] encoded features.
        """
        time_enc = self.time_enc(edge_ts)
        combined = torch.cat([time_enc, edge_feats], dim=-1)
        return self.linear(combined)


class FeedForward(nn.Module):
    """Two-layer MLP with GeLU activation and dropout."""

    def __init__(self, dim: int, expansion_factor: int = 2, dropout: float = 0.1):
        super().__init__()
        hidden = dim * expansion_factor
        self.net = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MixerBlock(nn.Module):
    """One MLP-Mixer block: token-mixing + channel-mixing with residuals."""

    def __init__(
        self,
        num_tokens: int,
        hidden_dim: int,
        token_expansion: int = 2,
        channel_expansion: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.token_norm = nn.LayerNorm(hidden_dim)
        self.token_ff = FeedForward(num_tokens, token_expansion, dropout)
        self.channel_norm = nn.LayerNorm(hidden_dim)
        self.channel_ff = FeedForward(hidden_dim, channel_expansion, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply token and channel mixing.

        Args:
            x: [B, num_tokens, hidden_dim]

        Returns:
            [B, num_tokens, hidden_dim]
        """
        residual = x
        x = self.token_norm(x)
        x = x.permute(0, 2, 1)
        x = self.token_ff(x)
        x = x.permute(0, 2, 1)
        x = x + residual

        residual = x
        x = self.channel_norm(x)
        x = self.channel_ff(x)
        return x + residual


class LinkEncoder(nn.Module):
    """MLP-Mixer based encoder for temporal link information.

    Takes K most-recent temporal edges per node (with delta-time encodings
    and optional edge features), applies MLP-Mixer blocks, and returns
    a fixed-size node embedding via masked mean-pooling.
    """

    def __init__(
        self,
        edge_feat_dim: int,
        time_dim: int = 100,
        hidden_dim: int = 100,
        num_neighbors: int = 20,
        num_mixer_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_neighbors = num_neighbors
        self.hidden_dim = hidden_dim

        self.feat_encoder = FeatEncoder(edge_feat_dim, time_dim, hidden_dim)
        self.mixer_blocks = nn.ModuleList([
            MixerBlock(num_tokens=num_neighbors, hidden_dim=hidden_dim, dropout=dropout)
            for _ in range(num_mixer_layers)
        ])
        self.final_norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        edge_feats: torch.Tensor,
        delta_times: torch.Tensor,
        lengths: torch.Tensor,
    ) -> torch.Tensor:
        """Encode temporal link information for a batch of nodes.

        Args:
            edge_feats: [B, K, edge_feat_dim] padded edge features.
            delta_times: [B, K] relative timestamps (query_time - edge_time).
            lengths: [B] actual number of valid neighbors.

        Returns:
            [B, hidden_dim] temporal link encoding.
        """
        B = edge_feats.shape[0]

        flat_feats = edge_feats.reshape(-1, edge_feats.shape[-1])
        flat_ts = delta_times.reshape(-1)
        encoded = self.feat_encoder(flat_feats, flat_ts)
        encoded = encoded.reshape(B, self.num_neighbors, self.hidden_dim)

        mask = torch.arange(self.num_neighbors, device=edge_feats.device).unsqueeze(0)
        mask = mask >= lengths.unsqueeze(1)
        encoded = encoded.masked_fill(mask.unsqueeze(-1), 0.0)

        for mixer in self.mixer_blocks:
            encoded = mixer(encoded)

        encoded = self.final_norm(encoded)
        return encoded.mean(dim=1)


class NodeEncoder(nn.Module):
    """Node encoder: own features + mean of 1-hop neighbor features."""

    def __init__(self, node_feat_dim: int, out_dim: int = 100):
        super().__init__()
        self.linear = nn.Linear(node_feat_dim, out_dim)

    def forward(
        self,
        node_feats: torch.Tensor,
        neighbor_feats: torch.Tensor,
        lengths: torch.Tensor,
    ) -> torch.Tensor:
        """Encode node information.

        Args:
            node_feats: [B, node_feat_dim] features of query nodes.
            neighbor_feats: [B, K, node_feat_dim] padded neighbor node features.
            lengths: [B] number of valid neighbors.

        Returns:
            [B, out_dim] node encodings.
        """
        mask = (
            torch.arange(neighbor_feats.shape[1], device=neighbor_feats.device)
            .unsqueeze(0)
            < lengths.unsqueeze(1)
        )
        neighbor_feats_masked = neighbor_feats * mask.unsqueeze(-1).float()
        safe_lengths = lengths.clamp(min=1).unsqueeze(-1).float()
        mean_neighbor = neighbor_feats_masked.sum(dim=1) / safe_lengths

        combined = node_feats + mean_neighbor
        return self.linear(combined)


class LinkClassifier(nn.Module):
    """Additive link predictor: score = out(ReLU(fc_src(h_src) + fc_dst(h_dst)))."""

    def __init__(self, input_dim: int, hidden_dim: int = 100):
        super().__init__()
        self.fc_src = nn.Linear(input_dim, hidden_dim)
        self.fc_dst = nn.Linear(input_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, 1)

    def forward(self, h_src: torch.Tensor, h_dst: torch.Tensor) -> torch.Tensor:
        """Predict link logit.

        Args:
            h_src: [B, input_dim]
            h_dst: [B, input_dim]

        Returns:
            [B] logits.
        """
        combined = F.relu(self.fc_src(h_src) + self.fc_dst(h_dst))
        return self.out(combined).squeeze(-1)


class GraphMixerTime(nn.Module):
    """GraphMixer adapted for stream-graph temporal link prediction.

    Provides a stream-graph-compatible API matching EAGLE / GLFormer:
        encode_nodes(...) → [B, repr_dim] node embeddings
        edge_predictor(h_src, h_dst) → [B] logits

    Args:
        time_dim: Dimension of the fixed cosine time encoding.
        hidden_dim: Hidden dimension for all components.
        num_neighbors: K most-recent neighbors sampled per node.
        num_mixer_layers: Number of MLP-Mixer blocks in LinkEncoder.
        dropout: Dropout rate.
        edge_feat_dim: Per-neighbor edge feature dimension (2 = btc+usd, 0 = time-only).
        node_feat_dim: Query-node own feature dimension. When 0 (default),
            NodeEncoder is disabled and repr_dim = hidden_dim.
            When > 0, NodeEncoder is used and repr_dim = 2 * hidden_dim.
    """

    def __init__(
        self,
        time_dim: int = 100,
        hidden_dim: int = 100,
        num_neighbors: int = 20,
        num_mixer_layers: int = 2,
        dropout: float = 0.1,
        edge_feat_dim: int = 2,
        node_feat_dim: int = 0,
    ):
        super().__init__()
        self.edge_feat_dim = edge_feat_dim
        self.node_feat_dim = node_feat_dim
        self.num_neighbors = num_neighbors
        self.hidden_dim = hidden_dim

        self.link_encoder = LinkEncoder(
            edge_feat_dim=edge_feat_dim,
            time_dim=time_dim,
            hidden_dim=hidden_dim,
            num_neighbors=num_neighbors,
            num_mixer_layers=num_mixer_layers,
            dropout=dropout,
        )

        if node_feat_dim > 0:
            self.node_encoder = NodeEncoder(node_feat_dim, hidden_dim)
        else:
            self.node_encoder = None

        repr_dim = hidden_dim * 2 if node_feat_dim > 0 else hidden_dim
        self.link_classifier = LinkClassifier(repr_dim, hidden_dim)

    def encode_nodes(
        self,
        delta_times: torch.Tensor,
        lengths: torch.Tensor,
        edge_feats: Optional[torch.Tensor] = None,
        node_feats: Optional[torch.Tensor] = None,
        neighbor_node_feats: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute node embeddings from temporal neighborhood.

        Args:
            delta_times: [B, K] float32, query_time - neighbor_time per slot.
            lengths: [B] int64, number of valid (non-padded) neighbors.
            edge_feats: [B, K, edge_feat_dim] or None (uses zeros when None).
            node_feats: [B, node_feat_dim] own features; required when
                node_feat_dim > 0.
            neighbor_node_feats: [B, K, node_feat_dim] neighbor features;
                required when node_feat_dim > 0.

        Returns:
            [B, hidden_dim] when node_feat_dim == 0, else [B, 2*hidden_dim].
        """
        B = delta_times.shape[0]

        if edge_feats is None:
            edge_feats = torch.zeros(
                B, self.num_neighbors, self.edge_feat_dim,
                dtype=torch.float32, device=delta_times.device,
            )

        h_link = self.link_encoder(edge_feats, delta_times, lengths)

        if self.node_encoder is not None:
            if neighbor_node_feats is None:
                neighbor_node_feats = torch.zeros(
                    B, self.num_neighbors, self.node_feat_dim,
                    dtype=torch.float32, device=delta_times.device,
                )
            if node_feats is None:
                node_feats = torch.zeros(
                    B, self.node_feat_dim,
                    dtype=torch.float32, device=delta_times.device,
                )
            h_node = self.node_encoder(node_feats, neighbor_node_feats, lengths)
            return torch.cat([h_link, h_node], dim=-1)

        return h_link

    def edge_predictor(
        self, h_src: torch.Tensor, h_dst: torch.Tensor
    ) -> torch.Tensor:
        """Score a batch of (src, dst) pairs.

        Args:
            h_src: [B, repr_dim]
            h_dst: [B, repr_dim]

        Returns:
            [B] logits.
        """
        return self.link_classifier(h_src, h_dst)

    def forward(
        self,
        src_delta_times: torch.Tensor,
        src_lengths: torch.Tensor,
        dst_delta_times: torch.Tensor,
        dst_lengths: torch.Tensor,
        src_edge_feats: Optional[torch.Tensor] = None,
        dst_edge_feats: Optional[torch.Tensor] = None,
        src_node_feats: Optional[torch.Tensor] = None,
        dst_node_feats: Optional[torch.Tensor] = None,
        src_neighbor_node_feats: Optional[torch.Tensor] = None,
        dst_neighbor_node_feats: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass for a batch of (src, dst) pairs.

        Args:
            src_delta_times: [B, K] source neighbor delta times.
            src_lengths: [B] source valid neighbor counts.
            dst_delta_times: [B, K] destination neighbor delta times.
            dst_lengths: [B] destination valid neighbor counts.
            src_edge_feats: [B, K, edge_feat_dim] or None.
            dst_edge_feats: [B, K, edge_feat_dim] or None.
            src_node_feats: [B, node_feat_dim] or None.
            dst_node_feats: [B, node_feat_dim] or None.
            src_neighbor_node_feats: [B, K, node_feat_dim] or None.
            dst_neighbor_node_feats: [B, K, node_feat_dim] or None.

        Returns:
            [B] logits.
        """
        h_src = self.encode_nodes(
            src_delta_times, src_lengths, src_edge_feats,
            src_node_feats, src_neighbor_node_feats,
        )
        h_dst = self.encode_nodes(
            dst_delta_times, dst_lengths, dst_edge_feats,
            dst_node_feats, dst_neighbor_node_feats,
        )
        return self.edge_predictor(h_src, h_dst)

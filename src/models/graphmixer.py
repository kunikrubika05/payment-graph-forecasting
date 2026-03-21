"""GraphMixer model for temporal link prediction.

Based on: "Do We Really Need Complicated Model Architectures For Temporal Networks?"
(Cong et al., ICLR 2023, arXiv:2302.11636)

Architecture:
    1. Link-encoder: MLP-Mixer over K most recent temporal edges per node
    2. Node-encoder: node features + mean-pooling of 1-hop neighbor features
    3. Link classifier: MLP over concatenated src/dst representations
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class FixedTimeEncoding(nn.Module):
    """Fixed (non-trainable) time encoding: cos(t * omega).

    Uses exponentially spaced frequencies omega = {alpha^(-(i-1)/beta)}_{i=1}^{d}.
    With alpha = beta = sqrt(d), d=100 by default.
    """

    def __init__(self, dim: int = 100):
        super().__init__()
        self.dim = dim
        alpha = math.sqrt(dim)
        beta = math.sqrt(dim)
        omega = torch.tensor(
            [alpha ** (-(i) / beta) for i in range(dim)],
            dtype=torch.float32,
        )
        self.register_buffer("omega", omega)

    def forward(self, timestamps: torch.Tensor) -> torch.Tensor:
        """Encode timestamps into d-dimensional vectors.

        Args:
            timestamps: Tensor of shape [...] with timestamp values.

        Returns:
            Tensor of shape [..., dim] with cosine time encodings.
        """
        t = timestamps.unsqueeze(-1).float()
        return torch.cos(t * self.omega)


class FeatEncoder(nn.Module):
    """Combines time encoding with raw edge features via linear projection."""

    def __init__(self, edge_feat_dim: int, time_dim: int = 100, out_dim: int = 100):
        super().__init__()
        self.time_enc = FixedTimeEncoding(time_dim)
        self.linear = nn.Linear(time_dim + edge_feat_dim, out_dim)

    def forward(
        self, edge_feats: torch.Tensor, edge_ts: torch.Tensor
    ) -> torch.Tensor:
        """Encode edge features with temporal information.

        Args:
            edge_feats: [batch_edges, edge_feat_dim] raw edge features.
            edge_ts: [batch_edges] relative timestamps (t_query - t_edge).

        Returns:
            [batch_edges, out_dim] encoded features.
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
            x: [batch, num_tokens, hidden_dim]

        Returns:
            [batch, num_tokens, hidden_dim]
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
        x = x + residual

        return x


class LinkEncoder(nn.Module):
    """MLP-Mixer based encoder for temporal link information.

    For each node, takes the K most recent edges (with time encodings and features),
    stacks them into a matrix, zero-pads to fixed length K, and applies
    MLP-Mixer blocks followed by mean-pooling.
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
            MixerBlock(
                num_tokens=num_neighbors,
                hidden_dim=hidden_dim,
                dropout=dropout,
            )
            for _ in range(num_mixer_layers)
        ])

        self.final_norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        edge_feats: torch.Tensor,
        edge_ts: torch.Tensor,
        lengths: torch.Tensor,
    ) -> torch.Tensor:
        """Encode temporal link information for a batch of nodes.

        Args:
            edge_feats: [batch, num_neighbors, edge_feat_dim] padded edge features.
            edge_ts: [batch, num_neighbors] relative timestamps (query_time - edge_time).
            lengths: [batch] actual number of valid neighbors per node.

        Returns:
            [batch, hidden_dim] temporal link encoding per node.
        """
        batch_size = edge_feats.shape[0]

        flat_feats = edge_feats.reshape(-1, edge_feats.shape[-1])
        flat_ts = edge_ts.reshape(-1)
        encoded = self.feat_encoder(flat_feats, flat_ts)
        encoded = encoded.reshape(batch_size, self.num_neighbors, self.hidden_dim)

        mask = torch.arange(self.num_neighbors, device=edge_feats.device).unsqueeze(0)
        mask = mask >= lengths.unsqueeze(1)
        encoded = encoded.masked_fill(mask.unsqueeze(-1), 0.0)

        for mixer in self.mixer_blocks:
            encoded = mixer(encoded)

        encoded = self.final_norm(encoded)
        output = encoded.mean(dim=1)

        return output


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
            node_feats: [batch, node_feat_dim] features of query nodes.
            neighbor_feats: [batch, num_neighbors, node_feat_dim] neighbor features.
            lengths: [batch] number of valid neighbors.

        Returns:
            [batch, out_dim] node encodings.
        """
        mask = torch.arange(
            neighbor_feats.shape[1], device=neighbor_feats.device
        ).unsqueeze(0) < lengths.unsqueeze(1)

        neighbor_feats_masked = neighbor_feats * mask.unsqueeze(-1).float()
        safe_lengths = lengths.clamp(min=1).unsqueeze(-1).float()
        mean_neighbor = neighbor_feats_masked.sum(dim=1) / safe_lengths

        combined = node_feats + mean_neighbor
        return self.linear(combined)


class LinkClassifier(nn.Module):
    """MLP-based link classifier: predicts link from src/dst representations."""

    def __init__(self, input_dim: int, hidden_dim: int = 100):
        super().__init__()
        self.fc_src = nn.Linear(input_dim, hidden_dim)
        self.fc_dst = nn.Linear(input_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, 1)

    def forward(
        self, h_src: torch.Tensor, h_dst: torch.Tensor
    ) -> torch.Tensor:
        """Predict link score.

        Args:
            h_src: [batch, input_dim] source node representations.
            h_dst: [batch, input_dim] or [batch, num_candidates, input_dim]
                   destination node representations.

        Returns:
            [batch] or [batch, num_candidates] logits.
        """
        src_proj = self.fc_src(h_src)

        if h_dst.dim() == 3:
            dst_proj = self.fc_dst(h_dst)
            src_proj = src_proj.unsqueeze(1)
            combined = F.relu(src_proj + dst_proj)
            return self.out(combined).squeeze(-1)
        else:
            dst_proj = self.fc_dst(h_dst)
            combined = F.relu(src_proj + dst_proj)
            return self.out(combined).squeeze(-1)


class GraphMixer(nn.Module):
    """Full GraphMixer model for temporal link prediction.

    Combines LinkEncoder (MLP-Mixer over temporal edges),
    NodeEncoder (features + neighbor pooling), and LinkClassifier.
    """

    def __init__(
        self,
        edge_feat_dim: int = 2,
        node_feat_dim: int = 25,
        time_dim: int = 100,
        hidden_dim: int = 100,
        num_neighbors: int = 20,
        num_mixer_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_neighbors = num_neighbors

        self.link_encoder = LinkEncoder(
            edge_feat_dim=edge_feat_dim,
            time_dim=time_dim,
            hidden_dim=hidden_dim,
            num_neighbors=num_neighbors,
            num_mixer_layers=num_mixer_layers,
            dropout=dropout,
        )

        self.node_encoder = NodeEncoder(
            node_feat_dim=node_feat_dim,
            out_dim=hidden_dim,
        )

        repr_dim = hidden_dim * 2
        self.link_classifier = LinkClassifier(repr_dim, hidden_dim)

    def encode_node(
        self,
        node_feats: torch.Tensor,
        neighbor_node_feats: torch.Tensor,
        neighbor_edge_feats: torch.Tensor,
        neighbor_rel_ts: torch.Tensor,
        neighbor_lengths: torch.Tensor,
    ) -> torch.Tensor:
        """Compute full node representation (link-encoding + node-encoding).

        Args:
            node_feats: [batch, node_feat_dim]
            neighbor_node_feats: [batch, K, node_feat_dim]
            neighbor_edge_feats: [batch, K, edge_feat_dim]
            neighbor_rel_ts: [batch, K] relative timestamps
            neighbor_lengths: [batch] valid neighbor counts

        Returns:
            [batch, hidden_dim * 2] node representations.
        """
        link_enc = self.link_encoder(
            neighbor_edge_feats, neighbor_rel_ts, neighbor_lengths
        )

        node_enc = self.node_encoder(
            node_feats, neighbor_node_feats, neighbor_lengths
        )

        return torch.cat([link_enc, node_enc], dim=-1)

    def forward(
        self,
        src_feats: torch.Tensor,
        src_neighbor_node_feats: torch.Tensor,
        src_neighbor_edge_feats: torch.Tensor,
        src_neighbor_rel_ts: torch.Tensor,
        src_neighbor_lengths: torch.Tensor,
        dst_feats: torch.Tensor,
        dst_neighbor_node_feats: torch.Tensor,
        dst_neighbor_edge_feats: torch.Tensor,
        dst_neighbor_rel_ts: torch.Tensor,
        dst_neighbor_lengths: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass for link prediction.

        Args:
            src_*: Source node features and neighbor data.
            dst_*: Destination node features and neighbor data.
                   dst can be [batch, ...] for single dst or
                   [batch, num_candidates, ...] for ranking.

        Returns:
            Logits tensor.
        """
        h_src = self.encode_node(
            src_feats, src_neighbor_node_feats,
            src_neighbor_edge_feats, src_neighbor_rel_ts,
            src_neighbor_lengths,
        )

        if dst_feats.dim() == 3:
            batch_size, num_candidates = dst_feats.shape[0], dst_feats.shape[1]
            dst_flat_feats = dst_feats.reshape(-1, dst_feats.shape[-1])
            dst_flat_nn_feats = dst_neighbor_node_feats.reshape(
                -1, dst_neighbor_node_feats.shape[-2], dst_neighbor_node_feats.shape[-1]
            )
            dst_flat_ee_feats = dst_neighbor_edge_feats.reshape(
                -1, dst_neighbor_edge_feats.shape[-2], dst_neighbor_edge_feats.shape[-1]
            )
            dst_flat_ts = dst_neighbor_rel_ts.reshape(-1, dst_neighbor_rel_ts.shape[-1])
            dst_flat_lens = dst_neighbor_lengths.reshape(-1)

            h_dst_flat = self.encode_node(
                dst_flat_feats, dst_flat_nn_feats,
                dst_flat_ee_feats, dst_flat_ts,
                dst_flat_lens,
            )
            h_dst = h_dst_flat.reshape(batch_size, num_candidates, -1)
            return self.link_classifier(h_src, h_dst)
        else:
            h_dst = self.encode_node(
                dst_feats, dst_neighbor_node_feats,
                dst_neighbor_edge_feats, dst_neighbor_rel_ts,
                dst_neighbor_lengths,
            )
            return self.link_classifier(h_src, h_dst)

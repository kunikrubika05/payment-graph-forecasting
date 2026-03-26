"""HyperEvent model for temporal link prediction via relative structural encoding.

Based on: "HyperEvent: A Strong Baseline for Dynamic Link Prediction via
Relative Structural Encoding" (Gao et al., 2025, arXiv:2507.11836).

Core idea: For each query event (u*, v*, t*), extract n_latest recent context
events from the adjacency tables of u* and v*. For each context event e = (u, v),
compute a 12-dimensional relational vector capturing 0-hop, 1-hop, and 2-hop
structural correlations between {u, v} and {u*, v*}. A Transformer encoder
discriminates whether the resulting hyper-event sequence is plausible.

Architecture:
    Input projection:  Linear(feat_dim=12, d_model)
    Positional enc:    Fixed sinusoidal (paper: max_len=512, pos_dim=64, but
                       we tie pos_dim to d_model for simplicity)
    Transformer enc:   L layers, H heads, GELU, d_model=64 (paper default)
    Classifier:        masked mean-pool → Linear(d_model, 1)

Relational vector (12-dim) per context event e = (u, v):
    [d0(u*,u), d0(u*,v), d0(v*,u), d0(v*,v),
     d1(u*,u), d1(u*,v), d1(v*,u), d1(v*,v),
     d2(u*,u), d2(u*,v), d2(v*,u), d2(v*,v)]

    d0(a,b) = fraction of adj[b] equal to a  (direct co-occurrence)
    d1(a,b) = |adj[a] ∩ adj[b]| / (|adj[a]| * |adj[b]|)  (1-hop overlap)
    d2(a,b) = |Ã_a ∩ Ã_b| / (|Ã_a| * |Ã_b|)  (2-hop overlap, top-√K per neighbor)
"""

import math
from typing import Optional

import torch
import torch.nn as nn


class SinusoidalPositionalEncoding(nn.Module):
    """Fixed sinusoidal positional encoding (Vaswani et al., 2017)."""

    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float)
            * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model % 2 == 0:
            pe[:, 1::2] = torch.cos(position * div_term)
        else:
            pe[:, 1::2] = torch.cos(position * div_term[:-1])
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to input.

        Args:
            x: [B, seq_len, d_model].

        Returns:
            [B, seq_len, d_model] with positional encoding added.
        """
        x = x + self.pe[: x.size(1)]
        return self.dropout(x)


class HyperEventModel(nn.Module):
    """HyperEvent: Transformer classifier over relative structural encoding sequences.

    Receives a sequence of 12-dimensional relational vectors for a (u*, v*, t*)
    query and outputs a link existence logit. Both the adjacency table management
    and relational vector computation live in hyperevent_train.py; this class
    is a pure PyTorch module.

    Args:
        feat_dim: Relational feature vector dimension (12 as per paper).
        d_model: Transformer hidden dimension (paper default: 64).
        n_heads: Number of attention heads (paper default: 4).
        n_layers: Number of Transformer encoder layers (paper default: 3).
        dropout: Dropout rate (paper default: 0.1).
        max_seq_len: Maximum sequence length for positional encoding (512).
    """

    def __init__(
        self,
        feat_dim: int = 12,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 3,
        dropout: float = 0.1,
        max_seq_len: int = 512,
    ):
        super().__init__()
        self.feat_dim = feat_dim
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers

        self.input_proj = nn.Linear(feat_dim, d_model)
        self.pos_enc = SinusoidalPositionalEncoding(d_model, max_seq_len, dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=False,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_model)
        self.classifier = nn.Linear(d_model, 1)

    def forward(
        self,
        rel_vecs: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute link existence logits.

        Args:
            rel_vecs: [B, seq_len, feat_dim] relational feature sequences.
            padding_mask: [B, seq_len] boolean tensor where True = padding
                position (ignored by Transformer). None means no masking.

        Returns:
            [B] logits (before sigmoid).
        """
        x = self.input_proj(rel_vecs)
        x = self.pos_enc(x)
        x = self.transformer(x, src_key_padding_mask=padding_mask)
        x = self.norm(x)

        if padding_mask is not None:
            valid = (~padding_mask).float().unsqueeze(-1)
            pooled = (x * valid).sum(dim=1) / valid.sum(dim=1).clamp(min=1.0)
        else:
            pooled = x.mean(dim=1)

        return self.classifier(pooled).squeeze(-1)

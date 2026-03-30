"""PairMLP model architecture.

A shallow MLP that scores a (src, dst) pair from 7 pairwise structural
features. Designed to be trained with BPR loss (see train.py).

Architecture:
    BatchNorm1d(7)
    → Linear(7, 64) → ReLU
    → Linear(64, 32) → ReLU
    → Linear(32, 1)

BatchNorm on the input layer handles the large range difference between
features (cn_uu ∈ [0, 1000+], jaccard ∈ [0, 1], log1p_pa ∈ [0, 15]).
"""

import torch
import torch.nn as nn

from src.models.pairwise_mlp.config import N_FEATURES


class PairMLP(nn.Module):
    """MLP ranker over pairwise structural features.

    Args:
        n_features:   Input dimensionality (default: N_FEATURES=7).
        hidden_dims:  Sequence of hidden layer widths (default: [64, 32]).
    """

    def __init__(
        self,
        n_features: int = N_FEATURES,
        hidden_dims: list[int] | None = None,
    ) -> None:
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [64, 32]

        self.bn_input = nn.BatchNorm1d(n_features)

        layers: list[nn.Module] = []
        in_dim = n_features
        for h in hidden_dims:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.ReLU(inplace=True))
            in_dim = h
        layers.append(nn.Linear(in_dim, 1))

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Score a batch of feature vectors.

        Args:
            x: Float32 tensor of shape (B, n_features).

        Returns:
            Score tensor of shape (B,).
        """
        return self.net(self.bn_input(x)).squeeze(-1)

    def score_candidates(
        self,
        cand_features: torch.Tensor,
    ) -> torch.Tensor:
        """Score a batch of candidate sets (used at eval time).

        Args:
            cand_features: Float32 tensor of shape (B, n_cand, n_features).

        Returns:
            Scores of shape (B, n_cand).
        """
        B, n_cand, _ = cand_features.shape
        flat = cand_features.view(B * n_cand, -1)
        scores_flat = self.forward(flat)
        return scores_flat.view(B, n_cand)


def build_model(
    hidden_dims: list[int] | None = None,
    n_features: int = N_FEATURES,
) -> PairMLP:
    """Construct a PairMLP from config parameters."""
    return PairMLP(n_features=n_features, hidden_dims=hidden_dims)

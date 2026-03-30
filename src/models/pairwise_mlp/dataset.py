"""PyTorch Dataset for precomputed pairwise features.

Loads the artifacts produced by precompute.py:
  pos_features.npy  — (N_train, 7)          float32
  neg_features.npy  — (N_train, K, 7)       float32

When active_feature_indices is provided, only those columns are exposed.
This enables ablation experiments without recomputing features.

Each item returned by __getitem__:
  pos_feat: (n_feat,)    float32 — features of positive (src, dst_true)
  neg_feat: (K, n_feat)  float32 — features of K negative (src, dst_neg_i)
"""

import os
from typing import List, Optional

import numpy as np
import torch
from torch.utils.data import Dataset

from src.models.pairwise_mlp.config import N_FEATURES


class PairwiseDataset(Dataset):
    """Dataset of precomputed pairwise features for BPR/BCE training.

    Args:
        pos_features:          Float32 array (N, N_ALL_FEATURES).
        neg_features:          Float32 array (N, K, N_ALL_FEATURES).
        active_feature_indices: Which columns to expose (None or [] = all).
    """

    def __init__(
        self,
        pos_features: np.ndarray,
        neg_features: np.ndarray,
        active_feature_indices: Optional[List[int]] = None,
    ) -> None:
        assert pos_features.ndim == 2
        assert neg_features.ndim == 3
        n = pos_features.shape[0]
        assert neg_features.shape[0] == n
        assert pos_features.shape[1] == neg_features.shape[2] == N_FEATURES
        assert pos_features.dtype == neg_features.dtype == np.float32

        if active_feature_indices:
            pos_features = pos_features[:, active_feature_indices]
            neg_features = neg_features[:, :, active_feature_indices]

        self.pos = torch.from_numpy(pos_features)   # (N, n_feat)
        self.neg = torch.from_numpy(neg_features)   # (N, K, n_feat)

    def __len__(self) -> int:
        return len(self.pos)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.pos[idx], self.neg[idx]


def load_dataset(
    precompute_dir: str,
    active_feature_indices: Optional[List[int]] = None,
) -> PairwiseDataset:
    """Load PairwiseDataset from a precompute output directory.

    Args:
        precompute_dir:        Path from precompute.py (contains .npy files).
        active_feature_indices: Columns to expose. None/[] = all.

    Returns:
        PairwiseDataset ready for DataLoader.
    """
    pos_path = os.path.join(precompute_dir, "pos_features.npy")
    neg_path = os.path.join(precompute_dir, "neg_features.npy")

    assert os.path.exists(pos_path), f"Missing {pos_path}"
    assert os.path.exists(neg_path), f"Missing {neg_path}"

    pos = np.load(pos_path)
    neg = np.load(neg_path)

    n_feat = len(active_feature_indices) if active_feature_indices else N_FEATURES
    print(f"  Loaded pos_features: {pos.shape}, neg_features: {neg.shape}")
    print(f"  Active features ({n_feat}): "
          f"{active_feature_indices if active_feature_indices else 'all'}")

    return PairwiseDataset(pos, neg, active_feature_indices)

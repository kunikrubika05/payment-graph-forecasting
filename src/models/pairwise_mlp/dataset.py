"""PyTorch Dataset for precomputed pairwise features.

Loads the artifacts produced by precompute.py:
  pos_features.npy  — (N_train, 7)          float32
  neg_features.npy  — (N_train, K, 7)       float32
  neg_dst.npy       — (N_train, K)           int64   (needed for node features)

When active_feature_indices is provided, only those columns are exposed.
This enables ablation experiments without recomputing features.

When node_features + local-index arrays are provided, the dataset appends
15-dim src and 15-dim dst node features to the pair feature vector at batch
time. The final feature vector becomes:
  [pair_features | node_src_features | node_dst_features]

Each item returned by __getitem__:
  pos_feat: (n_feat,)    float32
  neg_feat: (K, n_feat)  float32
where n_feat = n_pair_feat (+ 30 if node features enabled).
"""

import os
from typing import List, Optional

import numpy as np
import torch
from torch.utils.data import Dataset

from src.models.pairwise_mlp.config import N_FEATURES

N_NODE_FEATURES = 15


class PairwiseDataset(Dataset):
    """Dataset of precomputed pairwise features for BPR/BCE training.

    Args:
        pos_features:           Float32 array (N, N_PAIR_FEAT).
        neg_features:           Float32 array (N, K, N_PAIR_FEAT).
        active_feature_indices: Which pair-feature columns to expose. None/[] = all.
        node_features:          Optional float32 array (n_nodes_local, 15).
                                If provided, src+dst node features are appended.
        src_local:              Optional int64 array (N,) — local src indices.
        dst_local:              Optional int64 array (N,) — local pos-dst indices.
        neg_dst_local:          Optional int64 array (N, K) — local neg-dst indices.
    """

    def __init__(
        self,
        pos_features: np.ndarray,
        neg_features: np.ndarray,
        active_feature_indices: Optional[List[int]] = None,
        node_features: Optional[np.ndarray] = None,
        src_local: Optional[np.ndarray] = None,
        dst_local: Optional[np.ndarray] = None,
        neg_dst_local: Optional[np.ndarray] = None,
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

        self.pos = torch.from_numpy(pos_features)   # (N, n_pair_feat)
        self.neg = torch.from_numpy(neg_features)   # (N, K, n_pair_feat)
        self.neg_k = neg_features.shape[1]          # K (exposed for train.py)

        if node_features is not None:
            assert src_local is not None and dst_local is not None and neg_dst_local is not None, (
                "src_local, dst_local, neg_dst_local are required when node_features is provided"
            )
            assert node_features.ndim == 2 and node_features.shape[1] == N_NODE_FEATURES
            assert node_features.dtype == np.float32
            self.node_features = torch.from_numpy(node_features)   # (n_nodes, 15)
            self.src_local     = torch.from_numpy(src_local.astype(np.int64))       # (N,)
            self.dst_local     = torch.from_numpy(dst_local.astype(np.int64))       # (N,)
            self.neg_dst_local = torch.from_numpy(neg_dst_local.astype(np.int64))   # (N, K)
        else:
            self.node_features = None
            self.src_local     = None
            self.dst_local     = None
            self.neg_dst_local = None

    def __len__(self) -> int:
        return len(self.pos)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        pos_feat = self.pos[idx]   # (n_pair_feat,)
        neg_feat = self.neg[idx]   # (K, n_pair_feat)

        if self.node_features is not None:
            K = neg_feat.shape[0]
            src_feat     = self.node_features[self.src_local[idx]]      # (15,)
            dst_pos_feat = self.node_features[self.dst_local[idx]]      # (15,)
            dst_neg_feat = self.node_features[self.neg_dst_local[idx]]  # (K, 15)

            pos_feat = torch.cat([pos_feat, src_feat, dst_pos_feat])                      # (n_pair+30,)
            neg_feat = torch.cat([neg_feat, src_feat.unsqueeze(0).expand(K, -1),
                                  dst_neg_feat], dim=1)                                   # (K, n_pair+30)

        return pos_feat, neg_feat


def load_dataset(
    precompute_dir: str,
    active_feature_indices: Optional[List[int]] = None,
    node_features: Optional[np.ndarray] = None,
    src_local: Optional[np.ndarray] = None,
    dst_local: Optional[np.ndarray] = None,
    neg_dst_local: Optional[np.ndarray] = None,
) -> PairwiseDataset:
    """Load PairwiseDataset from a precompute output directory.

    Args:
        precompute_dir:         Path from precompute.py (contains .npy files).
        active_feature_indices: Columns to expose. None/[] = all.
        node_features:          Optional (n_nodes_local, 15) float32 array.
                                When provided, node features are appended to pair features.
        src_local:              Local src index per train edge (N,) int64.
        dst_local:              Local pos-dst index per train edge (N,) int64.
        neg_dst_local:          Local neg-dst indices (N, K) int64.

    Returns:
        PairwiseDataset ready for DataLoader.
    """
    pos_path = os.path.join(precompute_dir, "pos_features.npy")
    neg_path = os.path.join(precompute_dir, "neg_features.npy")

    assert os.path.exists(pos_path), f"Missing {pos_path}"
    assert os.path.exists(neg_path), f"Missing {neg_path}"

    pos = np.load(pos_path)
    neg = np.load(neg_path)

    n_pair = len(active_feature_indices) if active_feature_indices else N_FEATURES
    n_node = (N_NODE_FEATURES * 2) if node_features is not None else 0
    print(f"  Loaded pos_features: {pos.shape}, neg_features: {neg.shape}")
    print(f"  Active features ({n_pair}): "
          f"{active_feature_indices if active_feature_indices else 'all'}"
          + (f" + node_features({n_node})" if n_node else ""))

    return PairwiseDataset(
        pos, neg,
        active_feature_indices=active_feature_indices,
        node_features=node_features,
        src_local=src_local,
        dst_local=dst_local,
        neg_dst_local=neg_dst_local,
    )

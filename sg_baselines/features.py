"""Pair feature construction for stream graph baselines.

For each edge (src, dst), builds a feature vector of size 34:
- 15 node features for src
- 15 node features for dst
- CN_undirected, AA_undirected, CN_directed, AA_directed (4 pair features)

Node features are pre-computed from TRAIN edges only (no leakage).
Adjacency matrices are built from TRAIN edges only (no leakage).
Nodes unseen in train get zero features and zero CN/AA scores.
"""

import numpy as np
import pandas as pd
from scipy import sparse

from scripts.compute_stream_adjacency import compute_cn, compute_aa


N_NODE_FEATURES = 15
N_PAIR_FEATURES = 4
N_TOTAL_FEATURES = 2 * N_NODE_FEATURES + N_PAIR_FEATURES  # 34


def build_pair_features(
    src_global: np.ndarray,
    dst_global: np.ndarray,
    node_idx: np.ndarray,
    node_features: np.ndarray,
    node_mapping: np.ndarray,
    adj_directed: sparse.csr_matrix,
    adj_undirected: sparse.csr_matrix,
) -> np.ndarray:
    """Build feature vectors for (src, dst) pairs.

    Args:
        src_global: Source node global indices (int64).
        dst_global: Destination node global indices (int64).
        node_idx: Sorted global indices of nodes with features (from features parquet).
        node_features: Float32 array (n_active, 15) of node features.
        node_mapping: Local->global mapping from adjacency (same as node_idx).
        adj_directed: Directed CSR adjacency (local indices).
        adj_undirected: Undirected CSR adjacency (local indices).

    Returns:
        Float32 array of shape (n_pairs, 34).
    """
    n_pairs = len(src_global)
    assert len(dst_global) == n_pairs

    features = np.zeros((n_pairs, N_TOTAL_FEATURES), dtype=np.float32)

    src_feat = _lookup_node_features(src_global, node_idx, node_features)
    dst_feat = _lookup_node_features(dst_global, node_idx, node_features)
    features[:, :N_NODE_FEATURES] = src_feat
    features[:, N_NODE_FEATURES:2 * N_NODE_FEATURES] = dst_feat

    pair_feat = _compute_pair_features(
        src_global, dst_global, node_mapping, adj_directed, adj_undirected
    )
    features[:, 2 * N_NODE_FEATURES:] = pair_feat

    return features


def _lookup_node_features(
    global_indices: np.ndarray,
    node_idx: np.ndarray,
    node_features: np.ndarray,
) -> np.ndarray:
    """Look up node features by global index. Unknown nodes get zeros."""
    n = len(global_indices)
    n_feat = node_features.shape[1]
    result = np.zeros((n, n_feat), dtype=np.float32)

    positions = np.searchsorted(node_idx, global_indices)
    valid = (positions < len(node_idx)) & (node_idx[np.minimum(positions, len(node_idx) - 1)] == global_indices)

    if valid.any():
        result[valid] = node_features[positions[valid]]

    return result


def _compute_pair_features(
    src_global: np.ndarray,
    dst_global: np.ndarray,
    node_mapping: np.ndarray,
    adj_directed: sparse.csr_matrix,
    adj_undirected: sparse.csr_matrix,
) -> np.ndarray:
    """Compute CN and AA pair features (4 values per pair).

    Nodes not in node_mapping (unseen in train) get zero scores.
    """
    n = len(src_global)
    result = np.zeros((n, N_PAIR_FEATURES), dtype=np.float32)

    src_pos = np.searchsorted(node_mapping, src_global)
    dst_pos = np.searchsorted(node_mapping, dst_global)

    src_valid = (src_pos < len(node_mapping)) & (
        node_mapping[np.minimum(src_pos, len(node_mapping) - 1)] == src_global
    )
    dst_valid = (dst_pos < len(node_mapping)) & (
        node_mapping[np.minimum(dst_pos, len(node_mapping) - 1)] == dst_global
    )
    valid = src_valid & dst_valid

    if valid.any():
        src_local = src_pos[valid]
        dst_local = dst_pos[valid]

        result[valid, 0] = compute_cn(adj_undirected, src_local, dst_local)
        result[valid, 1] = compute_aa(adj_undirected, src_local, dst_local)
        result[valid, 2] = compute_cn(adj_directed, src_local, dst_local)
        result[valid, 3] = compute_aa(adj_directed, src_local, dst_local)

    return result


def get_feature_names() -> list[str]:
    """Return ordered list of feature names (34 total)."""
    from scripts.compute_stream_node_features import FEATURE_COLUMNS

    names = []
    for prefix in ["src", "dst"]:
        for col in FEATURE_COLUMNS:
            names.append(f"{prefix}_{col}")
    names.extend(["cn_undirected", "aa_undirected", "cn_directed", "aa_directed"])
    return names

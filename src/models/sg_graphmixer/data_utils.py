"""Data loading and preparation for GraphMixer on stream graph.

Converts stream graph data (loaded via sg_baselines.data) into TemporalEdgeData
and TemporalCSR structures expected by GraphMixer.

Key design decisions:
- Dense node mapping uses node_mapping from adjacency (= train nodes only)
- Graph is made undirected for neighbor sampling (both src->dst and dst->src)
- Node features are 15-dim static features from features parquet
- Edge features are [btc, usd] (2-dim)
- Nodes unseen in train get zero features and empty neighbor lists
"""

import os
import time

import numpy as np
import pandas as pd

from src.models.data_utils import TemporalEdgeData, TemporalCSR


def build_stream_graph_data(
    train_edges: pd.DataFrame,
    val_edges: pd.DataFrame,
    test_edges: pd.DataFrame,
    node_mapping: np.ndarray,
    node_features: np.ndarray,
    undirected: bool = True,
) -> tuple[TemporalEdgeData, np.ndarray, np.ndarray, np.ndarray]:
    """Convert stream graph splits into TemporalEdgeData with dense indices.

    Args:
        train_edges: Train DataFrame with [src_idx, dst_idx, timestamp, btc, usd].
        val_edges: Validation DataFrame.
        test_edges: Test DataFrame.
        node_mapping: Sorted global indices of active train nodes (local->global).
        node_features: Float32 array (n_active, 15) of node features.
        undirected: Whether to add reverse edges for neighbor sampling.

    Returns:
        (data, train_mask, val_mask, test_mask) where masks are boolean arrays
        over ALL edges in data (including reverse edges if undirected).
    """
    print("  Building TemporalEdgeData...", flush=True)
    t0 = time.time()

    global_to_dense = _build_global_to_dense(node_mapping)
    num_active = len(node_mapping)

    all_edges = pd.concat([train_edges, val_edges, test_edges], ignore_index=True)
    n_train = len(train_edges)
    n_val = len(val_edges)
    n_test = len(test_edges)

    src_global = all_edges["src_idx"].values.astype(np.int64)
    dst_global = all_edges["dst_idx"].values.astype(np.int64)
    timestamps = all_edges["timestamp"].values.astype(np.float64)
    btc = all_edges["btc"].values.astype(np.float32)
    usd = all_edges["usd"].values.astype(np.float32)

    src_dense = _map_to_dense(src_global, global_to_dense, num_active)
    dst_dense = _map_to_dense(dst_global, global_to_dense, num_active)

    btc = np.sign(btc) * np.log1p(np.abs(btc))
    usd = np.sign(usd) * np.log1p(np.abs(usd))
    edge_feats = np.stack([btc, usd], axis=1)

    split_labels = np.zeros(len(all_edges), dtype=np.int8)
    split_labels[:n_train] = 0
    split_labels[n_train:n_train + n_val] = 1
    split_labels[n_train + n_val:] = 2

    if undirected:
        src_dense_all = np.concatenate([src_dense, dst_dense])
        dst_dense_all = np.concatenate([dst_dense, src_dense])
        timestamps = np.concatenate([timestamps, timestamps])
        edge_feats = np.concatenate([edge_feats, edge_feats])
        split_labels = np.concatenate([split_labels, split_labels])
        src_dense = src_dense_all
        dst_dense = dst_dense_all

    sort_idx = np.argsort(timestamps, kind="stable")
    src_dense = src_dense[sort_idx]
    dst_dense = dst_dense[sort_idx]
    timestamps = timestamps[sort_idx]
    edge_feats = edge_feats[sort_idx]
    split_labels = split_labels[sort_idx]

    node_feats_dense = np.zeros((num_active, node_features.shape[1]), dtype=np.float32)
    node_feats_dense[:len(node_features)] = node_features

    node_id_map = {int(g): d for d, g in enumerate(node_mapping)}
    reverse_node_map = node_mapping.copy()

    data = TemporalEdgeData(
        src=src_dense,
        dst=dst_dense,
        timestamps=timestamps,
        edge_feats=edge_feats,
        node_feats=node_feats_dense,
        node_id_map=node_id_map,
        reverse_node_map=reverse_node_map,
    )

    train_mask = split_labels == 0
    val_mask = split_labels == 1
    test_mask = split_labels == 2

    n_total = len(src_dense)
    print(f"  TemporalEdgeData: {num_active:,} nodes, {n_total:,} edges "
          f"(train={train_mask.sum():,}, val={val_mask.sum():,}, test={test_mask.sum():,})"
          f" [{time.time() - t0:.1f}s]", flush=True)

    return data, train_mask, val_mask, test_mask


def filter_eval_edges_by_known_nodes(
    edges: pd.DataFrame,
    node_mapping: np.ndarray,
) -> pd.DataFrame:
    """Filter evaluation edges to only those with both src and dst in train nodes.

    Edges with unknown nodes get zero features and empty history, producing
    arbitrary scores and inflated/deflated MRR. Filtering ensures fair evaluation.
    """
    src = edges["src_idx"].values
    dst = edges["dst_idx"].values

    src_known = np.isin(src, node_mapping)
    dst_known = np.isin(dst, node_mapping)
    valid = src_known & dst_known

    n_before = len(edges)
    filtered = edges[valid].copy()
    n_after = len(filtered)

    if n_before > n_after:
        print(f"  Filtered eval edges: {n_before:,} -> {n_after:,} "
              f"({n_before - n_after:,} with unknown nodes removed)", flush=True)

    return filtered


def _build_global_to_dense(node_mapping: np.ndarray) -> dict[int, int]:
    """Build global->dense index mapping from sorted node_mapping."""
    return {int(g): i for i, g in enumerate(node_mapping)}


def _map_to_dense(
    global_indices: np.ndarray,
    global_to_dense: dict[int, int],
    num_active: int,
) -> np.ndarray:
    """Map global node indices to dense 0..N-1. Unknown nodes get index 0.

    Note: unknown nodes (not in train) will have zero features anyway.
    They only appear in val/test edges which are filtered before eval.
    """
    result = np.zeros(len(global_indices), dtype=np.int32)
    for i, g in enumerate(global_indices):
        result[i] = global_to_dense.get(int(g), 0)
    return result

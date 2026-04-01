"""Stream graph data utilities for EAGLE-Time.

Loads timestamped transaction streams from parquet files into TemporalData
(torch_geometric format) and converts them into TemporalEdgeData for use
with the EAGLE training pipeline.

Format of stream graph parquet files:
    src_idx   — source node index (int, dense 0..N-1 from node_mapping)
    dst_idx   — destination node index (int, dense 0..N-1)
    timestamp — Unix timestamp of transaction (int64)
    btc       — transaction value in BTC (float)
    usd       — transaction value in USD (float)

Re-exports from src.models.data_utils:
    TemporalEdgeData, TemporalCSR, build_temporal_csr,
    sample_neighbors_batch, generate_negatives_for_eval, chronological_split
"""

import logging
from typing import Tuple, Optional

import numpy as np
import pandas as pd
import torch

try:
    from torch_geometric.data import TemporalData
except ModuleNotFoundError:
    class TemporalData:  # type: ignore[override]
        """Small fallback used when torch_geometric is unavailable.

        It only implements the attribute container behavior required by tests and
        by the current stream-graph conversion helpers.
        """

        def __init__(self, src, dst, t, msg):
            self.src = src
            self.dst = dst
            self.t = t
            self.msg = msg

from src.models.data_utils import (
    TemporalEdgeData,
    TemporalCSR,
    build_temporal_csr,
    chronological_split,
    generate_negatives_for_eval,
    sample_neighbors_batch,
)

logger = logging.getLogger(__name__)

__all__ = [
    "load_temporal_data",
    "temporal_data_to_edge_data",
    "load_stream_graph_data",
    "load_node_features_for_eagle",
    "TemporalEdgeData",
    "TemporalCSR",
    "build_temporal_csr",
    "chronological_split",
    "generate_negatives_for_eval",
    "sample_neighbors_batch",
]


def load_temporal_data(
    parquet_path: str,
    fraction: Optional[float] = None,
) -> TemporalData:
    """Load a stream graph parquet file as TemporalData.

    Expected parquet columns: src_idx, dst_idx, timestamp, btc, usd.

    Args:
        parquet_path: Path to the stream graph parquet file.
        fraction: If set, take only first fraction of edges (period).

    Returns:
        TemporalData with fields src, dst, t (timestamps), msg ([btc, usd]).
    """
    df = pd.read_parquet(parquet_path)
    if fraction is not None and 0.0 < fraction < 1.0:
        period_end = int(len(df) * fraction)
        logger.info(
            "Applying fraction=%.2f: %d -> %d edges",
            fraction, len(df), period_end,
        )
        df = df.iloc[:period_end]
    return TemporalData(
        src=torch.tensor(df["src_idx"].values, dtype=torch.long),
        dst=torch.tensor(df["dst_idx"].values, dtype=torch.long),
        t=torch.tensor(df["timestamp"].values, dtype=torch.long),
        msg=torch.tensor(df[["btc", "usd"]].values, dtype=torch.float),
    )


def temporal_data_to_edge_data(
    data: TemporalData,
    undirected: bool = True,
    external_node_feats: Optional[np.ndarray] = None,
) -> TemporalEdgeData:
    """Convert TemporalData to TemporalEdgeData for EAGLE training.

    Sorts edges by timestamp and optionally adds reverse edges for
    bidirectional temporal neighbor lookup.

    Args:
        data: TemporalData loaded from a stream graph parquet file.
        undirected: If True, add reverse edges so each node accumulates
                    neighbors from both directions (matches GraphMixer protocol).
        external_node_feats: Pre-computed node features array of shape
                    [num_nodes, feat_dim]. If None, uses zeros (shape [N, 1]).

    Returns:
        TemporalEdgeData sorted by timestamp with identity node mapping.
    """
    src = data.src.numpy().astype(np.int32)
    dst = data.dst.numpy().astype(np.int32)
    timestamps = data.t.numpy().astype(np.float64)
    edge_feats = np.log1p(data.msg.numpy().astype(np.float32))

    order = np.argsort(timestamps, kind="stable")
    src = src[order]
    dst = dst[order]
    timestamps = timestamps[order]
    edge_feats = edge_feats[order]

    if undirected:
        orig_src = src.copy()
        orig_dst = dst.copy()
        orig_ts = timestamps.copy()
        orig_ef = edge_feats.copy()
        src = np.concatenate([orig_src, orig_dst])
        dst = np.concatenate([orig_dst, orig_src])
        timestamps = np.concatenate([orig_ts, orig_ts])
        edge_feats = np.concatenate([orig_ef, orig_ef])

        re_order = np.argsort(timestamps, kind="stable")
        src = src[re_order]
        dst = dst[re_order]
        timestamps = timestamps[re_order]
        edge_feats = edge_feats[re_order]

    unique_nodes = np.unique(np.concatenate([src, dst])).astype(np.int64)
    dense_idx = np.arange(len(unique_nodes), dtype=np.int32)
    src = np.searchsorted(unique_nodes, src).astype(np.int32)
    dst = np.searchsorted(unique_nodes, dst).astype(np.int32)

    num_nodes = int(len(unique_nodes))
    node_id_map = {int(orig): int(dense) for dense, orig in enumerate(unique_nodes)}
    reverse_node_map = unique_nodes.copy()

    if external_node_feats is not None:
        node_feats = external_node_feats[reverse_node_map]
        logger.info(
            "Using external node features: shape %s", node_feats.shape,
        )
    else:
        node_feats = np.zeros((num_nodes, 1), dtype=np.float32)

    logger.info(
        "TemporalEdgeData: %d nodes, %d edges (undirected=%s)",
        num_nodes,
        len(src),
        undirected,
    )

    return TemporalEdgeData(
        src=src,
        dst=dst,
        timestamps=timestamps,
        edge_feats=edge_feats,
        node_feats=node_feats,
        node_id_map=node_id_map,
        reverse_node_map=reverse_node_map,
    )


def load_node_features_for_eagle(
    features_path: str,
    num_nodes: int,
) -> np.ndarray:
    """Load sparse node features and expand to dense array.

    Reads features_{label}.parquet with columns [node_idx, feat1, ..., feat15].
    Returns dense float32 array of shape [num_nodes, 15] with zeros for
    nodes not present in the features file.

    Args:
        features_path: Path to features parquet (e.g. features_10.parquet).
        num_nodes: Total number of nodes in the graph (for dense array size).

    Returns:
        Dense float32 array [num_nodes, n_features].
    """
    df = pd.read_parquet(features_path)
    node_idx = df["node_idx"].values.astype(np.int64)
    feat_cols = [c for c in df.columns if c != "node_idx"]
    features = df[feat_cols].values.astype(np.float32)
    n_feats = features.shape[1]

    dense = np.zeros((num_nodes, n_feats), dtype=np.float32)
    valid = node_idx < num_nodes
    dense[node_idx[valid]] = features[valid]

    n_active = valid.sum()
    logger.info(
        "Loaded node features: %d/%d active nodes, %d features",
        n_active, num_nodes, n_feats,
    )
    return dense


def load_stream_graph_data(
    parquet_path: str,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    undirected: bool = True,
    fraction: Optional[float] = None,
    features_path: Optional[str] = None,
) -> Tuple[TemporalEdgeData, np.ndarray, np.ndarray, np.ndarray]:
    """Load a stream graph parquet file and split chronologically.

    Convenience wrapper that combines load_temporal_data,
    temporal_data_to_edge_data, and chronological_split.

    Args:
        parquet_path: Path to stream graph parquet file.
        train_ratio: Fraction of edges for training (default 0.70).
        val_ratio: Fraction of edges for validation (default 0.15).
                   Remaining edges form the test set.
        undirected: Add reverse edges for bidirectional lookup.
        fraction: If set, take only first fraction of edges (period).
        features_path: Path to features parquet. If None, node features
                       are set to zeros.

    Returns:
        Tuple of (data, train_mask, val_mask, test_mask) where masks
        are boolean arrays of shape [num_edges].
    """
    logger.info("Loading stream graph: %s", parquet_path)
    td = load_temporal_data(parquet_path, fraction=fraction)

    num_nodes_hint = int(max(td.src.max(), td.dst.max())) + 1
    external_feats = None
    if features_path is not None:
        external_feats = load_node_features_for_eagle(
            features_path, num_nodes_hint,
        )

    data = temporal_data_to_edge_data(
        td, undirected=undirected, external_node_feats=external_feats,
    )
    train_mask, val_mask, test_mask = chronological_split(
        data, train_ratio=train_ratio, val_ratio=val_ratio
    )
    return data, train_mask, val_mask, test_mask

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
from typing import Tuple

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import TemporalData

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
    "TemporalEdgeData",
    "TemporalCSR",
    "build_temporal_csr",
    "chronological_split",
    "generate_negatives_for_eval",
    "sample_neighbors_batch",
]


def load_temporal_data(parquet_path: str) -> TemporalData:
    """Load a stream graph parquet file as TemporalData.

    Expected parquet columns: src_idx, dst_idx, timestamp, btc, usd.

    Args:
        parquet_path: Path to the stream graph parquet file.

    Returns:
        TemporalData with fields src, dst, t (timestamps), msg ([btc, usd]).
    """
    df = pd.read_parquet(parquet_path)
    return TemporalData(
        src=torch.tensor(df["src_idx"].values, dtype=torch.long),
        dst=torch.tensor(df["dst_idx"].values, dtype=torch.long),
        t=torch.tensor(df["timestamp"].values, dtype=torch.long),
        msg=torch.tensor(df[["btc", "usd"]].values, dtype=torch.float),
    )


def temporal_data_to_edge_data(
    data: TemporalData,
    undirected: bool = True,
) -> TemporalEdgeData:
    """Convert TemporalData to TemporalEdgeData for EAGLE training.

    Sorts edges by timestamp and optionally adds reverse edges for
    bidirectional temporal neighbor lookup.

    EAGLE-Time does not use edge or node features — edge_feats stores
    [btc, usd] for completeness but is ignored during training. Node
    features are set to zeros (shape [num_nodes, 1]).

    Args:
        data: TemporalData loaded from a stream graph parquet file.
        undirected: If True, add reverse edges so each node accumulates
                    neighbors from both directions (matches GraphMixer protocol).

    Returns:
        TemporalEdgeData sorted by timestamp with identity node mapping.
    """
    src = data.src.numpy().astype(np.int32)
    dst = data.dst.numpy().astype(np.int32)
    timestamps = data.t.numpy().astype(np.float64)
    edge_feats = data.msg.numpy().astype(np.float32)

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

    num_nodes = int(max(src.max(), dst.max())) + 1
    node_id_map = {i: i for i in range(num_nodes)}
    reverse_node_map = np.arange(num_nodes, dtype=np.int64)
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


def load_stream_graph_data(
    parquet_path: str,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    undirected: bool = True,
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

    Returns:
        Tuple of (data, train_mask, val_mask, test_mask) where masks
        are boolean arrays of shape [num_edges].
    """
    logger.info("Loading stream graph: %s", parquet_path)
    td = load_temporal_data(parquet_path)
    data = temporal_data_to_edge_data(td, undirected=undirected)
    train_mask, val_mask, test_mask = chronological_split(
        data, train_ratio=train_ratio, val_ratio=val_ratio
    )
    return data, train_mask, val_mask, test_mask

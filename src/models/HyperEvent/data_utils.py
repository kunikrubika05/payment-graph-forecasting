"""Stream graph data utilities for HyperEvent.

Re-exports from src.models.EAGLE.data_utils so HyperEvent uses the same
stream graph loading and chronological splitting as EAGLE and GLFormer.

Format of stream graph parquet files:
    src_idx   — source node index (int, dense 0..N-1 from node_mapping)
    dst_idx   — destination node index (int, dense 0..N-1)
    timestamp — Unix timestamp of transaction (int64)
    btc       — transaction value in BTC (float)
    usd       — transaction value in USD (float)

HyperEvent does not use edge or node feature vectors during training.
The relational structural encoding is computed from adjacency table structure only.
"""

from src.models.stream_graph_data import (
    load_temporal_data,
    temporal_data_to_edge_data,
    load_stream_graph_data,
    TemporalEdgeData,
    TemporalCSR,
    build_temporal_csr,
    chronological_split,
    generate_negatives_for_eval,
    sample_neighbors_batch,
)
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

"""Stream graph data utilities for GraphMixer.

Re-exports all data utilities from EAGLE — the stream graph format
and data infrastructure is identical across all models.

Format of stream graph parquet files:
    src_idx   — source node index (int, dense 0..N-1)
    dst_idx   — destination node index (int, dense 0..N-1)
    timestamp — Unix timestamp of transaction (int64)
    btc       — transaction value in BTC (float)
    usd       — transaction value in USD (float)
"""

from src.models.EAGLE.data_utils import (
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

"""Neutral stream-graph data bridge shared across temporal models.

This module gives product-facing and cross-model code a model-agnostic import
path for the common stream-graph loaders and temporal CSR helpers.
"""

from src.models.EAGLE.data_utils import (
    TemporalCSR,
    TemporalEdgeData,
    build_temporal_csr,
    chronological_split,
    generate_negatives_for_eval,
    load_stream_graph_data,
    load_temporal_data,
    sample_neighbors_batch,
    temporal_data_to_edge_data,
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

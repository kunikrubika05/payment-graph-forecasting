"""CUDA-accelerated data utilities for GLFormer.

This compatibility module now delegates to the shared stream-graph bridge and
the unified sampler builder. It remains only so legacy GLFormer CUDA code can
keep importing the historical module path.
"""

from src.models.data_utils import build_unified_sampler
from src.models.stream_graph_data import (
    load_temporal_data,
    temporal_data_to_edge_data,
    load_stream_graph_data,
    TemporalEdgeData,
    chronological_split,
)
from src.models.temporal_graph_sampler import TemporalGraphSampler

__all__ = [
    "load_temporal_data",
    "temporal_data_to_edge_data",
    "load_stream_graph_data",
    "TemporalEdgeData",
    "chronological_split",
    "TemporalGraphSampler",
    "build_cuda_sampler",
]


def build_cuda_sampler(data: TemporalEdgeData, mask=None,
                       backend: str = "auto") -> TemporalGraphSampler:
    """Build a TemporalGraphSampler from TemporalEdgeData.

    Legacy GLFormer CUDA bridge over the shared sampler builder.
    """

    return build_unified_sampler(data, mask=mask, backend=backend)

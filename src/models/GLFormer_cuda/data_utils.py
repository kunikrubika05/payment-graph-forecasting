"""CUDA-accelerated data utilities for GLFormer.

Wraps TemporalGraphSampler (CUDA backend) to provide the same interface
as the standard GLFormer data pipeline, but with GPU-accelerated neighbor
sampling and feature gathering.

Also re-exports loading functions from EAGLE for data preparation.
"""

from src.models.EAGLE.data_utils import (
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

    Args:
        data: Temporal edge data with src, dst, timestamps, edge features.
        mask: Optional boolean mask to select a subset of edges
            (e.g. train-only edges).
        backend: Backend to use ('auto', 'cuda', 'cpp', 'python').

    Returns:
        TemporalGraphSampler configured with the requested backend.
    """
    import numpy as np

    if mask is not None:
        src = data.src[mask]
        dst = data.dst[mask]
        ts = data.timestamps[mask]
        eids = np.where(mask)[0].astype(np.int64)
    else:
        src = data.src
        dst = data.dst
        ts = data.timestamps
        eids = np.arange(len(src), dtype=np.int64)

    return TemporalGraphSampler(
        num_nodes=data.num_nodes,
        src=src,
        dst=dst,
        timestamps=ts,
        edge_ids=eids,
        node_feats=data.node_feats,
        edge_feats=data.edge_feats,
        backend=backend,
    )

"""Tests for shared temporal context helpers."""

from __future__ import annotations

import numpy as np

from payment_graph_forecasting.training.temporal_context import sample_node_contexts
from src.models.data_utils import TemporalEdgeData, TemporalCSR, sample_neighbors_batch


def _make_data():
    src = np.array([0, 0, 1, 2], dtype=np.int32)
    dst = np.array([1, 2, 2, 3], dtype=np.int32)
    timestamps = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64)
    edge_feats = np.arange(8, dtype=np.float32).reshape(4, 2)
    node_feats = np.arange(20, dtype=np.float32).reshape(5, 4)
    return TemporalEdgeData(
        src=src,
        dst=dst,
        timestamps=timestamps,
        edge_feats=edge_feats,
        node_feats=node_feats,
        node_id_map={i: i for i in range(5)},
        reverse_node_map=np.arange(5, dtype=np.int64),
    )


def test_sample_node_contexts_without_optional_features():
    data = _make_data()
    csr = TemporalCSR(data.num_nodes, data.src, data.dst, data.timestamps, np.arange(data.num_edges, dtype=np.int64))
    nodes = np.array([0], dtype=np.int32)
    query_ts = np.array([3.5], dtype=np.float64)
    context = sample_node_contexts(
        csr=csr,
        data=data,
        sample_neighbors_fn=sample_neighbors_batch,
        nodes=nodes,
        query_timestamps=query_ts,
        num_neighbors=3,
    )
    assert context.delta_times.shape == (1, 3)
    assert context.lengths.shape == (1,)
    assert context.edge_features is None
    assert context.node_features is None


def test_sample_node_contexts_with_features_and_zero_padding():
    data = _make_data()
    csr = TemporalCSR(data.num_nodes, data.src, data.dst, data.timestamps, np.arange(data.num_edges, dtype=np.int64))
    nodes = np.array([2], dtype=np.int32)
    query_ts = np.array([10.0], dtype=np.float64)
    context = sample_node_contexts(
        csr=csr,
        data=data,
        sample_neighbors_fn=sample_neighbors_batch,
        nodes=nodes,
        query_timestamps=query_ts,
        num_neighbors=4,
        use_edge_feats=True,
        use_node_feats=True,
        zero_pad_delta=True,
    )
    assert context.edge_features is not None
    assert context.node_features is not None
    assert context.edge_features.shape[:2] == (1, 4)
    assert context.node_features.shape == (1, 4)
    length = int(context.lengths[0])
    if length < 4:
        assert np.all(context.delta_times[0, length:] == 0.0)

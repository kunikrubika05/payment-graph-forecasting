"""Tests for shared temporal ranking helpers."""

from __future__ import annotations

import numpy as np
import torch

from payment_graph_forecasting.evaluation.temporal_ranking import (
    conservative_rank_from_scores,
    score_candidate_contexts,
)
from payment_graph_forecasting.training.temporal_context import NodeTemporalContext
from src.models.EAGLE.eagle import EAGLETime
from src.models.GLFormer.glformer import GLFormerTime


def _make_context(batch: int, k: int, edge_dim: int = 0, node_dim: int = 0) -> NodeTemporalContext:
    edge_features = None if edge_dim == 0 else np.zeros((batch, k, edge_dim), dtype=np.float32)
    node_features = None if node_dim == 0 else np.zeros((batch, node_dim), dtype=np.float32)
    return NodeTemporalContext(
        neighbor_ids=np.zeros((batch, k), dtype=np.int32),
        delta_times=np.zeros((batch, k), dtype=np.float32),
        lengths=np.full(batch, k, dtype=np.int32),
        edge_features=edge_features,
        node_features=node_features,
    )


def test_conservative_rank_from_scores():
    scores = np.array([0.5, 0.7, 0.4, 0.5], dtype=np.float32)
    assert conservative_rank_from_scores(scores) == 2.0


def test_score_candidate_contexts_with_eagle():
    model = EAGLETime(hidden_dim=16, num_neighbors=4)
    src_context = _make_context(batch=1, k=4)
    dst_context = _make_context(batch=3, k=4)
    scores = score_candidate_contexts(
        model=model,
        device=torch.device("cpu"),
        src_context=src_context,
        dst_context=dst_context,
        amp_enabled=False,
    )
    assert scores.shape == (3,)


def test_score_candidate_contexts_with_glformer_and_cooc():
    model = GLFormerTime(hidden_dim=16, num_neighbors=4, use_cooccurrence=True, cooc_dim=8)
    src_context = _make_context(batch=1, k=4, edge_dim=2)
    dst_context = _make_context(batch=2, k=4, edge_dim=2)
    scores = score_candidate_contexts(
        model=model,
        device=torch.device("cpu"),
        src_context=src_context,
        dst_context=dst_context,
        amp_enabled=False,
        cooc_counts=torch.zeros(2, dtype=torch.float32),
    )
    assert scores.shape == (2,)

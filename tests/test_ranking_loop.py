"""Tests for shared ranking loop helpers."""

from __future__ import annotations

import numpy as np

from payment_graph_forecasting.evaluation.ranking_loop import (
    choose_query_indices,
    evaluate_ranking_loop,
)


def test_choose_query_indices_without_subsampling():
    rng = np.random.RandomState(42)
    chosen = choose_query_indices(5, 10, rng=rng)
    np.testing.assert_array_equal(chosen, np.arange(5))


def test_choose_query_indices_with_subsampling_sorted():
    rng = np.random.RandomState(42)
    chosen = choose_query_indices(20, 5, rng=rng)
    assert len(chosen) == 5
    assert np.all(chosen[:-1] <= chosen[1:])


def test_evaluate_ranking_loop_aggregates_metrics():
    metrics, elapsed = evaluate_ranking_loop(
        np.array([0, 1, 2]),
        score_rank_fn=lambda idx: float(idx + 1),
    )
    assert metrics["n_queries"] == 3
    assert metrics["mrr"] > 0
    assert "hits@10" in metrics
    assert elapsed >= 0.0

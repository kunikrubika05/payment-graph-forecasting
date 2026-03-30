"""Shared loop helpers for ranking-based temporal link prediction evaluation."""

from __future__ import annotations

import time
from collections.abc import Callable

import numpy as np

from src.baselines.evaluation import compute_ranking_metrics


def choose_query_indices(
    n_total: int,
    max_queries: int,
    *,
    rng: np.random.RandomState,
) -> np.ndarray:
    """Return sorted evaluation indices, optionally subsampled."""

    if n_total > max_queries:
        chosen = rng.choice(n_total, size=max_queries, replace=False)
        chosen.sort()
        return chosen
    return np.arange(n_total)


def evaluate_ranking_loop(
    chosen_indices: np.ndarray,
    *,
    score_rank_fn: Callable[[int], float],
) -> tuple[dict[str, float], float]:
    """Run a per-query rank loop and aggregate ranking metrics."""

    all_ranks: list[float] = []
    start_time = time.time()
    for idx in chosen_indices:
        all_ranks.append(float(score_rank_fn(int(idx))))
    elapsed = time.time() - start_time
    ranks_arr = np.array(all_ranks, dtype=np.float64)
    metrics = compute_ranking_metrics(ranks_arr)
    metrics["eval_time_sec"] = elapsed
    metrics["edges_per_sec"] = len(chosen_indices) / elapsed if elapsed > 0 else 0.0
    return metrics, elapsed

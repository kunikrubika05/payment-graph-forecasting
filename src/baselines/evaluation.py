"""Evaluation metrics for link prediction and time series forecasting."""

import numpy as np
from typing import List, Optional, Dict


def compute_ranking_metrics(
    ranks: np.ndarray,
    k_values: Optional[List[int]] = None,
) -> Dict[str, float]:
    """Compute ranking metrics from per-query ranks (TGB-style protocol).

    Each rank represents the position (1-based) of the true destination
    within a per-source candidate set.

    Args:
        ranks: Array of 1-based ranks for each query (positive edge).
        k_values: List of K values for Hits@K. Defaults to [1, 3, 10].

    Returns:
        Dictionary with MRR, Hits@K for each K, and statistics.
    """
    if k_values is None:
        k_values = [1, 3, 10]

    if len(ranks) == 0:
        result = {"n_queries": 0, "mrr": float("nan")}
        for k in k_values:
            result[f"hits@{k}"] = float("nan")
        return result

    ranks = np.asarray(ranks, dtype=np.float64)

    metrics = {
        "n_queries": len(ranks),
        "mrr": float(np.mean(1.0 / ranks)),
        "mean_rank": float(np.mean(ranks)),
        "median_rank": float(np.median(ranks)),
    }

    for k in k_values:
        metrics[f"hits@{k}"] = float(np.mean(ranks <= k))

    return metrics


def compute_ts_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Compute time series forecasting metrics.

    Args:
        y_true: Actual values.
        y_pred: Predicted values.

    Returns:
        Dictionary with MAE, RMSE, MAPE, sMAPE.
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)

    errors = y_true - y_pred
    abs_errors = np.abs(errors)

    metrics = {
        "n_samples": len(y_true),
        "mae": float(np.mean(abs_errors)),
        "rmse": float(np.sqrt(np.mean(errors ** 2))),
    }

    nonzero_mask = y_true != 0
    if nonzero_mask.any():
        metrics["mape"] = float(np.mean(abs_errors[nonzero_mask] / np.abs(y_true[nonzero_mask])) * 100)
    else:
        metrics["mape"] = float("nan")

    denom = np.abs(y_true) + np.abs(y_pred)
    nonzero_denom = denom != 0
    if nonzero_denom.any():
        metrics["smape"] = float(
            np.mean(2 * abs_errors[nonzero_denom] / denom[nonzero_denom]) * 100
        )
    else:
        metrics["smape"] = float("nan")

    return metrics

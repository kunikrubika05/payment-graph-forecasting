"""Evaluation metrics for link prediction and time series forecasting."""

import numpy as np
from typing import List, Optional, Dict

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_recall_curve,
    f1_score,
)


def compute_classification_metrics(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    k_values: Optional[List[int]] = None,
) -> Dict[str, float]:
    """Compute classification metrics for link prediction.

    Args:
        y_true: Binary ground truth labels.
        y_proba: Predicted probabilities for positive class.
        k_values: List of K values for Precision@K / Recall@K.
            If None, uses [100, 500, 1000, 5000, 10000, n_positive].

    Returns:
        Dictionary with all computed metrics.
    """
    n_positive = int(y_true.sum())
    n_negative = len(y_true) - n_positive

    if k_values is None:
        k_values = [100, 500, 1000, 5000, 10000, n_positive]
    else:
        k_values = list(k_values) + [n_positive]

    k_values = sorted(set(k for k in k_values if 0 < k <= len(y_true)))

    metrics = {
        "n_samples": len(y_true),
        "n_positive": n_positive,
        "n_negative": n_negative,
        "pos_ratio": n_positive / len(y_true) if len(y_true) > 0 else 0.0,
    }

    if n_positive == 0 or n_negative == 0:
        metrics["roc_auc"] = float("nan")
        metrics["pr_auc"] = float("nan")
        metrics["f1_optimal"] = float("nan")
        metrics["optimal_threshold"] = float("nan")
        return metrics

    metrics["roc_auc"] = float(roc_auc_score(y_true, y_proba))
    metrics["pr_auc"] = float(average_precision_score(y_true, y_proba))

    precision_curve, recall_curve, thresholds = precision_recall_curve(y_true, y_proba)
    f1_scores = np.where(
        (precision_curve + recall_curve) > 0,
        2 * precision_curve * recall_curve / (precision_curve + recall_curve),
        0.0,
    )
    best_idx = np.argmax(f1_scores)
    metrics["f1_optimal"] = float(f1_scores[best_idx])
    metrics["optimal_threshold"] = float(
        thresholds[best_idx] if best_idx < len(thresholds) else 0.5
    )

    ranked_indices = np.argsort(-y_proba)
    sorted_labels = y_true[ranked_indices]

    for k in k_values:
        top_k_labels = sorted_labels[:k]
        tp_at_k = int(top_k_labels.sum())
        prec_k = tp_at_k / k
        rec_k = tp_at_k / n_positive if n_positive > 0 else 0.0
        suffix = f"@{k}" if k != n_positive else "@n_pos"
        metrics[f"precision{suffix}"] = float(prec_k)
        metrics[f"recall{suffix}"] = float(rec_k)

    reciprocal_ranks = []
    for i, idx in enumerate(ranked_indices):
        if y_true[idx] == 1:
            reciprocal_ranks.append(1.0 / (i + 1))
            break
    metrics["mrr"] = float(reciprocal_ranks[0]) if reciprocal_ranks else 0.0

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

"""Shared result-building helpers for library experiment runners."""

from __future__ import annotations

from typing import Any

import numpy as np


def build_dry_run_result(*, experiment: str, output_dir: str, **payload: Any) -> dict[str, Any]:
    """Build a standardized dry-run response payload."""

    return {
        "mode": "dry_run",
        "experiment": experiment,
        "output_dir": output_dir,
        **payload,
    }


def history_best_epoch(history: dict[str, list[float]]) -> int | None:
    """Return the 1-based best validation epoch from training history."""

    val_mrr = history.get("val_mrr", [])
    if not val_mrr:
        return None
    return int(np.argmax(val_mrr) + 1)


def history_best_val_mrr(history: dict[str, list[float]]) -> float | None:
    """Return the best validation MRR from training history."""

    val_mrr = history.get("val_mrr", [])
    if not val_mrr:
        return None
    return float(val_mrr[int(np.argmax(val_mrr))])


def build_final_results(
    *,
    experiment: str,
    model: str,
    history: dict[str, list[float]],
    timing: dict[str, float],
    args: dict[str, Any],
    device_info: dict[str, Any],
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a standardized final result payload."""

    result = {
        "experiment": experiment,
        "model": model,
        "best_val_mrr": history_best_val_mrr(history),
        "best_epoch": history_best_epoch(history),
        "total_epochs": len(history.get("train_loss", [])),
        "timing": timing,
        **device_info,
        "args": args,
    }
    if extra:
        result.update(extra)
    return result

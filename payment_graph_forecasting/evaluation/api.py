"""High-level evaluation API used by library-facing runners."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class EvaluationRunResult:
    """Structured evaluation result returned by model-specific wrappers."""

    metrics: dict[str, Any]


def evaluate_graphmixer_model(**kwargs: Any) -> EvaluationRunResult:
    """Evaluate GraphMixer through the stable library API."""

    from src.models.evaluate import evaluate_tgb_style

    metrics = evaluate_tgb_style(**kwargs)
    return EvaluationRunResult(metrics=metrics)


def evaluate_eagle_model(**kwargs: Any) -> EvaluationRunResult:
    """Evaluate EAGLE through the stable library API."""

    from src.models.EAGLE.eagle_evaluate import evaluate_tgb_style

    metrics = evaluate_tgb_style(**kwargs)
    return EvaluationRunResult(metrics=metrics)


def evaluate_glformer_model(**kwargs: Any) -> EvaluationRunResult:
    """Evaluate GLFormer through the stable library API."""

    from src.models.GLFormer.glformer_evaluate import evaluate_tgb_style

    metrics = evaluate_tgb_style(**kwargs)
    return EvaluationRunResult(metrics=metrics)

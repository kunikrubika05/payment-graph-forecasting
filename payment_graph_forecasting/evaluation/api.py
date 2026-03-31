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

    if kwargs.get("sampler") is not None:
        from src.models.GLFormer_cuda.glformer_evaluate import evaluate_tgb_style

        metrics = evaluate_tgb_style(**kwargs)
        return EvaluationRunResult(metrics=metrics)

    from src.models.GLFormer.glformer_evaluate import evaluate_tgb_style

    metrics = evaluate_tgb_style(**kwargs)
    return EvaluationRunResult(metrics=metrics)


def evaluate_hyperevent_model(**kwargs: Any) -> EvaluationRunResult:
    """Evaluate HyperEvent through the stable library API."""

    from src.models.HyperEvent.hyperevent_evaluate import evaluate_tgb_style

    metrics = evaluate_tgb_style(**kwargs)
    return EvaluationRunResult(metrics=metrics)


def evaluate_sg_graphmixer_model(**kwargs: Any) -> EvaluationRunResult:
    """Evaluate the sg-baselines-aligned GraphMixer through the stable library API."""

    from src.models.sg_graphmixer.evaluate import evaluate_tgb_style

    metrics = evaluate_tgb_style(**kwargs)
    return EvaluationRunResult(metrics=metrics)

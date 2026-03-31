"""High-level training API used by library-facing runners."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class TrainingRunResult:
    """Structured training result returned by model-specific wrappers."""

    model: Any
    history: dict[str, list[float]]


def train_graphmixer_model(**kwargs: Any) -> TrainingRunResult:
    """Train GraphMixer through the stable library API."""

    from src.models.train import train_graphmixer

    model, history = train_graphmixer(**kwargs)
    return TrainingRunResult(model=model, history=history)


def train_eagle_model(**kwargs: Any) -> TrainingRunResult:
    """Train EAGLE through the stable library API."""

    from src.models.EAGLE.eagle_train import train_eagle

    model, history = train_eagle(**kwargs)
    return TrainingRunResult(model=model, history=history)


def train_dygformer_model(**kwargs: Any) -> TrainingRunResult:
    """Train DyGFormer through the stable library API."""

    sampling_backend = kwargs.get("sampling_backend")
    if sampling_backend not in (None, "auto"):
        from src.models.DyGFormer.dygformer_cuda_train import train_dygformer_cuda

        model, history = train_dygformer_cuda(**kwargs)
        return TrainingRunResult(model=model, history=history)

    from src.models.DyGFormer.dygformer_train import train_dygformer

    legacy_kwargs = dict(kwargs)
    legacy_kwargs.pop("sampling_backend", None)
    model, history = train_dygformer(**legacy_kwargs)
    return TrainingRunResult(model=model, history=history)


def train_glformer_model(**kwargs: Any) -> TrainingRunResult:
    """Train GLFormer through the stable library API."""

    sampling_backend = kwargs.get("sampling_backend")
    if sampling_backend not in (None, "auto"):
        from src.models.GLFormer_cuda.glformer_train import train_glformer_cuda

        sampler_kwargs = dict(kwargs)
        sampler_kwargs.pop("adj", None)
        sampler_kwargs.pop("node_mapping", None)
        model, history = train_glformer_cuda(**sampler_kwargs)
        return TrainingRunResult(model=model, history=history)

    from src.models.GLFormer.glformer_train import train_glformer

    legacy_kwargs = dict(kwargs)
    legacy_kwargs.pop("sampling_backend", None)
    model, history = train_glformer(**legacy_kwargs)
    return TrainingRunResult(model=model, history=history)


def train_hyperevent_model(**kwargs: Any) -> TrainingRunResult:
    """Train HyperEvent through the stable library API."""

    from src.models.HyperEvent.hyperevent_train import train_hyperevent

    model, history = train_hyperevent(**kwargs)
    return TrainingRunResult(model=model, history=history)


def train_sg_graphmixer_model(**kwargs: Any) -> TrainingRunResult:
    """Train the sg-baselines-aligned GraphMixer through the stable library API."""

    from src.models.sg_graphmixer.train import train_graphmixer

    model, history = train_graphmixer(**kwargs)
    return TrainingRunResult(model=model, history=history)

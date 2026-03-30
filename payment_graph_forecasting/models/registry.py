"""Registry of model adapters exposed through the new package API."""

from __future__ import annotations

from payment_graph_forecasting.models.base import BaseModelAdapter
from payment_graph_forecasting.models.eagle import EAGLEAdapter
from payment_graph_forecasting.models.glformer import GLFormerAdapter
from payment_graph_forecasting.models.graphmixer import GraphMixerAdapter
from payment_graph_forecasting.models.pairwise_mlp import PairwiseMLPAdapter


MODEL_REGISTRY: dict[str, BaseModelAdapter] = {
    "graphmixer": GraphMixerAdapter(),
    "eagle": EAGLEAdapter(),
    "glformer": GLFormerAdapter(),
    "pairwise_mlp": PairwiseMLPAdapter(),
}


def get_model_adapter(model_name: str) -> BaseModelAdapter:
    """Return a registered model adapter by canonical name."""

    try:
        return MODEL_REGISTRY[model_name]
    except KeyError as exc:
        raise KeyError(f"Unknown model '{model_name}'. Known models: {sorted(MODEL_REGISTRY)}") from exc

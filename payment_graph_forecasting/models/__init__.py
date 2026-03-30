"""Public model API for the new library package."""

from payment_graph_forecasting.models.eagle import EAGLETime, TPPR
from payment_graph_forecasting.models.glformer import GLFormerTime
from payment_graph_forecasting.models.graphmixer import GraphMixer, GraphMixerTime
from payment_graph_forecasting.models.registry import MODEL_REGISTRY, get_model_adapter

__all__ = [
    "EAGLETime",
    "GLFormerTime",
    "GraphMixer",
    "GraphMixerTime",
    "MODEL_REGISTRY",
    "TPPR",
    "get_model_adapter",
]

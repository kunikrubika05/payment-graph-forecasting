"""Legacy compatibility wrapper for EAGLE exports.

This module preserves historical import paths used by tests and older scripts.
"""

from payment_graph_forecasting.models.eagle import (
    EAGLEEdgePredictor,
    EAGLEFeedForward,
    EAGLEMixerBlock,
    EAGLETime,
    EAGLETimeEncoder,
    EAGLETimeEncoding,
)

__all__ = [
    "EAGLETimeEncoding",
    "EAGLEFeedForward",
    "EAGLEMixerBlock",
    "EAGLETimeEncoder",
    "EAGLEEdgePredictor",
    "EAGLETime",
]

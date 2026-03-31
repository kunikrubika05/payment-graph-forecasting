"""Legacy compatibility wrapper for EAGLE exports.

This module preserves historical import paths used by tests and older scripts.
"""

# TODO(REFACTORING): remove legacy src.models.eagle adapter after callers migrate to payment_graph_forecasting.models.eagle.

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

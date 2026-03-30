"""Legacy compatibility wrapper for EAGLE training utilities."""

# TODO(REFACTORING): remove legacy src.models.eagle_train adapter after callers migrate to payment_graph_forecasting models/training API.

from src.models.EAGLE.eagle_train import *  # noqa: F401,F403

"""Legacy compatibility wrapper for TPPR exports."""

# TODO(REFACTORING): remove legacy src.models.tppr adapter after callers migrate to payment_graph_forecasting.models.eagle.

from payment_graph_forecasting.models.eagle import TPPR, get_forward_edge_mask

__all__ = ["TPPR", "get_forward_edge_mask"]

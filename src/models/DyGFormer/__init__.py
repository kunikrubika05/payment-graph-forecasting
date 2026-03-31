"""DyGFormer package for temporal link prediction on stream graphs.

Based on: "Towards Better Dynamic Graph Learning: New Architecture and
Unified Library" (Yu et al., NeurIPS 2023).
"""

from src.models.DyGFormer.dygformer import DyGFormerTime

__all__ = ["DyGFormerTime"]

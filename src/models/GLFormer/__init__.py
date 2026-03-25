"""GLFormer package for temporal link prediction on stream graphs.

Based on: "Global-Lens Transformers: Adaptive Token Mixing for Dynamic
Link Prediction" (Zou et al., AAAI 2026).
"""

from src.models.GLFormer.glformer import GLFormerTime

__all__ = ["GLFormerTime"]

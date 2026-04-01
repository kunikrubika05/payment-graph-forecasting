"""Package-facing graph metrics wrappers with optional C++/CUDA backends."""

from __future__ import annotations

from src.models.graph_metrics import CommonNeighbors, has_cpp, has_cuda

__all__ = ["CommonNeighbors", "has_cpp", "has_cuda"]

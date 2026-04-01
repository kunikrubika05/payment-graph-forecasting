"""Package-facing CUDA capability helpers.

YAML `sampling.backend` is the experiment-spec control surface for model
training paths. This module exposes the corresponding runtime capabilities as a
library-facing API so callers can inspect what CUDA-backed features are
available before selecting a backend.
"""

from __future__ import annotations

from dataclasses import dataclass

from payment_graph_forecasting.graph_metrics import has_cpp as has_graph_metrics_cpp
from payment_graph_forecasting.graph_metrics import has_cuda as has_graph_metrics_cuda
from payment_graph_forecasting.sampling import has_cpp as has_temporal_sampling_cpp_backend
from payment_graph_forecasting.sampling import has_cuda as has_temporal_sampling_cuda_backend


@dataclass(frozen=True, slots=True)
class CudaCapabilities:
    """Summary of optional C++/CUDA acceleration availability."""

    cuda_runtime_available: bool
    temporal_sampling_cpp_available: bool
    temporal_sampling_cuda_available: bool
    graph_metrics_cpp_available: bool
    graph_metrics_cuda_available: bool


def has_temporal_sampling_cuda() -> bool:
    """Return whether the CUDA temporal sampling backend is currently usable."""

    return bool(has_temporal_sampling_cuda_backend())


def has_temporal_sampling_cpp() -> bool:
    """Return whether the C++ temporal sampling backend is currently usable."""

    return bool(has_temporal_sampling_cpp_backend())


def describe_cuda_capabilities() -> CudaCapabilities:
    """Return the package-facing CUDA acceleration summary."""

    import torch

    return CudaCapabilities(
        cuda_runtime_available=bool(torch.cuda.is_available()),
        temporal_sampling_cpp_available=has_temporal_sampling_cpp(),
        temporal_sampling_cuda_available=has_temporal_sampling_cuda(),
        graph_metrics_cpp_available=bool(has_graph_metrics_cpp()),
        graph_metrics_cuda_available=bool(has_graph_metrics_cuda()),
    )


__all__ = [
    "CudaCapabilities",
    "describe_cuda_capabilities",
    "has_temporal_sampling_cpp",
    "has_temporal_sampling_cuda",
]

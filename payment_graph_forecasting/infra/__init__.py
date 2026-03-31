"""Shared infrastructure helpers for the library-facing package."""

from payment_graph_forecasting.infra.datasets import (
    ResolvedStreamGraphDataset,
    SUPPORTED_STREAM_GRAPH_SOURCES,
    resolve_stream_graph_dataset,
)
from payment_graph_forecasting.infra.data_access import (
    DEFAULT_DATA_CACHE_DIR,
    resolve_data_path,
)
from payment_graph_forecasting.infra.runtime import (
    RuntimeEnvironment,
    describe_runtime_environment,
    resolve_runtime_environment,
)

__all__ = [
    "ExtensionBuildSpec",
    "build_extension",
    "build_temporal_sampling_cpp",
    "build_temporal_sampling_cuda",
    "build_graph_metrics_cpp",
    "build_graph_metrics_cuda",
    "selected_build_specs",
    "DEFAULT_DATA_CACHE_DIR",
    "ResolvedStreamGraphDataset",
    "RuntimeEnvironment",
    "SUPPORTED_STREAM_GRAPH_SOURCES",
    "describe_runtime_environment",
    "resolve_stream_graph_dataset",
    "resolve_data_path",
    "resolve_runtime_environment",
]


def __getattr__(name: str):
    if name in {
        "ExtensionBuildSpec",
        "build_extension",
        "build_temporal_sampling_cpp",
        "build_temporal_sampling_cuda",
        "build_graph_metrics_cpp",
        "build_graph_metrics_cuda",
        "selected_build_specs",
    }:
        from payment_graph_forecasting.infra import extensions as _extensions

        return getattr(_extensions, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

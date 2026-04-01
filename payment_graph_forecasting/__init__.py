"""Top-level package for payment graph forecasting.

This package is the new library-facing API. Legacy ``src.*`` modules remain
available through compatibility adapters during the migration.
"""

from payment_graph_forecasting.analysis import StreamGraphAnalysisReport, analyze_stream_graph
from payment_graph_forecasting.config.base import ExperimentSpec
from payment_graph_forecasting.config.yaml_io import load_experiment_spec
from payment_graph_forecasting.cuda import CudaCapabilities, describe_cuda_capabilities
from payment_graph_forecasting.data import (
    StreamGraphDataset,
    StreamGraphSelection,
    open_stream_graph,
)
from payment_graph_forecasting.graph_metrics import CommonNeighbors
from payment_graph_forecasting.models.base import ModelExecutionPlan
from payment_graph_forecasting.models.registry import MODEL_REGISTRY, get_model_adapter
from payment_graph_forecasting.sampling.temporal import TemporalGraphSampler


def build_execution_plan(spec: ExperimentSpec) -> ModelExecutionPlan:
    """Lazy import to avoid pre-loading CLI modules during ``python -m``."""

    from payment_graph_forecasting.experiments.launcher import build_execution_plan as _build_execution_plan

    return _build_execution_plan(spec)


def launch_experiment(spec: ExperimentSpec):
    """Lazy import to avoid pre-loading CLI modules during ``python -m``."""

    from payment_graph_forecasting.experiments.launcher import launch_experiment as _launch_experiment

    return _launch_experiment(spec)


__all__ = [
    "ModelExecutionPlan",
    "ExperimentSpec",
    "MODEL_REGISTRY",
    "CommonNeighbors",
    "CudaCapabilities",
    "analyze_stream_graph",
    "build_execution_plan",
    "describe_cuda_capabilities",
    "get_model_adapter",
    "launch_experiment",
    "load_experiment_spec",
    "open_stream_graph",
    "StreamGraphAnalysisReport",
    "StreamGraphDataset",
    "StreamGraphSelection",
    "TemporalGraphSampler",
]

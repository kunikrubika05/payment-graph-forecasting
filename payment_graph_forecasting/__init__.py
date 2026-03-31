"""Top-level package for payment graph forecasting.

This package is the new library-facing API. Legacy ``src.*`` modules remain
available through compatibility adapters during the migration.
"""

from payment_graph_forecasting.config.base import ExperimentSpec
from payment_graph_forecasting.config.yaml_io import load_experiment_spec
from payment_graph_forecasting.models.base import ModelExecutionPlan
from payment_graph_forecasting.models.registry import MODEL_REGISTRY, get_model_adapter


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
    "build_execution_plan",
    "get_model_adapter",
    "launch_experiment",
    "load_experiment_spec",
]

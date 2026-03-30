"""Base interfaces for model adapters during the migration."""

from __future__ import annotations

from argparse import Namespace
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

from payment_graph_forecasting.config.base import ExperimentSpec


@dataclass(slots=True)
class LaunchResult:
    """Structured result returned by model adapters."""

    model_name: str
    experiment_name: str
    mode: str
    payload: dict[str, Any]


class BaseModelAdapter(ABC):
    """Abstract adapter between the new library API and legacy runners."""

    model_name: str

    @abstractmethod
    def run(self, spec: ExperimentSpec) -> LaunchResult:
        """Run or dry-run an experiment from a typed spec."""


class BaseRunnerAdapter(BaseModelAdapter):
    """Shared adapter for models backed by a runner taking argparse Namespace."""

    default_output_dir: str

    def common_runtime_kwargs(self, spec: ExperimentSpec) -> dict[str, Any]:
        """Build common runtime arguments shared by runner-backed adapters."""

        return {
            "output": spec.runtime.output_dir or self.default_output_dir,
            "dry_run": spec.runtime.dry_run,
            "no_amp": not spec.runtime.amp,
        }

    def common_training_kwargs(
        self,
        spec: ExperimentSpec,
        *,
        include_weight_decay: bool = True,
    ) -> dict[str, Any]:
        """Build common training arguments shared by runner-backed adapters."""

        kwargs = {
            "epochs": spec.training.epochs,
            "batch_size": spec.training.batch_size,
            "lr": spec.training.lr,
            "patience": spec.training.patience,
            "seed": spec.training.seed,
        }
        if include_weight_decay:
            kwargs["weight_decay"] = spec.training.weight_decay
        return kwargs

    @abstractmethod
    def build_runner_kwargs(self, spec: ExperimentSpec) -> dict[str, Any]:
        """Map a typed experiment spec to runner kwargs."""

    @abstractmethod
    def run_runner(self, args: Namespace) -> Any:
        """Execute the underlying runner."""

    def run(self, spec: ExperimentSpec) -> LaunchResult:
        args = Namespace(**self.build_runner_kwargs(spec))
        result = self.run_runner(args)
        return LaunchResult(
            self.model_name,
            spec.experiment.name,
            "dry_run" if spec.runtime.dry_run else "run",
            result if isinstance(result, dict) else {"result": result},
        )

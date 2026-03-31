"""Base interfaces for model adapters during the migration."""

from __future__ import annotations

from argparse import Namespace
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from payment_graph_forecasting.config.base import ExperimentSpec


@dataclass(slots=True)
class ModelExecutionPlan:
    """Canonical library-level execution contract for a model launch."""

    model_name: str
    experiment_name: str
    mode: str
    runner_name: str | None
    output_dir: str | None
    runner_kwargs: dict[str, Any]
    metadata: dict[str, Any] = field(default_factory=dict)

    def as_namespace(self) -> Namespace:
        """Materialize runner kwargs as an argparse namespace for legacy runners."""

        return Namespace(**self.runner_kwargs)


@dataclass(slots=True)
class LaunchResult:
    """Structured result returned by model adapters."""

    model_name: str
    experiment_name: str
    mode: str
    payload: dict[str, Any]


class BaseModelAdapter(ABC):
    """Abstract adapter between the new library API and concrete model backends."""

    model_name: str

    @abstractmethod
    def build_execution_plan(self, spec: ExperimentSpec) -> ModelExecutionPlan:
        """Build the canonical library-facing execution plan for a typed spec."""

    @abstractmethod
    def execute_plan(self, plan: ModelExecutionPlan) -> LaunchResult:
        """Execute a pre-built execution plan."""

    def run(self, spec: ExperimentSpec) -> LaunchResult:
        """Run or dry-run an experiment from a typed spec."""

        return self.execute_plan(self.build_execution_plan(spec))


class BaseRunnerAdapter(BaseModelAdapter):
    """Shared adapter for models backed by a runner taking argparse Namespace."""

    default_output_dir: str

    def common_runtime_kwargs(self, spec: ExperimentSpec) -> dict[str, Any]:
        """Build common runtime arguments shared by runner-backed adapters."""

        return {
            "output": spec.runtime.output_dir or self.default_output_dir,
            "device": spec.runtime.device,
            "dry_run": spec.runtime.dry_run,
            "no_amp": not spec.runtime.amp,
            "upload": spec.upload.enabled,
            "upload_backend": spec.upload.backend,
            "remote_dir": spec.upload.remote_dir,
            "token_env": spec.upload.token_env,
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

    def build_execution_plan(self, spec: ExperimentSpec) -> ModelExecutionPlan:
        """Translate a typed spec into the canonical execution plan."""

        runner_kwargs = self.build_runner_kwargs(spec)
        return ModelExecutionPlan(
            model_name=self.model_name,
            experiment_name=spec.experiment.name,
            mode="dry_run" if spec.runtime.dry_run else "run",
            runner_name=self.__class__.__name__,
            output_dir=runner_kwargs.get("output"),
            runner_kwargs=runner_kwargs,
        )

    def execute_plan(self, plan: ModelExecutionPlan) -> LaunchResult:
        # TODO(REFACTORING): remove the argparse bridge once all models execute from
        # the library-level execution plan directly instead of legacy runner CLIs.
        args = plan.as_namespace()
        result = self.run_runner(args)
        return LaunchResult(
            plan.model_name,
            plan.experiment_name,
            plan.mode,
            result if isinstance(result, dict) else {"result": result},
        )

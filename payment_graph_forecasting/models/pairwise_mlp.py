"""PairwiseMLP adapter wiring."""

from __future__ import annotations

from payment_graph_forecasting.config.base import ExperimentSpec
from payment_graph_forecasting.models.base import BaseRunnerAdapter
from payment_graph_forecasting.sampling.strategy import sampling_strategy_from_config
from src.models.pairwise_mlp.config import PairMLPConfig, PERIODS


class PairwiseMLPAdapter(BaseRunnerAdapter):
    """New API adapter over the legacy PairwiseMLP runner."""

    model_name = "pairwise_mlp"
    default_output_dir = "/tmp/pairmlp_results"

    def build_runner_kwargs(self, spec: ExperimentSpec) -> dict[str, object]:
        period_name = spec.data.period or "period_10"
        sampling = sampling_strategy_from_config(spec.sampling)
        return {
            "period": period_name,
            "data_dir": spec.data.data_dir or "/tmp/pairmlp_data",
            "precompute_dir": spec.model.get("precompute_dir", "/tmp/pairmlp_precompute"),
            "precompute_remote_dir": spec.model.get("precompute_remote_dir"),
            "exp_tag": spec.experiment.name,
            "loss": spec.model.get("loss", "bpr"),
            "features": list(spec.model.get("features", [])) or None,
            "feature_indices": spec.model.get("feature_indices"),
            "dropout": float(spec.training.dropout if spec.training.dropout is not None else 0.0),
            "n_negatives": sampling.total_negatives,
            **self.common_training_kwargs(spec),
            **self.common_runtime_kwargs(spec),
        }

    def run_runner(self, args):
        # TODO(REFACTORING): replace PairwiseMLP's legacy core pipeline with the unified trainer/evaluator stack.
        from payment_graph_forecasting.experiments.runners.pairwise_mlp import run_pairwise_mlp_experiment

        return run_pairwise_mlp_experiment(args)


__all__ = ["PairwiseMLPAdapter"]

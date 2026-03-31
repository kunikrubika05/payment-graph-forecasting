"""Stream-graph GraphMixer adapter wiring."""

from __future__ import annotations

from payment_graph_forecasting.config.base import ExperimentSpec
from payment_graph_forecasting.models.base import BaseRunnerAdapter
from sg_baselines.config import PERIODS
from src.models.graphmixer import GraphMixer

SGGraphMixer = GraphMixer


class SGGraphMixerAdapter(BaseRunnerAdapter):
    """Library adapter for the sg-baselines-aligned GraphMixer variant."""

    model_name = "sg_graphmixer"
    default_output_dir = "/tmp/sg_graphmixer_results"

    def build_runner_kwargs(self, spec: ExperimentSpec) -> dict[str, object]:
        period_name = spec.data.period if spec.data.period in PERIODS else "period_10"
        n_negatives = int(spec.model.get("n_negatives", spec.sampling.n_random_neg + spec.sampling.n_hist_neg))
        return {
            "period": period_name,
            "data_dir": spec.data.data_dir or "/tmp/sg_baselines_data",
            "upload": spec.upload.enabled,
            "token_env": spec.upload.token_env,
            "hidden_dim": spec.training.hidden_dim or 200,
            "num_neighbors": spec.sampling.num_neighbors,
            "num_mixer_layers": int(spec.model.get("num_mixer_layers", 1)),
            "lr": spec.training.lr,
            "weight_decay": spec.training.weight_decay,
            "dropout": spec.training.dropout if spec.training.dropout is not None else 0.2,
            "num_epochs": spec.training.epochs,
            "max_val_queries": int(spec.model.get("max_val_queries", 10_000)),
            "max_test_queries": int(spec.model.get("max_test_queries", 50_000)),
            "n_negatives": n_negatives,
            "train_ratio": spec.data.train_ratio,
            "val_ratio": spec.data.val_ratio,
            **self.common_runtime_kwargs(spec),
            "batch_size": spec.training.batch_size,
            "patience": spec.training.patience,
            "seed": spec.training.seed,
        }

    def run_runner(self, args):
        from payment_graph_forecasting.experiments.runners.sg_graphmixer import run_sg_graphmixer_experiment

        return run_sg_graphmixer_experiment(args)


__all__ = ["GraphMixer", "SGGraphMixer", "SGGraphMixerAdapter"]

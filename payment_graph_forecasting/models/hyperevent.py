"""HyperEvent exports and adapter wiring."""

from __future__ import annotations

from payment_graph_forecasting.config.base import ExperimentSpec
from payment_graph_forecasting.models.base import BaseRunnerAdapter
from payment_graph_forecasting.sampling.strategy import sampling_strategy_from_config
from src.models.HyperEvent.hyperevent import HyperEventModel


class HyperEventAdapter(BaseRunnerAdapter):
    """New API adapter over the migrated HyperEvent runner."""

    model_name = "hyperevent"
    default_output_dir = "/tmp/hyperevent_results"

    def build_runner_kwargs(self, spec: ExperimentSpec) -> dict[str, object]:
        sampling = sampling_strategy_from_config(spec.sampling)
        return {
            "data_source": spec.data.source,
            "raw_path": spec.data.raw_path,
            "raw_remote_path": spec.data.raw_remote_path,
            "data_extra": spec.data.extra,
            "parquet_path": spec.data.parquet_path,
            "parquet_remote_path": spec.data.parquet_remote_path,
            "fraction": spec.data.fraction,
            "data_backend": spec.data.download_backend,
            "data_cache_dir": spec.data.cache_dir,
            "data_token_env": spec.data.token_env,
            "train_ratio": spec.data.train_ratio,
            "val_ratio": spec.data.val_ratio,
            "n_neighbor": int(spec.model.get("n_neighbor", spec.sampling.num_neighbors)),
            "n_latest": int(spec.model.get("n_latest", 10)),
            "d_model": int(spec.model.get("d_model", spec.training.hidden_dim or 64)),
            "n_heads": int(spec.model.get("n_heads", 4)),
            "n_layers": int(spec.model.get("n_layers", 3)),
            "dropout": spec.training.dropout if spec.training.dropout is not None else 0.1,
            "n_hist_neg": sampling.n_hist_neg,
            "n_random_neg": sampling.n_random_neg,
            "max_val_edges": int(spec.model.get("max_val_edges", 5000)),
            **self.common_training_kwargs(spec),
            **self.common_runtime_kwargs(spec),
        }

    def run_runner(self, args):
        from payment_graph_forecasting.experiments.runners.hyperevent import run_hyperevent_experiment

        return run_hyperevent_experiment(args)


__all__ = ["HyperEventModel", "HyperEventAdapter"]

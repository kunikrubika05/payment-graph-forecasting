"""DyGFormer exports and adapter wiring."""

from __future__ import annotations

from payment_graph_forecasting.config.base import ExperimentSpec
from payment_graph_forecasting.models.base import BaseRunnerAdapter
from payment_graph_forecasting.sampling.strategy import sampling_strategy_from_config
from src.models.DyGFormer.dygformer import DyGFormerTime


class DyGFormerAdapter(BaseRunnerAdapter):
    """New API adapter over the migrated DyGFormer runner."""

    model_name = "dygformer"
    default_output_dir = "/tmp/dygformer_results"

    def build_runner_kwargs(self, spec: ExperimentSpec) -> dict[str, object]:
        sampling = sampling_strategy_from_config(spec.sampling)
        return {
            "data_source": spec.data.source,
            "raw_path": spec.data.raw_path,
            "raw_remote_path": spec.data.raw_remote_path,
            "data_extra": spec.data.extra,
            "parquet_path": spec.data.parquet_path,
            "parquet_remote_path": spec.data.parquet_remote_path,
            "features_path": spec.data.features_path,
            "features_remote_path": spec.data.features_remote_path,
            "node_mapping_path": spec.data.node_mapping_path,
            "node_mapping_remote_path": spec.data.node_mapping_remote_path,
            "data_backend": spec.data.download_backend,
            "data_cache_dir": spec.data.cache_dir,
            "data_token_env": spec.data.token_env,
            "train_ratio": spec.data.train_ratio,
            "val_ratio": spec.data.val_ratio,
            "fraction": spec.data.fraction,
            "patch_size": int(spec.model.get("patch_size", 1)),
            "time_dim": int(spec.model.get("time_dim", 100)),
            "aligned_dim": int(spec.model.get("aligned_dim", 50)),
            "num_transformer_layers": int(spec.model.get("num_transformer_layers", 2)),
            "num_attention_heads": int(spec.model.get("num_attention_heads", 2)),
            "cooc_dim": int(spec.model.get("cooc_dim", 50)),
            "output_dim": int(spec.model.get("output_dim", 172)),
            "num_neighbors": spec.sampling.num_neighbors,
            "dropout": spec.training.dropout if spec.training.dropout is not None else 0.1,
            "neg_per_positive": int(spec.model.get("neg_per_positive", max(1, sampling.total_negatives // 20 or 5))),
            "n_hist_neg": sampling.n_hist_neg,
            "n_random_neg": sampling.n_random_neg,
            "max_val_edges": int(spec.model.get("max_val_edges", 5000)),
            "max_test_edges": spec.model.get("max_test_edges"),
            "node_feat_dim": int(spec.model.get("node_feat_dim", 0)),
            "edge_feat_dim": int(spec.model.get("edge_feat_dim", 2)),
            **self.common_training_kwargs(spec),
            **self.common_runtime_kwargs(spec),
        }

    def run_runner(self, args):
        from payment_graph_forecasting.experiments.runners.dygformer import run_dygformer_experiment

        return run_dygformer_experiment(args)


__all__ = ["DyGFormerAdapter", "DyGFormerTime"]

"""GLFormer exports and adapter wiring."""

from __future__ import annotations

from payment_graph_forecasting.config.base import ExperimentSpec
from payment_graph_forecasting.models.base import BaseRunnerAdapter
from payment_graph_forecasting.sampling.strategy import sampling_strategy_from_config
from src.models.GLFormer.glformer import GLFormerTime


class GLFormerAdapter(BaseRunnerAdapter):
    """New API adapter over the migrated GLFormer runner."""

    model_name = "glformer"
    default_output_dir = "/tmp/glformer_results"

    def build_runner_kwargs(self, spec: ExperimentSpec) -> dict[str, object]:
        sampling = sampling_strategy_from_config(spec.sampling)
        return {
            "parquet_path": spec.data.parquet_path,
            "train_ratio": spec.data.train_ratio,
            "val_ratio": spec.data.val_ratio,
            "node_feats_path": spec.data.features_path,
            "adj_path": spec.model.get("adj_path"),
            "node_mapping_path": spec.data.node_mapping_path,
            "use_cooccurrence": bool(spec.model.get("use_cooccurrence", False)),
            "edge_feat_dim": int(spec.model.get("edge_feat_dim", 2)),
            "node_feat_dim": int(spec.model.get("node_feat_dim", 0)),
            "cooc_dim": int(spec.model.get("cooc_dim", 16)),
            "num_neighbors": spec.sampling.num_neighbors,
            "hidden_dim": spec.training.hidden_dim or 100,
            "num_glformer_layers": int(spec.model.get("num_glformer_layers", 2)),
            "channel_expansion": float(spec.model.get("channel_expansion", 4.0)),
            "dropout": spec.training.dropout if spec.training.dropout is not None else 0.1,
            "max_val_edges": int(spec.model.get("max_val_edges", 5000)),
            "max_test_edges": spec.model.get("max_test_edges"),
            "n_hist_neg": sampling.n_hist_neg,
            "n_random_neg": sampling.n_random_neg,
            "exp_name": spec.experiment.name,
            **self.common_training_kwargs(spec),
            **self.common_runtime_kwargs(spec),
        }

    def run_runner(self, args):
        from payment_graph_forecasting.experiments.runners.glformer import run_glformer_experiment

        return run_glformer_experiment(args)


__all__ = ["GLFormerTime", "GLFormerAdapter"]

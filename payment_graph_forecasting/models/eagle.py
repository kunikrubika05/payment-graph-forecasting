"""EAGLE exports and adapter wiring."""

from __future__ import annotations

from payment_graph_forecasting.config.base import ExperimentSpec
from payment_graph_forecasting.models.base import BaseRunnerAdapter
from payment_graph_forecasting.sampling.strategy import sampling_strategy_from_config
from src.models.EAGLE.eagle import (
    EAGLEEdgePredictor,
    EAGLEFeedForward,
    EAGLEMixerBlock,
    EAGLETime,
    EAGLETimeEncoder,
    EAGLETimeEncoding,
)
from src.models.EAGLE.tppr import TPPR, get_forward_edge_mask


class EAGLEAdapter(BaseRunnerAdapter):
    """New API adapter over the migrated EAGLE runner."""

    model_name = "eagle"
    default_output_dir = "/tmp/eagle_results"

    def build_runner_kwargs(self, spec: ExperimentSpec) -> dict[str, object]:
        sampling = sampling_strategy_from_config(spec.sampling)
        return {
            "parquet_path": spec.data.parquet_path,
            "features_path": spec.data.features_path,
            "node_mapping_path": spec.data.node_mapping_path,
            "fraction": spec.data.fraction,
            "train_ratio": spec.data.train_ratio,
            "val_ratio": spec.data.val_ratio,
            "node_feat_dim": int(spec.model.get("node_feat_dim", 0)),
            "edge_feat_dim": int(spec.model.get("edge_feat_dim", 2)),
            "num_neighbors": spec.sampling.num_neighbors,
            "hidden_dim": spec.training.hidden_dim or 100,
            "num_mixer_layers": int(spec.model.get("num_mixer_layers", 1)),
            "token_expansion": float(spec.model.get("token_expansion", 0.5)),
            "channel_expansion": float(spec.model.get("channel_expansion", 4.0)),
            "dropout": spec.training.dropout if spec.training.dropout is not None else 0.1,
            "max_val_edges": int(spec.model.get("max_val_edges", 5000)),
            "max_test_edges": int(spec.model.get("max_test_edges", 50_000)),
            "n_negatives": sampling.total_negatives,
            **self.common_training_kwargs(spec),
            **self.common_runtime_kwargs(spec),
        }

    def run_runner(self, args):
        from payment_graph_forecasting.experiments.runners.eagle import run_eagle_experiment

        return run_eagle_experiment(args)


__all__ = [
    "EAGLETimeEncoding",
    "EAGLEFeedForward",
    "EAGLEMixerBlock",
    "EAGLETimeEncoder",
    "EAGLEEdgePredictor",
    "EAGLETime",
    "TPPR",
    "get_forward_edge_mask",
    "EAGLEAdapter",
]

"""GraphMixer exports and adapter wiring."""

from __future__ import annotations

from payment_graph_forecasting.config.base import ExperimentSpec
from payment_graph_forecasting.models.base import BaseRunnerAdapter
from payment_graph_forecasting.sampling.strategy import sampling_strategy_from_config
from src.models.GraphMixer.graphmixer import GraphMixerTime
from src.models.graphmixer import (
    FeatEncoder,
    FeedForward,
    FixedTimeEncoding,
    GraphMixer,
    LinkClassifier,
    LinkEncoder,
    MixerBlock,
    NodeEncoder,
)


class GraphMixerAdapter(BaseRunnerAdapter):
    """New API adapter over the migrated GraphMixer runner."""

    model_name = "graphmixer"
    default_output_dir = "/tmp/graphmixer_results"

    def build_runner_kwargs(self, spec: ExperimentSpec) -> dict[str, object]:
        sampling = sampling_strategy_from_config(spec.sampling)
        return {
            "period": spec.data.period or "mature_2020q2",
            "window": spec.data.window or 7,
            "data_dir": spec.data.data_dir or "/tmp/graphmixer_data",
            "num_neighbors": spec.sampling.num_neighbors,
            "n_hist_neg": sampling.n_hist_neg,
            "n_random_neg": sampling.n_random_neg,
            "hidden_dim": spec.training.hidden_dim or 100,
            "num_mixer_layers": int(spec.model.get("num_mixer_layers", 2)),
            "dropout": spec.training.dropout if spec.training.dropout is not None else 0.1,
            "max_val_edges": int(spec.model.get("max_val_edges", 5000)),
            "eval_batch_size": int(spec.model.get("eval_batch_size", 32)),
            **self.common_training_kwargs(spec, include_weight_decay=False),
            **self.common_runtime_kwargs(spec),
        }

    def run_runner(self, args):
        from payment_graph_forecasting.experiments.runners.graphmixer import run_graphmixer_experiment

        return run_graphmixer_experiment(args)


__all__ = [
    "FixedTimeEncoding",
    "FeatEncoder",
    "FeedForward",
    "MixerBlock",
    "LinkEncoder",
    "NodeEncoder",
    "LinkClassifier",
    "GraphMixer",
    "GraphMixerTime",
    "GraphMixerAdapter",
]

from __future__ import annotations

from payment_graph_forecasting.config.base import (
    DataConfig,
    ExperimentMetadata,
    ExperimentSpec,
    RuntimeConfig,
    SamplingConfig,
    TrainingConfig,
)
from payment_graph_forecasting.models.base import BaseRunnerAdapter
from payment_graph_forecasting.models.eagle import EAGLEAdapter
from payment_graph_forecasting.models.glformer import GLFormerAdapter
from payment_graph_forecasting.models.graphmixer import GraphMixerAdapter


class DummyRunnerAdapter(BaseRunnerAdapter):
    model_name = "dummy"
    default_output_dir = "/tmp/dummy"

    def build_runner_kwargs(self, spec: ExperimentSpec) -> dict[str, object]:
        return {
            "foo": spec.experiment.name,
            **self.common_training_kwargs(spec),
            **self.common_runtime_kwargs(spec),
        }

    def run_runner(self, args):
        return vars(args)


def _make_spec(model: str) -> ExperimentSpec:
    return ExperimentSpec(
        experiment=ExperimentMetadata(name="exp_name", model=model),
        data=DataConfig(period="mature_2020q2", window=7, parquet_path="/tmp/data.parquet"),
        sampling=SamplingConfig(num_neighbors=17),
        training=TrainingConfig(
            epochs=3,
            batch_size=8,
            lr=0.002,
            patience=4,
            seed=5,
            weight_decay=0.01,
            hidden_dim=64,
            dropout=0.25,
        ),
        runtime=RuntimeConfig(amp=False, dry_run=True, output_dir=None),
        model={},
    )


def test_base_runner_adapter_builds_namespace_and_launch_result():
    adapter = DummyRunnerAdapter()
    result = adapter.run(_make_spec("dummy"))

    assert result.model_name == "dummy"
    assert result.mode == "dry_run"
    assert result.payload["foo"] == "exp_name"
    assert result.payload["epochs"] == 3
    assert result.payload["output"] == "/tmp/dummy"
    assert result.payload["no_amp"] is True


def test_graphmixer_adapter_uses_shared_defaults():
    payload = GraphMixerAdapter().build_runner_kwargs(_make_spec("graphmixer"))

    assert payload["output"] == "/tmp/graphmixer_results"
    assert payload["dry_run"] is True
    assert payload["epochs"] == 3
    assert payload["num_neighbors"] == 17


def test_eagle_adapter_uses_shared_defaults():
    payload = EAGLEAdapter().build_runner_kwargs(_make_spec("eagle"))

    assert payload["output"] == "/tmp/eagle_results"
    assert payload["no_amp"] is True
    assert payload["weight_decay"] == 0.01
    assert payload["parquet_path"] == "/tmp/data.parquet"


def test_glformer_adapter_uses_shared_defaults_and_experiment_name():
    payload = GLFormerAdapter().build_runner_kwargs(_make_spec("glformer"))

    assert payload["output"] == "/tmp/glformer_results"
    assert payload["exp_name"] == "exp_name"
    assert payload["weight_decay"] == 0.01
    assert payload["node_feats_path"] is None

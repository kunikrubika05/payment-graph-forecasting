from __future__ import annotations

from payment_graph_forecasting.config.base import (
    DataConfig,
    ExperimentMetadata,
    ExperimentSpec,
    RuntimeConfig,
    SamplingConfig,
    TrainingConfig,
    UploadConfig,
)
from payment_graph_forecasting.models.base import BaseRunnerAdapter
from payment_graph_forecasting.models.eagle import EAGLEAdapter
from payment_graph_forecasting.models.glformer import GLFormerAdapter
from payment_graph_forecasting.models.graphmixer import GraphMixerAdapter
from payment_graph_forecasting.models.hyperevent import HyperEventAdapter
from payment_graph_forecasting.models.sg_graphmixer import SGGraphMixerAdapter


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
        data=DataConfig(
            source="orbitaal_stream_graph",
            raw_path="/tmp/raw.csv",
            raw_remote_path="remote/raw.csv",
            period="mature_2020q2",
            window=7,
            parquet_path="/tmp/data.parquet",
            parquet_remote_path="remote/data.parquet",
            features_remote_path="remote/features.parquet",
            node_mapping_remote_path="remote/node_mapping.npy",
            cache_dir="/tmp/data_cache",
            token_env="DATA_TOKEN",
        ),
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
        upload=UploadConfig(enabled=True, remote_dir="remote/root", token_env="TOKEN_ENV"),
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
    assert result.payload["upload"] is True
    assert result.payload["remote_dir"] == "remote/root"


def test_base_runner_adapter_builds_execution_plan():
    adapter = DummyRunnerAdapter()
    plan = adapter.build_execution_plan(_make_spec("dummy"))

    assert plan.model_name == "dummy"
    assert plan.experiment_name == "exp_name"
    assert plan.mode == "dry_run"
    assert plan.runner_name == "DummyRunnerAdapter"
    assert plan.output_dir == "/tmp/dummy"
    assert plan.runner_kwargs["foo"] == "exp_name"
    assert plan.as_namespace().epochs == 3


def test_graphmixer_adapter_uses_shared_defaults():
    payload = GraphMixerAdapter().build_runner_kwargs(_make_spec("graphmixer"))

    assert payload["output"] == "/tmp/graphmixer_results"
    assert payload["device"] == "auto"
    assert payload["dry_run"] is True
    assert payload["epochs"] == 3
    assert payload["num_neighbors"] == 17
    assert payload["upload"] is True
    assert payload["remote_dir"] == "remote/root"


def test_eagle_adapter_uses_shared_defaults():
    payload = EAGLEAdapter().build_runner_kwargs(_make_spec("eagle"))

    assert payload["output"] == "/tmp/eagle_results"
    assert payload["device"] == "auto"
    assert payload["no_amp"] is True
    assert payload["weight_decay"] == 0.01
    assert payload["data_source"] == "orbitaal_stream_graph"
    assert payload["raw_path"] == "/tmp/raw.csv"
    assert payload["parquet_path"] == "/tmp/data.parquet"
    assert payload["parquet_remote_path"] == "remote/data.parquet"
    assert payload["data_cache_dir"] == "/tmp/data_cache"
    assert payload["data_token_env"] == "DATA_TOKEN"


def test_glformer_adapter_uses_shared_defaults_and_experiment_name():
    payload = GLFormerAdapter().build_runner_kwargs(_make_spec("glformer"))

    assert payload["output"] == "/tmp/glformer_results"
    assert payload["device"] == "auto"
    assert payload["sampling_backend"] == "auto"
    assert payload["exp_name"] == "exp_name"
    assert payload["weight_decay"] == 0.01
    assert payload["data_source"] == "orbitaal_stream_graph"
    assert payload["node_feats_path"] is None
    assert payload["node_feats_remote_path"] == "remote/features.parquet"
    assert payload["node_mapping_remote_path"] == "remote/node_mapping.npy"


def test_hyperevent_adapter_uses_shared_defaults():
    payload = HyperEventAdapter().build_runner_kwargs(_make_spec("hyperevent"))

    assert payload["output"] == "/tmp/hyperevent_results"
    assert payload["device"] == "auto"
    assert payload["data_source"] == "orbitaal_stream_graph"
    assert payload["parquet_path"] == "/tmp/data.parquet"
    assert payload["parquet_remote_path"] == "remote/data.parquet"
    assert payload["n_neighbor"] == 17
    assert payload["n_latest"] == 10


def test_sg_graphmixer_adapter_uses_stream_graph_defaults():
    payload = SGGraphMixerAdapter().build_runner_kwargs(_make_spec("sg_graphmixer"))

    assert payload["output"] == "/tmp/sg_graphmixer_results"
    assert payload["device"] == "auto"
    assert payload["period"] == "period_10"
    assert payload["num_neighbors"] == 17
    assert payload["n_negatives"] == 100

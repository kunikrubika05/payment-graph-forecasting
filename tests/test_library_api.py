"""Tests for the new payment_graph_forecasting library API."""

from __future__ import annotations

from pathlib import Path

import pytest

import payment_graph_forecasting as pgf
import payment_graph_forecasting.models as pgf_models
from payment_graph_forecasting.config.base import ExperimentSpec
from payment_graph_forecasting.config.yaml_io import load_experiment_spec
from payment_graph_forecasting.experiments.launcher import build_execution_plan, launch_experiment, main
from payment_graph_forecasting.models.registry import MODEL_REGISTRY, get_model_adapter
from payment_graph_forecasting.sampling.strategy import NegativeSamplingStrategy


def test_registry_contains_priority_models():
    assert {"graphmixer", "dygformer", "eagle", "glformer", "hyperevent", "sg_graphmixer", "pairwise_mlp"} <= set(MODEL_REGISTRY)


def test_get_model_adapter_returns_expected_adapter():
    adapter = get_model_adapter("graphmixer")
    assert adapter.model_name == "graphmixer"


def test_unknown_model_raises_key_error():
    with pytest.raises(KeyError):
        get_model_adapter("unknown_model")


def test_negative_sampling_strategy_total():
    strategy = NegativeSamplingStrategy(n_random_neg=40, n_hist_neg=60)
    assert strategy.total_negatives == 100


def test_load_experiment_spec_from_yaml_subset(tmp_path: Path):
    spec_path = tmp_path / "exp.yaml"
    spec_path.write_text(
        "\n".join(
            [
                "experiment:",
                "  name: gm_smoke",
                "  model: graphmixer",
                "data:",
                "  period: mature_2020q2",
                "  window: 7",
                "sampling:",
                "  num_neighbors: 24",
                "  n_random_neg: 60",
                "  n_hist_neg: 40",
                "training:",
                "  epochs: 3",
                "  batch_size: 16",
                "runtime:",
                "  dry_run: true",
            ]
        )
    )

    spec = load_experiment_spec(spec_path)
    assert isinstance(spec, ExperimentSpec)
    assert spec.experiment.name == "gm_smoke"
    assert spec.model_name == "graphmixer"
    assert spec.data.period == "mature_2020q2"
    assert spec.sampling.num_neighbors == 24
    assert spec.sampling.n_random_neg == 60
    assert spec.sampling.n_hist_neg == 40
    assert spec.runtime.dry_run is True


def test_load_experiment_spec_supports_remote_data_fields(tmp_path: Path):
    spec_path = tmp_path / "exp.yaml"
    spec_path.write_text(
        "\n".join(
            [
                "experiment:",
                "  name: remote_smoke",
                "  model: eagle",
                "data:",
                "  parquet_remote_path: orbitaal_processed/stream_graph/2020.parquet",
                "  features_remote_path: orbitaal_processed/stream_graph/features_25.parquet",
                "  node_mapping_remote_path: orbitaal_processed/stream_graph/node_mapping_25.npy",
                "  cache_dir: /tmp/pfg_data",
                "  token_env: DATA_TOKEN",
                "  download_backend: yadisk",
                "runtime:",
                "  dry_run: true",
            ]
        )
    )

    spec = load_experiment_spec(spec_path)
    assert spec.data.parquet_remote_path == "orbitaal_processed/stream_graph/2020.parquet"
    assert spec.data.features_remote_path == "orbitaal_processed/stream_graph/features_25.parquet"
    assert spec.data.node_mapping_remote_path == "orbitaal_processed/stream_graph/node_mapping_25.npy"
    assert spec.data.cache_dir == "/tmp/pfg_data"
    assert spec.data.token_env == "DATA_TOKEN"


def test_load_experiment_spec_supports_dataset_source_fields(tmp_path: Path):
    spec_path = tmp_path / "exp.yaml"
    spec_path.write_text(
        "\n".join(
            [
                "experiment:",
                "  name: wiki_smoke",
                "  model: hyperevent",
                "data:",
                "  source: jodie_csv",
                "  raw_path: /tmp/wikipedia.csv",
                "  cache_dir: /tmp/pfg_cache",
                "runtime:",
                "  dry_run: true",
            ]
        )
    )

    spec = load_experiment_spec(spec_path)
    assert spec.data.source == "jodie_csv"
    assert spec.data.raw_path == "/tmp/wikipedia.csv"
    assert spec.data.cache_dir == "/tmp/pfg_cache"


def test_launch_experiment_dry_run(tmp_path: Path):
    spec_path = tmp_path / "exp.yaml"
    spec_path.write_text(
        "\n".join(
            [
                "experiment:",
                "  name: eagle_smoke",
                "  model: eagle",
                "data:",
                "  parquet_path: /tmp/stream.parquet",
                "runtime:",
                "  dry_run: true",
            ]
        )
    )
    spec = load_experiment_spec(spec_path)
    result = launch_experiment(spec)
    assert result.mode == "dry_run"
    assert result.model_name == "eagle"
    assert result.payload["parquet_path"] == "/tmp/stream.parquet"


def test_build_execution_plan_exposes_canonical_model_contract(tmp_path: Path):
    spec_path = tmp_path / "exp.yaml"
    spec_path.write_text(
        "\n".join(
            [
                "experiment:",
                "  name: graphmixer_plan",
                "  model: graphmixer",
                "runtime:",
                "  device: cpu",
                "  dry_run: true",
            ]
        )
    )

    spec = load_experiment_spec(spec_path)
    plan = build_execution_plan(spec)
    assert plan.model_name == "graphmixer"
    assert plan.experiment_name == "graphmixer_plan"
    assert plan.mode == "dry_run"
    assert plan.output_dir == "/tmp/graphmixer_results"
    assert plan.runner_kwargs["device"] == "cpu"
    assert plan.runner_kwargs["dry_run"] is True


def test_dygformer_execution_plan_carries_dropout(tmp_path: Path):
    spec_path = tmp_path / "exp.yaml"
    spec_path.write_text(
        "\n".join(
            [
                "experiment:",
                "  name: dygformer_plan",
                "  model: dygformer",
                "training:",
                "  dropout: 0.25",
                "runtime:",
                "  dry_run: true",
            ]
        )
    )

    spec = load_experiment_spec(spec_path)
    plan = build_execution_plan(spec)
    assert plan.runner_kwargs["dropout"] == 0.25


def test_launcher_main_dry_run(tmp_path: Path):
    spec_path = tmp_path / "exp.yaml"
    spec_path.write_text(
        "\n".join(
            [
                "experiment:",
                "  name: glformer_smoke",
                "  model: glformer",
                "runtime:",
                "  dry_run: true",
            ]
        )
    )
    assert main([str(spec_path), "--dry-run"]) == 0


def test_repository_example_spec_loads():
    spec = load_experiment_spec(Path("exps/examples/graphmixer_library.yaml"))
    assert spec.experiment.name == "graphmixer_library_smoke"
    assert spec.model_name == "graphmixer"
    assert spec.runtime.dry_run is True


@pytest.mark.parametrize(
    ("path", "model_name"),
    [
        ("exps/examples/graphmixer_library.yaml", "graphmixer"),
        ("exps/examples/dygformer_library.yaml", "dygformer"),
        ("exps/examples/sg_graphmixer_library.yaml", "sg_graphmixer"),
        ("exps/examples/eagle_library.yaml", "eagle"),
        ("exps/examples/glformer_library.yaml", "glformer"),
        ("exps/examples/hyperevent_library.yaml", "hyperevent"),
        ("exps/examples/pairwise_mlp_library.yaml", "pairwise_mlp"),
    ],
)
def test_repository_example_specs_launch_in_dry_run(path: str, model_name: str):
    spec = load_experiment_spec(Path(path))
    result = launch_experiment(spec)
    assert result.mode == "dry_run"
    assert result.model_name == model_name


def test_top_level_package_exports_execution_plan_api():
    assert hasattr(pgf, "build_execution_plan")
    assert hasattr(pgf, "ModelExecutionPlan")


def test_models_package_exports_sg_graphmixer_variant():
    assert hasattr(pgf_models, "GraphMixer")
    assert hasattr(pgf_models, "SGGraphMixer")

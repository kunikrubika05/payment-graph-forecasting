"""Tests for the new payment_graph_forecasting library API."""

from __future__ import annotations

from pathlib import Path

import pytest

from payment_graph_forecasting.config.base import ExperimentSpec
from payment_graph_forecasting.config.yaml_io import load_experiment_spec
from payment_graph_forecasting.experiments.launcher import launch_experiment, main
from payment_graph_forecasting.models.registry import MODEL_REGISTRY, get_model_adapter
from payment_graph_forecasting.sampling.strategy import NegativeSamplingStrategy


def test_registry_contains_priority_models():
    assert {"graphmixer", "eagle", "glformer", "pairwise_mlp"} <= set(MODEL_REGISTRY)


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

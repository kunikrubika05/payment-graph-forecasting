from __future__ import annotations

from argparse import Namespace

from payment_graph_forecasting.config.base import (
    DataConfig,
    ExperimentMetadata,
    ExperimentSpec,
    RuntimeConfig,
    SamplingConfig,
    TrainingConfig,
    UploadConfig,
)
from payment_graph_forecasting.config.yaml_io import load_experiment_spec, save_experiment_spec
from payment_graph_forecasting.experiments.hpo_artifacts import (
    build_best_hpo_spec,
    write_best_training_artifacts,
)


def test_save_experiment_spec_roundtrip(tmp_path):
    spec = ExperimentSpec(
        experiment=ExperimentMetadata(
            name="roundtrip",
            model="eagle",
            tags=["test", "yaml"],
        ),
        data=DataConfig(
            source="orbitaal_stream_graph",
            raw_path="/tmp/raw.csv",
            raw_remote_path="remote/raw.csv",
            parquet_path="/tmp/stream.parquet",
            parquet_remote_path="remote/stream.parquet",
            features_path="/tmp/features.parquet",
            features_remote_path="remote/features.parquet",
            node_mapping_remote_path="remote/node_mapping.npy",
            cache_dir="/tmp/cache",
            token_env="DATA_TOKEN",
            train_ratio=0.8,
            val_ratio=0.1,
        ),
        sampling=SamplingConfig(
            strategy="tgb_mixed",
            num_neighbors=15,
            n_random_neg=70,
            n_hist_neg=30,
            backend="cpp",
        ),
        training=TrainingConfig(
            epochs=12,
            batch_size=64,
            lr=0.001,
            patience=5,
            seed=7,
            weight_decay=1e-4,
            hidden_dim=128,
            dropout=0.2,
        ),
        runtime=RuntimeConfig(
            device="cpu",
            amp=False,
            dry_run=True,
            output_dir="/tmp/out",
        ),
        upload=UploadConfig(
            enabled=True,
            backend="yadisk",
            remote_dir="remote/path",
            token_env="TOKEN_ENV",
        ),
        model={"token_expansion": 0.5},
    )

    path = save_experiment_spec(spec, tmp_path / "spec.yaml")
    loaded = load_experiment_spec(path)

    assert loaded.experiment.name == "roundtrip"
    assert loaded.experiment.tags == ["test", "yaml"]
    assert loaded.data.source == "orbitaal_stream_graph"
    assert loaded.data.raw_path == "/tmp/raw.csv"
    assert loaded.data.raw_remote_path == "remote/raw.csv"
    assert loaded.data.parquet_path == "/tmp/stream.parquet"
    assert loaded.data.parquet_remote_path == "remote/stream.parquet"
    assert loaded.data.features_remote_path == "remote/features.parquet"
    assert loaded.data.node_mapping_remote_path == "remote/node_mapping.npy"
    assert loaded.data.cache_dir == "/tmp/cache"
    assert loaded.data.token_env == "DATA_TOKEN"
    assert loaded.sampling.backend == "cpp"
    assert loaded.training.hidden_dim == 128
    assert loaded.runtime.output_dir == "/tmp/out"
    assert loaded.upload.enabled is True
    assert loaded.model["token_expansion"] == 0.5


def test_build_best_hpo_spec_for_eagle_preserves_core_fields():
    args = Namespace(
        parquet_path="/tmp/stream.parquet",
        features_path="/tmp/features.parquet",
        node_mapping_path="/tmp/node_mapping.npy",
        fraction=0.1,
        train_ratio=0.7,
        val_ratio=0.15,
        no_amp=False,
        seed=42,
        edge_feat_dim=2,
        node_feat_dim=15,
        max_val_edges=3000,
    )
    best_params = {
        "hidden_dim": 100,
        "num_neighbors": 20,
        "num_mixer_layers": 2,
        "lr": 0.001,
        "weight_decay": 1e-5,
        "dropout": 0.1,
        "batch_size": 400,
        "token_expansion": 1.0,
        "channel_expansion": 4.0,
    }

    spec = build_best_hpo_spec("eagle", args, best_params)

    assert spec.experiment.model == "eagle"
    assert spec.data.parquet_path == "/tmp/stream.parquet"
    assert spec.data.features_path == "/tmp/features.parquet"
    assert spec.data.node_mapping_path == "/tmp/node_mapping.npy"
    assert spec.sampling.num_neighbors == 20
    assert spec.training.batch_size == 400
    assert spec.training.hidden_dim == 100
    assert spec.model["token_expansion"] == 1.0
    assert spec.runtime.output_dir == "/tmp/eagle_results"


def test_write_best_training_artifacts_creates_package_launcher_bundle(tmp_path):
    args = Namespace(
        parquet_path="/tmp/stream.parquet",
        train_ratio=0.7,
        val_ratio=0.15,
        no_amp=False,
        seed=42,
        max_val_edges=3000,
    )
    best_params = {
        "n_neighbor": 20,
        "n_latest": 10,
        "d_model": 64,
        "n_heads": 4,
        "n_layers": 2,
        "lr": 0.001,
        "weight_decay": 1e-5,
        "dropout": 0.1,
        "batch_size": 200,
    }

    artifacts = write_best_training_artifacts(
        "hyperevent",
        args,
        best_params,
        output_dir=tmp_path,
    )

    command_text = (tmp_path / "best_train_command.sh").read_text()
    yaml_text = (tmp_path / "best_experiment.yaml").read_text()

    assert artifacts["spec_path"].endswith("best_experiment.yaml")
    assert "payment_graph_forecasting.experiments.launcher --config" in artifacts["command"]
    assert "payment_graph_forecasting.experiments.launcher --config" in command_text
    assert "model: hyperevent" in yaml_text

    loaded = load_experiment_spec(tmp_path / "best_experiment.yaml")
    assert loaded.training.hidden_dim == 64
    assert loaded.model["n_heads"] == 4

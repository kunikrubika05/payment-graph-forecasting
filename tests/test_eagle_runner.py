"""Tests for the migrated EAGLE runner."""

from argparse import Namespace

from payment_graph_forecasting.experiments.runners.eagle import (
    build_eagle_arg_parser,
    run_eagle_experiment,
)


def test_eagle_arg_parser_supports_dry_run():
    parser = build_eagle_arg_parser()
    args = parser.parse_args(["--parquet-path", "/tmp/stream.parquet", "--dry-run"])
    assert args.parquet_path == "/tmp/stream.parquet"
    assert args.dry_run is True


def test_eagle_runner_dry_run_returns_payload(tmp_path):
    args = Namespace(
        data_source="stream_graph",
        raw_path=None,
        raw_remote_path=None,
        parquet_path="/tmp/stream.parquet",
        parquet_remote_path=None,
        features_path="/tmp/features.parquet",
        features_remote_path=None,
        node_mapping_path=None,
        node_mapping_remote_path=None,
        data_backend="yadisk",
        data_cache_dir=None,
        data_token_env="YADISK_TOKEN",
        fraction=0.1,
        train_ratio=0.7,
        val_ratio=0.15,
        output=str(tmp_path),
        device="cpu",
        upload=False,
        remote_dir=None,
        epochs=1,
        batch_size=4,
        lr=1e-3,
        weight_decay=0.0,
        num_neighbors=12,
        hidden_dim=32,
        num_mixer_layers=1,
        token_expansion=0.5,
        channel_expansion=4.0,
        dropout=0.1,
        patience=1,
        seed=42,
        no_amp=True,
        max_val_edges=10,
        max_test_edges=12,
        edge_feat_dim=2,
        node_feat_dim=0,
        dry_run=True,
    )
    result = run_eagle_experiment(args)
    assert result["mode"] == "dry_run"
    assert result["parquet_path"] == "/tmp/stream.parquet"
    assert result["parquet_remote_path"] is None
    assert result["features_path"] == "/tmp/features.parquet"

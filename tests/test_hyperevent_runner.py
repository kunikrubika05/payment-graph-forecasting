"""Tests for the migrated HyperEvent runner."""

from argparse import Namespace

from payment_graph_forecasting.experiments.runners.hyperevent import (
    build_hyperevent_arg_parser,
    run_hyperevent_experiment,
)


def test_hyperevent_arg_parser_supports_dry_run():
    parser = build_hyperevent_arg_parser()
    args = parser.parse_args(["--parquet-path", "/tmp/stream.parquet", "--fraction", "0.1", "--dry-run"])
    assert args.parquet_path == "/tmp/stream.parquet"
    assert args.fraction == 0.1
    assert args.dry_run is True


def test_hyperevent_runner_dry_run_returns_payload(tmp_path):
    args = Namespace(
        data_source="stream_graph",
        raw_path=None,
        raw_remote_path=None,
        parquet_path="/tmp/stream.parquet",
        parquet_remote_path=None,
        fraction=0.25,
        train_ratio=0.7,
        val_ratio=0.15,
        output=str(tmp_path),
        device="cpu",
        data_backend="yadisk",
        data_cache_dir=None,
        data_token_env="YADISK_TOKEN",
        upload=False,
        remote_dir=None,
        upload_backend="yadisk",
        token_env="YADISK_TOKEN",
        epochs=1,
        batch_size=4,
        lr=1e-4,
        weight_decay=0.0,
        n_neighbor=12,
        n_latest=6,
        d_model=32,
        n_heads=2,
        n_layers=1,
        dropout=0.1,
        patience=1,
        seed=42,
        no_amp=True,
        max_val_edges=10,
        n_hist_neg=7,
        n_random_neg=9,
        dry_run=True,
    )
    result = run_hyperevent_experiment(args)
    assert result["mode"] == "dry_run"
    assert result["parquet_path"] == "/tmp/stream.parquet"
    assert result["parquet_remote_path"] is None
    assert result["fraction"] == 0.25
    assert result["device"] == "cpu"
    assert result["n_neighbor"] == 12
    assert result["n_latest"] == 6

"""Tests for the migrated sg-GraphMixer runner."""

from argparse import Namespace

from payment_graph_forecasting.experiments.runners.sg_graphmixer import (
    build_sg_graphmixer_arg_parser,
    run_sg_graphmixer_experiment,
)


def test_sg_graphmixer_arg_parser_supports_dry_run():
    parser = build_sg_graphmixer_arg_parser()
    args = parser.parse_args(["--period", "period_10", "--dry-run"])
    assert args.period == "period_10"
    assert args.dry_run is True


def test_sg_graphmixer_runner_dry_run_returns_payload(tmp_path):
    args = Namespace(
        period="period_10",
        output=str(tmp_path),
        data_dir="/tmp/sg_baselines_data",
        upload=False,
        token_env="YADISK_TOKEN",
        device="cpu",
        seed=42,
        hidden_dim=200,
        num_neighbors=30,
        num_mixer_layers=1,
        lr=1e-3,
        weight_decay=3e-6,
        dropout=0.2,
        batch_size=4000,
        num_epochs=10,
        patience=5,
        max_val_queries=10_000,
        max_test_queries=50_000,
        n_negatives=100,
        train_ratio=0.70,
        val_ratio=0.15,
        dry_run=True,
    )
    result = run_sg_graphmixer_experiment(args)
    assert result["mode"] == "dry_run"
    assert result["period"] == "period_10"
    assert result["device"] == "cpu"
    assert result["n_negatives"] == 100

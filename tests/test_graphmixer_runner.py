"""Tests for the migrated GraphMixer runner."""

from argparse import Namespace

from payment_graph_forecasting.experiments.runners.graphmixer import (
    build_graphmixer_arg_parser,
    run_graphmixer_experiment,
)


def test_graphmixer_arg_parser_supports_dry_run():
    parser = build_graphmixer_arg_parser()
    args = parser.parse_args(["--period", "mature_2020q2", "--dry-run"])
    assert args.period == "mature_2020q2"
    assert args.dry_run is True


def test_graphmixer_runner_dry_run_returns_payload(tmp_path):
    args = Namespace(
        period="mature_2020q2",
        window=7,
        output=str(tmp_path),
        data_dir="/tmp/graphmixer_data",
        epochs=1,
        batch_size=4,
        lr=1e-4,
        num_neighbors=12,
        hidden_dim=32,
        num_mixer_layers=1,
        dropout=0.1,
        patience=1,
        seed=42,
        max_val_edges=10,
        eval_batch_size=2,
        dry_run=True,
    )
    result = run_graphmixer_experiment(args)
    assert result["mode"] == "dry_run"
    assert result["period"] == "mature_2020q2"
    assert result["num_neighbors"] == 12

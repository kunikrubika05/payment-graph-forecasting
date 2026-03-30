"""PairwiseMLP runner on top of the new package layout."""

from __future__ import annotations

import argparse
import json
import os

from payment_graph_forecasting.experiments.results import build_dry_run_result
from payment_graph_forecasting.experiments.runner_utils import ensure_output_dir
from src.models.pairwise_mlp.config import (
    FEATURE_NAMES,
    PairMLPConfig,
    PERIODS,
    resolve_feature_indices,
)
from src.models.pairwise_mlp.run import run_experiment


def build_pairwise_mlp_arg_parser() -> argparse.ArgumentParser:
    """Build the PairwiseMLP CLI parser."""

    parser = argparse.ArgumentParser(description="PairwiseMLP experiment runner")
    parser.add_argument("--period", type=str, default="period_10", choices=list(PERIODS.keys()))
    parser.add_argument("--output", type=str, default="/tmp/pairmlp_results")
    parser.add_argument("--data-dir", type=str, default="/tmp/pairmlp_data")
    parser.add_argument("--precompute-dir", type=str, default="/tmp/pairmlp_precompute")
    parser.add_argument("--exp-tag", type=str, default="")
    parser.add_argument("--loss", type=str, default="bpr")
    parser.add_argument("--features", nargs="*", default=None)
    parser.add_argument("--feature-indices", nargs="*", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-negatives", type=int, default=100)
    parser.add_argument("--upload", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def run_pairwise_mlp_experiment(args: argparse.Namespace):
    """Run a PairwiseMLP experiment through the new runner contract."""

    period = PERIODS[args.period]
    cfg = PairMLPConfig(
        period_name=args.period,
        fraction=float(period["fraction"]),
        label=str(period["label"]),
        random_seed=args.seed,
        hidden_dims=[64, 32],
        dropout=args.dropout,
        lr=args.lr,
        weight_decay=args.weight_decay,
        batch_size=args.batch_size,
        n_epochs=args.epochs,
        patience=args.patience,
        exp_tag=args.exp_tag,
        loss=args.loss,
        active_feature_indices=resolve_feature_indices(
            feature_names=args.features,
            feature_indices=args.feature_indices,
        ),
        n_negatives=args.n_negatives,
        local_data_dir=args.data_dir,
        local_precompute_dir=args.precompute_dir,
        local_output_dir=args.output,
        upload=args.upload,
    )
    output_dir = os.path.join(cfg.local_output_dir, cfg.exp_name)

    if args.dry_run:
        ensure_output_dir(output_dir)
        return build_dry_run_result(
            experiment=cfg.exp_name,
            output_dir=output_dir,
            period=args.period,
            loss=args.loss,
            n_negatives=args.n_negatives,
            selected_features=cfg.selected_feature_names,
        )

    token = os.environ.get("YADISK_TOKEN", "")
    run_experiment(cfg, token)

    summary_path = os.path.join(output_dir, "summary.json")
    summary = None
    if os.path.exists(summary_path):
        with open(summary_path, encoding="utf-8") as handle:
            summary = json.load(handle)

    return {
        "experiment": cfg.exp_name,
        "output_dir": output_dir,
        "summary_path": summary_path,
        "summary": summary,
        "selected_features": cfg.selected_feature_names,
        "n_negatives": args.n_negatives,
    }

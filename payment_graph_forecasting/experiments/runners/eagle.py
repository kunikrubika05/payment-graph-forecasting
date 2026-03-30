"""EAGLE runner on top of the new package layout."""

from __future__ import annotations

import argparse
import logging
import os
import time
from pathlib import Path

import numpy as np
import torch

from payment_graph_forecasting.experiments.runner_utils import (
    attach_file_logger,
    configure_root_logging,
    describe_device,
    ensure_output_dir,
    maybe_upload_output,
    resolve_device,
    save_json,
    save_training_curves,
)
from payment_graph_forecasting.experiments.results import (
    build_dry_run_result,
    build_final_results,
)
from payment_graph_forecasting.evaluation.api import evaluate_eagle_model
from payment_graph_forecasting.training.api import train_eagle_model
from src.models.EAGLE.data_utils import TemporalCSR, load_stream_graph_data

logger = configure_root_logging()

YADISK_EXPERIMENTS_BASE = "orbitaal_processed/experiments/exp_sg_eagle"


def build_eagle_arg_parser() -> argparse.ArgumentParser:
    """Build the EAGLE CLI parser."""

    parser = argparse.ArgumentParser(description="EAGLE temporal link prediction on stream graphs")
    parser.add_argument("--parquet-path", type=str, required=True)
    parser.add_argument("--features-path", type=str, default=None)
    parser.add_argument("--node-mapping-path", type=str, default=None)
    parser.add_argument("--fraction", type=float, default=None)
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--output", type=str, default="/tmp/eagle_results")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=200)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=5e-5)
    parser.add_argument("--num-neighbors", type=int, default=20)
    parser.add_argument("--hidden-dim", type=int, default=100)
    parser.add_argument("--num-mixer-layers", type=int, default=1)
    parser.add_argument("--token-expansion", type=float, default=0.5)
    parser.add_argument("--channel-expansion", type=float, default=4.0)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--max-val-edges", type=int, default=5000)
    parser.add_argument("--max-test-edges", type=int, default=50_000)
    parser.add_argument("--n-negatives", type=int, default=100)
    parser.add_argument("--edge-feat-dim", type=int, default=0)
    parser.add_argument("--node-feat-dim", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def _build_train_neighbors(data, train_mask, is_undirected=True):
    train_indices = np.where(train_mask)[0]
    if is_undirected:
        train_indices = train_indices[: len(train_indices) // 2]

    src = data.src[train_indices].astype(np.int64)
    dst = data.dst[train_indices].astype(np.int64)
    neighbors = {}
    for s, d in zip(src, dst):
        neighbors.setdefault(int(s), set()).add(int(d))
    return neighbors


def _compute_active_nodes(node_mapping_path, data, train_mask, is_undirected=True):
    if node_mapping_path is not None:
        active = np.load(node_mapping_path).astype(np.int64)
        logger.info("Loaded node_mapping: %d active nodes", len(active))
        return np.sort(active)

    train_indices = np.where(train_mask)[0]
    if is_undirected:
        train_indices = train_indices[: len(train_indices) // 2]
    src = data.src[train_indices].astype(np.int64)
    dst = data.dst[train_indices].astype(np.int64)
    active = np.unique(np.concatenate([src, dst]))
    logger.info("Computed active_nodes from train: %d nodes", len(active))
    return active

def _build_data_summary(data, train_mask, val_mask, test_mask):
    return {
        "num_nodes": int(data.num_nodes),
        "num_edges": int(data.num_edges),
        "train_edges": int(train_mask.sum()),
        "val_edges": int(val_mask.sum()),
        "test_edges": int(test_mask.sum()),
        "timestamp_min": float(data.timestamps.min()),
        "timestamp_max": float(data.timestamps.max()),
    }


def run_eagle_experiment(args: argparse.Namespace):
    """Run a full EAGLE experiment using the new package runner."""

    parquet_name = Path(args.parquet_path).stem
    frac_str = f"f{int(args.fraction * 100)}" if args.fraction else "full"
    exp_name = f"eagle_{parquet_name}_{frac_str}"
    output_dir = os.path.join(args.output, exp_name)
    n_negatives = getattr(args, "n_negatives", 100)

    if args.dry_run:
        ensure_output_dir(output_dir)
        return build_dry_run_result(
            experiment=exp_name,
            output_dir=output_dir,
            parquet_path=args.parquet_path,
            features_path=args.features_path,
            fraction=args.fraction,
            n_negatives=n_negatives,
        )

    total_start = time.time()
    ensure_output_dir(output_dir)
    attach_file_logger(output_dir)

    device = resolve_device()

    data_start = time.time()
    data, train_mask, val_mask, test_mask = load_stream_graph_data(
        args.parquet_path,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        undirected=True,
        fraction=args.fraction,
        features_path=args.features_path,
    )
    data_time = time.time() - data_start
    save_json(os.path.join(output_dir, "data_summary.json"), _build_data_summary(data, train_mask, val_mask, test_mask))

    active_nodes = _compute_active_nodes(args.node_mapping_path, data, train_mask, is_undirected=True)
    train_neighbors = _build_train_neighbors(data, train_mask, is_undirected=True)

    node_feat_dim = args.node_feat_dim
    if args.features_path and node_feat_dim == 0:
        node_feat_dim = data.node_feats.shape[1]

    train_start = time.time()
    training_result = train_eagle_model(
        data=data,
        train_mask=train_mask,
        val_mask=val_mask,
        output_dir=output_dir,
        device=device,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        num_neighbors=args.num_neighbors,
        hidden_dim=args.hidden_dim,
        num_mixer_layers=args.num_mixer_layers,
        token_expansion=args.token_expansion,
        channel_expansion=args.channel_expansion,
        dropout=args.dropout,
        patience=args.patience,
        seed=args.seed,
        max_val_edges=args.max_val_edges,
        use_amp=not args.no_amp,
        edge_feat_dim=args.edge_feat_dim,
        node_feat_dim=node_feat_dim,
        active_nodes=active_nodes,
    )
    model = training_result.model
    history = training_result.history
    train_time = time.time() - train_start
    save_training_curves(output_dir, history)

    eval_start = time.time()
    all_before_test = train_mask | val_mask
    test_csr = TemporalCSR(
        data.num_nodes,
        data.src[all_before_test],
        data.dst[all_before_test],
        data.timestamps[all_before_test],
        np.where(all_before_test)[0].astype(np.int64),
    )
    val_csr = TemporalCSR(
        data.num_nodes,
        data.src[train_mask],
        data.dst[train_mask],
        data.timestamps[train_mask],
        np.where(train_mask)[0].astype(np.int64),
    )

    val_metrics = evaluate_eagle_model(
        model=model,
        data=data,
        csr=val_csr,
        eval_mask=val_mask,
        device=device,
        train_neighbors=train_neighbors,
        active_nodes=active_nodes,
        num_neighbors=args.num_neighbors,
        n_negatives=n_negatives,
        use_amp=not args.no_amp,
        seed=args.seed + 300,
        max_queries=args.max_test_edges,
        is_undirected=True,
    ).metrics
    test_metrics = evaluate_eagle_model(
        model=model,
        data=data,
        csr=test_csr,
        eval_mask=test_mask,
        device=device,
        train_neighbors=train_neighbors,
        active_nodes=active_nodes,
        num_neighbors=args.num_neighbors,
        n_negatives=n_negatives,
        use_amp=not args.no_amp,
        seed=args.seed + 400,
        max_queries=args.max_test_edges,
        is_undirected=True,
    ).metrics
    eval_time = time.time() - eval_start

    final_results = build_final_results(
        experiment=exp_name,
        model="EAGLE-Time",
        history=history,
        timing={
            "data_prep_sec": data_time,
            "training_sec": train_time,
            "evaluation_sec": eval_time,
            "total_sec": time.time() - total_start,
        },
        args=vars(args),
        device_info=describe_device(device),
        extra={
            "parquet_path": args.parquet_path,
            "fraction": args.fraction,
            "features_path": args.features_path,
            "val_metrics": val_metrics,
            "test_metrics": test_metrics,
        },
    )
    save_json(os.path.join(output_dir, "final_results.json"), final_results)

    if maybe_upload_output(output_dir, f"{YADISK_EXPERIMENTS_BASE}/{exp_name}"):
        try:
            logger.info("Uploaded results to %s", f"{YADISK_EXPERIMENTS_BASE}/{exp_name}")
        except Exception as exc:
            logger.error("Upload failed: %s", exc)

    return final_results


def main(argv: list[str] | None = None) -> int:
    parser = build_eagle_arg_parser()
    args = parser.parse_args(argv)
    run_eagle_experiment(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

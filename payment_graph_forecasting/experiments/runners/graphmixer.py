"""GraphMixer runner on top of the new package layout."""

from __future__ import annotations

import argparse
import logging
import os
import time

import numpy as np
import torch

from payment_graph_forecasting.experiments.runner_utils import (
    attach_file_logger,
    configure_root_logging,
    describe_runtime,
    ensure_output_dir,
    maybe_upload_from_args,
    save_json,
    save_training_curves,
)
from payment_graph_forecasting.experiments.results import (
    build_dry_run_result,
    build_final_results,
)
from payment_graph_forecasting.evaluation.api import evaluate_graphmixer_model
from payment_graph_forecasting.training.api import train_graphmixer_model
from src.baselines.config import PERIODS
from src.models.data_utils import TemporalCSR, prepare_sliding_window

logger = configure_root_logging()


def build_graphmixer_arg_parser() -> argparse.ArgumentParser:
    """Build the GraphMixer CLI parser."""

    parser = argparse.ArgumentParser(description="GraphMixer temporal link prediction")
    parser.add_argument("--period", type=str, default="mature_2020q2", choices=list(PERIODS.keys()))
    parser.add_argument("--window", type=int, default=7)
    parser.add_argument("--output", type=str, default="/tmp/graphmixer_results")
    parser.add_argument("--data-dir", type=str, default="/tmp/graphmixer_data")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--upload", action="store_true")
    parser.add_argument("--upload-backend", type=str, default="yadisk")
    parser.add_argument("--remote-dir", type=str, default=None)
    parser.add_argument("--token-env", type=str, default="YADISK_TOKEN")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=600)
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--num-neighbors", type=int, default=20)
    parser.add_argument("--hidden-dim", type=int, default=100)
    parser.add_argument("--num-mixer-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-val-edges", type=int, default=5000)
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument("--n-hist-neg", type=int, default=50)
    parser.add_argument("--n-random-neg", type=int, default=50)
    parser.add_argument("--dry-run", action="store_true")
    return parser

def _build_data_summary(data, dates, train_mask, val_mask, test_mask) -> dict:
    return {
        "num_nodes": int(data.num_nodes),
        "num_edges": int(data.num_edges),
        "edge_feat_dim": int(data.edge_feats.shape[1]),
        "node_feat_dim": int(data.node_feats.shape[1]),
        "num_dates": len(dates),
        "date_range": [dates[0], dates[-1]],
        "train_edges": int(train_mask.sum()),
        "val_edges": int(val_mask.sum()),
        "test_edges": int(test_mask.sum()),
        "timestamp_range": [float(data.timestamps.min()), float(data.timestamps.max())],
    }


def run_graphmixer_experiment(args: argparse.Namespace):
    """Run a full GraphMixer experiment using the new package runner."""

    exp_name = f"graphmixer_{args.period}_w{args.window}"
    output_dir = os.path.join(args.output, exp_name)
    n_hist_neg = getattr(args, "n_hist_neg", 50)
    n_random_neg = getattr(args, "n_random_neg", 50)

    if args.dry_run:
        ensure_output_dir(output_dir)
        return build_dry_run_result(
            experiment=exp_name,
            output_dir=output_dir,
            period=args.period,
            device=getattr(args, "device", "auto"),
            window=args.window,
            num_neighbors=args.num_neighbors,
            upload=bool(getattr(args, "upload", False)),
            remote_dir=getattr(args, "remote_dir", None),
            n_hist_neg=n_hist_neg,
            n_random_neg=n_random_neg,
        )

    total_start = time.time()
    ensure_output_dir(output_dir)
    attach_file_logger(output_dir)

    logger.info("=" * 60)
    logger.info("GraphMixer Experiment: %s", exp_name)
    logger.info("Args: %s", vars(args))
    logger.info("=" * 60)

    runtime = describe_runtime(getattr(args, "device", "auto"), amp=not getattr(args, "no_amp", False))
    device = runtime.device
    logger.info("Device: %s", device)
    if device.type == "cuda":
        logger.info("GPU: %s", torch.cuda.get_device_name(0))
        logger.info("GPU memory: %.1f GB", torch.cuda.get_device_properties(0).total_memory / 1e9)

    logger.info("Step 1: Preparing data for period '%s' (window=%d)...", args.period, args.window)
    data_start = time.time()
    data, dates, train_mask, val_mask, test_mask = prepare_sliding_window(
        args.period, window=args.window, local_dir=args.data_dir, undirected=True
    )
    data_time = time.time() - data_start
    logger.info("Data: %s (%.1f sec)", data, data_time)

    save_json(os.path.join(output_dir, "data_summary.json"), _build_data_summary(data, dates, train_mask, val_mask, test_mask))

    logger.info("Step 2: Training GraphMixer...")
    train_start = time.time()
    training_result = train_graphmixer_model(
        data=data,
        train_mask=train_mask,
        val_mask=val_mask,
        output_dir=output_dir,
        device=device,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        num_neighbors=args.num_neighbors,
        hidden_dim=args.hidden_dim,
        num_mixer_layers=args.num_mixer_layers,
        dropout=args.dropout,
        patience=args.patience,
        seed=args.seed,
        max_val_edges=args.max_val_edges,
    )
    model = training_result.model
    history = training_result.history
    train_time = time.time() - train_start

    save_training_curves(output_dir, history)

    logger.info("Step 3: TGB-style evaluation on test set...")
    eval_start = time.time()
    all_before_test = train_mask | val_mask
    test_csr = TemporalCSR(
        data.num_nodes,
        data.src[all_before_test],
        data.dst[all_before_test],
        data.timestamps[all_before_test],
        np.where(all_before_test)[0].astype(np.int64),
    )
    test_metrics = evaluate_graphmixer_model(
        model=model,
        data=data,
        csr=test_csr,
        eval_mask=test_mask,
        device=device,
        num_neighbors=args.num_neighbors,
        n_hist_neg=n_hist_neg,
        n_random_neg=n_random_neg,
        eval_batch_size=args.eval_batch_size,
        seed=args.seed,
    ).metrics
    eval_time = time.time() - eval_start

    total_time = time.time() - total_start
    final_results = build_final_results(
        experiment=exp_name,
        model="GraphMixer",
        history=history,
        timing={
            "data_prep_sec": data_time,
            "training_sec": train_time,
            "evaluation_sec": eval_time,
            "total_sec": total_time,
        },
        args=vars(args),
        device_info={
            "device": runtime.resolved_device,
            "requested_device": runtime.requested_device,
            "cuda_available": runtime.cuda_available,
            "amp_enabled": runtime.amp_enabled,
            "gpu_name": runtime.gpu_name,
        },
        extra={
            "period": args.period,
            "test_metrics": test_metrics,
        },
    )
    save_json(os.path.join(output_dir, "final_results.json"), final_results)

    logger.info("Step 4: Uploading results...")
    if maybe_upload_from_args(output_dir, args, experiment_name=exp_name, logger=logger):
        logger.info("Uploaded results for %s", exp_name)

    return final_results


def main(argv: list[str] | None = None) -> int:
    parser = build_graphmixer_arg_parser()
    args = parser.parse_args(argv)
    run_graphmixer_experiment(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

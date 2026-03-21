"""CLI launcher for GraphMixer temporal link prediction experiments.

Usage:
    PYTHONPATH=. python src/models/launcher.py --period mature_2020q2 --output /tmp/graphmixer_results
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

from src.models.data_utils import (
    prepare_period_data,
    chronological_split,
    build_temporal_csr,
    TemporalCSR,
)
from src.models.train import train_graphmixer
from src.models.evaluate import evaluate_tgb_style
from src.baselines.config import PERIODS

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def run_experiment(args: argparse.Namespace) -> None:
    """Run a full GraphMixer experiment: data prep, train, evaluate."""
    total_start = time.time()

    exp_name = f"graphmixer_{args.period}_w{args.window}"
    output_dir = os.path.join(args.output, exp_name)
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    log_file = os.path.join(output_dir, "experiment.log")
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
    logging.getLogger().addHandler(file_handler)

    logger.info("=" * 60)
    logger.info("GraphMixer Experiment: %s", exp_name)
    logger.info("Args: %s", vars(args))
    logger.info("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)
    if device.type == "cuda":
        logger.info("GPU: %s", torch.cuda.get_device_name(0))
        logger.info("GPU memory: %.1f GB", torch.cuda.get_device_properties(0).total_mem / 1e9)

    logger.info("Step 1: Preparing data for period '%s'...", args.period)
    data, dates = prepare_period_data(
        args.period, local_dir=args.data_dir, undirected=True,
    )
    logger.info("Data: %s", data)

    window_dates = dates[-args.window:]
    logger.info("Using last %d days as context window: %s to %s",
                args.window, window_dates[0], window_dates[-1])

    logger.info("Step 2: Chronological split...")
    train_mask, val_mask, test_mask = chronological_split(
        data, train_ratio=args.train_ratio, val_ratio=args.val_ratio,
    )

    logger.info("Step 3: Training GraphMixer...")
    model, history = train_graphmixer(
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

    logger.info("Step 4: TGB-style evaluation on test set...")
    all_before_test = train_mask | val_mask
    test_csr = TemporalCSR(
        data.num_nodes,
        data.src[all_before_test],
        data.dst[all_before_test],
        data.timestamps[all_before_test],
        np.where(all_before_test)[0].astype(np.int64),
    )

    test_metrics = evaluate_tgb_style(
        model=model,
        data=data,
        csr=test_csr,
        eval_mask=test_mask,
        device=device,
        num_neighbors=args.num_neighbors,
        n_hist_neg=50,
        n_random_neg=50,
        eval_batch_size=args.eval_batch_size,
        seed=args.seed,
    )

    total_time = time.time() - total_start

    final_results = {
        "experiment": exp_name,
        "period": args.period,
        "model": "GraphMixer",
        "test_metrics": test_metrics,
        "best_val_mrr": history["val_mrr"][np.argmax(history["val_mrr"])] if history["val_mrr"] else None,
        "total_time_sec": total_time,
        "device": str(device),
        "args": vars(args),
    }
    with open(os.path.join(output_dir, "final_results.json"), "w") as f:
        json.dump(final_results, f, indent=2)

    logger.info("=" * 60)
    logger.info("RESULTS: %s", exp_name)
    logger.info("  Test MRR:     %.4f", test_metrics["mrr"])
    logger.info("  Test Hits@1:  %.4f", test_metrics["hits@1"])
    logger.info("  Test Hits@3:  %.4f", test_metrics["hits@3"])
    logger.info("  Test Hits@10: %.4f", test_metrics["hits@10"])
    logger.info("  Total time:   %.1f min", total_time / 60)
    logger.info("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="GraphMixer temporal link prediction")
    parser.add_argument("--period", type=str, default="mature_2020q2",
                        choices=list(PERIODS.keys()),
                        help="Bitcoin period to train on")
    parser.add_argument("--window", type=int, default=7,
                        help="Context window size in days (for data prep)")
    parser.add_argument("--output", type=str, default="/tmp/graphmixer_results",
                        help="Output directory for results")
    parser.add_argument("--data-dir", type=str, default="/tmp/graphmixer_data",
                        help="Local directory for downloaded data")

    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=600)
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--num-neighbors", type=int, default=20)
    parser.add_argument("--hidden-dim", type=int, default=100)
    parser.add_argument("--num-mixer-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--train-ratio", type=float, default=0.6)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--max-val-edges", type=int, default=5000)
    parser.add_argument("--eval-batch-size", type=int, default=32)

    args = parser.parse_args()
    run_experiment(args)


if __name__ == "__main__":
    main()

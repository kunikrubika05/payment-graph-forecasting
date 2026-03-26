"""CLI launcher for HyperEvent temporal link prediction on stream graphs.

Usage:
    YADISK_TOKEN="..." PYTHONPATH=. python src/models/HyperEvent/hyperevent_launcher.py \\
        --parquet-path stream_graph/2020-06-01_2020-08-31.parquet \\
        --n-neighbor 20 --n-latest 10 --epochs 50

Stream graph parquet format expected:
    src_idx, dst_idx, timestamp, btc, usd

Results are saved locally and optionally uploaded to Yandex.Disk.
"""

import argparse
import csv
import json
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

from src.models.HyperEvent.data_utils import load_stream_graph_data
from src.models.HyperEvent.hyperevent_train import train_hyperevent
from src.models.HyperEvent.hyperevent_evaluate import evaluate_tgb_style
from src.yadisk_utils import upload_directory

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

YADISK_EXPERIMENTS_BASE = "orbitaal_processed/experiments/exp_008_hyperevent"


def _save_data_summary(output_dir, data, train_mask, val_mask, test_mask):
    summary = {
        "num_nodes": int(data.num_nodes),
        "num_edges": int(data.num_edges),
        "train_edges": int(train_mask.sum()),
        "val_edges": int(val_mask.sum()),
        "test_edges": int(test_mask.sum()),
        "timestamp_min": float(data.timestamps.min()),
        "timestamp_max": float(data.timestamps.max()),
    }
    with open(os.path.join(output_dir, "data_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)


def _save_training_curves(output_dir, history):
    n_epochs = len(history["train_loss"])
    rows = []
    for i in range(n_epochs):
        rows.append({
            "epoch": i + 1,
            "train_loss": history["train_loss"][i],
            "val_mrr": history["val_mrr"][i],
            "val_hits@1": history["val_hits@1"][i],
            "val_hits@3": history["val_hits@3"][i],
            "val_hits@10": history["val_hits@10"][i],
            "epoch_time_sec": history["epoch_time"][i],
        })
    csv_path = os.path.join(output_dir, "training_curves.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def run_experiment(args):
    """Run a full HyperEvent experiment: load data, train, evaluate, upload."""
    total_start = time.time()

    parquet_name = Path(args.parquet_path).stem
    exp_name = f"hyperevent_{parquet_name}"
    output_dir = os.path.join(args.output, exp_name)
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    log_file = os.path.join(output_dir, "experiment.log")
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    )
    logging.getLogger().addHandler(file_handler)

    logger.info("=" * 60)
    logger.info("HyperEvent Experiment: %s", exp_name)
    logger.info("Args: %s", vars(args))
    logger.info("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)
    if device.type == "cuda":
        logger.info("GPU: %s", torch.cuda.get_device_name(0))
        logger.info(
            "GPU memory: %.1f GB",
            torch.cuda.get_device_properties(0).total_memory / 1e9,
        )

    logger.info("Step 1: Loading stream graph from %s...", args.parquet_path)
    data_start = time.time()
    data, train_mask, val_mask, test_mask = load_stream_graph_data(
        args.parquet_path,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        undirected=True,
    )
    data_time = time.time() - data_start
    logger.info("Data: %s (%.1f sec)", data, data_time)

    _save_data_summary(output_dir, data, train_mask, val_mask, test_mask)

    logger.info("Step 2: Training HyperEvent...")
    train_start = time.time()
    model, history = train_hyperevent(
        data=data,
        train_mask=train_mask,
        val_mask=val_mask,
        output_dir=output_dir,
        device=device,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        n_neighbor=args.n_neighbor,
        n_latest=args.n_latest,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        dropout=args.dropout,
        patience=args.patience,
        seed=args.seed,
        max_val_edges=args.max_val_edges,
        use_amp=not args.no_amp,
    )
    train_time = time.time() - train_start

    _save_training_curves(output_dir, history)

    logger.info("Step 3: TGB-style evaluation on test set...")
    eval_start = time.time()
    history_mask = train_mask | val_mask

    test_metrics = evaluate_tgb_style(
        model=model,
        data=data,
        eval_mask=test_mask,
        history_mask=history_mask,
        device=device,
        n_neighbor=args.n_neighbor,
        n_latest=args.n_latest,
        n_hist_neg=50,
        n_random_neg=50,
        use_amp=not args.no_amp,
        seed=args.seed,
    )
    eval_time = time.time() - eval_start
    total_time = time.time() - total_start

    final_results = {
        "experiment": exp_name,
        "parquet_path": args.parquet_path,
        "model": "HyperEvent",
        "test_metrics": test_metrics,
        "best_val_mrr": (
            float(history["val_mrr"][np.argmax(history["val_mrr"])])
            if history["val_mrr"] else None
        ),
        "best_epoch": (
            int(np.argmax(history["val_mrr"]) + 1)
            if history["val_mrr"] else None
        ),
        "total_epochs": len(history["train_loss"]),
        "timing": {
            "data_prep_sec": data_time,
            "training_sec": train_time,
            "evaluation_sec": eval_time,
            "total_sec": total_time,
        },
        "device": str(device),
        "gpu_name": (
            torch.cuda.get_device_name(0) if device.type == "cuda" else None
        ),
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
    logger.info(
        "  Best val MRR: %.4f (epoch %d)",
        final_results["best_val_mrr"],
        final_results["best_epoch"],
    )
    logger.info("  Training:     %.1f min", train_time / 60)
    logger.info("  Evaluation:   %.1f min", eval_time / 60)
    logger.info("  Total time:   %.1f min", total_time / 60)
    logger.info("=" * 60)

    logger.info("Step 4: Uploading results to Yandex.Disk...")
    token = os.environ.get("YADISK_TOKEN", "")
    if token:
        remote_dir = f"{YADISK_EXPERIMENTS_BASE}/{exp_name}"
        try:
            count = upload_directory(output_dir, remote_dir, token)
            logger.info("Uploaded %d files to %s", count, remote_dir)
        except Exception as e:
            logger.error("Upload failed: %s", e)
    else:
        logger.warning(
            "YADISK_TOKEN not set — skipping upload. Results: %s", output_dir
        )


def main():
    parser = argparse.ArgumentParser(
        description="HyperEvent temporal link prediction on stream graphs"
    )
    parser.add_argument(
        "--parquet-path", type=str, required=True,
        help="Path to stream graph parquet file (src_idx, dst_idx, timestamp, btc, usd)",
    )
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--output", type=str, default="/tmp/hyperevent_results")

    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=200)
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument(
        "--n-neighbor", type=int, default=20,
        help="Adjacency table size per node (paper recommends 10-50).",
    )
    parser.add_argument(
        "--n-latest", type=int, default=10,
        help="Context events taken per query node (paper default: 10).",
    )
    parser.add_argument(
        "--d-model", type=int, default=64,
        help="Transformer hidden dimension (paper default: 64).",
    )
    parser.add_argument(
        "--n-heads", type=int, default=4,
        help="Number of attention heads (paper default: 4).",
    )
    parser.add_argument(
        "--n-layers", type=int, default=3,
        help="Number of Transformer encoder layers (paper default: 3).",
    )
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--max-val-edges", type=int, default=5000)

    args = parser.parse_args()
    run_experiment(args)


if __name__ == "__main__":
    main()

"""CLI launcher for the GraphMixer CUDA sampler benchmark on A10.

Trains GraphMixerTime with one sampling backend (python/cpp/cuda) and
reports per-epoch breakdown of sampling_time vs forward_time.

Expected results on A10, batch=2000, K=20, 3-month stream graph:
  python: ~5 min/epoch  sampling ~81%  forward ~19%
  cpp:    ~35 sec/epoch  sampling ~81%  forward ~19%   (← 8.5x faster than python)
  cuda:   ~9 sec/epoch   sampling ~22%  forward ~78%   (← 3.8x faster than cpp)

Usage:
    YADISK_TOKEN="..." PYTHONPATH=. python \\
        src/models/cuda_exp_graphmixer_a10/launcher.py \\
        --parquet-path stream_graph/2020-06-01_2020-08-31.parquet \\
        --sampling-backend cpp --epochs 3
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

from src.models.EAGLE.data_utils import load_stream_graph_data
from src.models.cuda_exp_graphmixer_a10.train import train_graphmixer
from src.yadisk_utils import upload_directory

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

YADISK_EXPERIMENTS_BASE = "orbitaal_processed/experiments/exp_cuda_graphmixer_a10"


def _save_curves(output_dir: str, history: dict) -> None:
    """Save per-epoch timing and metrics as CSV for plotting."""
    n = len(history["train_loss"])
    rows = []
    for i in range(n):
        s = history["sampling_time_sec"][i]
        f = history["forward_time_sec"][i]
        rows.append({
            "epoch": i + 1,
            "train_loss": history["train_loss"][i],
            "sampling_time_sec": s,
            "forward_time_sec": f,
            "sampling_fraction_pct": 100 * s / (s + f + 1e-9),
            "val_mrr": history["val_mrr"][i],
            "val_hits@1": history["val_hits@1"][i],
            "val_hits@3": history["val_hits@3"][i],
            "val_hits@10": history["val_hits@10"][i],
            "epoch_time_sec": history["epoch_time_sec"][i],
        })
    with open(os.path.join(output_dir, "training_curves.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def run_experiment(args: argparse.Namespace) -> None:
    """Load data, train with chosen backend, and upload results."""
    total_start = time.time()

    parquet_stem = Path(args.parquet_path).stem
    exp_name = f"graphmixer_{args.sampling_backend}_{parquet_stem}"
    output_dir = os.path.join(args.output, exp_name)
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    fh = logging.FileHandler(os.path.join(output_dir, "experiment.log"))
    fh.setFormatter(
        logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    )
    logging.getLogger().addHandler(fh)

    logger.info("=" * 60)
    logger.info("GraphMixer CUDA Benchmark | backend=%s", args.sampling_backend)
    logger.info("Args: %s", vars(args))
    logger.info("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)
    if device.type == "cuda":
        logger.info("GPU: %s  (%.1f GB)",
                    torch.cuda.get_device_name(0),
                    torch.cuda.get_device_properties(0).total_memory / 1e9)

    logger.info("Loading %s ...", args.parquet_path)
    data, train_mask, val_mask, test_mask = load_stream_graph_data(
        args.parquet_path,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        undirected=True,
    )
    logger.info(
        "Edges: %d total (train=%d val=%d test=%d) | nodes=%d",
        data.num_edges, train_mask.sum(), val_mask.sum(),
        test_mask.sum(), data.num_nodes,
    )

    batches_per_epoch = int(np.ceil(train_mask.sum() / args.batch_size))
    logger.info(
        "Batches/epoch: %d  (train_edges=%d / batch=%d)",
        batches_per_epoch, train_mask.sum(), args.batch_size,
    )

    with open(os.path.join(output_dir, "data_summary.json"), "w") as f:
        json.dump({
            "num_nodes": int(data.num_nodes),
            "num_edges": int(data.num_edges),
            "train_edges": int(train_mask.sum()),
            "val_edges": int(val_mask.sum()),
            "test_edges": int(test_mask.sum()),
            "batches_per_epoch": batches_per_epoch,
        }, f, indent=2)

    model, history = train_graphmixer(
        data=data,
        train_mask=train_mask,
        val_mask=val_mask,
        output_dir=output_dir,
        sampling_backend=args.sampling_backend,
        device=device,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        num_neighbors=args.num_neighbors,
        hidden_dim=args.hidden_dim,
        num_mixer_layers=args.num_mixer_layers,
        dropout=args.dropout,
        patience=args.patience,
        seed=args.seed,
        max_val_edges=args.max_val_edges,
        use_amp=not args.no_amp,
        edge_feat_dim=args.edge_feat_dim,
        node_feat_dim=0,
    )

    _save_curves(output_dir, history)

    total_time = time.time() - total_start
    avg_s = float(np.mean(history["sampling_time_sec"]))
    avg_f = float(np.mean(history["forward_time_sec"]))
    avg_e = float(np.mean(history["epoch_time_sec"]))

    final = {
        "experiment": exp_name,
        "sampling_backend": args.sampling_backend,
        "parquet_path": args.parquet_path,
        "model": "GraphMixerTime",
        "best_val_mrr": float(max(history["val_mrr"])) if history["val_mrr"] else None,
        "best_epoch": int(np.argmax(history["val_mrr"]) + 1) if history["val_mrr"] else None,
        "total_epochs": len(history["train_loss"]),
        "timing": {
            "avg_epoch_sec": avg_e,
            "avg_sampling_sec": avg_s,
            "avg_forward_sec": avg_f,
            "sampling_fraction_pct": 100 * avg_s / (avg_s + avg_f + 1e-9),
            "total_experiment_sec": total_time,
        },
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        "args": vars(args),
    }
    with open(os.path.join(output_dir, "final_results.json"), "w") as f:
        json.dump(final, f, indent=2)

    t = final["timing"]
    logger.info("=" * 60)
    logger.info("RESULTS | backend=%s", args.sampling_backend)
    logger.info("  Avg epoch:      %.0f sec", t["avg_epoch_sec"])
    logger.info("  Avg sampling:   %.1f sec (%.0f%%)",
                t["avg_sampling_sec"], t["sampling_fraction_pct"])
    logger.info("  Avg forward:    %.1f sec (%.0f%%)",
                t["avg_forward_sec"], 100 - t["sampling_fraction_pct"])
    logger.info("  Best val MRR:   %.4f (epoch %d)",
                final["best_val_mrr"] or 0.0, final["best_epoch"] or 0)
    logger.info("  Total time:     %.1f min", total_time / 60)
    logger.info("=" * 60)

    token = os.environ.get("YADISK_TOKEN", "")
    if token:
        remote = f"{YADISK_EXPERIMENTS_BASE}/{exp_name}"
        try:
            count = upload_directory(output_dir, remote, token)
            logger.info("Uploaded %d files → %s", count, remote)
        except Exception as e:
            logger.error("Upload failed: %s", e)
    else:
        logger.warning("YADISK_TOKEN not set. Results at: %s", output_dir)


def main() -> None:
    """Parse CLI and run experiment."""
    parser = argparse.ArgumentParser(
        description="GraphMixer CUDA sampler benchmark: python vs cpp vs cuda"
    )
    parser.add_argument("--parquet-path", type=str, required=True)
    parser.add_argument("--sampling-backend", type=str, default="cpp",
                        choices=["python", "cpp", "cuda"])
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--output", type=str, default="/tmp/cuda_exp_a10")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=2000,
                        help="2000 recommended: shows 3-4x C++→CUDA speedup. "
                             "200 shows 1.7x (forward overhead dominates).")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--num-neighbors", type=int, default=20)
    parser.add_argument("--hidden-dim", type=int, default=100)
    parser.add_argument("--num-mixer-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--max-val-edges", type=int, default=5000)
    parser.add_argument("--edge-feat-dim", type=int, default=2)

    args = parser.parse_args()
    run_experiment(args)


if __name__ == "__main__":
    main()

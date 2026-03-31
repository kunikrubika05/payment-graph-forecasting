"""CLI launcher for DyGFormer temporal link prediction on stream graphs.

Usage:
    YADISK_TOKEN="..." PYTHONPATH=. python src/models/DyGFormer/dygformer_launcher.py \\
        --parquet-path stream_graph/2020-06-01_2020-08-31.parquet \\
        --edge-feat-dim 2 --epochs 100

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
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.models.DyGFormer.data_utils import load_stream_graph_data, TemporalCSR
from src.models.DyGFormer.dygformer_train import train_dygformer
from src.models.DyGFormer.dygformer_evaluate import evaluate_tgb_style
from src.yadisk_utils import upload_directory

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

YADISK_EXPERIMENTS_BASE = "orbitaal_processed/experiments"


def _save_data_summary(output_dir, data, train_mask, val_mask, test_mask):
    """Save dataset statistics for reproducibility."""
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
    """Save per-epoch training metrics as CSV for plotting."""
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


def _build_eval_infrastructure(parquet_path, train_ratio, val_ratio):
    """Build train_neighbors, active_nodes, and directed eval edges from parquet.

    Reads the DIRECTED parquet to build evaluation infrastructure matching
    the baseline protocol (sg_baselines).

    Returns:
        dict with keys: train_neighbors, active_nodes,
        val_src, val_dst, val_ts, test_src, test_dst, test_ts,
        n_train, n_val, n_test.
    """
    df = pd.read_parquet(parquet_path)
    n_total = len(df)
    train_end = int(n_total * train_ratio)
    val_end = int(n_total * (train_ratio + val_ratio))

    train_df = df.iloc[:train_end]
    val_df = df.iloc[train_end:val_end]
    test_df = df.iloc[val_end:]

    train_src = train_df["src_idx"].values
    train_dst = train_df["dst_idx"].values
    train_neighbors: dict[int, set[int]] = defaultdict(set)
    for s, d in zip(train_src, train_dst):
        train_neighbors[int(s)].add(int(d))
    train_neighbors = dict(train_neighbors)

    active_nodes = np.unique(
        np.concatenate([train_src, train_dst])
    ).astype(np.int64)

    return {
        "train_neighbors": train_neighbors,
        "active_nodes": active_nodes,
        "val_src": val_df["src_idx"].values.astype(np.int32),
        "val_dst": val_df["dst_idx"].values.astype(np.int32),
        "val_ts": val_df["timestamp"].values.astype(np.float64),
        "test_src": test_df["src_idx"].values.astype(np.int32),
        "test_dst": test_df["dst_idx"].values.astype(np.int32),
        "test_ts": test_df["timestamp"].values.astype(np.float64),
        "n_train": len(train_df),
        "n_val": len(val_df),
        "n_test": len(test_df),
    }


def run_experiment(args):
    """Run a full DyGFormer experiment: load data, train, evaluate, upload."""
    total_start = time.time()

    parquet_name = Path(args.parquet_path).stem
    exp_name = args.exp_name if args.exp_name else f"dygformer_{parquet_name}"
    output_dir = os.path.join(args.output, exp_name)
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    log_file = os.path.join(output_dir, "experiment.log")
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    )
    logging.getLogger().addHandler(file_handler)

    logger.info("=" * 60)
    logger.info("DyGFormer Experiment: %s", exp_name)
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

    logger.info("Step 1a: Loading stream graph (undirected) from %s...", args.parquet_path)
    data_start = time.time()
    data, train_mask, val_mask, test_mask = load_stream_graph_data(
        args.parquet_path,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        undirected=True,
    )
    logger.info("Data: %s (%.1f sec)", data, time.time() - data_start)

    logger.info("Step 1b: Building eval infrastructure (directed edges)...")
    eval_infra = _build_eval_infrastructure(
        args.parquet_path, args.train_ratio, args.val_ratio
    )
    logger.info(
        "Directed split: train=%d, val=%d, test=%d. Active nodes: %d",
        eval_infra["n_train"], eval_infra["n_val"], eval_infra["n_test"],
        len(eval_infra["active_nodes"]),
    )
    data_time = time.time() - data_start

    if args.node_feats_path:
        from scripts.compute_stream_node_features import load_node_features as _load_nf
        logger.info("Loading node features from %s...", args.node_feats_path)
        node_feats = _load_nf(args.node_feats_path, data.num_nodes)
        data.node_feats = node_feats
        if args.node_feat_dim == 0:
            args.node_feat_dim = node_feats.shape[1]
        logger.info("Node features: shape=%s, dim=%d", node_feats.shape, args.node_feat_dim)

    _save_data_summary(output_dir, data, train_mask, val_mask, test_mask)

    logger.info("Step 2: Training DyGFormer...")
    train_start = time.time()
    model, history = train_dygformer(
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
        patch_size=args.patch_size,
        time_dim=args.time_dim,
        aligned_dim=args.aligned_dim,
        num_transformer_layers=args.num_transformer_layers,
        num_attention_heads=args.num_attention_heads,
        cooc_dim=args.cooc_dim,
        output_dim=args.output_dim,
        dropout=args.dropout,
        patience=args.patience,
        seed=args.seed,
        max_val_edges=args.max_val_edges,
        use_amp=not args.no_amp,
        edge_feat_dim=args.edge_feat_dim,
        node_feat_dim=args.node_feat_dim,
        neg_per_positive=args.neg_per_positive,
    )
    train_time = time.time() - train_start

    _save_training_curves(output_dir, history)

    from src.models.DyGFormer.data_utils import build_temporal_csr

    logger.info("Step 3a: TGB-style evaluation on val set (directed, baseline protocol)...")
    eval_start = time.time()
    train_csr_for_val = build_temporal_csr(data, train_mask)
    val_metrics = evaluate_tgb_style(
        model=model,
        data=data,
        csr=train_csr_for_val,
        eval_src=eval_infra["val_src"],
        eval_dst=eval_infra["val_dst"],
        eval_ts=eval_infra["val_ts"],
        train_neighbors=eval_infra["train_neighbors"],
        active_nodes=eval_infra["active_nodes"],
        device=device,
        num_neighbors=args.num_neighbors,
        use_amp=not args.no_amp,
        seed=args.seed + 200,
        max_edges=50_000,
    )

    logger.info("Step 3b: TGB-style evaluation on test set (directed, baseline protocol)...")
    all_before_test = train_mask | val_mask
    test_csr = build_temporal_csr(data, all_before_test)
    test_metrics = evaluate_tgb_style(
        model=model,
        data=data,
        csr=test_csr,
        eval_src=eval_infra["test_src"],
        eval_dst=eval_infra["test_dst"],
        eval_ts=eval_infra["test_ts"],
        train_neighbors=eval_infra["train_neighbors"],
        active_nodes=eval_infra["active_nodes"],
        device=device,
        num_neighbors=args.num_neighbors,
        use_amp=not args.no_amp,
        seed=args.seed + 400,
        max_edges=args.max_test_edges if args.max_test_edges else 50_000,
    )
    eval_time = time.time() - eval_start
    total_time = time.time() - total_start

    final_results = {
        "experiment": exp_name,
        "parquet_path": args.parquet_path,
        "model": "DyGFormerTime",
        "val_metrics": val_metrics,
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
    logger.info("  Val MRR:      %.4f", val_metrics["mrr"])
    logger.info("  Val Hits@1:   %.4f", val_metrics["hits@1"])
    logger.info("  Val Hits@3:   %.4f", val_metrics["hits@3"])
    logger.info("  Val Hits@10:  %.4f", val_metrics["hits@10"])
    logger.info("  Test MRR:     %.4f", test_metrics["mrr"])
    logger.info("  Test Hits@1:  %.4f", test_metrics["hits@1"])
    logger.info("  Test Hits@3:  %.4f", test_metrics["hits@3"])
    logger.info("  Test Hits@10: %.4f", test_metrics["hits@10"])
    logger.info(
        "  Best val MRR (training): %.4f (epoch %d)",
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
        description="DyGFormer temporal link prediction on stream graphs"
    )
    parser.add_argument(
        "--parquet-path",
        type=str,
        required=True,
        help="Path to stream graph parquet file (src_idx, dst_idx, timestamp, btc, usd)",
    )
    parser.add_argument(
        "--train-ratio", type=float, default=0.7,
        help="Fraction of edges for training (chronological split)",
    )
    parser.add_argument(
        "--val-ratio", type=float, default=0.15,
        help="Fraction of edges for validation",
    )
    parser.add_argument(
        "--output", type=str, default="/tmp/dygformer_results",
    )
    parser.add_argument(
        "--exp-name",
        type=str,
        default=None,
        help="Experiment name. Defaults to 'dygformer_<parquet_stem>'.",
    )

    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=200)
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument(
        "--num-neighbors", type=int, default=32,
        help="K most-recent neighbors sampled per node. Paper uses 32-4096. "
             "Increase for datasets with longer interaction histories.",
    )
    parser.add_argument(
        "--patch-size", type=int, default=1,
        help="Patch size P. Paper uses 1-128, scaling with num_neighbors. "
             "P=1 means no patching (each neighbor is a separate token). "
             "For num_neighbors=32, P=1; for 512, P=16; for 4096, P=128.",
    )
    parser.add_argument(
        "--time-dim", type=int, default=100,
        help="Time encoding dimension d_T (paper: 100).",
    )
    parser.add_argument(
        "--aligned-dim", type=int, default=50,
        help="Per-channel aligned encoding dimension d (paper: 50).",
    )
    parser.add_argument(
        "--num-transformer-layers", type=int, default=2,
        help="Number of Transformer layers L (paper: 2).",
    )
    parser.add_argument(
        "--num-attention-heads", type=int, default=2,
        help="Number of attention heads I (paper: 2).",
    )
    parser.add_argument(
        "--cooc-dim", type=int, default=50,
        help="Co-occurrence encoding dimension d_C (paper: 50).",
    )
    parser.add_argument(
        "--output-dim", type=int, default=172,
        help="Output embedding dimension d_out (paper: 172).",
    )
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--max-val-edges", type=int, default=2000)
    parser.add_argument(
        "--max-test-edges", type=int, default=None,
        help="Subsample test set for fast evaluation (default: full test set)",
    )
    parser.add_argument(
        "--edge-feat-dim", type=int, default=2,
        help="Dimension of per-neighbor edge features. "
             "2 = use btc+usd. 0 = time-only mode.",
    )
    parser.add_argument(
        "--neg-per-positive", type=int, default=5,
        help="Number of random negative edges per positive during training "
             "(checklist: 5). Eval always uses 50 hist + 50 rand.",
    )
    parser.add_argument(
        "--node-feat-dim", type=int, default=0,
        help="Dimension of per-neighbor node features (0 = disabled). "
             "Auto-detected from --node-feats-path if not set.",
    )
    parser.add_argument(
        "--node-feats-path",
        type=str,
        default=None,
        help="Path to node features parquet (features_10.parquet or features_25.parquet).",
    )

    args = parser.parse_args()
    run_experiment(args)


if __name__ == "__main__":
    main()

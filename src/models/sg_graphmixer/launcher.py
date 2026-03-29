"""CLI launcher for GraphMixer on stream graph data.

Usage:
    YADISK_TOKEN="..." PYTHONPATH=. python -u src/models/sg_graphmixer/launcher.py \
        --period period_10 --output /tmp/sg_graphmixer_results --upload \
        2>&1 | tee /tmp/sg_graphmixer.log
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

from sg_baselines.config import ExperimentConfig, PERIODS, YADISK_EXPERIMENTS_BASE
from sg_baselines.data import (
    build_train_neighbor_sets,
    load_adjacency,
    load_node_features_sparse,
    load_stream_graph,
    split_stream_graph,
)
from src.models.sg_graphmixer.data_utils import (
    build_stream_graph_data,
    filter_eval_edges_by_known_nodes,
)
from src.models.sg_graphmixer.train import train_graphmixer
from src.models.sg_graphmixer.evaluate import evaluate_tgb_style
from src.models.data_utils import TemporalCSR


def main():
    parser = argparse.ArgumentParser(
        description="Train and evaluate GraphMixer on stream graph"
    )
    parser.add_argument("--period", type=str, default="period_10",
                        choices=list(PERIODS.keys()))
    parser.add_argument("--output", type=str, default="/tmp/sg_graphmixer_results")
    parser.add_argument("--data-dir", type=str, default="/tmp/sg_baselines_data")
    parser.add_argument("--upload", action="store_true")
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--hidden-dim", type=int, default=200)
    parser.add_argument("--num-neighbors", type=int, default=30)
    parser.add_argument("--num-mixer-layers", type=int, default=1)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=3e-6)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--batch-size", type=int, default=4000)
    parser.add_argument("--num-epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--neg-per-positive", type=int, default=1)
    parser.add_argument("--max-val-queries", type=int, default=10_000,
                        help="Val queries per epoch for early stopping (10K for speed)")
    parser.add_argument("--max-test-queries", type=int, default=50_000,
                        help="Final val+test queries (50K for accuracy)")
    parser.add_argument("--n-negatives", type=int, default=100)
    parser.add_argument("--device", type=str, default="auto")

    args = parser.parse_args()

    token = os.environ.get("YADISK_TOKEN", "")
    if not token:
        print("ERROR: YADISK_TOKEN environment variable required", file=sys.stderr)
        sys.exit(1)

    period_cfg = PERIODS[args.period]
    config = ExperimentConfig(
        period_name=args.period,
        fraction=period_cfg["fraction"],
        label=period_cfg["label"],
        local_data_dir=args.data_dir,
        output_dir=args.output,
        upload=args.upload,
        random_seed=args.seed,
        n_negatives=args.n_negatives,
    )

    exp_name = f"exp_sg_graphmixer_{config.label}"
    exp_dir = os.path.join(args.output, exp_name)
    os.makedirs(exp_dir, exist_ok=True)

    summary_path = os.path.join(exp_dir, "summary.json")
    if os.path.exists(summary_path):
        print(f"SKIP: {summary_path} already exists (resume mode)")
        return

    t_total_start = time.time()

    print(f"\n{'='*60}", flush=True)
    print(f"GraphMixer on Stream Graph — {exp_name}", flush=True)
    print(f"Period: {config.period_name} (fraction={config.fraction})", flush=True)
    print(f"{'='*60}\n", flush=True)

    print("[1/7] Loading stream graph...", flush=True)
    df = load_stream_graph(config, token)
    train_edges, val_edges, test_edges = split_stream_graph(df, config)
    del df

    print("\n[2/7] Loading features and adjacency...", flush=True)
    node_idx, features_df = load_node_features_sparse(config, token)
    node_features = features_df.values.astype(np.float32)
    node_mapping, adj_directed, adj_undirected = load_adjacency(config, token)
    assert np.array_equal(node_idx, node_mapping), \
        "Node features and adjacency mappings don't match!"
    active_nodes = node_mapping

    print("\n[3/7] Building train neighbor sets...", flush=True)
    t0 = time.time()
    train_neighbors = build_train_neighbor_sets(train_edges)
    print(f"  {len(train_neighbors):,} sources ({time.time() - t0:.1f}s)", flush=True)

    print("\n[4/7] Building TemporalEdgeData...", flush=True)
    data, train_mask, val_mask, test_mask = build_stream_graph_data(
        train_edges, val_edges, test_edges,
        node_mapping, node_features, undirected=True,
    )
    del train_edges, val_edges, test_edges, node_features

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    print(f"\n[5/7] Training GraphMixer (device={device})...", flush=True)
    model, history = train_graphmixer(
        data=data,
        train_mask=train_mask,
        val_mask=val_mask,
        train_neighbors=train_neighbors,
        active_nodes=active_nodes,
        node_mapping=node_mapping,
        output_dir=exp_dir,
        device=device,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        num_neighbors=args.num_neighbors,
        hidden_dim=args.hidden_dim,
        num_mixer_layers=args.num_mixer_layers,
        dropout=args.dropout,
        patience=args.patience,
        seed=args.seed,
        max_val_queries=args.max_val_queries,
        n_negatives=args.n_negatives,
        neg_per_positive=args.neg_per_positive,
    )

    print("\n[6/7] Final evaluation...", flush=True)
    train_csr = TemporalCSR(
        data.num_nodes,
        data.src[train_mask],
        data.dst[train_mask],
        data.timestamps[train_mask],
        np.where(train_mask)[0].astype(np.int64),
    )

    print("  Evaluating on val (final)...", flush=True)
    val_metrics = evaluate_tgb_style(
        model=model, data=data, csr=train_csr,
        eval_mask=val_mask, device=device,
        num_neighbors=args.num_neighbors,
        train_neighbors=train_neighbors,
        active_nodes=active_nodes,
        node_mapping=node_mapping,
        n_negatives=args.n_negatives,
        max_queries=args.max_test_queries,
        seed=args.seed + 300,
    )
    print(f"  Val: MRR={val_metrics['mrr']:.4f} "
          f"H@1={val_metrics['hits@1']:.3f} H@10={val_metrics['hits@10']:.3f}", flush=True)

    print("  Evaluating on test...", flush=True)
    test_metrics = evaluate_tgb_style(
        model=model, data=data, csr=train_csr,
        eval_mask=test_mask, device=device,
        num_neighbors=args.num_neighbors,
        train_neighbors=train_neighbors,
        active_nodes=active_nodes,
        node_mapping=node_mapping,
        n_negatives=args.n_negatives,
        max_queries=args.max_test_queries,
        seed=args.seed + 400,
    )
    print(f"  Test: MRR={test_metrics['mrr']:.4f} "
          f"H@1={test_metrics['hits@1']:.3f} H@10={test_metrics['hits@10']:.3f}", flush=True)

    elapsed_total = time.time() - t_total_start

    summary = {
        "experiment": exp_name,
        "period": config.period_name,
        "fraction": config.fraction,
        "model": "graphmixer",
        "hyperparameters": {
            "hidden_dim": args.hidden_dim,
            "num_neighbors": args.num_neighbors,
            "num_mixer_layers": args.num_mixer_layers,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "dropout": args.dropout,
            "batch_size": args.batch_size,
            "neg_per_positive": args.neg_per_positive,
        },
        "training": {
            "best_epoch": history.get("val_mrr", [0]).index(max(history.get("val_mrr", [0]))) + 1,
            "total_epochs": len(history.get("train_loss", [])),
            "best_val_mrr_during_training": max(history.get("val_mrr", [0])),
        },
        "val": val_metrics,
        "test": test_metrics,
        "elapsed_seconds": elapsed_total,
        "baseline_reference": {
            "cn_test_mrr": 0.8725,
            "aa_test_mrr": 0.8709,
            "jaccard_test_mrr": 0.8621,
            "pa_test_mrr": 0.8036,
        },
    }

    print(f"\n[7/7] Saving results...", flush=True)
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  Saved {summary_path}", flush=True)

    if config.upload:
        _upload_results(exp_dir, exp_name, token)

    print(f"\n{'='*60}", flush=True)
    print(f"DONE: {exp_name}", flush=True)
    print(f"  Val  MRR={val_metrics['mrr']:.4f}  H@1={val_metrics['hits@1']:.3f}  "
          f"H@10={val_metrics['hits@10']:.3f}", flush=True)
    print(f"  Test MRR={test_metrics['mrr']:.4f}  H@1={test_metrics['hits@1']:.3f}  "
          f"H@10={test_metrics['hits@10']:.3f}", flush=True)
    print(f"  Baselines: CN={0.8725:.4f}  AA={0.8709:.4f}  PA={0.8036:.4f}", flush=True)
    print(f"  Total time: {elapsed_total / 60:.1f} min", flush=True)
    print(f"{'='*60}", flush=True)


def _upload_results(exp_dir: str, exp_name: str, token: str):
    """Upload experiment results to Yandex.Disk."""
    from src.yadisk_utils import upload_directory, create_remote_folder_recursive

    remote_dir = f"{YADISK_EXPERIMENTS_BASE}/{exp_name}"
    print(f"  Uploading to {remote_dir}...", flush=True)
    create_remote_folder_recursive(remote_dir, token)
    n = upload_directory(exp_dir, remote_dir, token)
    print(f"  Uploaded {n} files", flush=True)


if __name__ == "__main__":
    main()

"""Main entry point for stream graph baseline experiments.

Usage:
    YADISK_TOKEN="..." PYTHONPATH=. python -u sg_baselines/run.py \
        --period period_10 --output /tmp/sg_baselines_results --upload \
        2>&1 | tee /tmp/sg_baselines.log
"""

import argparse
import json
import os
import sys
import time

import numpy as np

from sg_baselines.config import (
    ExperimentConfig,
    PERIODS,
    YADISK_EXPERIMENTS_BASE,
    get_experiment_configs,
)
from sg_baselines.data import (
    build_train_neighbor_sets,
    load_adjacency,
    load_node_features_sparse,
    load_stream_graph,
    split_stream_graph,
)
from sg_baselines.heuristics import evaluate_heuristics
from sg_baselines.ml_pipeline import hp_search, prepare_training_data, train_and_evaluate


def run_experiment(config: ExperimentConfig, token: str):
    """Run all baselines for one period configuration."""
    exp_name = f"exp_sg_{config.label}"
    exp_dir = os.path.join(config.output_dir, exp_name)
    os.makedirs(exp_dir, exist_ok=True)

    summary_path = os.path.join(exp_dir, "summary.json")
    if os.path.exists(summary_path):
        print(f"\nSKIP {exp_name}: summary.json already exists (resume mode)", flush=True)
        return

    print(f"\n{'='*60}", flush=True)
    print(f"EXPERIMENT: {exp_name}", flush=True)
    print(f"Period: {config.period_name} (fraction={config.fraction})", flush=True)
    print(f"Split: train={config.train_ratio}, val={config.val_ratio}, "
          f"test={config.test_ratio:.2f}", flush=True)
    print(f"{'='*60}", flush=True)

    t_start = time.time()

    config_path = os.path.join(exp_dir, "config.json")
    with open(config_path, "w") as f:
        json.dump(config.to_dict(), f, indent=2)

    print("\n[1/6] Loading data...", flush=True)
    df = load_stream_graph(config, token)
    train_edges, val_edges, test_edges = split_stream_graph(df, config)
    del df

    print("\n[2/6] Loading features and adjacency...", flush=True)
    node_idx, features_df = load_node_features_sparse(config, token)
    node_features = features_df.values.astype(np.float32)
    node_mapping, adj_directed, adj_undirected = load_adjacency(config, token)

    assert np.array_equal(node_idx, node_mapping), (
        "Node features and adjacency node mappings don't match!"
    )
    active_nodes = node_mapping

    print("\n[3/6] Building train neighbor sets...", flush=True)
    t0 = time.time()
    train_neighbors = build_train_neighbor_sets(train_edges)
    print(f"  Built neighbor sets for {len(train_neighbors):,} sources "
          f"({time.time() - t0:.1f}s)", flush=True)

    all_results = {"config": config.to_dict(), "heuristics": {}, "ml": {}}

    if config.heuristics:
        print("\n[4/6] Heuristic baselines...", flush=True)
        for split_name, edges in [("val", val_edges), ("test", test_edges)]:
            heur_results = evaluate_heuristics(
                edges, train_neighbors, active_nodes,
                node_mapping, adj_undirected,
                config.heuristics, config.n_negatives,
                seed=config.random_seed + (10 if split_name == "val" else 20),
                split_name=split_name,
                max_queries=50_000,
            )
            for heur_name, metrics in heur_results.items():
                all_results["heuristics"].setdefault(heur_name, {})[split_name] = metrics
    else:
        print("\n[4/6] Skipping heuristics", flush=True)

    if config.models:
        print("\n[5/6] ML baselines...", flush=True)
        print("  Preparing training data...", flush=True)
        X_train, y_train = prepare_training_data(
            train_edges, node_idx, node_features,
            node_mapping, adj_directed, adj_undirected,
            train_neighbors, active_nodes, config,
        )

        for model_name in config.models:
            print(f"\n  --- {model_name} ---", flush=True)
            best_params, hp_results = hp_search(
                model_name, X_train, y_train, val_edges,
                node_idx, node_features,
                node_mapping, adj_directed, adj_undirected,
                train_neighbors, active_nodes, config,
            )

            hp_path = os.path.join(exp_dir, f"hp_search_{model_name}.json")
            with open(hp_path, "w") as f:
                json.dump(hp_results, f, indent=2)

            eval_results = train_and_evaluate(
                model_name, best_params,
                X_train, y_train, val_edges, test_edges,
                node_idx, node_features,
                node_mapping, adj_directed, adj_undirected,
                train_neighbors, active_nodes, config,
            )

            all_results["ml"][model_name] = {
                "best_params": best_params,
                "val": eval_results["val"],
                "test": eval_results["test"],
            }
    else:
        print("\n[5/6] Skipping ML baselines", flush=True)

    elapsed = time.time() - t_start
    all_results["elapsed_seconds"] = elapsed

    print(f"\n[6/6] Saving results...", flush=True)
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"  Saved {summary_path}", flush=True)

    _print_summary_table(all_results)

    if config.upload:
        _upload_results(exp_dir, exp_name, token)

    print(f"\n  Total time: {elapsed / 60:.1f} min", flush=True)


def _print_summary_table(results: dict):
    """Print a summary table of all results."""
    print(f"\n{'='*70}", flush=True)
    print(f"{'Method':<15} {'Val MRR':>10} {'Test MRR':>10} "
          f"{'Test H@1':>10} {'Test H@10':>10}", flush=True)
    print(f"{'-'*70}", flush=True)

    for heur_name, splits in results.get("heuristics", {}).items():
        val_mrr = splits.get("val", {}).get("mrr", float("nan"))
        test_mrr = splits.get("test", {}).get("mrr", float("nan"))
        test_h1 = splits.get("test", {}).get("hits@1", float("nan"))
        test_h10 = splits.get("test", {}).get("hits@10", float("nan"))
        print(f"{heur_name:<15} {val_mrr:>10.4f} {test_mrr:>10.4f} "
              f"{test_h1:>10.4f} {test_h10:>10.4f}", flush=True)

    for model_name, data in results.get("ml", {}).items():
        val_mrr = data.get("val", {}).get("mrr", float("nan"))
        test_mrr = data.get("test", {}).get("mrr", float("nan"))
        test_h1 = data.get("test", {}).get("hits@1", float("nan"))
        test_h10 = data.get("test", {}).get("hits@10", float("nan"))
        print(f"{model_name:<15} {val_mrr:>10.4f} {test_mrr:>10.4f} "
              f"{test_h1:>10.4f} {test_h10:>10.4f}", flush=True)

    print(f"{'='*70}", flush=True)


def _upload_results(exp_dir: str, exp_name: str, token: str):
    """Upload experiment results to Yandex.Disk."""
    from src.yadisk_utils import upload_directory, create_remote_folder_recursive

    remote_dir = f"{YADISK_EXPERIMENTS_BASE}/{exp_name}"
    print(f"  Uploading to {remote_dir}...", flush=True)
    create_remote_folder_recursive(remote_dir, token)
    n = upload_directory(exp_dir, remote_dir, token)
    print(f"  Uploaded {n} files", flush=True)


def main():
    parser = argparse.ArgumentParser(
        description="Run stream graph baseline experiments"
    )
    parser.add_argument(
        "--period", type=str, default="all",
        choices=["all", "period_10", "period_25"],
        help="Which period to run (default: all)",
    )
    parser.add_argument(
        "--output", type=str, default="/tmp/sg_baselines_results",
        help="Output directory for results",
    )
    parser.add_argument(
        "--data-dir", type=str, default="/tmp/sg_baselines_data",
        help="Directory for cached data files",
    )
    parser.add_argument("--upload", action="store_true", help="Upload results to Yandex.Disk")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--models", type=str, nargs="*", default=None,
        help="ML models to run (default: all). E.g., --models logreg catboost",
    )
    parser.add_argument(
        "--heuristics", type=str, nargs="*", default=None,
        help="Heuristics to run (default: all). E.g., --heuristics cn aa",
    )
    parser.add_argument(
        "--skip-ml", action="store_true", help="Skip ML baselines (heuristics only)",
    )
    parser.add_argument(
        "--skip-heuristics", action="store_true", help="Skip heuristics (ML only)",
    )
    args = parser.parse_args()

    token = os.environ.get("YADISK_TOKEN", "")
    if not token:
        print("ERROR: YADISK_TOKEN environment variable required", flush=True)
        sys.exit(1)

    configs = get_experiment_configs()
    if args.period != "all":
        configs = [c for c in configs if c.period_name == args.period]

    for config in configs:
        config.output_dir = args.output
        config.local_data_dir = args.data_dir
        config.upload = args.upload
        config.random_seed = args.seed

        if args.models is not None:
            config.models = args.models
        if args.heuristics is not None:
            config.heuristics = args.heuristics
        if args.skip_ml:
            config.models = []
        if args.skip_heuristics:
            config.heuristics = []

        try:
            run_experiment(config, token)
        except Exception as e:
            import traceback
            error_msg = f"ERROR in {config.period_name}: {e}"
            print(error_msg, flush=True)
            traceback.print_exc()
            error_path = os.path.join(args.output, f"exp_sg_{config.label}", "error.txt")
            os.makedirs(os.path.dirname(error_path), exist_ok=True)
            with open(error_path, "w") as f:
                f.write(traceback.format_exc())

    print("\nAll experiments done.", flush=True)


if __name__ == "__main__":
    main()

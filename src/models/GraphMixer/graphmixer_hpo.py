"""Optuna hyperparameter optimization for GraphMixer on stream graphs.

Uses TemporalGraphSampler (cpp backend) for fast neighbor sampling.
Builds samplers once, creates fresh model per trial.

Usage:
    PYTHONPATH=. python src/models/GraphMixer/graphmixer_hpo.py \\
        --parquet-path stream_graph/week.parquet \\
        --n-trials 20 --hpo-epochs 10 --edge-feat-dim 2 \\
        --output /tmp/graphmixer_hpo 2>&1 | tee /tmp/hpo.log

Key hyperparameters searched:
    - hidden_dim: {50, 100, 200}
    - num_neighbors: {10, 20, 30}
    - num_mixer_layers: {1, 2, 3}
    - lr: log-uniform [1e-4, 1e-2]
    - weight_decay: log-uniform [1e-6, 1e-3]
    - dropout: [0.0, 0.3]
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

try:
    import optuna
    from optuna.pruners import MedianPruner
    from optuna.samplers import TPESampler
except ImportError:
    raise ImportError("Optuna is required for HPO: pip install optuna")

from src.models.stream_graph_data import load_stream_graph_data
from src.models.GraphMixer.graphmixer import GraphMixerTime
from src.models.cuda_exp_graphmixer_a10.train import build_sampler, train_epoch, validate

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

YADISK_HPO_BASE = "orbitaal_processed/experiments/graphmixer_hpo"


def create_objective(
    data,
    train_mask,
    val_mask,
    device,
    hpo_epochs,
    max_val_edges,
    use_amp,
    edge_feat_dim=2,
    node_feat_dim=0,
    batch_size=2000,
):
    """Create an Optuna objective function with pre-loaded data and samplers.

    Samplers are built once and shared across all trials to avoid redundant I/O.
    Each trial creates a fresh model and optimizer.

    Args:
        data: TemporalEdgeData loaded from stream graph.
        train_mask: Boolean training edge mask.
        val_mask: Boolean validation edge mask.
        device: Torch device.
        hpo_epochs: Maximum epochs per trial.
        max_val_edges: Maximum validation edges per evaluation.
        use_amp: Enable mixed precision.
        edge_feat_dim: Per-neighbor edge feature dimension.
        node_feat_dim: Query-node feature dimension (unused, reserved).
        batch_size: Training batch size (fixed, not searched).

    Returns:
        Objective function for optuna.study.optimize().
    """
    logger.info("Building train sampler (cpp)...")
    train_sampler = build_sampler(data, train_mask, backend="cpp")
    logger.info("Building val sampler (cpp)...")
    val_sampler = build_sampler(data, train_mask | val_mask, backend="cpp")

    train_indices = np.where(train_mask)[0]
    val_indices = np.where(val_mask)[0]
    amp_enabled = use_amp and device.type == "cuda"

    def objective(trial):
        hidden_dim = trial.suggest_categorical("hidden_dim", [50, 100, 200])
        num_neighbors = trial.suggest_categorical("num_neighbors", [10, 20, 30])
        num_mixer_layers = trial.suggest_int("num_mixer_layers", 1, 3)
        lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
        weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)
        dropout = trial.suggest_float("dropout", 0.0, 0.3, step=0.05)

        seed = 42
        rng = np.random.default_rng(seed)
        torch.manual_seed(seed)
        if device.type == "cuda":
            torch.cuda.manual_seed(seed)

        model = GraphMixerTime(
            hidden_dim=hidden_dim,
            num_neighbors=num_neighbors,
            num_mixer_layers=num_mixer_layers,
            dropout=dropout,
            edge_feat_dim=edge_feat_dim,
            node_feat_dim=node_feat_dim,
        ).to(device)

        optimizer = torch.optim.Adam(
            model.parameters(), lr=lr, weight_decay=weight_decay
        )
        scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

        best_mrr = 0.0

        for epoch in range(1, hpo_epochs + 1):
            train_epoch(
                model, data, train_sampler, train_indices,
                optimizer, scaler, device, batch_size,
                num_neighbors, amp_enabled, rng,
            )

            val_metrics = validate(
                model, data, val_sampler, val_indices,
                device, num_neighbors, max_val_edges, amp_enabled,
                np.random.default_rng(42),
            )

            mrr = val_metrics["mrr"]
            best_mrr = max(best_mrr, mrr)

            trial.report(mrr, epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()

        return best_mrr

    return objective


def main():
    parser = argparse.ArgumentParser(
        description="GraphMixer hyperparameter optimization via Optuna"
    )
    parser.add_argument("--parquet-path", type=str, required=True)
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--output", type=str, default="/tmp/graphmixer_hpo")
    parser.add_argument("--n-trials", type=int, default=20)
    parser.add_argument("--hpo-epochs", type=int, default=10)
    parser.add_argument("--max-val-edges", type=int, default=3000)
    parser.add_argument("--batch-size", type=int, default=2000,
                        help="Fixed batch size for HPO (not searched).")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--edge-feat-dim", type=int, default=2,
                        help="2 = btc+usd, 0 = time-only.")
    parser.add_argument("--node-feat-dim", type=int, default=0)

    args = parser.parse_args()
    Path(args.output).mkdir(parents=True, exist_ok=True)

    log_file = os.path.join(args.output, "hpo.log")
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    )
    logging.getLogger().addHandler(file_handler)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)
    if device.type == "cuda":
        logger.info("GPU: %s  (%.1f GB)",
                    torch.cuda.get_device_name(0),
                    torch.cuda.get_device_properties(0).total_memory / 1e9)

    logger.info("Loading stream graph: %s", args.parquet_path)
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

    objective = create_objective(
        data, train_mask, val_mask, device,
        hpo_epochs=args.hpo_epochs,
        max_val_edges=args.max_val_edges,
        use_amp=not args.no_amp,
        edge_feat_dim=args.edge_feat_dim,
        node_feat_dim=args.node_feat_dim,
        batch_size=args.batch_size,
    )

    parquet_name = Path(args.parquet_path).stem
    study = optuna.create_study(
        direction="maximize",
        sampler=TPESampler(seed=args.seed),
        pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=3),
        study_name=f"graphmixer_hpo_{parquet_name}",
    )

    logger.info("Starting HPO: %d trials × %d epochs", args.n_trials, args.hpo_epochs)
    start = time.time()
    study.optimize(objective, n_trials=args.n_trials, show_progress_bar=True)
    elapsed = time.time() - start

    best = study.best_trial
    logger.info("=" * 60)
    logger.info("HPO COMPLETE (%.1f min)", elapsed / 60)
    logger.info("Best trial #%d: MRR=%.4f", best.number, best.value)
    logger.info("Best params: %s", best.params)
    logger.info("=" * 60)

    results = {
        "best_trial": best.number,
        "best_mrr": best.value,
        "best_params": best.params,
        "n_trials": len(study.trials),
        "n_completed": sum(
            1 for t in study.trials
            if t.state == optuna.trial.TrialState.COMPLETE
        ),
        "n_pruned": sum(
            1 for t in study.trials
            if t.state == optuna.trial.TrialState.PRUNED
        ),
        "total_time_sec": elapsed,
        "parquet_path": args.parquet_path,
        "hpo_epochs": args.hpo_epochs,
        "batch_size": args.batch_size,
        "all_trials": [
            {
                "number": t.number,
                "value": t.value,
                "params": t.params,
                "state": t.state.name,
            }
            for t in study.trials
        ],
    }
    with open(os.path.join(args.output, "hpo_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    p = best.params
    # TODO(REFACTORING): switch this recommendation to the package launcher once
    # the GraphMixerTime + CUDA sampler training path has a first-class
    # payment_graph_forecasting model contract distinct from snapshot GraphMixer
    # and sg_graphmixer.
    train_cmd = (
        f"YADISK_TOKEN=\"$YADISK_TOKEN\" PYTHONPATH=. python"
        f" src/models/cuda_exp_graphmixer_a10/launcher.py"
        f" --parquet-path {args.parquet_path}"
        f" --sampling-backend cuda"
        f" --epochs 100"
        f" --batch-size {args.batch_size}"
        f" --edge-feat-dim {args.edge_feat_dim}"
        f" --hidden-dim {p['hidden_dim']}"
        f" --num-neighbors {p['num_neighbors']}"
        f" --num-mixer-layers {p['num_mixer_layers']}"
        f" --lr {p['lr']:.6f}"
        f" --weight-decay {p['weight_decay']:.8f}"
        f" --dropout {p['dropout']:.2f}"
        f" --patience 15"
        f" --output /tmp/graphmixer_final"
        f" 2>&1 | tee /tmp/graphmixer_final.log"
    )

    logger.info("Recommended training command:\n%s", train_cmd)

    cmd_path = os.path.join(args.output, "best_train_command.sh")
    with open(cmd_path, "w") as f:
        f.write("#!/bin/bash\n")
        f.write(train_cmd + "\n")
    os.chmod(cmd_path, 0o755)
    logger.info("Command saved to %s", cmd_path)

    token = os.environ.get("YADISK_TOKEN", "")
    if token:
        from src.yadisk_utils import upload_directory
        remote = f"{YADISK_HPO_BASE}/{parquet_name}"
        try:
            count = upload_directory(args.output, remote, token)
            logger.info("Uploaded %d files → %s", count, remote)
        except Exception as e:
            logger.error("Upload failed: %s", e)
    else:
        logger.warning("YADISK_TOKEN not set — results only at: %s", args.output)


if __name__ == "__main__":
    main()

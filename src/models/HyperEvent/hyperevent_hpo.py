"""Optuna hyperparameter optimization for HyperEvent on stream graphs.

Usage:
    PYTHONPATH=. python src/models/HyperEvent/hyperevent_hpo.py \\
        --parquet-path stream_graph/2020-06-01_2020-08-31.parquet \\
        --n-trials 30 --hpo-epochs 10 \\
        --output /tmp/hyperevent_hpo 2>&1 | tee /tmp/hyperevent_hpo.log

Designed to run within ~2 hours on a GPU. Uses MedianPruner to terminate
underperforming trials early.

Key hyperparameters searched:
    - n_neighbor:  {10, 15, 20, 30, 50}
    - n_latest:    {5, 8, 10}
    - d_model:     {32, 64, 128}
    - n_heads:     {2, 4, 8}
    - n_layers:    {1, 2, 3}
    - lr:          log-uniform [1e-4, 1e-2]
    - weight_decay: log-uniform [1e-6, 1e-3]
    - dropout:     [0.0, 0.3]
    - batch_size:  {100, 200, 400}

Note: n_heads must divide d_model; invalid combinations are pruned automatically.
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
from payment_graph_forecasting.experiments.hpo_artifacts import (
    write_best_training_artifacts,
)

try:
    import optuna
    from optuna.pruners import MedianPruner
    from optuna.samplers import TPESampler
except ImportError:
    raise ImportError("Optuna is required for HPO: pip install optuna")

from src.models.HyperEvent.data_utils import load_stream_graph_data
from src.models.HyperEvent.hyperevent import HyperEventModel
from src.models.HyperEvent.hyperevent_train import (
    AdjacencyTable,
    train_epoch,
    validate,
    build_adj_from_mask,
    RELATIONAL_DIM,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def create_objective(
    data,
    train_mask,
    val_mask,
    device,
    hpo_epochs,
    max_val_edges,
    use_amp,
):
    """Create an Optuna objective with pre-loaded data.

    Data is loaded once and shared across all trials to avoid I/O overhead.
    Each trial creates a fresh model, adjacency table, and optimizer.

    Args:
        data: TemporalEdgeData loaded from stream graph.
        train_mask: Boolean training edge mask.
        val_mask: Boolean validation edge mask.
        device: Torch device.
        hpo_epochs: Maximum epochs per trial.
        max_val_edges: Maximum validation edges per epoch (for speed).
        use_amp: Enable mixed precision.

    Returns:
        Objective function for optuna.study.optimize().
    """
    train_indices = np.where(train_mask)[0]
    val_indices = np.where(val_mask)[0]

    def objective(trial):
        n_neighbor = trial.suggest_categorical("n_neighbor", [10, 15, 20, 30, 50])
        n_latest = trial.suggest_categorical("n_latest", [5, 8, 10])
        d_model = trial.suggest_categorical("d_model", [32, 64, 128])
        n_heads = trial.suggest_categorical("n_heads", [2, 4, 8])
        n_layers = trial.suggest_int("n_layers", 1, 3)
        lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
        weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)
        dropout = trial.suggest_float("dropout", 0.0, 0.3, step=0.05)
        batch_size = trial.suggest_categorical("batch_size", [100, 200, 400])

        # n_heads must divide d_model
        if d_model % n_heads != 0:
            raise optuna.TrialPruned()

        seed = 42
        rng = np.random.default_rng(seed)
        torch.manual_seed(seed)
        if device.type == "cuda":
            torch.cuda.manual_seed(seed)

        model = HyperEventModel(
            feat_dim=RELATIONAL_DIM,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            dropout=dropout,
        ).to(device)

        optimizer = torch.optim.Adam(
            model.parameters(), lr=lr, weight_decay=weight_decay
        )
        amp_enabled = use_amp and device.type == "cuda"
        scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)

        best_mrr = 0.0

        for epoch in range(1, hpo_epochs + 1):
            adj = AdjacencyTable(data.num_nodes, n_neighbor)

            train_epoch(
                model, data, adj, train_indices, optimizer, device,
                batch_size=batch_size,
                n_latest=n_latest,
                use_amp=use_amp,
                scaler=scaler,
                rng=rng,
            )

            val_adj = build_adj_from_mask(data, train_mask, n_neighbor)
            val_metrics = validate(
                model, data, val_adj, val_indices, device,
                n_latest=n_latest,
                max_eval_edges=max_val_edges,
                use_amp=use_amp,
                rng=np.random.default_rng(42),
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
        description="HyperEvent hyperparameter optimization via Optuna on stream graphs"
    )
    parser.add_argument(
        "--parquet-path", type=str, required=True,
        help="Path to stream graph parquet file",
    )
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--output", type=str, default="/tmp/hyperevent_hpo")
    parser.add_argument("--n-trials", type=int, default=30)
    parser.add_argument("--hpo-epochs", type=int, default=10)
    parser.add_argument("--max-val-edges", type=int, default=3000)
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--seed", type=int, default=42)

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
        logger.info("GPU: %s", torch.cuda.get_device_name(0))

    logger.info("Loading stream graph: %s", args.parquet_path)
    data, train_mask, val_mask, _ = load_stream_graph_data(
        args.parquet_path,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        undirected=True,
    )
    logger.info("Data: %s", data)

    objective = create_objective(
        data, train_mask, val_mask, device,
        hpo_epochs=args.hpo_epochs,
        max_val_edges=args.max_val_edges,
        use_amp=not args.no_amp,
    )

    parquet_name = Path(args.parquet_path).stem
    study = optuna.create_study(
        direction="maximize",
        sampler=TPESampler(seed=args.seed),
        pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=3),
        study_name=f"hyperevent_hpo_{parquet_name}",
    )

    logger.info(
        "Starting HPO: %d trials, %d epochs each",
        args.n_trials, args.hpo_epochs,
    )
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
        "n_completed": len([
            t for t in study.trials
            if t.state == optuna.trial.TrialState.COMPLETE
        ]),
        "n_pruned": len([
            t for t in study.trials
            if t.state == optuna.trial.TrialState.PRUNED
        ]),
        "total_time_sec": elapsed,
        "parquet_path": args.parquet_path,
        "hpo_epochs": args.hpo_epochs,
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

    artifact_paths = write_best_training_artifacts(
        "hyperevent",
        args,
        best.params,
        output_dir=args.output,
    )
    logger.info("Package-facing best spec saved to %s", artifact_paths["spec_path"])
    logger.info("Recommended training command:\n%s", artifact_paths["command"])
    logger.info("Command saved to %s", artifact_paths["command_path"])


if __name__ == "__main__":
    main()

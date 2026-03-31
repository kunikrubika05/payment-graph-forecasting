"""Optuna hyperparameter optimization for EAGLE-Time on stream graphs.

Usage:
    PYTHONPATH=. python src/models/EAGLE/eagle_hpo.py \
        --parquet-path stream_graph/2020-06-01_2020-08-31.parquet \
        --n-trials 30 --hpo-epochs 15 \
        --output /tmp/eagle_hpo 2>&1 | tee /tmp/eagle_hpo.log

Designed to run within ~3 hours on A100. Uses MedianPruner to
terminate underperforming trials early.
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
    raise ImportError(
        "Optuna is required for HPO: pip install optuna"
    )

from src.models.EAGLE.data_utils import load_stream_graph_data, build_temporal_csr
from src.models.EAGLE.eagle import EAGLETime
from src.models.EAGLE.eagle_train import train_epoch, validate

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def create_objective(data, train_mask, val_mask, device, hpo_epochs,
                     max_val_edges, use_amp, edge_feat_dim=0, node_feat_dim=0,
                     active_nodes=None):
    """Create an Optuna objective function with pre-loaded data.

    Args:
        data: TemporalEdgeData loaded from stream graph.
        train_mask: Boolean training edge mask.
        val_mask: Boolean validation edge mask.
        device: Torch device.
        hpo_epochs: Maximum epochs per trial.
        max_val_edges: Maximum validation edges (for speed).
        use_amp: Enable mixed precision.
        active_nodes: Sorted train node indices for negative sampling.

    Returns:
        Objective function for optuna.study.optimize().
    """
    train_csr = build_temporal_csr(data, train_mask)
    full_mask = train_mask | val_mask
    full_csr = build_temporal_csr(data, full_mask)
    train_indices = np.where(train_mask)[0]
    val_indices = np.where(val_mask)[0]

    def objective(trial):
        hidden_dim = trial.suggest_categorical(
            "hidden_dim", [50, 100, 200]
        )
        num_neighbors = trial.suggest_categorical(
            "num_neighbors", [10, 15, 20, 30]
        )
        num_mixer_layers = trial.suggest_int("num_mixer_layers", 1, 3)
        lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
        weight_decay = trial.suggest_float(
            "weight_decay", 1e-6, 1e-3, log=True
        )
        dropout = trial.suggest_float("dropout", 0.0, 0.3, step=0.05)
        batch_size = trial.suggest_categorical(
            "batch_size", [200, 400, 600]
        )
        token_expansion = trial.suggest_categorical(
            "token_expansion", [0.5, 1.0, 2.0]
        )
        channel_expansion = trial.suggest_categorical(
            "channel_expansion", [2.0, 4.0]
        )

        seed = 42
        rng = np.random.default_rng(seed)
        torch.manual_seed(seed)
        if device.type == "cuda":
            torch.cuda.manual_seed(seed)

        model = EAGLETime(
            hidden_dim=hidden_dim,
            num_neighbors=num_neighbors,
            num_mixer_layers=num_mixer_layers,
            token_expansion=token_expansion,
            channel_expansion=channel_expansion,
            dropout=dropout,
            edge_feat_dim=edge_feat_dim,
            node_feat_dim=node_feat_dim,
        ).to(device)

        optimizer = torch.optim.Adam(
            model.parameters(), lr=lr, weight_decay=weight_decay
        )
        amp_enabled = use_amp and device.type == "cuda"
        scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)

        best_mrr = 0.0

        for epoch in range(1, hpo_epochs + 1):
            train_epoch(
                model,
                data,
                train_csr,
                train_indices,
                optimizer,
                device,
                batch_size=batch_size,
                num_neighbors=num_neighbors,
                use_amp=use_amp,
                scaler=scaler,
                rng=rng,
                active_nodes=active_nodes,
            )

            val_metrics = validate(
                model,
                data,
                full_csr,
                val_indices,
                device,
                num_neighbors=num_neighbors,
                max_eval_edges=max_val_edges,
                use_amp=use_amp,
                rng=np.random.default_rng(42),
                active_nodes=active_nodes,
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
        description="EAGLE-Time hyperparameter optimization via Optuna on stream graphs"
    )
    parser.add_argument(
        "--parquet-path",
        type=str,
        required=True,
        help="Path to stream graph parquet file",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.7,
        help="Fraction of edges for training",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.15,
        help="Fraction of edges for validation",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="/tmp/eagle_hpo",
    )
    parser.add_argument("--n-trials", type=int, default=30)
    parser.add_argument("--hpo-epochs", type=int, default=15)
    parser.add_argument("--max-val-edges", type=int, default=3000)
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--edge-feat-dim",
        type=int,
        default=0,
        help="Dimension of edge features (0 = time-only). Set to 2 for [btc, usd].",
    )
    parser.add_argument(
        "--node-feat-dim",
        type=int,
        default=0,
        help="Dimension of node features (0 = auto-detect from features file).",
    )
    parser.add_argument(
        "--fraction",
        type=float,
        default=None,
        help="Fraction of stream graph to use as period (e.g. 0.10).",
    )
    parser.add_argument(
        "--features-path",
        type=str,
        default=None,
        help="Path to features_{label}.parquet with 15 node features.",
    )
    parser.add_argument(
        "--node-mapping-path",
        type=str,
        default=None,
        help="Path to node_mapping_{label}.npy.",
    )

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
        fraction=args.fraction,
        features_path=args.features_path,
    )
    logger.info("Data: %s", data)

    if args.node_mapping_path is not None:
        active_nodes = np.sort(np.load(args.node_mapping_path).astype(np.int64))
        logger.info("Loaded active_nodes: %d nodes", len(active_nodes))
    else:
        train_idx = np.where(train_mask)[0]
        train_idx = train_idx[:len(train_idx) // 2]
        src = data.src[train_idx].astype(np.int64)
        dst = data.dst[train_idx].astype(np.int64)
        active_nodes = np.unique(np.concatenate([src, dst]))
        logger.info("Computed active_nodes: %d nodes", len(active_nodes))

    node_feat_dim = args.node_feat_dim
    if args.features_path and node_feat_dim == 0:
        node_feat_dim = data.node_feats.shape[1]
        logger.info("Auto-detected node_feat_dim=%d", node_feat_dim)

    objective = create_objective(
        data,
        train_mask,
        val_mask,
        device,
        hpo_epochs=args.hpo_epochs,
        max_val_edges=args.max_val_edges,
        use_amp=not args.no_amp,
        edge_feat_dim=args.edge_feat_dim,
        node_feat_dim=node_feat_dim,
        active_nodes=active_nodes,
    )

    parquet_name = Path(args.parquet_path).stem
    study = optuna.create_study(
        direction="maximize",
        sampler=TPESampler(seed=args.seed),
        pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=3),
        study_name=f"eagle_hpo_{parquet_name}",
    )

    logger.info(
        "Starting HPO: %d trials, %d epochs each", args.n_trials, args.hpo_epochs
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
        "eagle",
        args,
        best.params,
        output_dir=args.output,
    )
    logger.info("Package-facing best spec saved to %s", artifact_paths["spec_path"])
    logger.info("Recommended training command:\n%s", artifact_paths["command"])
    logger.info("Command saved to %s", artifact_paths["command_path"])


if __name__ == "__main__":
    main()

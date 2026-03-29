"""Optuna hyperparameter optimization for GLFormer on stream graphs.

Usage:
    PYTHONPATH=. python src/models/GLFormer/glformer_hpo.py \\
        --parquet-path /tmp/stream_graph_10pct.parquet \\
        --node-feats-path /tmp/features_10.parquet \\
        --n-trials 6 --hpo-epochs 10 --edge-feat-dim 2 \\
        --output /tmp/glformer_hpo 2>&1 | tee /tmp/glformer_hpo.log

Designed to run within ~4 hours on V100 (batch_size=4000, 10% dataset,
1 epoch ≈ 4 min). Uses a 6-point grid search over the two parameters
that matter most per the GLFormer paper (Figure 4 + Implementation Details):

Fixed per paper:
    - lr: 0.0001  (fixed across all paper experiments)
    - weight_decay: 1e-5
    - dropout: 0.1
    - channel_expansion: 4.0
    - num_neighbors: 20  (same as prior works per paper)
    - batch_size: 4000   (passed via CLI, hardcoded in objective)

Searched (6 combinations = full grid):
    - hidden_dim: {100, 200}
    - num_glformer_layers: {1, 2, 3}
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
    from optuna.samplers import GridSampler
except ImportError:
    raise ImportError("Optuna is required for HPO: pip install optuna")

from src.models.GLFormer.data_utils import load_stream_graph_data, build_temporal_csr
from src.models.GLFormer.glformer import GLFormerTime
from src.models.GLFormer.glformer_train import train_epoch, validate

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
    edge_feat_dim=2,
    node_feat_dim=0,
    use_cooccurrence=False,
    cooc_dim=16,
):
    """Create an Optuna objective function with pre-loaded data.

    The data and CSR structures are built once and shared across all trials
    to avoid redundant I/O. Each trial creates a fresh model and optimizer.

    Args:
        data: TemporalEdgeData loaded from stream graph.
        train_mask: Boolean training edge mask.
        val_mask: Boolean validation edge mask.
        device: Torch device.
        hpo_epochs: Maximum epochs per trial.
        max_val_edges: Maximum validation edges per epoch (for speed).
        use_amp: Enable mixed precision.
        edge_feat_dim: Per-neighbor edge feature dimension.
        node_feat_dim: Query-node feature dimension.
        use_cooccurrence: Enable co-occurrence features.
        cooc_dim: Co-occurrence encoding dimension.

    Returns:
        Objective function for optuna.study.optimize().
    """
    train_csr = build_temporal_csr(data, train_mask)
    full_csr = build_temporal_csr(data, train_mask | val_mask)
    train_indices = np.where(train_mask)[0]
    val_indices = np.where(val_mask)[0]

    def objective(trial):
        hidden_dim = trial.suggest_categorical("hidden_dim", [100, 200])
        num_glformer_layers = trial.suggest_int("num_glformer_layers", 1, 3)
        num_neighbors = 20
        lr = 0.0001
        weight_decay = 1e-5
        dropout = 0.1
        batch_size = 4000
        channel_expansion = 4.0

        seed = 42
        rng = np.random.default_rng(seed)
        torch.manual_seed(seed)
        if device.type == "cuda":
            torch.cuda.manual_seed(seed)

        model = GLFormerTime(
            hidden_dim=hidden_dim,
            num_neighbors=num_neighbors,
            num_glformer_layers=num_glformer_layers,
            channel_expansion=channel_expansion,
            dropout=dropout,
            edge_feat_dim=edge_feat_dim,
            node_feat_dim=node_feat_dim,
            use_cooccurrence=use_cooccurrence,
            cooc_dim=cooc_dim,
        ).to(device)

        optimizer = torch.optim.Adam(
            model.parameters(), lr=lr, weight_decay=weight_decay
        )
        amp_enabled = use_amp and device.type == "cuda"
        scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

        best_mrr = 0.0

        for epoch in range(1, hpo_epochs + 1):
            train_epoch(
                model, data, train_csr, train_indices, optimizer, device,
                batch_size=batch_size,
                num_neighbors=num_neighbors,
                use_amp=use_amp,
                scaler=scaler,
                rng=rng,
            )

            val_metrics = validate(
                model, data, full_csr, val_indices, device,
                num_neighbors=num_neighbors,
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
        description="GLFormer hyperparameter optimization via Optuna on stream graphs"
    )
    parser.add_argument(
        "--parquet-path", type=str, required=True,
        help="Path to stream graph parquet file",
    )
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--output", type=str, default="/tmp/glformer_hpo")
    parser.add_argument("--n-trials", type=int, default=6)
    parser.add_argument("--hpo-epochs", type=int, default=10)
    parser.add_argument("--max-val-edges", type=int, default=3000)
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--edge-feat-dim", type=int, default=2,
        help="Per-neighbor edge feature dimension (2 = btc+usd, 0 = time-only).",
    )
    parser.add_argument(
        "--node-feat-dim", type=int, default=0,
        help="Query-node feature dimension (0 = disabled). Auto-detected from --node-feats-path.",
    )
    parser.add_argument(
        "--node-feats-path", type=str, default=None,
        help="Path to node features parquet (features_10.parquet or features_25.parquet).",
    )
    parser.add_argument(
        "--use-cooccurrence", action="store_true",
        help="Enable co-occurrence features. Slower per trial but may improve MRR.",
    )
    parser.add_argument("--cooc-dim", type=int, default=16)

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

    if args.node_feats_path:
        from scripts.compute_stream_node_features import load_node_features as _load_nf
        logger.info("Loading node features from %s...", args.node_feats_path)
        node_feats = _load_nf(args.node_feats_path, data.num_nodes)
        data.node_feats = node_feats
        if args.node_feat_dim == 0:
            args.node_feat_dim = node_feats.shape[1]
        logger.info("Node features: shape=%s, dim=%d", node_feats.shape, args.node_feat_dim)

    objective = create_objective(
        data, train_mask, val_mask, device,
        hpo_epochs=args.hpo_epochs,
        max_val_edges=args.max_val_edges,
        use_amp=not args.no_amp,
        edge_feat_dim=args.edge_feat_dim,
        node_feat_dim=args.node_feat_dim,
        use_cooccurrence=args.use_cooccurrence,
        cooc_dim=args.cooc_dim,
    )

    parquet_name = Path(args.parquet_path).stem
    search_space = {
        "hidden_dim": [100, 200],
        "num_glformer_layers": [1, 2, 3],
    }
    study = optuna.create_study(
        direction="maximize",
        sampler=GridSampler(search_space),
        study_name=f"glformer_hpo_{parquet_name}",
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

    train_cmd = (
        f"YADISK_TOKEN=\"...\" PYTHONPATH=. python src/models/GLFormer/glformer_launcher.py \\\n"
        f"    --parquet-path {args.parquet_path} --epochs 100 \\\n"
        f"    --edge-feat-dim {args.edge_feat_dim} \\\n"
    )
    for key, value in best.params.items():
        flag = key.replace("_", "-")
        train_cmd += f"    --{flag} {value} \\\n"
    if args.use_cooccurrence:
        train_cmd += "    --use-cooccurrence \\\n"
    train_cmd += "    --output /tmp/glformer_results 2>&1 | tee /tmp/glformer_train.log"

    logger.info("Recommended training command:\n%s", train_cmd)

    cmd_path = os.path.join(args.output, "best_train_command.sh")
    with open(cmd_path, "w") as f:
        f.write("#!/bin/bash\n")
        f.write(train_cmd + "\n")
    logger.info("Command saved to %s", cmd_path)


if __name__ == "__main__":
    main()

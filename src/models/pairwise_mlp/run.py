"""GPU training + evaluation runner for PairwiseMLP.

Downloads precomputed features from Yandex.Disk (produced by precompute.py),
trains PairMLP with the selected loss, evaluates TGB-style, uploads results.

All experiment variants are controlled by CLI arguments — no code changes needed.

Ablation examples (see FEATURE_NAMES in config.py for indices):
  # E1: CN only, BPR
  python run.py --features cn_uu --exp-tag E1

  # E2: CN + log(CN) + AA, BPR
  python run.py --features cn_uu log1p_cn_uu aa_uu --exp-tag E2

  # E3: all 7 features, BPR (default)
  python run.py --exp-tag E3

  # E4: all 7 features, BCE
  python run.py --loss bce --exp-tag E4

  # Custom: directed features only
  python run.py --features cn_dir aa_dir --exp-tag directed_only

  # Use preset
  python run.py --features E2 --exp-tag E2_preset

Feature selection:
  --features NAME [NAME ...]   — select by name (from FEATURE_NAMES)
  --feature-indices N [N ...]  — select by 0-based column index
  (omit both = use all 7 features)

Usage:
    YADISK_TOKEN="..." python -u src/models/pairwise_mlp/run.py \\
        --period period_10 \\
        --precompute-dir /tmp/pairmlp_precompute \\
        --data-dir /tmp/pairmlp_data \\
        --output-dir /tmp/pairmlp_results \\
        --features cn_uu log1p_cn_uu aa_uu \\
        --loss bpr --exp-tag E2 --upload \\
        2>&1 | tee /tmp/pairmlp_E2.log
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

from sg_baselines.config import ExperimentConfig
from sg_baselines.data import (
    build_train_neighbor_sets,
    load_adjacency,
    load_node_features_sparse,
    load_stream_graph,
    split_stream_graph,
)
from src.models.pairwise_mlp.config import (
    PairMLPConfig,
    PERIODS,
    FEATURE_NAMES,
    VALID_LOSSES,
    resolve_feature_indices,
)
from src.models.pairwise_mlp.dataset import load_dataset
from src.models.pairwise_mlp.evaluate import build_eval_cache, evaluate_split
from src.models.pairwise_mlp.features import (
    precompute_degrees,
    precompute_aa_weights,
)
from src.models.pairwise_mlp.model import build_model
from src.models.pairwise_mlp.train import train
from src.yadisk_utils import (
    create_remote_folder_recursive,
    download_file,
    upload_file,
    upload_directory,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def ensure_precomputed(precompute_dir: str, cfg: PairMLPConfig, token: str) -> None:
    """Download precomputed .npy artifacts from Yandex.Disk if not cached."""
    os.makedirs(precompute_dir, exist_ok=True)
    for fname in ["pos_features.npy", "neg_features.npy", "neg_dst.npy", "meta.json"]:
        local = os.path.join(precompute_dir, fname)
        if not os.path.exists(local):
            remote = f"{cfg.yadisk_precompute_dir}/{fname}"
            print(f"  Downloading {remote} ...", flush=True)
            ok = download_file(remote, local, token)
            if not ok:
                raise RuntimeError(f"Failed to download {remote}")
        else:
            print(f"  [cached] {local}")


def upload_results(out_dir: str, remote_dir: str, token: str) -> None:
    create_remote_folder_recursive(remote_dir, token)
    n = upload_directory(out_dir, remote_dir, token)
    print(f"  Uploaded {n} files to {remote_dir}", flush=True)


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------

def run_experiment(cfg: PairMLPConfig, token: str) -> None:
    """Full GPU training + evaluation pipeline for one config."""
    out_dir      = os.path.join(cfg.local_output_dir, cfg.exp_name)
    os.makedirs(out_dir, exist_ok=True)

    summary_path = os.path.join(out_dir, "summary.json")
    if os.path.exists(summary_path):
        print(f"\nSKIP {cfg.exp_name}: summary.json exists (resume mode)")
        return

    print(f"\n{'='*65}")
    print(f"PairMLP — {cfg.exp_name}")
    print(f"  period  : {cfg.period_name} (fraction={cfg.fraction})")
    print(f"  features: {cfg.selected_feature_names}")
    print(f"  loss    : {cfg.loss.upper()}")
    print(f"  arch    : {cfg.n_input_features} → {cfg.hidden_dims} → 1")
    print(f"  lr={cfg.lr}, epochs={cfg.n_epochs}, patience={cfg.patience}")
    print(f"{'='*65}")
    t_start = time.time()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}", flush=True)
    if device.type == "cuda":
        print(f"  {torch.cuda.get_device_name(0)}", flush=True)

    # ------------------------------------------------------------------
    # [1] Download precomputed features
    # ------------------------------------------------------------------
    print("\n[1/6] Precomputed features...", flush=True)
    ensure_precomputed(cfg.precompute_artifact_dir, cfg, token)

    meta_path = os.path.join(cfg.precompute_artifact_dir, "meta.json")
    with open(meta_path) as f:
        meta = json.load(f)
    for name, ok in meta.get("correctness_checks", {}).items():
        assert ok, f"Precompute correctness check FAILED: {name}"
    print(f"  Correctness checks: {len(meta['correctness_checks'])} passed", flush=True)

    # ------------------------------------------------------------------
    # [2] Load stream graph + adjacency (same loaders as sg_baselines)
    # ------------------------------------------------------------------
    print("\n[2/6] Loading stream graph + adjacency...", flush=True)
    sg_cfg = ExperimentConfig(
        period_name=cfg.period_name,
        fraction=cfg.fraction,
        label=cfg.label,
        train_ratio=cfg.train_ratio,
        val_ratio=cfg.val_ratio,
        local_data_dir=cfg.local_data_dir,
        models=[],
        heuristics=[],
    )
    df = load_stream_graph(sg_cfg, token)
    train_edges, val_edges, test_edges = split_stream_graph(df, sg_cfg)
    del df

    node_mapping, adj_dir, adj_undir = load_adjacency(sg_cfg, token)
    node_idx, _ = load_node_features_sparse(sg_cfg, token)

    # CORRECTNESS: node_mapping must equal node_idx (same train set)
    assert np.array_equal(node_idx, node_mapping), (
        "node_mapping != node_idx — data inconsistency!"
    )
    # CORRECTNESS: train size must match what precompute used
    N_train = len(train_edges)
    assert N_train == meta["n_train"], (
        f"Train size mismatch: {N_train} vs precomputed {meta['n_train']}. "
        "Re-run precompute for this period."
    )
    print(f"  train={N_train:,}, val={len(val_edges):,}, test={len(test_edges):,}",
          flush=True)
    print(f"  node_mapping == node_idx: OK", flush=True)

    # ------------------------------------------------------------------
    # [3] Build train neighbor sets + precompute adj helpers
    # ------------------------------------------------------------------
    print("\n[3/6] Building train neighbor sets...", flush=True)
    train_neighbors = build_train_neighbor_sets(train_edges)
    print(f"  {len(train_neighbors):,} sources", flush=True)

    deg_undir, deg_dir = precompute_degrees(adj_undir, adj_dir)
    w_undir, w_dir     = precompute_aa_weights(adj_undir, deg_undir, adj_dir, deg_dir)

    # ------------------------------------------------------------------
    # [4] Load dataset (with active feature column selection)
    # ------------------------------------------------------------------
    print("\n[4/6] Loading dataset + building model...", flush=True)
    dataset = load_dataset(
        cfg.precompute_artifact_dir,
        active_feature_indices=cfg.active_feature_indices or None,
    )

    model = build_model(
        hidden_dims=cfg.hidden_dims,
        n_features=cfg.n_input_features,
        dropout=cfg.dropout,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  PairMLP: {n_params:,} parameters", flush=True)

    # ------------------------------------------------------------------
    # [5] Build eval caches (one-time CPU cost; eval during training is then
    #     just a fast GPU forward pass).
    # ------------------------------------------------------------------
    cache_kwargs = dict(
        train_neighbors=train_neighbors,
        node_mapping=node_mapping,
        adj_undir=adj_undir,
        adj_dir=adj_dir,
        deg_undir=deg_undir,
        w_undir=w_undir,
        w_dir=w_dir,
        n_negatives=cfg.n_negatives,
        max_queries=cfg.max_eval_queries,
        active_feature_indices=cfg.active_feature_indices or None,
    )

    print("\n[5/6] Building eval feature caches...", flush=True)
    val_cache  = build_eval_cache(
        val_edges,  seed=cfg.val_seed,  split_name="val",  **cache_kwargs)
    test_cache = build_eval_cache(
        test_edges, seed=cfg.test_seed, split_name="test", **cache_kwargs)

    def eval_fn(mdl, split: str) -> dict:
        cache = val_cache if split == "val" else test_cache
        return evaluate_split(
            model=mdl, device=device,
            eval_cache=cache, split_name=split,
        )

    # ------------------------------------------------------------------
    # [6] Train
    # ------------------------------------------------------------------
    print(f"\n[6/7] Training ({cfg.loss.upper()} loss)...", flush=True)
    history = train(
        model=model, dataset=dataset, cfg=cfg,
        device=device, eval_fn=eval_fn,
        output_dir=out_dir, eval_every=cfg.eval_every,
    )

    # ------------------------------------------------------------------
    # [7] Final eval with best checkpoint
    # ------------------------------------------------------------------
    print("\n[7/7] Final evaluation (best checkpoint)...", flush=True)
    ckpt = torch.load(history["ckpt_path"], map_location=device)
    model.load_state_dict(ckpt["model_state"])
    print(f"  Best epoch={ckpt['epoch']}, val_MRR={ckpt['val_mrr']:.4f}",
          flush=True)

    val_metrics  = eval_fn(model, "val")
    test_metrics = eval_fn(model, "test")

    # ------------------------------------------------------------------
    # Save summary
    # ------------------------------------------------------------------
    elapsed = time.time() - t_start
    summary = {
        "config":        cfg.to_dict(),
        "val":           val_metrics,
        "test":          test_metrics,
        "train_history": {
            "best_val_mrr":  history["best_val_mrr"],
            "best_epoch":    ckpt["epoch"],
            "train_losses":  history["train_loss"],
            "val_mrr_curve": list(zip(
                history["val_mrr_epoch"], history["val_mrr"]
            )),
        },
        "baselines_for_comparison": {
            "cn_test_mrr":  0.6403,
            "aa_test_mrr":  0.6255,
            "rf_test_mrr":  0.5215,
        },
        "elapsed_seconds": elapsed,
    }
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*65}")
    print(f"RESULTS: {cfg.exp_name}")
    print(f"  features  : {cfg.selected_feature_names}")
    print(f"  loss      : {cfg.loss.upper()}")
    print(f"  Val  MRR  : {val_metrics['mrr']:.4f}")
    print(f"  Test MRR  : {test_metrics['mrr']:.4f}  (CN=0.6403, AA=0.6255)")
    print(f"  Test H@1  : {test_metrics['hits@1']:.4f}")
    print(f"  Test H@10 : {test_metrics['hits@10']:.4f}")
    print(f"  Total time: {elapsed / 60:.1f} min")
    print(f"{'='*65}")

    if cfg.upload:
        print("\nUploading results...", flush=True)
        upload_results(out_dir, cfg.yadisk_results_dir, token)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train + eval PairwiseMLP (GPU). All params via CLI.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Feature names: {FEATURE_NAMES}
Feature presets: E1 (cn_uu), E2 (cn+log_cn+aa), E3 (all)
Loss functions: {list(VALID_LOSSES)}

Examples:
  # E1: CN only, BPR
  python run.py --features cn_uu --exp-tag E1

  # E2: CN + log(CN) + AA
  python run.py --features cn_uu log1p_cn_uu aa_uu --exp-tag E2

  # E3: all features (default)
  python run.py --exp-tag E3

  # E4: all features + BCE
  python run.py --loss bce --exp-tag E4
""",
    )

    # Data
    parser.add_argument("--period", default="period_10",
                        choices=["period_10", "period_25"])
    parser.add_argument("--data-dir",        default="/tmp/pairmlp_data")
    parser.add_argument("--precompute-dir",  default=None,
                        help="Local dir with precomputed .npy (downloaded if absent)")
    parser.add_argument("--output-dir",      default="/tmp/pairmlp_results")
    parser.add_argument("--upload",          action="store_true")

    # Feature selection (mutually exclusive: names or indices)
    feat_group = parser.add_mutually_exclusive_group()
    feat_group.add_argument(
        "--features", nargs="+", metavar="NAME",
        help="Feature names to use (or preset: E1/E2/E3). Default: all.",
    )
    feat_group.add_argument(
        "--feature-indices", nargs="+", type=int, metavar="N",
        help="0-based column indices. Alternative to --features.",
    )

    # Loss
    parser.add_argument("--loss", default="bpr", choices=list(VALID_LOSSES))

    # Architecture
    parser.add_argument("--hidden", nargs="+", type=int, default=[64, 32],
                        metavar="DIM", help="Hidden layer widths (default: 64 32)")
    parser.add_argument("--dropout", type=float, default=0.0,
                        help="Dropout rate after each hidden ReLU (default: 0 = off)")

    # Training
    parser.add_argument("--lr",         type=float, default=1e-3)
    parser.add_argument("--epochs",     type=int,   default=50)
    parser.add_argument("--patience",   type=int,   default=10)
    parser.add_argument("--batch-size", type=int,   default=4096)
    parser.add_argument("--seed",       type=int,   default=42)
    parser.add_argument("--eval-every",   type=int, default=2,
                        help="Epochs between val evaluations (default: 2)")
    parser.add_argument("--k-neg-sample", type=int, default=10,
                        help="Negatives per step subsampled from precomputed K "
                             "(0 = use all K; default: 10 to prevent neg overfitting)")

    # Experiment tag for output naming
    parser.add_argument("--exp-tag", default="",
                        help="Tag appended to output dir name, e.g. 'E1', 'E3_bce'")

    args = parser.parse_args()

    token = os.environ.get("YADISK_TOKEN", "")
    if not token:
        print("ERROR: YADISK_TOKEN environment variable is required")
        sys.exit(1)

    # Resolve feature indices
    active_indices = resolve_feature_indices(
        feature_names=args.features,
        feature_indices=args.feature_indices,
    )
    # Empty list means "all" in config
    if active_indices == list(range(len(FEATURE_NAMES))):
        active_indices = []

    period = PERIODS[args.period]
    precompute_base = args.precompute_dir or args.output_dir

    cfg = PairMLPConfig(
        period_name=args.period,
        fraction=period["fraction"],
        label=period["label"],
        random_seed=args.seed,
        loss=args.loss,
        active_feature_indices=active_indices,
        hidden_dims=args.hidden,
        dropout=args.dropout,
        lr=args.lr,
        n_epochs=args.epochs,
        patience=args.patience,
        batch_size=args.batch_size,
        eval_every=args.eval_every,
        k_neg_sample=args.k_neg_sample,
        exp_tag=args.exp_tag,
        local_data_dir=args.data_dir,
        local_precompute_dir=precompute_base,
        local_output_dir=args.output_dir,
        upload=args.upload,
    )

    run_experiment(cfg, token)


if __name__ == "__main__":
    main()

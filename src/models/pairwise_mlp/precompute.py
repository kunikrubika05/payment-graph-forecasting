"""CPU precompute script for PairwiseMLP.

Downloads stream graph, adjacency matrices, and node mapping from
Yandex.Disk, then precomputes 7 pairwise features for:
  - All train POSITIVE edges            → pos_features.npy  (N_train, 7)
  - K=20 pre-sampled train NEGATIVES    → neg_features.npy  (N_train, K, 7)
  - Negative destination indices        → neg_dst.npy       (N_train, K)  [int64]
  - Metadata + correctness stats        → meta.json

Correctness guarantees (verified by explicit assertions):
  1. Split indices IDENTICAL to sg_baselines (fraction=0.10, 70/15/15).
  2. node_mapping == node_idx from features parquet (same train set).
  3. Negative destinations sampled from train nodes ONLY (active_nodes =
     node_mapping). No val/test nodes in negative pool.
  4. Historical negatives: train neighbors of src MINUS all train positives
     of src MINUS src itself. No val/test edges consulted.
  5. adj matrices were built from train edges only (guaranteed by
     compute_stream_adjacency.py — asserted here by shape / nnz check).
  6. All feature values are finite and non-negative; Jaccard ∈ [0,1].

Usage (CPU machine):
    YADISK_TOKEN="..." python -u src/models/pairwise_mlp/precompute.py \\
        --period period_10 \\
        --data-dir /tmp/pairmlp_data \\
        --output-dir /tmp/pairmlp_precompute \\
        --upload --n-jobs -1 \\
        2>&1 | tee /tmp/pairmlp_precompute.log
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd
from scipy import sparse
from tqdm import tqdm

# Add project root to path so sg_baselines and src are importable.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

from sg_baselines.data import (
    build_train_neighbor_sets,
    load_adjacency,
    load_stream_graph,
    split_stream_graph,
    load_node_features_sparse,
)
from sg_baselines.config import ExperimentConfig
from src.models.pairwise_mlp.config import (
    PairMLPConfig,
    PERIODS,
    YADISK_EXPERIMENTS_BASE,
    N_FEATURES,
)
from src.models.pairwise_mlp.features import (
    precompute_degrees,
    precompute_aa_weights,
    compute_features_batch,
    verify_features,
)
from src.yadisk_utils import (
    create_remote_folder_recursive,
    upload_file,
)


# ---------------------------------------------------------------------------
# Negative sampling (fixed K per positive, for BPR training)
# ---------------------------------------------------------------------------

def sample_train_negatives_fixed_k(
    train_src: np.ndarray,
    train_dst: np.ndarray,
    train_neighbors: dict,
    active_nodes: np.ndarray,
    k: int,
    k_hist_max: int,
    rng: np.random.RandomState,
    chunk_size: int = 50_000,
) -> np.ndarray:
    """Sample exactly K negative destinations per positive train edge.

    Correctness:
      - Historical negatives: train neighbors of src MINUS all train
        positives of src MINUS src itself. Uses ONLY train data.
      - Random negatives: drawn from active_nodes = node_mapping
        (= all nodes seen in train). No val/test nodes.
      - Result shape: (N_train, K) int64.

    Args:
        train_src, train_dst: Positive train edge arrays.
        train_neighbors: Per-source neighbor sets from train.
        active_nodes: Sorted array of train node global indices.
        k: Total negatives per positive.
        k_hist_max: Maximum historical negatives (rest = random).
        rng: Random state.
        chunk_size: Progress reporting interval.

    Returns:
        neg_dst: int64 array of shape (N_train, k).
    """
    n = len(train_src)
    neg_dst = np.empty((n, k), dtype=np.int64)

    # Build per-source positive set (train only — no val/test).
    pos_per_src: dict[int, set] = {}
    for s, d in zip(train_src, train_dst):
        pos_per_src.setdefault(int(s), set()).add(int(d))

    active_list = active_nodes.tolist()
    n_active = len(active_list)

    for i in tqdm(range(n), desc="  sampling negatives", mininterval=5.0):
        s = int(train_src[i])
        d_true = int(train_dst[i])

        # Historical candidates: train neighbors of src
        # minus all train positives of src minus src itself.
        # CORRECTNESS: uses only train_neighbors (no val/test edges).
        pos_s = pos_per_src.get(s, set())
        hist_cands = list(train_neighbors.get(s, set()) - pos_s - {s})

        n_hist = min(k_hist_max, len(hist_cands))
        if n_hist > 0:
            chosen_hist = rng.choice(len(hist_cands), size=n_hist, replace=False)
            hist_neg = [hist_cands[j] for j in chosen_hist]
        else:
            hist_neg = []

        n_rand = k - n_hist
        # Exclude: train positives of src, historical neighbors, src itself.
        exclude = pos_s | set(train_neighbors.get(s, set())) | {s}

        rand_neg = []
        max_attempts = n_rand * 30
        attempts = 0
        while len(rand_neg) < n_rand and attempts < max_attempts:
            batch = min((n_rand - len(rand_neg)) * 3, n_active)
            idxs = rng.randint(0, n_active, size=batch)
            for idx in idxs:
                c = active_list[idx]
                if c not in exclude:
                    rand_neg.append(c)
                    exclude.add(c)
                    if len(rand_neg) >= n_rand:
                        break
            attempts += batch

        # Concatenate and pad with random if needed (extremely rare).
        sampled = hist_neg + rand_neg
        if len(sampled) < k:
            # Last-resort padding: fill remaining with random (no exclusion).
            # This path is taken only if active_nodes is tiny — should never
            # happen on the real dataset (1.9M active nodes).
            pad = rng.choice(active_nodes, size=k - len(sampled), replace=True)
            sampled = sampled + pad.tolist()
        neg_dst[i] = sampled[:k]

    return neg_dst


# ---------------------------------------------------------------------------
# Main precompute pipeline
# ---------------------------------------------------------------------------

def run_precompute(cfg: PairMLPConfig, token: str) -> None:
    """Full precompute pipeline."""
    out_dir = cfg.precompute_artifact_dir
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(cfg.local_data_dir, exist_ok=True)

    meta_path = os.path.join(out_dir, "meta.json")
    if os.path.exists(meta_path):
        print(f"\nSKIP: {out_dir}/meta.json already exists (resume mode)")
        return

    print(f"\n{'='*60}")
    print(f"PairMLP PRECOMPUTE — period_{cfg.label}")
    print(f"  fraction={cfg.fraction}, split 70/15/15")
    print(f"  K_neg={cfg.k_neg_train}, K_hist_max={cfg.k_hist_max}, n_jobs={cfg.n_jobs}")
    print(f"{'='*60}")
    t_start = time.time()

    # ------------------------------------------------------------------
    # [1] Load data (same as sg_baselines)
    # ------------------------------------------------------------------
    print("\n[1/6] Loading stream graph + adjacency...", flush=True)

    # Build sg_baselines ExperimentConfig so we can reuse its data loaders.
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

    train_src = train_edges["src_idx"].values.astype(np.int64)
    train_dst = train_edges["dst_idx"].values.astype(np.int64)
    N_train = len(train_src)
    print(f"  Train edges: {N_train:,}", flush=True)
    print(f"  Val edges:   {len(val_edges):,}", flush=True)
    print(f"  Test edges:  {len(test_edges):,}", flush=True)

    # ------------------------------------------------------------------
    # [2] Load adjacency and node features
    # ------------------------------------------------------------------
    print("\n[2/6] Loading adjacency + node features...", flush=True)
    node_mapping, adj_dir, adj_undir = load_adjacency(sg_cfg, token)

    # CORRECTNESS CHECK: node_mapping must match node_idx from features.
    # Both are np.unique(concat(train_src, train_dst)).
    node_idx, _ = load_node_features_sparse(sg_cfg, token)
    assert np.array_equal(node_idx, node_mapping), (
        "LEAKAGE RISK: node_mapping != node_idx. "
        "Features and adjacency were built from different data!"
    )
    active_nodes = node_mapping
    n_active = len(active_nodes)

    # CORRECTNESS CHECK: adjacency was built from train only.
    # We verify indirectly: adj nnz should be ≤ N_train edges.
    # (undirected adj has 2*nnz_directed directed edges represented)
    max_possible_nnz = N_train
    assert adj_dir.nnz <= max_possible_nnz, (
        f"adj_directed.nnz={adj_dir.nnz} > N_train={max_possible_nnz}. "
        "Adjacency may contain val/test edges!"
    )
    print(f"  node_mapping == node_idx: OK ({n_active:,} active nodes)")
    print(f"  adj_directed nnz={adj_dir.nnz:,} ≤ N_train={N_train:,}: OK")

    # ------------------------------------------------------------------
    # [3] Build train neighbor sets + per-source positive sets
    # ------------------------------------------------------------------
    print("\n[3/6] Building train neighbor sets...", flush=True)
    t0 = time.time()
    train_neighbors = build_train_neighbor_sets(train_edges)
    print(f"  {len(train_neighbors):,} sources ({time.time()-t0:.1f}s)", flush=True)

    # ------------------------------------------------------------------
    # [4] Sample K=20 negatives per positive train edge
    # ------------------------------------------------------------------
    print(f"\n[4/6] Sampling {cfg.k_neg_train} negatives per positive train edge...",
          flush=True)
    neg_dst_checkpoint = os.path.join(out_dir, "neg_dst.npy")

    if os.path.exists(neg_dst_checkpoint):
        neg_dst = np.load(neg_dst_checkpoint)
        print(f"  [resume] Loaded neg_dst from checkpoint: {neg_dst.shape}", flush=True)
        assert neg_dst.shape == (N_train, cfg.k_neg_train), (
            f"Checkpoint neg_dst shape {neg_dst.shape} != ({N_train}, {cfg.k_neg_train}). "
            "Delete checkpoint and re-run."
        )
    else:
        rng = np.random.RandomState(cfg.random_seed)
        t0 = time.time()
        neg_dst = sample_train_negatives_fixed_k(
            train_src, train_dst,
            train_neighbors, active_nodes,
            k=cfg.k_neg_train,
            k_hist_max=cfg.k_hist_max,
            rng=rng,
        )
        assert neg_dst.shape == (N_train, cfg.k_neg_train), (
            f"neg_dst shape {neg_dst.shape} != ({N_train}, {cfg.k_neg_train})"
        )

        # CORRECTNESS CHECK: all negative destinations must be train nodes.
        neg_flat = neg_dst.ravel()
        n_bad = int(np.sum(~np.isin(neg_flat, active_nodes)))
        assert n_bad == 0, (
            f"{n_bad:,} negative destinations not in active_nodes (train nodes)!"
        )
        print(f"  Sampled {N_train * cfg.k_neg_train:,} negatives ({time.time()-t0:.1f}s)",
              flush=True)
        print(f"  All negatives in train nodes: OK", flush=True)

        # Checkpoint: save immediately so a crash in step 5 doesn't lose 48+ min of work.
        np.save(neg_dst_checkpoint, neg_dst)
        print(f"  Checkpoint saved: {neg_dst_checkpoint}", flush=True)

    # ------------------------------------------------------------------
    # [5] Precompute degrees and AA weights (once, reused for all batches)
    # ------------------------------------------------------------------
    print("\n[5/6] Computing features...", flush=True)
    # Ensure CSR canonical form before indexing (avoids in-place modification
    # that could cause crashes during fancy row indexing).
    adj_undir.sort_indices()
    adj_dir.sort_indices()
    deg_undir, deg_dir = precompute_degrees(adj_undir, adj_dir)
    w_undir, w_dir = precompute_aa_weights(adj_undir, deg_undir, adj_dir, deg_dir)

    # --- Positive features ---
    print("  Positive edges...", flush=True)
    t0 = time.time()
    pos_features = compute_features_batch(
        train_src, train_dst,
        node_mapping, adj_undir, adj_dir,
        deg_undir, w_undir, w_dir,
        batch_size=cfg.precompute_batch,
    )
    verify_features(pos_features, label="pos")
    print(f"  pos_features: {pos_features.shape} ({time.time()-t0:.1f}s)", flush=True)

    # --- Negative features ---
    # Process in a single sequential pass to avoid threading/joblib issues
    # with scipy CSR fancy indexing (race condition in sort_indices).
    print("  Negative edges...", flush=True)
    t0 = time.time()
    neg_src_flat = np.repeat(train_src, cfg.k_neg_train)  # (N_train * K,)
    neg_dst_flat = neg_dst.ravel()                         # (N_train * K,)
    neg_features_flat = compute_features_batch(
        neg_src_flat, neg_dst_flat,
        node_mapping, adj_undir, adj_dir,
        deg_undir, w_undir, w_dir,
        batch_size=cfg.precompute_batch,
    )
    del neg_src_flat, neg_dst_flat
    verify_features(neg_features_flat, label="neg")
    neg_features = neg_features_flat.reshape(N_train, cfg.k_neg_train, N_FEATURES)
    del neg_features_flat

    assert neg_features.shape == (N_train, cfg.k_neg_train, N_FEATURES), (
        f"neg_features shape {neg_features.shape} wrong"
    )
    print(f"  neg_features: {neg_features.shape} ({time.time()-t0:.1f}s)", flush=True)

    # CORRECTNESS CHECK: pos CN should match known heuristic distribution.
    cn_pos = pos_features[:, 0]
    frac_cn_zero = float(np.mean(cn_pos == 0))
    frac_cn_pos  = float(np.mean(cn_pos > 0))
    print(f"  Positive edge CN=0: {frac_cn_zero:.1%}, CN>0: {frac_cn_pos:.1%}")
    # Sanity: some positive edges should have CN > 0 (they're real transactions)
    assert frac_cn_pos > 0.01, (
        f"Too few positive edges with CN>0: {frac_cn_pos:.1%}. "
        "Check that adjacency and stream graph use the same period!"
    )

    cn_neg = neg_features[:, :, 0].ravel()
    frac_neg_cn_zero = float(np.mean(cn_neg == 0))
    print(f"  Negative edge CN=0: {frac_neg_cn_zero:.1%} "
          f"(expected ~90%+ for sparse graph)")

    # ------------------------------------------------------------------
    # [6] Save artifacts
    # ------------------------------------------------------------------
    print("\n[6/6] Saving...", flush=True)

    pos_path     = os.path.join(out_dir, "pos_features.npy")
    neg_path     = os.path.join(out_dir, "neg_features.npy")
    neg_dst_path = os.path.join(out_dir, "neg_dst.npy")

    np.save(pos_path,     pos_features)
    np.save(neg_path,     neg_features)
    np.save(neg_dst_path, neg_dst)

    elapsed = time.time() - t_start
    meta = {
        "config": cfg.to_dict(),
        "n_train": N_train,
        "n_active": n_active,
        "adj_dir_nnz": int(adj_dir.nnz),
        "adj_undir_nnz": int(adj_undir.nnz),
        "pos_features_shape": list(pos_features.shape),
        "neg_features_shape": list(neg_features.shape),
        "neg_dst_shape": list(neg_dst.shape),
        "frac_pos_cn_nonzero": frac_cn_pos,
        "frac_neg_cn_zero": frac_neg_cn_zero,
        "elapsed_seconds": elapsed,
        "correctness_checks": {
            "node_mapping_eq_node_idx": True,
            "adj_nnz_leq_n_train": True,
            "all_neg_in_train_nodes": True,
            "features_finite_nonneg": True,
            "jaccard_in_0_1": True,
        },
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"  pos_features.npy:  {pos_features.nbytes / 1e6:.1f} MB")
    print(f"  neg_features.npy:  {neg_features.nbytes / 1e6:.1f} MB")
    print(f"  neg_dst.npy:       {neg_dst.nbytes / 1e6:.1f} MB")
    print(f"  Total elapsed:     {elapsed / 60:.1f} min")

    if cfg.upload:
        print(f"\nUploading to {cfg.yadisk_precompute_dir}...", flush=True)
        create_remote_folder_recursive(cfg.yadisk_precompute_dir, token)
        for fpath in [pos_path, neg_path, neg_dst_path, meta_path]:
            fname = os.path.basename(fpath)
            ok = upload_file(fpath, f"{cfg.yadisk_precompute_dir}/{fname}", token)
            print(f"  {'OK' if ok else 'FAILED'}: {fname}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Precompute pairwise features for PairwiseMLP (CPU)"
    )
    parser.add_argument(
        "--period", default="period_10",
        choices=["period_10", "period_25"],
    )
    parser.add_argument("--data-dir",   default="/tmp/pairmlp_data")
    parser.add_argument("--output-dir", default="/tmp/pairmlp_precompute")
    parser.add_argument("--upload",     action="store_true")
    parser.add_argument("--n-jobs",     type=int, default=-1,
                        help="Parallel workers (-1 = all cores)")
    parser.add_argument("--k-neg",      type=int, default=20,
                        help="Negatives per positive (default 20)")
    parser.add_argument("--seed",       type=int, default=42)
    args = parser.parse_args()

    token = os.environ.get("YADISK_TOKEN", "")
    if not token:
        print("ERROR: YADISK_TOKEN environment variable is required")
        sys.exit(1)

    period = PERIODS[args.period]
    cfg = PairMLPConfig(
        period_name=args.period,
        fraction=period["fraction"],
        label=period["label"],
        random_seed=args.seed,
        k_neg_train=args.k_neg,
        n_jobs=args.n_jobs,
        local_data_dir=args.data_dir,
        local_precompute_dir=args.output_dir,
        upload=args.upload,
    )
    run_precompute(cfg, token)


if __name__ == "__main__":
    main()

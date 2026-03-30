"""Pairwise feature computation for PairwiseMLP.

7 features per (src, dst) pair, all derived from TRAIN adjacency only:
  0: cn_uu        — Common Neighbors, undirected (integer count ≥ 0)
  1: log1p_cn_uu  — log1p(cn_uu)
  2: aa_uu        — Adamic-Adar, undirected (weighted CN ≥ 0)
  3: cn_dir       — CN directed: count of nodes both src and dst point TO
  4: aa_dir       — AA directed (same edge set as cn_dir)
  5: jaccard_uu   — Jaccard coefficient, undirected ∈ [0, 1]
  6: log1p_pa_uu  — log1p(deg_src_uu * deg_dst_uu), undirected degrees

Correctness guarantee:
  All features computed from adj_undirected / adj_directed built on
  TRAIN edges only. Nodes not present in train → all 7 features = 0.
  This matches the zero-feature convention of sg_baselines.

Parallelisation:
  compute_features_parallel() splits pairs into chunks and dispatches to
  joblib workers. Each worker receives read-only scipy matrices.
  Safe: scipy CSR fancy indexing is read-only and re-entrant.
"""

import numpy as np
from scipy import sparse
from joblib import Parallel, delayed
from typing import Optional

from src.models.pairwise_mlp.config import N_FEATURES, FEATURE_NAMES


# ---------------------------------------------------------------------------
# Global→local index mapping helpers (identical to sg_baselines/features.py)
# ---------------------------------------------------------------------------

def global_to_local(
    global_indices: np.ndarray,
    node_mapping: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Map global node indices to local (adjacency) indices.

    Args:
        global_indices: Array of global node indices (int64).
        node_mapping:   Sorted array mapping local→global (from adj .npy).

    Returns:
        (local_indices, valid_mask) where valid_mask[i]=True iff
        global_indices[i] is present in node_mapping.
        local_indices[~valid_mask] are set to 0 (safe sentinel).
    """
    positions = np.searchsorted(node_mapping, global_indices)
    n = len(node_mapping)
    valid = (
        (positions < n)
        & (node_mapping[np.minimum(positions, n - 1)] == global_indices)
    )
    local = np.where(valid, positions, 0)
    return local, valid


# ---------------------------------------------------------------------------
# Per-degree caches (precomputed once per adjacency matrix)
# ---------------------------------------------------------------------------

def precompute_degrees(
    adj_undir: sparse.csr_matrix,
    adj_dir: sparse.csr_matrix,
) -> tuple[np.ndarray, np.ndarray]:
    """Precompute node degrees for fast PA/AA computation.

    Returns:
        (deg_undir, deg_dir): float64 arrays of shape (n_nodes,).
    """
    deg_undir = np.array(adj_undir.sum(axis=1), dtype=np.float64).ravel()
    deg_dir   = np.array(adj_dir.sum(axis=1),   dtype=np.float64).ravel()
    return deg_undir, deg_dir


def precompute_aa_weights(
    adj_undir: sparse.csr_matrix,
    deg_undir: np.ndarray,
    adj_dir: sparse.csr_matrix,
    deg_dir: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Precompute per-node Adamic-Adar weights (1/log(deg)) for both adjs.

    Nodes with deg ≤ 1 get weight 0 (log(1)=0, log(0) undefined).

    Returns:
        (w_undir, w_dir): float64 weight arrays of shape (n_nodes,).
    """
    n = adj_undir.shape[0]
    w_undir = np.zeros(n, dtype=np.float64)
    mask_u = deg_undir > 1
    w_undir[mask_u] = 1.0 / np.log(deg_undir[mask_u])

    w_dir = np.zeros(n, dtype=np.float64)
    mask_d = deg_dir > 1
    w_dir[mask_d] = 1.0 / np.log(deg_dir[mask_d])

    return w_undir, w_dir


# ---------------------------------------------------------------------------
# Core batched feature computation (operates on LOCAL indices only)
# ---------------------------------------------------------------------------

def _compute_features_local(
    src_local: np.ndarray,
    dst_local: np.ndarray,
    valid: np.ndarray,
    adj_undir: sparse.csr_matrix,
    adj_dir: sparse.csr_matrix,
    deg_undir: np.ndarray,
    w_undir: np.ndarray,
    w_dir: np.ndarray,
) -> np.ndarray:
    """Compute 7 features for pre-mapped local indices.

    Args:
        src_local, dst_local: Local index arrays (invalid → 0).
        valid: Boolean mask — only valid pairs get real features.
        adj_undir, adj_dir: CSR adjacency (local indices).
        deg_undir: Precomputed undirected degrees.
        w_undir, w_dir: Precomputed AA weights (per node).

    Returns:
        Float32 array of shape (n_pairs, 7). Invalid pairs = 0.
    """
    n = len(src_local)
    out = np.zeros((n, N_FEATURES), dtype=np.float32)

    if not valid.any():
        return out

    vs = src_local[valid]
    vd = dst_local[valid]
    nv = int(valid.sum())

    # --- cn_uu and log1p_cn_uu ---
    A_src_u = adj_undir[vs]        # (nv, N) sparse
    A_dst_u = adj_undir[vd]        # (nv, N) sparse
    cn_uu = np.array(
        A_src_u.multiply(A_dst_u).sum(axis=1), dtype=np.float32
    ).ravel()
    out[valid, 0] = cn_uu
    out[valid, 1] = np.log1p(cn_uu)

    # --- aa_uu ---
    # AA = common_adj @ w_undir  (common_adj is element-wise product)
    common_u = A_src_u.multiply(A_dst_u)           # (nv, N) sparse, 0/1
    aa_uu = np.array(
        common_u @ w_undir.reshape(-1, 1), dtype=np.float32
    ).ravel()
    out[valid, 2] = aa_uu

    # --- cn_dir and aa_dir ---
    A_src_d = adj_dir[vs]
    A_dst_d = adj_dir[vd]
    common_d = A_src_d.multiply(A_dst_d)
    cn_dir = np.array(
        common_d.sum(axis=1), dtype=np.float32
    ).ravel()
    aa_dir = np.array(
        common_d @ w_dir.reshape(-1, 1), dtype=np.float32
    ).ravel()
    out[valid, 3] = cn_dir
    out[valid, 4] = aa_dir

    # --- jaccard_uu ---
    deg_s = deg_undir[vs].astype(np.float32)
    deg_d = deg_undir[vd].astype(np.float32)
    union = deg_s + deg_d - cn_uu
    jaccard = np.zeros(nv, dtype=np.float32)
    m = union > 0
    jaccard[m] = cn_uu[m] / union[m]
    out[valid, 5] = jaccard

    # --- log1p_pa_uu ---
    pa = deg_s * deg_d
    out[valid, 6] = np.log1p(pa)

    assert np.isfinite(out).all(), "Non-finite values in computed features"
    return out


# ---------------------------------------------------------------------------
# Public API: batch computation with global indices
# ---------------------------------------------------------------------------

def compute_features_batch(
    src_global: np.ndarray,
    dst_global: np.ndarray,
    node_mapping: np.ndarray,
    adj_undir: sparse.csr_matrix,
    adj_dir: sparse.csr_matrix,
    deg_undir: np.ndarray,
    w_undir: np.ndarray,
    w_dir: np.ndarray,
    batch_size: int = 50_000,
) -> np.ndarray:
    """Compute 7 pairwise features for (src, dst) global-index pairs.

    Processes data in batches to control memory usage.

    Args:
        src_global, dst_global: Global node index arrays (int64).
        node_mapping:  Sorted local→global mapping from adjacency.
        adj_undir, adj_dir: CSR adjacency matrices (local indices).
        deg_undir: Precomputed undirected degree array (float64, local).
        w_undir, w_dir: Precomputed AA weight arrays (float64, local).
        batch_size: Pairs processed in one scipy call.

    Returns:
        Float32 array (n_pairs, 7). Pairs where src or dst not in
        node_mapping get all-zero features.
    """
    n = len(src_global)
    assert len(dst_global) == n, "src/dst length mismatch"

    out = np.empty((n, N_FEATURES), dtype=np.float32)
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        src_loc, valid = global_to_local(src_global[start:end], node_mapping)
        dst_loc, valid_d = global_to_local(dst_global[start:end], node_mapping)
        valid = valid & valid_d
        out[start:end] = _compute_features_local(
            src_loc, dst_loc, valid,
            adj_undir, adj_dir,
            deg_undir, w_undir, w_dir,
        )
    return out


def _worker_fn(
    chunk_src: np.ndarray,
    chunk_dst: np.ndarray,
    node_mapping: np.ndarray,
    adj_undir: sparse.csr_matrix,
    adj_dir: sparse.csr_matrix,
    deg_undir: np.ndarray,
    w_undir: np.ndarray,
    w_dir: np.ndarray,
    batch_size: int,
) -> np.ndarray:
    """Worker function for joblib parallelism."""
    return compute_features_batch(
        chunk_src, chunk_dst,
        node_mapping, adj_undir, adj_dir,
        deg_undir, w_undir, w_dir, batch_size,
    )


def compute_features_parallel(
    src_global: np.ndarray,
    dst_global: np.ndarray,
    node_mapping: np.ndarray,
    adj_undir: sparse.csr_matrix,
    adj_dir: sparse.csr_matrix,
    deg_undir: np.ndarray,
    w_undir: np.ndarray,
    w_dir: np.ndarray,
    batch_size: int = 50_000,
    n_jobs: int = -1,
) -> np.ndarray:
    """Parallelised version of compute_features_batch.

    Splits the pair array into n_jobs chunks and dispatches to separate
    processes via joblib. Each worker gets a read-only copy of the scipy
    matrices (serialised via pickle at dispatch time, ~50 MB total).

    Args:
        n_jobs: Number of parallel workers (-1 = all available cores).
        All other args: same as compute_features_batch.

    Returns:
        Float32 array (n_pairs, 7).
    """
    import os
    if n_jobs == -1:
        n_jobs = os.cpu_count() or 1
    n_jobs = max(1, min(n_jobs, len(src_global) // max(batch_size, 1) + 1))

    chunks_src = np.array_split(src_global, n_jobs)
    chunks_dst = np.array_split(dst_global, n_jobs)

    results = Parallel(n_jobs=n_jobs, backend="threading", verbose=0)(
        delayed(_worker_fn)(
            cs, cd, node_mapping, adj_undir, adj_dir,
            deg_undir, w_undir, w_dir, batch_size,
        )
        for cs, cd in zip(chunks_src, chunks_dst)
    )

    return np.concatenate(results, axis=0)


# ---------------------------------------------------------------------------
# Correctness sanity-checks (called from precompute.py)
# ---------------------------------------------------------------------------

def verify_features(
    features: np.ndarray,
    label: str = "",
) -> None:
    """Assert basic feature validity.

    Raises:
        AssertionError: on any invariant violation.
    """
    tag = f"[{label}] " if label else ""
    assert features.ndim == 2 and features.shape[1] == N_FEATURES, (
        f"{tag}Expected shape (N, {N_FEATURES}), got {features.shape}"
    )
    assert features.dtype == np.float32, (
        f"{tag}Expected float32, got {features.dtype}"
    )
    assert np.isfinite(features).all(), (
        f"{tag}Non-finite values in features"
    )
    # cn_uu ≥ 0, log1p_cn_uu ≥ 0, aa_uu ≥ 0, cn_dir ≥ 0, aa_dir ≥ 0
    for i, name in enumerate(FEATURE_NAMES):
        assert (features[:, i] >= 0).all(), (
            f"{tag}Feature {name} (col {i}) has negative values"
        )
    # jaccard ∈ [0, 1]
    assert (features[:, 5] <= 1.0 + 1e-5).all(), (
        f"{tag}Jaccard > 1 detected: max={features[:, 5].max()}"
    )

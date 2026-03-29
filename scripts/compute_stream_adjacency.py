"""Compute sparse binary adjacency matrices from stream graph train splits.

Builds directed and undirected CSR adjacency matrices for two period
configurations (10% and 25% of edges). Matrices are used to compute
pair-level features (Common Neighbors, Adamic-Adar) on the fly during
model training/evaluation.

Features are computed ONLY on the train split (first 70% of the period)
to avoid data leakage.

Usage:
    YADISK_TOKEN="..." PYTHONPATH=. python scripts/compute_stream_adjacency.py \
        --input /tmp/stream_graph_full.parquet \
        --output-dir /tmp/stream_adjacency/ \
        --upload

    YADISK_TOKEN="..." PYTHONPATH=. python scripts/compute_stream_adjacency.py \
        --yadisk-path orbitaal_processed/stream_graph/2020-06-01__2020-08-31.parquet \
        --output-dir /tmp/stream_adjacency/ \
        --upload
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd
from scipy import sparse


def build_adjacency_matrices(
    src: np.ndarray,
    dst: np.ndarray,
    num_nodes_global: int,
) -> tuple[np.ndarray, sparse.csr_matrix, sparse.csr_matrix]:
    """Build directed and undirected binary CSR adjacency matrices.

    Args:
        src: Source node indices (int64, global).
        dst: Destination node indices (int64, global).
        num_nodes_global: Total nodes in full graph (for metadata only).

    Returns:
        Tuple of (node_mapping, adj_directed, adj_undirected) where
        node_mapping maps local index -> global index,
        adj_directed is CSR (n_active, n_active) with A[i,j]=1 if edge i->j,
        adj_undirected is symmetric CSR with A[i,j]=A[j,i]=1 if edge i->j or j->i.
    """
    assert len(src) == len(dst), f"src/dst length mismatch: {len(src)} vs {len(dst)}"
    assert len(src) > 0, "Empty edge arrays"
    assert src.dtype == np.int64 and dst.dtype == np.int64, "src/dst must be int64"

    active_set = np.unique(np.concatenate([src, dst]))
    n_active = len(active_set)
    assert n_active >= 2, f"Need at least 2 active nodes, got {n_active}"

    src_local = np.searchsorted(active_set, src)
    dst_local = np.searchsorted(active_set, dst)

    assert src_local.max() < n_active, "searchsorted produced out-of-bounds local index (src)"
    assert dst_local.max() < n_active, "searchsorted produced out-of-bounds local index (dst)"

    edges_directed = np.column_stack([src_local, dst_local])
    edges_directed = np.unique(edges_directed, axis=0)

    ones_dir = np.ones(len(edges_directed), dtype=np.float32)
    adj_directed = sparse.csr_matrix(
        (ones_dir, (edges_directed[:, 0], edges_directed[:, 1])),
        shape=(n_active, n_active),
    )

    assert adj_directed.max() == 1.0, "Directed adjacency must be binary"
    assert adj_directed.min() >= 0.0, "Directed adjacency has negative values"

    edges_undirected = np.concatenate([
        edges_directed,
        edges_directed[:, ::-1],
    ], axis=0)
    edges_undirected = np.unique(edges_undirected, axis=0)

    ones_undir = np.ones(len(edges_undirected), dtype=np.float32)
    adj_undirected = sparse.csr_matrix(
        (ones_undir, (edges_undirected[:, 0], edges_undirected[:, 1])),
        shape=(n_active, n_active),
    )

    assert adj_undirected.max() == 1.0, "Undirected adjacency must be binary"
    diff = adj_undirected - adj_undirected.T
    assert diff.nnz == 0, f"Undirected adjacency is not symmetric, diff.nnz={diff.nnz}"
    assert adj_undirected.nnz >= adj_directed.nnz, (
        f"Undirected nnz ({adj_undirected.nnz}) < directed nnz ({adj_directed.nnz})"
    )

    return active_set, adj_directed, adj_undirected


def compute_cn(adj: sparse.csr_matrix, src_idx: np.ndarray, dst_idx: np.ndarray) -> np.ndarray:
    """Compute Common Neighbors scores for given pairs.

    Args:
        adj: Binary adjacency matrix (CSR, local indices).
        src_idx: Source local indices.
        dst_idx: Destination local indices.

    Returns:
        Array of CN scores (float32).
    """
    assert len(src_idx) == len(dst_idx), f"src/dst length mismatch: {len(src_idx)} vs {len(dst_idx)}"
    assert src_idx.max() < adj.shape[0], f"src_idx out of bounds: max={src_idx.max()}, n={adj.shape[0]}"
    assert dst_idx.max() < adj.shape[0], f"dst_idx out of bounds: max={dst_idx.max()}, n={adj.shape[0]}"
    result = np.array(adj[src_idx].multiply(adj[dst_idx]).sum(axis=1), dtype=np.float32).ravel()
    assert result.shape == (len(src_idx),), f"CN result shape mismatch: {result.shape}"
    assert np.isfinite(result).all(), "CN contains non-finite values"
    return result


def compute_aa(adj: sparse.csr_matrix, src_idx: np.ndarray, dst_idx: np.ndarray) -> np.ndarray:
    """Compute Adamic-Adar scores for given pairs.

    Args:
        adj: Binary adjacency matrix (CSR, local indices).
        src_idx: Source local indices.
        dst_idx: Destination local indices.

    Returns:
        Array of AA scores (float32).
    """
    degrees = np.array(adj.sum(axis=1)).ravel()
    weights = np.zeros(adj.shape[0], dtype=np.float64)
    mask = degrees > 1
    weights[mask] = 1.0 / np.log(degrees[mask])
    assert len(src_idx) == len(dst_idx), f"src/dst length mismatch: {len(src_idx)} vs {len(dst_idx)}"
    assert src_idx.max() < adj.shape[0], f"src_idx out of bounds: max={src_idx.max()}, n={adj.shape[0]}"
    assert dst_idx.max() < adj.shape[0], f"dst_idx out of bounds: max={dst_idx.max()}, n={adj.shape[0]}"
    common = adj[src_idx].multiply(adj[dst_idx])
    result = np.array(common @ weights.reshape(-1, 1), dtype=np.float32).ravel()
    assert result.shape == (len(src_idx),), f"AA result shape mismatch: {result.shape}"
    assert np.isfinite(result).all(), "AA contains non-finite values"
    return result


def process_period(
    df_full: pd.DataFrame,
    fraction: float,
    train_ratio: float,
    num_nodes_global: int,
    output_dir: str,
    label: str,
) -> dict:
    """Build and save adjacency matrices for one period configuration.

    Args:
        df_full: Full stream graph DataFrame.
        fraction: Fraction of edges to use as period (e.g. 0.10).
        train_ratio: Fraction of period edges for training (e.g. 0.70).
        num_nodes_global: Total number of nodes in the full graph.
        output_dir: Directory to save results.
        label: Label for output files (e.g. "10" or "25").

    Returns:
        Metadata dict.
    """
    total = len(df_full)
    period_end = int(total * fraction)
    period = df_full.iloc[:period_end]
    train_end = int(len(period) * train_ratio)
    train = period.iloc[:train_end]

    n_period = len(period)
    n_train = len(train)
    assert n_train > 0, f"Empty train set for fraction={fraction}"
    assert n_train < n_period, "Train must be a proper subset of period"
    print(f"  period={n_period:,} edges, train={n_train:,} edges")

    train_ts_max = train["timestamp"].iloc[-1]
    val_ts_min = period.iloc[train_end]["timestamp"]
    assert train_ts_max <= val_ts_min, (
        f"Train/val time overlap! train_max={train_ts_max}, val_min={val_ts_min}"
    )

    src = train["src_idx"].values.astype(np.int64)
    dst = train["dst_idx"].values.astype(np.int64)

    t0 = time.time()
    node_mapping, adj_dir, adj_undir = build_adjacency_matrices(src, dst, num_nodes_global)
    n_active = len(node_mapping)
    assert adj_dir.shape == (n_active, n_active), "Directed adj shape mismatch"
    assert adj_undir.shape == (n_active, n_active), "Undirected adj shape mismatch"
    print(f"  - Built adjacency matrices ({time.time() - t0:.1f}s)")
    print(f"    directed: {adj_dir.shape}, nnz={adj_dir.nnz:,}")
    print(f"    undirected: {adj_undir.shape}, nnz={adj_undir.nnz:,}")

    t0 = time.time()
    dir_path = os.path.join(output_dir, f"adj_{label}_directed.npz")
    sparse.save_npz(dir_path, adj_dir)
    dir_mb = os.path.getsize(dir_path) / 1e6
    print(f"  - Saved {dir_path} ({dir_mb:.1f} MB, {time.time() - t0:.1f}s)")

    t0 = time.time()
    undir_path = os.path.join(output_dir, f"adj_{label}_undirected.npz")
    sparse.save_npz(undir_path, adj_undir)
    undir_mb = os.path.getsize(undir_path) / 1e6
    print(f"  - Saved {undir_path} ({undir_mb:.1f} MB, {time.time() - t0:.1f}s)")

    t0 = time.time()
    mapping_path = os.path.join(output_dir, f"node_mapping_{label}.npy")
    np.save(mapping_path, node_mapping)
    print(f"  - Saved {mapping_path} ({time.time() - t0:.1f}s)")

    t0 = time.time()
    adj_dir_reload = sparse.load_npz(dir_path)
    adj_undir_reload = sparse.load_npz(undir_path)
    mapping_reload = np.load(mapping_path)
    assert adj_dir_reload.shape == adj_dir.shape, "Directed adj reload shape mismatch"
    assert adj_dir_reload.nnz == adj_dir.nnz, "Directed adj reload nnz mismatch"
    assert adj_undir_reload.shape == adj_undir.shape, "Undirected adj reload shape mismatch"
    assert adj_undir_reload.nnz == adj_undir.nnz, "Undirected adj reload nnz mismatch"
    assert np.array_equal(mapping_reload, node_mapping), "Mapping reload mismatch"
    print(f"  - Reload verification ✓ ({time.time() - t0:.1f}s)")

    t0 = time.time()
    _verify_mapping_matches_features(node_mapping, output_dir, label)
    print(f"  - Mapping vs features verification ({time.time() - t0:.1f}s)")

    metadata = {
        "period_fraction": fraction,
        "train_ratio": train_ratio,
        "num_edges_period": n_period,
        "num_edges_train": n_train,
        "num_nodes_global": num_nodes_global,
        "num_nodes_active": n_active,
        "nnz_directed": int(adj_dir.nnz),
        "nnz_undirected": int(adj_undir.nnz),
        "train_timestamp_min": int(train["timestamp"].min()),
        "train_timestamp_max": int(train["timestamp"].max()),
        "source_file": "2020-06-01__2020-08-31.parquet",
        "format": "scipy CSR in .npz, local indices, use node_mapping_{label}.npy for global mapping",
        "pair_features": "CN and AA computed on the fly via compute_cn() / compute_aa()",
    }

    json_path = os.path.join(output_dir, f"adj_{label}.json")
    with open(json_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"  - Saved {json_path}")

    del adj_dir, adj_undir
    return metadata


def _verify_mapping_matches_features(node_mapping: np.ndarray, output_dir: str, label: str):
    """Verify that adjacency node mapping matches features node_idx."""
    features_path = os.path.join(output_dir, f"features_{label}.parquet")
    if not os.path.exists(features_path):
        features_path_alt = os.path.join(
            os.path.dirname(output_dir.rstrip("/")), "stream_features", f"features_{label}.parquet"
        )
        if os.path.exists(features_path_alt):
            features_path = features_path_alt
        else:
            print(f"    (skipped — features_{label}.parquet not found locally)")
            return

    feat_df = pd.read_parquet(features_path, columns=["node_idx"])
    feat_nodes = feat_df["node_idx"].values
    if np.array_equal(node_mapping, feat_nodes):
        print(f"    ✓ mapping matches features_{label}.parquet ({len(node_mapping):,} nodes)")
    else:
        print(f"    ⚠ WARNING: mapping DOES NOT match features_{label}.parquet!")
        print(f"      adj nodes: {len(node_mapping):,}, features nodes: {len(feat_nodes):,}")


def main():
    parser = argparse.ArgumentParser(
        description="Compute sparse adjacency matrices from stream graph"
    )
    parser.add_argument("--input", type=str, default=None,
                        help="Local path to stream graph parquet")
    parser.add_argument("--yadisk-path", type=str,
                        default="orbitaal_processed/stream_graph/2020-06-01__2020-08-31.parquet",
                        help="Yandex.Disk path to download stream graph")
    parser.add_argument("--output-dir", type=str, default="/tmp/stream_adjacency/",
                        help="Output directory")
    parser.add_argument("--upload", action="store_true",
                        help="Upload results to Yandex.Disk")
    args = parser.parse_args()

    if args.input is None:
        token = os.environ.get("YADISK_TOKEN", "")
        if not token:
            print("ERROR: YADISK_TOKEN required when --input is not specified")
            sys.exit(1)
        from src.yadisk_utils import download_file
        args.input = "/tmp/stream_graph_full.parquet"
        if not os.path.exists(args.input):
            print(f"[1/4] Downloading stream graph from Yandex.Disk...")
            ok = download_file(args.yadisk_path, args.input, token)
            if not ok:
                print("ERROR: Failed to download stream graph")
                sys.exit(1)
            print(f"  Downloaded to {args.input}")
        else:
            print(f"[1/4] Stream graph already at {args.input}")
    else:
        print(f"[1/4] Loading stream graph...")

    df = pd.read_parquet(args.input)
    num_nodes_global = int(max(df["src_idx"].max(), df["dst_idx"].max()) + 1)
    print(f"  {len(df):,} edges, {num_nodes_global:,} nodes (global index space)")

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"[2/4] Building adjacency matrices for period_10...")
    process_period(df, 0.10, 0.70, num_nodes_global, args.output_dir, "10")

    print(f"[3/4] Building adjacency matrices for period_25...")
    process_period(df, 0.25, 0.70, num_nodes_global, args.output_dir, "25")

    if args.upload:
        token = os.environ.get("YADISK_TOKEN", "")
        if not token:
            print("ERROR: YADISK_TOKEN required for --upload")
            sys.exit(1)
        from src.yadisk_utils import upload_file
        print(f"[4/4] Uploading to Yandex.Disk...")
        for label in ["10", "25"]:
            for fname in [
                f"adj_{label}_directed.npz",
                f"adj_{label}_undirected.npz",
                f"node_mapping_{label}.npy",
                f"adj_{label}.json",
            ]:
                local = os.path.join(args.output_dir, fname)
                remote = f"orbitaal_processed/stream_graph/{fname}"
                upload_file(local, remote, token)
                print(f"  Uploaded {remote}")
        print("  Done!")
    else:
        print(f"[4/4] Skipping upload (use --upload to upload to Yandex.Disk)")

    print("\nAll done.")


if __name__ == "__main__":
    main()

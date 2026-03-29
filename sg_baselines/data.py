"""Data loading for stream graph baselines.

Downloads stream graph, node features, and adjacency matrices from Yandex.Disk.
Splits stream graph into train/val/test by chronological edge order.
"""

import os
import json
import time

import numpy as np
import pandas as pd
from scipy import sparse

from sg_baselines.config import ExperimentConfig, YADISK_STREAM_GRAPH, YADISK_STREAM_DIR


def download_if_missing(remote_path: str, local_path: str, token: str) -> str:
    """Download file from Yandex.Disk if not already present locally."""
    if os.path.exists(local_path):
        print(f"  [cached] {local_path}")
        return local_path
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    from src.yadisk_utils import download_file
    print(f"  Downloading {remote_path} -> {local_path}...")
    ok = download_file(remote_path, local_path, token)
    if not ok:
        raise RuntimeError(f"Failed to download {remote_path}")
    return local_path


def load_stream_graph(config: ExperimentConfig, token: str) -> pd.DataFrame:
    """Load full stream graph from Yandex.Disk or local cache."""
    local_path = os.path.join(config.local_data_dir, "stream_graph.parquet")
    download_if_missing(YADISK_STREAM_GRAPH, local_path, token)
    t0 = time.time()
    df = pd.read_parquet(local_path)
    print(f"  Loaded stream graph: {len(df):,} edges ({time.time() - t0:.1f}s)")
    assert list(df.columns) == ["src_idx", "dst_idx", "timestamp", "btc", "usd"], (
        f"Unexpected columns: {list(df.columns)}"
    )
    ts = df["timestamp"].values
    assert (ts[1:] >= ts[:-1]).all(), "Stream graph must be sorted by timestamp"
    return df


def split_stream_graph(
    df: pd.DataFrame, config: ExperimentConfig
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split stream graph into train/val/test for the given period.

    Returns:
        (train_edges, val_edges, test_edges) DataFrames.
    """
    total = len(df)
    period_end = int(total * config.fraction)
    period = df.iloc[:period_end]
    n_period = len(period)

    train_end = int(n_period * config.train_ratio)
    val_end = int(n_period * (config.train_ratio + config.val_ratio))

    train = period.iloc[:train_end]
    val = period.iloc[train_end:val_end]
    test = period.iloc[val_end:]

    assert len(train) > 0, "Empty train set"
    assert len(val) > 0, "Empty val set"
    assert len(test) > 0, "Empty test set"

    train_ts_max = train["timestamp"].iloc[-1]
    val_ts_min = val["timestamp"].iloc[0]
    val_ts_max = val["timestamp"].iloc[-1]
    test_ts_min = test["timestamp"].iloc[0]

    assert train_ts_max <= val_ts_min, (
        f"Train/val time overlap! train_max={train_ts_max}, val_min={val_ts_min}"
    )
    assert val_ts_max <= test_ts_min, (
        f"Val/test time overlap! val_max={val_ts_max}, test_min={test_ts_min}"
    )

    print(f"  Split {config.period_name}: "
          f"train={len(train):,}, val={len(val):,}, test={len(test):,}")

    return train, val, test


def load_node_features_sparse(
    config: ExperimentConfig, token: str
) -> tuple[np.ndarray, pd.DataFrame]:
    """Load sparse node features for the period.

    Returns:
        (node_idx_array, features_df) where node_idx_array is sorted global indices
        and features_df has 15 feature columns.
    """
    label = config.label
    remote = f"{YADISK_STREAM_DIR}/features_{label}.parquet"
    local = os.path.join(config.local_data_dir, f"features_{label}.parquet")
    download_if_missing(remote, local, token)

    df = pd.read_parquet(local)
    node_idx = df["node_idx"].values.astype(np.int64)
    features = df.drop(columns=["node_idx"])
    print(f"  Node features: {len(node_idx):,} active nodes, {features.shape[1]} features")
    return node_idx, features


def load_adjacency(
    config: ExperimentConfig, token: str
) -> tuple[np.ndarray, sparse.csr_matrix, sparse.csr_matrix]:
    """Load adjacency matrices and node mapping for the period.

    Returns:
        (node_mapping, adj_directed, adj_undirected) where node_mapping maps
        local index -> global index.
    """
    label = config.label
    files = {
        "mapping": (f"node_mapping_{label}.npy", f"node_mapping_{label}.npy"),
        "directed": (f"adj_{label}_directed.npz", f"adj_{label}_directed.npz"),
        "undirected": (f"adj_{label}_undirected.npz", f"adj_{label}_undirected.npz"),
    }

    local_paths = {}
    for key, (remote_name, local_name) in files.items():
        remote = f"{YADISK_STREAM_DIR}/{remote_name}"
        local = os.path.join(config.local_data_dir, local_name)
        download_if_missing(remote, local, token)
        local_paths[key] = local

    node_mapping = np.load(local_paths["mapping"])
    adj_dir = sparse.load_npz(local_paths["directed"])
    adj_undir = sparse.load_npz(local_paths["undirected"])

    n = len(node_mapping)
    assert adj_dir.shape == (n, n), f"Directed adj shape {adj_dir.shape} != ({n},{n})"
    assert adj_undir.shape == (n, n), f"Undirected adj shape {adj_undir.shape} != ({n},{n})"

    print(f"  Adjacency: {n:,} nodes, "
          f"directed nnz={adj_dir.nnz:,}, undirected nnz={adj_undir.nnz:,}")

    return node_mapping, adj_dir, adj_undir


def build_train_neighbor_sets(train_edges: pd.DataFrame) -> dict[int, set[int]]:
    """Build per-source neighbor sets from train edges.

    Used for historical negative sampling. Only train edges are included
    to prevent data leakage.
    """
    neighbors: dict[int, set[int]] = {}
    src = train_edges["src_idx"].values
    dst = train_edges["dst_idx"].values
    for s, d in zip(src, dst):
        neighbors.setdefault(s, set()).add(d)
    return neighbors

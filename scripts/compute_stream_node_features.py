"""Compute static node features from a stream graph for temporal link prediction.

Computes 15 node features from the train portion of the stream graph for two
configurations: features_10 (first 10% of edges) and features_25 (first 25%).

Features are computed ONLY on the train split (first 70% of the period) to
avoid data leakage into val/test.

Usage:
    YADISK_TOKEN="..." PYTHONPATH=. python scripts/compute_stream_node_features.py \
        --input /path/to/stream_graph.parquet \
        --output-dir /tmp/stream_features/ \
        --upload

    YADISK_TOKEN="..." PYTHONPATH=. python scripts/compute_stream_node_features.py \
        --yadisk-path orbitaal_processed/stream_graph/2020-06-01__2020-08-31.parquet \
        --output-dir /tmp/stream_features/ \
        --upload
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd


FEATURE_COLUMNS = [
    "log_in_degree",
    "log_out_degree",
    "in_out_ratio",
    "log_unique_in_cp",
    "log_unique_out_cp",
    "log_total_btc_in",
    "log_total_btc_out",
    "log_avg_btc_in",
    "log_avg_btc_out",
    "recency",
    "activity_span",
    "log_event_rate",
    "burstiness",
    "out_counterparty_entropy",
    "in_counterparty_entropy",
]


def compute_node_features(
    src: np.ndarray,
    dst: np.ndarray,
    ts: np.ndarray,
    btc: np.ndarray,
    num_nodes: int,
) -> np.ndarray:
    """Compute 15 node features from edge arrays.

    Args:
        src: Source node indices (int64).
        dst: Destination node indices (int64).
        ts: Timestamps (int64, UNIX seconds).
        btc: Transaction values in BTC (float32).
        num_nodes: Total number of nodes (features array size).

    Returns:
        Float32 array of shape (num_nodes, 15).
    """
    btc64 = btc.astype(np.float64)
    features = np.zeros((num_nodes, 15), dtype=np.float64)

    t0 = time.time()
    _compute_degree_features(src, dst, num_nodes, features)
    print(f"  - Degree features... done ({time.time() - t0:.1f}s)")

    t0 = time.time()
    _compute_volume_features(src, dst, btc64, num_nodes, features)
    print(f"  - Volume features... done ({time.time() - t0:.1f}s)")

    t0 = time.time()
    _compute_temporal_features(src, dst, ts, num_nodes, features)
    print(f"  - Temporal features... done ({time.time() - t0:.1f}s)")

    t0 = time.time()
    _compute_entropy_features(src, dst, num_nodes, features)
    print(f"  - Entropy features... done ({time.time() - t0:.1f}s)")

    result = features.astype(np.float32)
    result[~np.isfinite(result)] = 0.0
    return result


def _compute_degree_features(
    src: np.ndarray, dst: np.ndarray, num_nodes: int, features: np.ndarray
):
    """Compute degree and counterparty features (columns 0-4)."""
    out_deg = np.bincount(src, minlength=num_nodes).astype(np.float64)
    in_deg = np.bincount(dst, minlength=num_nodes).astype(np.float64)

    features[:, 0] = np.log1p(in_deg)
    features[:, 1] = np.log1p(out_deg)

    total_deg = in_deg + out_deg
    ratio = np.full(num_nodes, 0.5, dtype=np.float64)
    mask_active = total_deg > 0
    ratio[mask_active] = out_deg[mask_active] / total_deg[mask_active]
    features[:, 2] = ratio

    df_edges = pd.DataFrame({"src": src, "dst": dst})
    unique_out_cp = df_edges.groupby("src")["dst"].nunique()
    arr_unique_out = np.zeros(num_nodes, dtype=np.float64)
    arr_unique_out[unique_out_cp.index.values] = unique_out_cp.values

    unique_in_cp = df_edges.groupby("dst")["src"].nunique()
    arr_unique_in = np.zeros(num_nodes, dtype=np.float64)
    arr_unique_in[unique_in_cp.index.values] = unique_in_cp.values

    features[:, 3] = np.log1p(arr_unique_in)
    features[:, 4] = np.log1p(arr_unique_out)


def _compute_volume_features(
    src: np.ndarray, dst: np.ndarray, btc: np.ndarray, num_nodes: int,
    features: np.ndarray,
):
    """Compute BTC volume features (columns 5-8)."""
    total_in = np.zeros(num_nodes, dtype=np.float64)
    np.add.at(total_in, dst, btc)

    total_out = np.zeros(num_nodes, dtype=np.float64)
    np.add.at(total_out, src, btc)

    in_deg = np.bincount(dst, minlength=num_nodes).astype(np.float64)
    out_deg = np.bincount(src, minlength=num_nodes).astype(np.float64)

    avg_in = np.zeros(num_nodes, dtype=np.float64)
    mask_in = in_deg > 0
    avg_in[mask_in] = total_in[mask_in] / in_deg[mask_in]

    avg_out = np.zeros(num_nodes, dtype=np.float64)
    mask_out = out_deg > 0
    avg_out[mask_out] = total_out[mask_out] / out_deg[mask_out]

    features[:, 5] = np.log1p(total_in)
    features[:, 6] = np.log1p(total_out)
    features[:, 7] = np.log1p(avg_in)
    features[:, 8] = np.log1p(avg_out)


def _compute_temporal_features(
    src: np.ndarray, dst: np.ndarray, ts: np.ndarray, num_nodes: int,
    features: np.ndarray,
):
    """Compute temporal pattern features (columns 9-12)."""
    nodes = np.concatenate([src, dst])
    times = np.concatenate([ts, ts]).astype(np.float64)

    t_split = float(ts.max())
    t_min_global = float(ts.min())
    time_range = max(t_split - t_min_global, 1.0)

    events = pd.DataFrame({"node": nodes, "t": times})
    node_stats = events.groupby("node")["t"].agg(["min", "max", "count"])

    t_last = np.full(num_nodes, np.nan, dtype=np.float64)
    t_first = np.full(num_nodes, np.nan, dtype=np.float64)
    n_events = np.zeros(num_nodes, dtype=np.float64)

    idx = node_stats.index.values
    t_last[idx] = node_stats["max"].values
    t_first[idx] = node_stats["min"].values
    n_events[idx] = node_stats["count"].values

    active = n_events > 0
    recency = np.ones(num_nodes, dtype=np.float64)
    recency[active] = (t_split - t_last[active]) / time_range

    span_seconds = np.zeros(num_nodes, dtype=np.float64)
    span_seconds[active] = t_last[active] - t_first[active]

    activity_span = np.zeros(num_nodes, dtype=np.float64)
    activity_span[active] = span_seconds[active] / time_range

    event_rate = np.zeros(num_nodes, dtype=np.float64)
    mask_span = span_seconds > 0
    event_rate[mask_span] = n_events[mask_span] / span_seconds[mask_span]

    features[:, 9] = recency
    features[:, 10] = activity_span
    features[:, 11] = np.log1p(event_rate)

    events.sort_values(["node", "t"], inplace=True)
    events["dt"] = events.groupby("node")["t"].diff()
    dt_valid = events.dropna(subset=["dt"])
    dt_stats = dt_valid.groupby("node")["dt"].agg(["mean", "std", "count"])

    burstiness = np.zeros(num_nodes, dtype=np.float64)
    valid = dt_stats[dt_stats["count"] >= 2]
    if len(valid) > 0:
        idx_v = valid.index.values
        mu = valid["mean"].values.astype(np.float64)
        sigma = valid["std"].values.astype(np.float64)
        denom = sigma + mu + 1e-8
        burstiness[idx_v] = (sigma - mu) / denom

    features[:, 12] = burstiness


def _compute_entropy_features(
    src: np.ndarray, dst: np.ndarray, num_nodes: int, features: np.ndarray,
):
    """Compute counterparty entropy features (columns 13-14)."""
    df = pd.DataFrame({"src": src, "dst": dst})

    out_entropy = _normalized_entropy(df, group_col="src", value_col="dst", num_nodes=num_nodes)
    in_entropy = _normalized_entropy(df, group_col="dst", value_col="src", num_nodes=num_nodes)

    features[:, 13] = out_entropy
    features[:, 14] = in_entropy


def _normalized_entropy(
    df: pd.DataFrame, group_col: str, value_col: str, num_nodes: int,
) -> np.ndarray:
    """Compute normalized entropy of value distribution per group."""
    result = np.zeros(num_nodes, dtype=np.float64)

    counts = df.groupby([group_col, value_col]).size().reset_index(name="cnt")
    if len(counts) == 0:
        return result

    totals = counts.groupby(group_col)["cnt"].transform("sum")
    n_unique = counts.groupby(group_col)[value_col].transform("nunique")

    p = counts["cnt"].values.astype(np.float64) / totals.values.astype(np.float64)
    log_p = np.log(p)
    counts["neg_p_logp"] = -(p * log_p)
    counts["n_unique"] = n_unique

    entropy_per_group = counts.groupby(group_col).agg(
        H=("neg_p_logp", "sum"),
        n_unique=("n_unique", "first"),
    )

    idx = entropy_per_group.index.values
    H = entropy_per_group["H"].values.astype(np.float64)
    n_u = entropy_per_group["n_unique"].values.astype(np.float64)

    mask = n_u > 1
    normalized = np.zeros(len(H), dtype=np.float64)
    normalized[mask] = H[mask] / np.log(n_u[mask])

    result[idx] = normalized
    return result


def process_period(
    df_full: pd.DataFrame,
    fraction: float,
    train_ratio: float,
    num_nodes: int,
    output_dir: str,
    label: str,
) -> dict:
    """Process one period configuration and save results.

    Args:
        df_full: Full stream graph DataFrame.
        fraction: Fraction of edges to use as period (e.g. 0.10).
        train_ratio: Fraction of period edges for training (e.g. 0.70).
        num_nodes: Total number of nodes.
        output_dir: Directory to save results.
        label: Label for output files (e.g. "features_10").

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
    print(f"  period={n_period:,} edges, train={n_train:,} edges")

    src = train["src_idx"].values.astype(np.int64)
    dst = train["dst_idx"].values.astype(np.int64)
    ts = train["timestamp"].values.astype(np.int64)
    btc = train["btc"].values.astype(np.float32)

    feat = compute_node_features(src, dst, ts, btc, num_nodes)

    assert feat.shape == (num_nodes, 15), f"Shape mismatch: {feat.shape}"
    assert np.isfinite(feat).all(), "Non-finite values in features"

    n_active = int((feat.sum(axis=1) != 0).sum())
    print(f"  - Validation: all finite, shape {feat.shape}, active nodes: {n_active:,} ✓")

    feat_df = pd.DataFrame(feat, columns=FEATURE_COLUMNS)
    feat_df = feat_df.astype(np.float32)

    parquet_path = os.path.join(output_dir, f"{label}.parquet")
    feat_df.to_parquet(parquet_path, index=False)
    size_mb = os.path.getsize(parquet_path) / 1e6
    print(f"  Saved {parquet_path} ({size_mb:.1f} MB)")

    metadata = {
        "period_fraction": fraction,
        "train_ratio": train_ratio,
        "num_edges_period": n_period,
        "num_edges_train": n_train,
        "num_nodes_total": num_nodes,
        "num_nodes_active_in_train": n_active,
        "train_timestamp_min": int(ts.min()) if n_train > 0 else 0,
        "train_timestamp_max": int(ts.max()) if n_train > 0 else 0,
        "feature_columns": FEATURE_COLUMNS,
        "source_file": "2020-06-01__2020-08-31.parquet",
    }

    json_path = os.path.join(output_dir, f"{label}.json")
    with open(json_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"  Saved {json_path}")

    return metadata


def main():
    parser = argparse.ArgumentParser(
        description="Compute static node features from stream graph"
    )
    parser.add_argument("--input", type=str, default=None,
                        help="Local path to stream graph parquet")
    parser.add_argument("--yadisk-path", type=str,
                        default="orbitaal_processed/stream_graph/2020-06-01__2020-08-31.parquet",
                        help="Yandex.Disk path to download stream graph")
    parser.add_argument("--output-dir", type=str, default="/tmp/stream_features/",
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
        print(f"[1/6] Downloading stream graph from Yandex.Disk...")
        ok = download_file(args.yadisk_path, args.input, token)
        if not ok:
            print("ERROR: Failed to download stream graph")
            sys.exit(1)
        print(f"  Downloaded to {args.input}")
    else:
        print(f"[1/6] Loading stream graph...")

    df = pd.read_parquet(args.input)
    num_nodes = int(max(df["src_idx"].max(), df["dst_idx"].max()) + 1)
    print(f"  {len(df):,} edges, {num_nodes:,} nodes")

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"[2/6] Computing features_10...")
    process_period(df, 0.10, 0.70, num_nodes, args.output_dir, "features_10")

    print(f"[3/6] Saving features_10 — done (saved inside process_period)")

    print(f"[4/6] Computing features_25...")
    process_period(df, 0.25, 0.70, num_nodes, args.output_dir, "features_25")

    print(f"[5/6] Saving features_25 — done (saved inside process_period)")

    if args.upload:
        token = os.environ.get("YADISK_TOKEN", "")
        if not token:
            print("ERROR: YADISK_TOKEN required for --upload")
            sys.exit(1)
        from src.yadisk_utils import upload_file
        print(f"[6/6] Uploading to Yandex.Disk...")
        for label in ["features_10", "features_25"]:
            for ext in [".parquet", ".json"]:
                local = os.path.join(args.output_dir, f"{label}{ext}")
                remote = f"orbitaal_processed/stream_graph/{label}{ext}"
                upload_file(local, remote, token)
                print(f"  Uploaded {remote}")
        print("  Done!")
    else:
        print(f"[6/6] Skipping upload (use --upload to upload to Yandex.Disk)")

    print("\nAll done.")


if __name__ == "__main__":
    main()

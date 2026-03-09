"""Compute graph-level and node-level features from daily snapshot parquets.

Processes each daily snapshot, computes structural and transactional features
at both graph and node granularity. Graph-level features are saved as a single
CSV. Node-level features are saved as daily parquet files, optionally uploaded
to Yandex.Disk in batches with local cleanup to conserve disk space.

Output structure:
    data/processed/
        graph_features.csv              # one row per day, ~40 columns
        node_features/
            2009-01-03.parquet          # one file per day, active nodes only
            ...
            2021-01-25.parquet

Usage:
    python src/compute_features.py --input-dir data/processed/daily_snapshots \\
                                   --output-dir data/processed \\
                                   --upload --batch-size 100
"""

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components


def build_adjacency(src: np.ndarray, dst: np.ndarray, num_nodes: int):
    """Build sparse binary and weighted adjacency matrices.

    Args:
        src: Source node indices array.
        dst: Destination node indices array.
        num_nodes: Total number of nodes (matrix dimension).

    Returns:
        Tuple of (binary_adjacency, index_map) where binary_adjacency is a
        CSR matrix and index_map maps compressed indices back to original
        node indices.
    """
    unique_nodes = np.unique(np.concatenate([src, dst]))
    n = len(unique_nodes)
    remap = np.full(num_nodes, -1, dtype=np.int64)
    remap[unique_nodes] = np.arange(n)

    src_c = remap[src]
    dst_c = remap[dst]

    adj = sp.csr_matrix(
        (np.ones(len(src_c), dtype=np.float32), (src_c, dst_c)),
        shape=(n, n),
    )
    return adj, unique_nodes, src_c, dst_c


def compute_pagerank(adj, alpha=0.85, max_iter=100, tol=1e-6):
    """Compute PageRank via power iteration on a sparse adjacency matrix.

    Args:
        adj: Sparse CSR adjacency matrix (directed).
        alpha: Damping factor.
        max_iter: Maximum number of iterations.
        tol: Convergence tolerance (L1 norm).

    Returns:
        np.ndarray of PageRank scores, shape (num_nodes,).
    """
    n = adj.shape[0]
    if n == 0:
        return np.array([], dtype=np.float64)

    out_degree = np.array(adj.sum(axis=1)).flatten()
    dangling = out_degree == 0

    out_degree[dangling] = 1
    D_inv = sp.diags(1.0 / out_degree)
    M = (D_inv @ adj).T

    pr = np.full(n, 1.0 / n)
    for _ in range(max_iter):
        new_pr = alpha * M.dot(pr) + alpha * dangling.dot(pr) / n + (1 - alpha) / n
        if np.abs(new_pr - pr).sum() < tol:
            break
        pr = new_pr

    return pr / pr.sum()


def compute_clustering(adj):
    """Compute undirected clustering coefficient per node using sparse ops.

    Symmetrizes the adjacency matrix, then computes triangles via
    A_sym @ A_sym element-wise multiplied by A_sym.

    Args:
        adj: Sparse CSR adjacency matrix (directed).

    Returns:
        np.ndarray of clustering coefficients, shape (num_nodes,).
    """
    n = adj.shape[0]
    if n == 0:
        return np.array([], dtype=np.float64)

    A = (adj + adj.T)
    A = (A > 0).astype(np.float32)
    A.setdiag(0)
    A.eliminate_zeros()

    deg = np.array(A.sum(axis=1)).flatten()

    A2 = A @ A
    tri = np.array(A2.multiply(A).sum(axis=1)).flatten() / 2.0

    cc = np.zeros(n, dtype=np.float64)
    mask = deg > 1
    cc[mask] = 2.0 * tri[mask] / (deg[mask] * (deg[mask] - 1))
    return cc


def compute_k_core(adj):
    """Compute k-core number per node via Batagelj-Zaversnik peeling.

    Operates on the symmetrized (undirected) version of the graph.

    Args:
        adj: Sparse CSR adjacency matrix (directed).

    Returns:
        np.ndarray of k-core numbers, shape (num_nodes,).
    """
    n = adj.shape[0]
    if n == 0:
        return np.array([], dtype=np.int32)

    A = (adj + adj.T)
    A = (A > 0).astype(np.int32)
    A.setdiag(0)
    A.eliminate_zeros()
    A = A.tocsr()

    deg = np.array(A.sum(axis=1)).flatten().astype(np.int32)
    core = deg.copy()

    max_deg = int(deg.max()) if n > 0 else 0
    if max_deg == 0:
        return np.zeros(n, dtype=np.int32)

    bin_count = np.zeros(max_deg + 1, dtype=np.int32)
    for d in deg:
        bin_count[d] += 1

    bin_start = np.zeros(max_deg + 2, dtype=np.int32)
    for d in range(1, max_deg + 1):
        bin_start[d] = bin_start[d - 1] + bin_count[d - 1]
    bin_start[max_deg + 1] = n

    pos = np.zeros(n, dtype=np.int32)
    order = np.zeros(n, dtype=np.int32)
    bin_offset = bin_start[:-1].copy()

    for v in range(n):
        d = deg[v]
        pos[v] = bin_offset[d]
        order[bin_offset[d]] = v
        bin_offset[d] += 1

    for i in range(n):
        v = order[i]
        for j_ptr in range(A.indptr[v], A.indptr[v + 1]):
            u = A.indices[j_ptr]
            if core[u] > core[v]:
                du = core[u]
                pw = bin_start[du]
                w = order[pw]
                if u != w:
                    order[pos[u]] = w
                    order[pw] = u
                    pos[w] = pos[u]
                    pos[u] = pw
                bin_start[du] += 1
                core[u] -= 1

    return core


def compute_triangle_counts(adj):
    """Compute per-node triangle count using sparse matrix multiplication.

    Args:
        adj: Sparse CSR adjacency matrix (directed).

    Returns:
        np.ndarray of triangle counts per node, shape (num_nodes,).
    """
    n = adj.shape[0]
    if n == 0:
        return np.array([], dtype=np.int64)

    A = (adj + adj.T)
    A = (A > 0).astype(np.float32)
    A.setdiag(0)
    A.eliminate_zeros()

    A2 = A @ A
    tri = np.array(A2.multiply(A).sum(axis=1)).flatten() / 2.0
    return tri.astype(np.int64)


def gini_coefficient(values):
    """Compute Gini coefficient of an array.

    Args:
        values: 1-D array of non-negative values.

    Returns:
        Float Gini coefficient in [0, 1].
    """
    if len(values) == 0:
        return 0.0
    v = np.sort(values)
    n = len(v)
    total = v.sum()
    if total == 0:
        return 0.0
    index = np.arange(1, n + 1)
    return (2.0 * (index * v).sum() / (n * total)) - (n + 1) / n


def compute_node_features(df, adj, unique_nodes, src_c, dst_c):
    """Compute all node-level features for one day.

    Args:
        df: Daily snapshot DataFrame with columns src_idx, dst_idx, btc, usd.
        adj: Sparse CSR binary adjacency matrix (compressed indices).
        unique_nodes: Array mapping compressed index -> original node index.
        src_c: Compressed source indices.
        dst_c: Compressed destination indices.

    Returns:
        DataFrame with node_idx and feature columns for all active nodes.
    """
    n = len(unique_nodes)
    if n == 0:
        return pd.DataFrame(columns=[
            "node_idx", "in_degree", "out_degree", "total_degree",
            "weighted_in_btc", "weighted_out_btc", "weighted_in_usd", "weighted_out_usd",
            "balance_btc", "balance_usd",
            "avg_in_btc", "avg_out_btc", "median_in_btc", "median_out_btc",
            "max_in_btc", "max_out_btc", "min_in_btc", "min_out_btc",
            "std_in_btc", "std_out_btc",
            "unique_in_counterparties", "unique_out_counterparties",
            "pagerank", "clustering_coeff", "k_core", "triangle_count",
        ])

    in_deg = np.array(adj.sum(axis=0)).flatten().astype(np.int32)
    out_deg = np.array(adj.sum(axis=1)).flatten().astype(np.int32)

    remap = np.full(int(unique_nodes.max()) + 1, -1, dtype=np.int64)
    remap[unique_nodes] = np.arange(n)

    btc = df["btc"].values.astype(np.float64)
    usd = df["usd"].values if "usd" in df.columns else np.zeros(len(df), dtype=np.float64)

    w_in_btc = np.zeros(n, dtype=np.float64)
    w_out_btc = np.zeros(n, dtype=np.float64)
    w_in_usd = np.zeros(n, dtype=np.float64)
    w_out_usd = np.zeros(n, dtype=np.float64)
    np.add.at(w_in_btc, dst_c, btc)
    np.add.at(w_out_btc, src_c, btc)
    np.add.at(w_in_usd, dst_c, usd)
    np.add.at(w_out_usd, src_c, usd)

    src_orig = df["src_idx"].values
    dst_orig = df["dst_idx"].values
    src_series = pd.Series(src_c)
    dst_series = pd.Series(dst_c)
    btc_series = pd.Series(btc)

    in_stats = pd.DataFrame({"node": dst_c, "btc": btc}).groupby("node")["btc"].agg(
        ["mean", "median", "max", "min", "std"]
    )
    out_stats = pd.DataFrame({"node": src_c, "btc": btc}).groupby("node")["btc"].agg(
        ["mean", "median", "max", "min", "std"]
    )

    avg_in_btc = np.zeros(n, dtype=np.float64)
    med_in_btc = np.zeros(n, dtype=np.float64)
    max_in_btc = np.zeros(n, dtype=np.float64)
    min_in_btc = np.zeros(n, dtype=np.float64)
    std_in_btc = np.zeros(n, dtype=np.float64)
    if len(in_stats) > 0:
        idx = in_stats.index.values
        avg_in_btc[idx] = in_stats["mean"].values
        med_in_btc[idx] = in_stats["median"].values
        max_in_btc[idx] = in_stats["max"].values
        min_in_btc[idx] = in_stats["min"].values
        std_in_btc[idx] = in_stats["std"].fillna(0).values

    avg_out_btc = np.zeros(n, dtype=np.float64)
    med_out_btc = np.zeros(n, dtype=np.float64)
    max_out_btc = np.zeros(n, dtype=np.float64)
    min_out_btc = np.zeros(n, dtype=np.float64)
    std_out_btc = np.zeros(n, dtype=np.float64)
    if len(out_stats) > 0:
        idx = out_stats.index.values
        avg_out_btc[idx] = out_stats["mean"].values
        med_out_btc[idx] = out_stats["median"].values
        max_out_btc[idx] = out_stats["max"].values
        min_out_btc[idx] = out_stats["min"].values
        std_out_btc[idx] = out_stats["std"].fillna(0).values

    unique_in = np.zeros(n, dtype=np.int32)
    unique_out = np.zeros(n, dtype=np.int32)
    in_counts = pd.Series(dst_c).value_counts()
    out_counts = pd.Series(src_c).value_counts()

    in_cp = pd.DataFrame({"dst": dst_c, "src": src_c}).groupby("dst")["src"].nunique()
    out_cp = pd.DataFrame({"src": src_c, "dst": dst_c}).groupby("src")["dst"].nunique()
    if len(in_cp) > 0:
        unique_in[in_cp.index.values] = in_cp.values.astype(np.int32)
    if len(out_cp) > 0:
        unique_out[out_cp.index.values] = out_cp.values.astype(np.int32)

    pr = compute_pagerank(adj)
    cc = compute_clustering(adj)
    kcore = compute_k_core(adj)
    tri = compute_triangle_counts(adj)

    result = pd.DataFrame({
        "node_idx": unique_nodes,
        "in_degree": in_deg,
        "out_degree": out_deg,
        "total_degree": in_deg + out_deg,
        "weighted_in_btc": w_in_btc,
        "weighted_out_btc": w_out_btc,
        "weighted_in_usd": w_in_usd,
        "weighted_out_usd": w_out_usd,
        "balance_btc": w_in_btc - w_out_btc,
        "balance_usd": w_in_usd - w_out_usd,
        "avg_in_btc": avg_in_btc,
        "avg_out_btc": avg_out_btc,
        "median_in_btc": med_in_btc,
        "median_out_btc": med_out_btc,
        "max_in_btc": max_in_btc,
        "max_out_btc": max_out_btc,
        "min_in_btc": min_in_btc,
        "min_out_btc": min_out_btc,
        "std_in_btc": std_in_btc,
        "std_out_btc": std_out_btc,
        "unique_in_counterparties": unique_in,
        "unique_out_counterparties": unique_out,
        "pagerank": pr,
        "clustering_coeff": cc,
        "k_core": kcore,
        "triangle_count": tri,
    })

    for col in result.columns:
        if result[col].dtype == np.float64:
            result[col] = result[col].astype(np.float32)

    return result


def compute_graph_features(df, date_str, adj, unique_nodes, src_c, dst_c,
                           node_features_df):
    """Compute graph-level features for one day.

    Args:
        df: Daily snapshot DataFrame.
        date_str: Date string YYYY-MM-DD.
        adj: Sparse CSR binary adjacency matrix (compressed).
        unique_nodes: Array of active node indices.
        src_c: Compressed source indices.
        dst_c: Compressed destination indices.
        node_features_df: Pre-computed node-level features DataFrame.

    Returns:
        Dict of graph-level features.
    """
    n = len(unique_nodes)
    m = len(df)

    if n == 0:
        return {"date": date_str, "num_nodes": 0, "num_edges": 0}

    in_deg = node_features_df["in_degree"].values
    out_deg = node_features_df["out_degree"].values
    total_deg = node_features_df["total_degree"].values
    pr = node_features_df["pagerank"].values
    cc = node_features_df["clustering_coeff"].values
    kcore = node_features_df["k_core"].values
    tri = node_features_df["triangle_count"].values

    density = m / (n * (n - 1)) if n > 1 else 0.0

    n_wcc, labels_wcc = connected_components(adj, directed=True, connection="weak")
    n_scc, labels_scc = connected_components(adj, directed=True, connection="strong")

    wcc_sizes = np.bincount(labels_wcc)
    scc_sizes = np.bincount(labels_scc)
    largest_wcc = int(wcc_sizes.max())
    largest_scc = int(scc_sizes.max())

    A_sym = (adj + adj.T)
    A_sym = (A_sym > 0).astype(np.float32)
    A_sym.setdiag(0)
    sym_deg = np.array(A_sym.sum(axis=1)).flatten()

    adj_no_diag = adj.copy()
    adj_no_diag.setdiag(0)
    adj_no_diag.eliminate_zeros()
    m_directed = adj_no_diag.nnz
    reciprocal = adj_no_diag.multiply(adj_no_diag.T).nnz
    reciprocity = reciprocal / m_directed if m_directed > 0 else 0.0

    btc_values = df["btc"].values
    usd_values = df["usd"].values if "usd" in df.columns else np.zeros(len(df))

    try:
        src_deg = out_deg[node_features_df.index]
        dst_remap = np.full(unique_nodes.max() + 1, -1, dtype=np.int64)
        dst_remap[unique_nodes] = np.arange(n)
        s_deg = out_deg[dst_remap[df["src_idx"].values]]
        d_deg = in_deg[dst_remap[df["dst_idx"].values]]
        if len(s_deg) > 1 and np.std(s_deg) > 0 and np.std(d_deg) > 0:
            assortativity = float(np.corrcoef(s_deg, d_deg)[0, 1])
        else:
            assortativity = 0.0
    except Exception:
        assortativity = 0.0

    features = {
        "date": date_str,
        "num_nodes": n,
        "num_edges": m,
        "density": density,

        "avg_in_degree": float(np.mean(in_deg)),
        "median_in_degree": float(np.median(in_deg)),
        "max_in_degree": int(np.max(in_deg)),
        "std_in_degree": float(np.std(in_deg)),
        "avg_out_degree": float(np.mean(out_deg)),
        "median_out_degree": float(np.median(out_deg)),
        "max_out_degree": int(np.max(out_deg)),
        "std_out_degree": float(np.std(out_deg)),

        "avg_total_degree": float(np.mean(total_deg)),
        "max_total_degree": int(np.max(total_deg)),
        "gini_total_degree": float(gini_coefficient(total_deg.astype(np.float64))),

        "avg_weighted_in_btc": float(node_features_df["weighted_in_btc"].mean()),
        "avg_weighted_out_btc": float(node_features_df["weighted_out_btc"].mean()),

        "total_btc": float(btc_values.sum()),
        "total_usd": float(usd_values.sum()),
        "avg_btc": float(btc_values.mean()),
        "median_btc": float(np.median(btc_values)),
        "max_btc": float(btc_values.max()),
        "std_btc": float(np.std(btc_values)),

        "num_wcc": int(n_wcc),
        "largest_wcc_size": largest_wcc,
        "largest_wcc_fraction": largest_wcc / n,
        "num_scc": int(n_scc),
        "largest_scc_size": largest_scc,
        "largest_scc_fraction": largest_scc / n,

        "avg_clustering": float(np.mean(cc)),
        "num_triangles": int(tri.sum() // 3),

        "avg_pagerank": float(np.mean(pr)),
        "max_pagerank": float(np.max(pr)),
        "std_pagerank": float(np.std(pr)),
        "gini_pagerank": float(gini_coefficient(pr.astype(np.float64))),

        "max_k_core": int(np.max(kcore)),
        "avg_k_core": float(np.mean(kcore)),

        "assortativity": assortativity,
        "reciprocity": reciprocity,

        "avg_neighbor_degree": float(np.mean(sym_deg)) if n > 0 else 0.0,
    }

    return features


def process_single_day(filepath):
    """Process one daily snapshot: compute graph and node features.

    Args:
        filepath: Path to daily snapshot parquet file.

    Returns:
        Tuple of (date_str, graph_features_dict, node_features_dataframe).
    """
    date_str = filepath.stem
    df = pd.read_parquet(filepath)

    if len(df) == 0:
        return date_str, {"date": date_str, "num_nodes": 0, "num_edges": 0}, None

    src = df["src_idx"].values.astype(np.int64)
    dst = df["dst_idx"].values.astype(np.int64)
    num_nodes = max(src.max(), dst.max()) + 1

    adj, unique_nodes, src_c, dst_c = build_adjacency(src, dst, num_nodes)

    node_feat = compute_node_features(df, adj, unique_nodes, src_c, dst_c)
    graph_feat = compute_graph_features(
        df, date_str, adj, unique_nodes, src_c, dst_c, node_feat
    )

    return date_str, graph_feat, node_feat


def upload_file_to_yadisk(local_path, remote_path, token):
    """Upload a single file to Yandex.Disk.

    Args:
        local_path: Path to local file.
        remote_path: Remote path on Yandex.Disk.
        token: OAuth token.

    Returns:
        True on success, False on failure.
    """
    api_base = "https://cloud-api.yandex.net/v1/disk/resources"
    headers = {"Authorization": f"OAuth {token}"}

    url = (f"{api_base}/upload"
           f"?path={urllib.parse.quote(remote_path)}"
           f"&overwrite=true")

    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req) as resp:
            data = json.loads(resp.read())
    except urllib.error.HTTPError as e:
        print(f"  [upload error] {remote_path}: {e}")
        return False

    if "href" not in data:
        return False

    with open(local_path, "rb") as f:
        file_data = f.read()

    put_req = urllib.request.Request(data["href"], data=file_data, method="PUT")
    try:
        urllib.request.urlopen(put_req)
    except urllib.error.HTTPError as e:
        print(f"  [upload error] PUT {remote_path}: {e}")
        return False

    return True


def ensure_remote_folder(folder_path, token):
    """Create a folder on Yandex.Disk if it doesn't exist.

    Args:
        folder_path: Remote folder path.
        token: OAuth token.
    """
    api_base = "https://cloud-api.yandex.net/v1/disk/resources"
    headers = {"Authorization": f"OAuth {token}"}
    url = f"{api_base}?path={urllib.parse.quote(folder_path)}"
    req = urllib.request.Request(url, headers=headers, method="PUT")
    try:
        urllib.request.urlopen(req)
    except urllib.error.HTTPError:
        pass


def upload_batch_and_cleanup(file_paths, remote_dir, token):
    """Upload a batch of files to Yandex.Disk and delete local copies.

    Args:
        file_paths: List of Path objects to upload.
        remote_dir: Remote directory on Yandex.Disk.
        token: OAuth token.

    Returns:
        Number of successfully uploaded and deleted files.
    """
    success = 0
    for p in file_paths:
        remote_path = f"{remote_dir}/{p.name}"
        if upload_file_to_yadisk(p, remote_path, token):
            p.unlink()
            success += 1
    return success


def format_eta(seconds):
    """Format seconds into human-readable HH:MM:SS string.

    Args:
        seconds: Number of seconds.

    Returns:
        Formatted string like "2h 15m 30s".
    """
    if seconds < 0:
        return "?"
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    if h > 0:
        return f"{h}h {m:02d}m {s:02d}s"
    if m > 0:
        return f"{m}m {s:02d}s"
    return f"{s}s"


def run_pipeline(input_dir, output_dir, upload=False, batch_size=100,
                 remote_dir="orbitaal_processed/node_features"):
    """Main processing loop: compute features for all daily snapshots.

    Args:
        input_dir: Directory containing daily snapshot parquet files.
        output_dir: Base output directory for graph_features.csv and node_features/.
        upload: Whether to upload node features to Yandex.Disk and delete locally.
        batch_size: Number of days to accumulate before uploading.
        remote_dir: Remote directory on Yandex.Disk for node features.
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    node_dir = output_dir / "node_features"
    node_dir.mkdir(parents=True, exist_ok=True)
    graph_csv = output_dir / "graph_features.csv"

    all_files = sorted(input_dir.glob("*.parquet"))
    if not all_files:
        print(f"[error] No parquet files found in {input_dir}")
        sys.exit(1)

    existing_graph = set()
    if graph_csv.exists():
        old = pd.read_csv(graph_csv)
        existing_graph = set(old["date"].values)

    pending = [f for f in all_files if f.stem not in existing_graph]

    if not pending:
        print(f"[skip] All {len(all_files)} days already processed")
        return

    print(f"[features] {len(pending)} days to process ({len(existing_graph)} already done)")
    print(f"[features] Output: {output_dir}")
    if upload:
        print(f"[features] Upload enabled, batch size: {batch_size}")

    token = None
    if upload:
        token = os.environ.get("YADISK_TOKEN")
        if not token:
            print("[error] YADISK_TOKEN not set, disabling upload")
            upload = False
        else:
            ensure_remote_folder("orbitaal_processed", token)
            ensure_remote_folder(remote_dir, token)

    graph_features_list = []
    if graph_csv.exists():
        graph_features_list = pd.read_csv(graph_csv).to_dict("records")

    batch_files = []
    times = []
    t_start = time.time()

    for i, filepath in enumerate(pending):
        t0 = time.time()

        date_str, graph_feat, node_feat = process_single_day(filepath)

        graph_features_list.append(graph_feat)

        if node_feat is not None:
            node_path = node_dir / f"{date_str}.parquet"
            node_feat.to_parquet(node_path, index=False)
            batch_files.append(node_path)

        elapsed = time.time() - t0
        times.append(elapsed)

        avg_time = np.mean(times[-100:])
        remaining = (len(pending) - i - 1) * avg_time
        eta_str = format_eta(remaining)
        total_elapsed = time.time() - t_start

        n_nodes = graph_feat.get("num_nodes", 0)
        n_edges = graph_feat.get("num_edges", 0)

        pct = (i + 1) / len(pending) * 100
        bar_len = 30
        filled = int(bar_len * (i + 1) / len(pending))
        bar = "█" * filled + "░" * (bar_len - filled)

        print(
            f"\r[{bar}] {i+1}/{len(pending)} ({pct:.1f}%) | "
            f"{date_str} | {n_nodes:,}n {n_edges:,}e | "
            f"{elapsed:.1f}s/day | ETA: {eta_str} | "
            f"elapsed: {format_eta(total_elapsed)}",
            end="",
            flush=True,
        )

        if upload and len(batch_files) >= batch_size:
            print(f"\n[upload] Uploading {len(batch_files)} files...")
            uploaded = upload_batch_and_cleanup(batch_files, remote_dir, token)
            print(f"[upload] {uploaded}/{len(batch_files)} uploaded, local copies deleted")
            batch_files = []

        if (i + 1) % 50 == 0 or i == len(pending) - 1:
            graph_df = pd.DataFrame(graph_features_list)
            graph_df.sort_values("date").to_csv(graph_csv, index=False)

    print()

    graph_df = pd.DataFrame(graph_features_list)
    graph_df.sort_values("date").to_csv(graph_csv, index=False)
    print(f"[features] Saved graph features: {graph_csv} ({len(graph_df)} rows)")

    if upload and batch_files:
        print(f"[upload] Final batch: {len(batch_files)} files...")
        uploaded = upload_batch_and_cleanup(batch_files, remote_dir, token)
        print(f"[upload] {uploaded}/{len(batch_files)} uploaded")

    if upload:
        upload_file_to_yadisk(
            graph_csv,
            "orbitaal_processed/graph_features.csv",
            token,
        )
        print("[upload] graph_features.csv uploaded")

    total = time.time() - t_start
    print(f"[features] Done in {format_eta(total)}")


def main():
    """CLI entry point for feature computation pipeline."""
    parser = argparse.ArgumentParser(description="Compute graph and node features")
    parser.add_argument(
        "--input-dir", type=Path,
        default=Path("data/processed/daily_snapshots"),
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("data/processed"),
    )
    parser.add_argument(
        "--upload", action="store_true",
        help="Upload node features to Yandex.Disk and delete local copies",
    )
    parser.add_argument(
        "--batch-size", type=int, default=100,
        help="Number of days per upload batch",
    )
    parser.add_argument(
        "--remote-dir", type=str,
        default="orbitaal_processed/node_features",
    )
    args = parser.parse_args()

    run_pipeline(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        upload=args.upload,
        batch_size=args.batch_size,
        remote_dir=args.remote_dir,
    )


if __name__ == "__main__":
    main()

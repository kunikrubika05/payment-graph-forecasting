"""Data utilities for temporal link prediction models.

Handles conversion of daily snapshots to event streams, CSR construction,
temporal neighbor sampling, and mini-batch preparation.

Uses C++ extension (temporal_sampling_cpp) when available for 3-5x speedup.
Falls back to pure Python/NumPy otherwise.
Build C++ extension: python src/models/build_ext.py
"""

import os
import logging
from pathlib import Path
from typing import List, Dict, Optional, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.baselines.config import PERIODS, NODE_FEATURE_COLUMNS
from src.baselines.data_loader import (
    get_available_dates,
    download_period_data,
    load_daily_snapshot,
    load_node_features,
)

logger = logging.getLogger(__name__)

_cpp_ext = None
_cpp_ext_loaded = False


def _load_cpp_extension():
    """Try to load the C++ temporal sampling extension."""
    global _cpp_ext, _cpp_ext_loaded
    if _cpp_ext_loaded:
        return _cpp_ext
    _cpp_ext_loaded = True
    try:
        from torch.utils.cpp_extension import load as _load_ext
        cpp_path = str(Path(__file__).parent / "csrc" / "temporal_sampling.cpp")
        build_dir = str(Path(__file__).parent / "csrc" / "build")
        if Path(cpp_path).exists():
            _cpp_ext = _load_ext(
                name="temporal_sampling_cpp",
                sources=[cpp_path],
                build_directory=build_dir,
                extra_cflags=["-O3"],
                verbose=False,
            )
            logger.info("Loaded C++ temporal sampling extension")
    except Exception as e:
        logger.info("C++ extension unavailable (%s), using Python fallback", e)
    return _cpp_ext


class TemporalEdgeData:
    """Container for temporal edge data in event-stream format.

    Attributes:
        src: Source node indices (int32).
        dst: Destination node indices (int32).
        timestamps: Edge timestamps (float64, day indices).
        edge_feats: Edge features array (float32), shape [num_edges, feat_dim].
        node_feats: Node features array (float32), shape [num_nodes, node_feat_dim].
        num_nodes: Total number of unique nodes.
        num_edges: Total number of temporal edges.
        node_id_map: Mapping from original node_idx to dense 0..N-1.
        reverse_node_map: Mapping from dense index back to original node_idx.
    """

    def __init__(
        self,
        src: np.ndarray,
        dst: np.ndarray,
        timestamps: np.ndarray,
        edge_feats: np.ndarray,
        node_feats: np.ndarray,
        node_id_map: Dict[int, int],
        reverse_node_map: np.ndarray,
    ):
        self.src = src.astype(np.int32)
        self.dst = dst.astype(np.int32)
        self.timestamps = timestamps.astype(np.float64)
        self.edge_feats = edge_feats.astype(np.float32)
        self.node_feats = node_feats.astype(np.float32)
        self.num_nodes = len(node_id_map)
        self.num_edges = len(src)
        self.node_id_map = node_id_map
        self.reverse_node_map = reverse_node_map

    def __repr__(self) -> str:
        return (
            f"TemporalEdgeData(num_nodes={self.num_nodes}, "
            f"num_edges={self.num_edges}, "
            f"edge_feat_dim={self.edge_feats.shape[1]}, "
            f"node_feat_dim={self.node_feats.shape[1]})"
        )


def build_event_stream(
    dates: List[str],
    local_dir: str,
    undirected: bool = True,
) -> TemporalEdgeData:
    """Convert daily snapshots + node features into a single event stream.

    Each day's edges become events with timestamp = day index (0, 1, 2, ...).
    If undirected=True, reverse edges are added (GraphMixer default).
    Node features are aggregated by mean across all days a node appears.

    Args:
        dates: Sorted list of date strings (YYYY-MM-DD).
        local_dir: Local directory with node_features/ and daily_snapshots/.
        undirected: Whether to add reverse edges.

    Returns:
        TemporalEdgeData with all edges and features.
    """
    all_src = []
    all_dst = []
    all_ts = []
    all_btc = []
    all_usd = []

    node_feat_accum: Dict[int, List[np.ndarray]] = {}
    active_nodes = set()

    for day_idx, date in enumerate(tqdm(dates, desc="Loading snapshots")):
        snap = load_daily_snapshot(date, local_dir)
        if snap is None or len(snap) == 0:
            continue

        src = snap["src_idx"].values
        dst = snap["dst_idx"].values
        btc = snap["btc"].values.astype(np.float32)
        usd = snap["usd"].values.astype(np.float32)
        ts = np.full(len(src), day_idx, dtype=np.float64)

        all_src.append(src)
        all_dst.append(dst)
        all_ts.append(ts)
        all_btc.append(btc)
        all_usd.append(usd)

        if undirected:
            all_src.append(dst)
            all_dst.append(src)
            all_ts.append(ts.copy())
            all_btc.append(btc.copy())
            all_usd.append(usd.copy())

        active_nodes.update(src)
        active_nodes.update(dst)

        nf = load_node_features(date, local_dir)
        if nf is not None:
            present_cols = [c for c in NODE_FEATURE_COLUMNS if c in nf.columns]
            col_indices = [NODE_FEATURE_COLUMNS.index(c) for c in present_cols]
            feat_matrix = np.zeros((len(nf), len(NODE_FEATURE_COLUMNS)), dtype=np.float32)
            feat_matrix[:, col_indices] = nf[present_cols].values.astype(np.float32)
            for row_idx, node_idx in enumerate(nf.index):
                if node_idx not in node_feat_accum:
                    node_feat_accum[node_idx] = []
                node_feat_accum[node_idx].append(feat_matrix[row_idx])

    src_all = np.concatenate(all_src)
    dst_all = np.concatenate(all_dst)
    ts_all = np.concatenate(all_ts)
    btc_all = np.concatenate(all_btc)
    usd_all = np.concatenate(all_usd)

    edge_feats = np.stack([btc_all, usd_all], axis=1)

    sorted_nodes = sorted(active_nodes)
    node_id_map = {orig: dense for dense, orig in enumerate(sorted_nodes)}
    reverse_node_map = np.array(sorted_nodes, dtype=np.int64)

    src_dense = np.array([node_id_map[n] for n in src_all], dtype=np.int32)
    dst_dense = np.array([node_id_map[n] for n in dst_all], dtype=np.int32)

    num_nodes = len(sorted_nodes)
    node_feat_dim = len(NODE_FEATURE_COLUMNS)
    node_feats = np.zeros((num_nodes, node_feat_dim), dtype=np.float32)
    for orig_idx, dense_idx in node_id_map.items():
        if orig_idx in node_feat_accum:
            stacked = np.stack(node_feat_accum[orig_idx])
            node_feats[dense_idx] = np.nanmean(stacked, axis=0)

    sort_idx = np.argsort(ts_all, kind="stable")
    src_dense = src_dense[sort_idx]
    dst_dense = dst_dense[sort_idx]
    ts_all = ts_all[sort_idx]
    edge_feats = edge_feats[sort_idx]

    nan_mask = np.isnan(node_feats)
    if nan_mask.any():
        node_feats[nan_mask] = 0.0
        logger.warning("Replaced %d NaN values in node features with 0", nan_mask.sum())

    logger.info(
        "Built event stream: %d nodes, %d edges, edge_feat_dim=%d, node_feat_dim=%d",
        num_nodes, len(src_dense), edge_feats.shape[1], node_feats.shape[1],
    )

    return TemporalEdgeData(
        src=src_dense,
        dst=dst_dense,
        timestamps=ts_all,
        edge_feats=edge_feats,
        node_feats=node_feats,
        node_id_map=node_id_map,
        reverse_node_map=reverse_node_map,
    )


def chronological_split(
    data: TemporalEdgeData,
    train_ratio: float = 0.6,
    val_ratio: float = 0.2,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split edge indices chronologically into train/val/test.

    Args:
        data: TemporalEdgeData (edges already sorted by timestamp).
        train_ratio: Fraction of edges for training.
        val_ratio: Fraction of edges for validation.

    Returns:
        Tuple of (train_mask, val_mask, test_mask) boolean arrays.
    """
    n = data.num_edges
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    train_mask = np.zeros(n, dtype=bool)
    val_mask = np.zeros(n, dtype=bool)
    test_mask = np.zeros(n, dtype=bool)

    train_mask[:train_end] = True
    val_mask[train_end:val_end] = True
    test_mask[val_end:] = True

    logger.info(
        "Split: train=%d, val=%d, test=%d edges",
        train_mask.sum(), val_mask.sum(), test_mask.sum(),
    )
    return train_mask, val_mask, test_mask


class TemporalCSR:
    """CSR-like structure for efficient temporal neighbor lookup.

    Stores adjacency sorted by timestamp for fast "most recent K neighbors" queries.
    Uses C++ backend when available for binary search and batch operations.
    """

    def __init__(self, num_nodes: int, src: np.ndarray, dst: np.ndarray,
                 timestamps: np.ndarray, edge_ids: np.ndarray):
        """Build CSR from edge arrays.

        Args:
            num_nodes: Total number of nodes.
            src: Source node indices.
            dst: Destination node indices (neighbors).
            timestamps: Edge timestamps.
            edge_ids: Original edge indices.
        """
        self.num_nodes = num_nodes

        cpp = _load_cpp_extension()
        if cpp is not None:
            self._cpp_csr = cpp.TemporalCSR(
                num_nodes,
                np.ascontiguousarray(src, dtype=np.int32),
                np.ascontiguousarray(dst, dtype=np.int32),
                np.ascontiguousarray(timestamps, dtype=np.float64),
                np.ascontiguousarray(edge_ids, dtype=np.int64),
            )
            self._use_cpp = True
        else:
            self._cpp_csr = None
            self._use_cpp = False

            sort_idx = np.lexsort((timestamps, src))
            src_sorted = src[sort_idx]
            dst_sorted = dst[sort_idx]
            ts_sorted = timestamps[sort_idx]
            eid_sorted = edge_ids[sort_idx]

            self.indptr = np.zeros(num_nodes + 1, dtype=np.int64)
            for s in src_sorted:
                self.indptr[s + 1] += 1
            np.cumsum(self.indptr, out=self.indptr)

            self.neighbors = dst_sorted.astype(np.int32)
            self.timestamps = ts_sorted
            self.edge_ids = eid_sorted.astype(np.int64)

    def get_temporal_neighbors(
        self,
        node: int,
        before_time: float,
        k: int = 20,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Get the K most recent neighbors of node before a given time.

        Args:
            node: Node index.
            before_time: Only return neighbors with timestamp < before_time.
            k: Maximum number of neighbors to return.

        Returns:
            Tuple of (neighbor_ids, timestamps, edge_ids), each of length <= k.
        """
        if self._use_cpp:
            return self._cpp_csr.get_temporal_neighbors(node, before_time, k)

        start = self.indptr[node]
        end = self.indptr[node + 1]

        if start == end:
            return (
                np.array([], dtype=np.int32),
                np.array([], dtype=np.float64),
                np.array([], dtype=np.int64),
            )

        ts_slice = self.timestamps[start:end]
        valid_end = np.searchsorted(ts_slice, before_time, side="left")

        if valid_end == 0:
            return (
                np.array([], dtype=np.int32),
                np.array([], dtype=np.float64),
                np.array([], dtype=np.int64),
            )

        actual_start = max(start, start + valid_end - k)
        actual_end = start + valid_end

        return (
            self.neighbors[actual_start:actual_end].copy(),
            self.timestamps[actual_start:actual_end].copy(),
            self.edge_ids[actual_start:actual_end].copy(),
        )


def build_temporal_csr(data: TemporalEdgeData, mask: Optional[np.ndarray] = None) -> TemporalCSR:
    """Build TemporalCSR from edge data, optionally filtering by mask.

    Args:
        data: TemporalEdgeData.
        mask: Optional boolean mask to select subset of edges (e.g., train only).

    Returns:
        TemporalCSR for neighbor lookups.
    """
    if mask is not None:
        src = data.src[mask]
        dst = data.dst[mask]
        ts = data.timestamps[mask]
        eids = np.where(mask)[0].astype(np.int64)
    else:
        src = data.src
        dst = data.dst
        ts = data.timestamps
        eids = np.arange(len(src), dtype=np.int64)

    return TemporalCSR(data.num_nodes, src, dst, ts, eids)


def sample_neighbors_batch(
    csr: TemporalCSR,
    nodes: np.ndarray,
    timestamps: np.ndarray,
    num_neighbors: int = 20,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Sample temporal neighbors for a batch of (node, time) queries.

    Args:
        csr: TemporalCSR structure.
        nodes: Array of node indices to query.
        timestamps: Array of query timestamps (one per node).
        num_neighbors: Max neighbors per node.

    Returns:
        Tuple of (neighbor_nodes, neighbor_ts, neighbor_eids, lengths):
            - neighbor_nodes: [batch_size, num_neighbors] padded with -1
            - neighbor_ts: [batch_size, num_neighbors] padded with 0
            - neighbor_eids: [batch_size, num_neighbors] padded with -1
            - lengths: [batch_size] actual number of neighbors per node
    """
    cpp = _load_cpp_extension()
    if cpp is not None and csr._use_cpp:
        return cpp.sample_neighbors_batch(
            csr._cpp_csr,
            np.ascontiguousarray(nodes, dtype=np.int32),
            np.ascontiguousarray(timestamps, dtype=np.float64),
            num_neighbors,
        )

    batch_size = len(nodes)
    neighbor_nodes = np.full((batch_size, num_neighbors), -1, dtype=np.int32)
    neighbor_ts = np.zeros((batch_size, num_neighbors), dtype=np.float64)
    neighbor_eids = np.full((batch_size, num_neighbors), -1, dtype=np.int64)
    lengths = np.zeros(batch_size, dtype=np.int32)

    for i in range(batch_size):
        nids, nts, neids = csr.get_temporal_neighbors(
            nodes[i], timestamps[i], k=num_neighbors
        )
        length = len(nids)
        lengths[i] = length
        if length > 0:
            neighbor_nodes[i, :length] = nids
            neighbor_ts[i, :length] = nts
            neighbor_eids[i, :length] = neids

    return neighbor_nodes, neighbor_ts, neighbor_eids, lengths


def featurize_neighbors(
    neighbor_nodes: np.ndarray,
    neighbor_eids: np.ndarray,
    lengths: np.ndarray,
    neighbor_ts: np.ndarray,
    query_ts: np.ndarray,
    node_feats: np.ndarray,
    edge_feats: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fill neighbor node/edge feature arrays and compute relative timestamps.

    Args:
        neighbor_nodes: [batch, K] neighbor node indices (padded with -1).
        neighbor_eids: [batch, K] neighbor edge indices (padded with -1).
        lengths: [batch] actual neighbor counts.
        neighbor_ts: [batch, K] neighbor timestamps.
        query_ts: [batch] query timestamps.
        node_feats: [num_nodes, node_feat_dim] all node features.
        edge_feats: [num_edges, edge_feat_dim] all edge features.

    Returns:
        Tuple of:
            - neighbor_node_feats: [batch, K, node_feat_dim]
            - neighbor_edge_feats: [batch, K, edge_feat_dim]
            - neighbor_rel_ts: [batch, K] relative timestamps (query_t - edge_t)
    """
    cpp = _load_cpp_extension()
    if cpp is not None:
        return cpp.featurize_neighbors(
            np.ascontiguousarray(neighbor_nodes, dtype=np.int32),
            np.ascontiguousarray(neighbor_eids, dtype=np.int64),
            np.ascontiguousarray(lengths, dtype=np.int32),
            np.ascontiguousarray(neighbor_ts, dtype=np.float64),
            np.ascontiguousarray(query_ts, dtype=np.float64),
            np.ascontiguousarray(node_feats, dtype=np.float32),
            np.ascontiguousarray(edge_feats, dtype=np.float32),
        )

    batch_size = neighbor_nodes.shape[0]
    K = neighbor_nodes.shape[1]
    node_feat_dim = node_feats.shape[1]
    edge_feat_dim = edge_feats.shape[1]

    out_nnf = np.zeros((batch_size, K, node_feat_dim), dtype=np.float32)
    out_nef = np.zeros((batch_size, K, edge_feat_dim), dtype=np.float32)
    out_nrt = np.zeros((batch_size, K), dtype=np.float64)

    for i in range(batch_size):
        length = lengths[i]
        if length == 0:
            continue

        valid_nids = neighbor_nodes[i, :length]
        out_nnf[i, :length] = node_feats[valid_nids]

        valid_eids = neighbor_eids[i, :length]
        for j in range(length):
            if valid_eids[j] >= 0:
                out_nef[i, j] = edge_feats[valid_eids[j]]

        out_nrt[i, :length] = query_ts[i] - neighbor_ts[i, :length]

    return out_nnf, out_nef, out_nrt


def generate_negatives_for_eval(
    src_node: int,
    true_dst: int,
    timestamp: float,
    csr: TemporalCSR,
    num_nodes: int,
    n_hist: int = 50,
    n_random: int = 50,
    rng: np.random.Generator = None,
) -> np.ndarray:
    """Generate negative candidates for TGB-style evaluation.

    50% historical neighbors (interacted before but not at this time) +
    50% random nodes. Ensures true_dst is not in negatives.

    Args:
        src_node: Source node.
        true_dst: True destination (to exclude from negatives).
        timestamp: Query timestamp.
        csr: TemporalCSR with historical edges.
        num_nodes: Total nodes in graph.
        n_hist: Number of historical negatives.
        n_random: Number of random negatives.
        rng: Random number generator.

    Returns:
        Array of negative node indices, length n_hist + n_random.
    """
    if rng is None:
        rng = np.random.default_rng(42)

    hist_neighbors, _, _ = csr.get_temporal_neighbors(
        src_node, before_time=timestamp, k=500
    )
    hist_set = set(hist_neighbors.tolist()) - {true_dst, src_node}

    hist_list = list(hist_set)
    if len(hist_list) >= n_hist:
        hist_neg = rng.choice(hist_list, size=n_hist, replace=False)
    elif len(hist_list) > 0:
        hist_neg = rng.choice(hist_list, size=n_hist, replace=True)
    else:
        hist_neg = rng.integers(0, num_nodes, size=n_hist)

    exclude = set(hist_neg.tolist()) | {true_dst, src_node}
    rand_neg = []
    while len(rand_neg) < n_random:
        candidates = rng.integers(0, num_nodes, size=n_random * 2)
        for c in candidates:
            if c not in exclude:
                rand_neg.append(c)
                exclude.add(c)
                if len(rand_neg) >= n_random:
                    break

    return np.concatenate([hist_neg, np.array(rand_neg[:n_random], dtype=np.int32)])


def prepare_period_data(
    period_name: str,
    local_dir: str = "/tmp/graphmixer_data",
    token: Optional[str] = None,
    undirected: bool = True,
) -> Tuple[TemporalEdgeData, List[str]]:
    """Download and prepare data for a specific period (all days).

    Args:
        period_name: Key from PERIODS dict (e.g., "mature_2020q2").
        local_dir: Local directory for downloaded files.
        token: Yandex.Disk token. If None, reads from YADISK_TOKEN env.
        undirected: Whether to make graph undirected.

    Returns:
        Tuple of (TemporalEdgeData, dates_list).
    """
    if token is None:
        token = os.environ.get("YADISK_TOKEN", "")

    period = PERIODS[period_name]
    dates = get_available_dates(period["start"], period["end"])
    logger.info("Period %s: %d available dates (%s to %s)",
                period_name, len(dates), dates[0], dates[-1])

    download_period_data(
        dates, local_dir, token,
        need_node_features=True, need_snapshots=True,
    )

    data = build_event_stream(dates, local_dir, undirected=undirected)
    return data, dates


def prepare_sliding_window(
    period_name: str,
    window: int = 7,
    target_offset: int = 0,
    local_dir: str = "/tmp/graphmixer_data",
    token: Optional[str] = None,
    undirected: bool = True,
) -> Tuple[TemporalEdgeData, List[str], np.ndarray, np.ndarray, np.ndarray]:
    """Prepare data using sliding window: W days train, next day val, day after test.

    Matches baseline protocol: train on window of W days, predict edges of next day.

    Args:
        period_name: Key from PERIODS dict.
        window: Number of days for training context.
        target_offset: Which window position within the period (0 = last possible).
        local_dir: Local directory for downloaded files.
        token: Yandex.Disk token.
        undirected: Whether to make graph undirected.

    Returns:
        Tuple of (TemporalEdgeData, all_dates, train_mask, val_mask, test_mask).
    """
    if token is None:
        token = os.environ.get("YADISK_TOKEN", "")

    period = PERIODS[period_name]
    all_dates = get_available_dates(period["start"], period["end"])
    n_dates = len(all_dates)

    needed = window + 2
    if n_dates < needed:
        raise ValueError(
            f"Period {period_name} has {n_dates} dates, need at least {needed} "
            f"(window={window} + 1 val + 1 test)"
        )

    max_offset = n_dates - needed
    if target_offset > max_offset:
        target_offset = max_offset
    start_idx = max_offset - target_offset

    selected_dates = all_dates[start_idx:start_idx + needed]
    train_dates = selected_dates[:window]
    val_date = selected_dates[window]
    test_date = selected_dates[window + 1]

    logger.info(
        "Sliding window: train %s..%s (%d days), val %s, test %s",
        train_dates[0], train_dates[-1], window, val_date, test_date,
    )

    download_period_data(
        selected_dates, local_dir, token,
        need_node_features=True, need_snapshots=True,
    )

    data = build_event_stream(selected_dates, local_dir, undirected=undirected)

    train_mask = data.timestamps < window
    val_mask = (data.timestamps >= window) & (data.timestamps < window + 1)
    test_mask = data.timestamps >= window + 1

    logger.info(
        "Split: train=%d edges (%d days), val=%d edges (1 day), test=%d edges (1 day)",
        train_mask.sum(), window, val_mask.sum(), test_mask.sum(),
    )

    return data, selected_dates, train_mask, val_mask, test_mask


def build_unified_sampler(
    data: TemporalEdgeData,
    mask: Optional[np.ndarray] = None,
    backend: str = "auto",
):
    """Create a TemporalGraphSampler from TemporalEdgeData.

    Drop-in bridge between existing pipeline and the new unified sampler.
    Supports 'python', 'cpp', 'cuda', or 'auto' backends.

    Args:
        data: TemporalEdgeData (from build_event_stream or prepare_period_data).
        mask: Optional boolean mask to select subset of edges (e.g., train only).
        backend: 'auto', 'python', 'cpp', or 'cuda'.

    Returns:
        TemporalGraphSampler instance with the selected backend.
    """
    from src.models.temporal_graph_sampler import TemporalGraphSampler

    if mask is not None:
        src = data.src[mask]
        dst = data.dst[mask]
        ts = data.timestamps[mask]
        eids = np.where(mask)[0].astype(np.int64)
    else:
        src = data.src
        dst = data.dst
        ts = data.timestamps
        eids = np.arange(len(data.src), dtype=np.int64)

    return TemporalGraphSampler(
        num_nodes=data.num_nodes,
        src=src,
        dst=dst,
        timestamps=ts,
        edge_ids=eids,
        node_feats=data.node_feats,
        edge_feats=data.edge_feats,
        backend=backend,
    )

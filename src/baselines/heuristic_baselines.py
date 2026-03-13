"""Heuristic link prediction baselines: Common Neighbors, Jaccard, Adamic-Adar, PA."""

import gc
import logging
import os
import time
from typing import Dict, List, Set, Tuple

import numpy as np
import pandas as pd
from scipy import sparse

from src.baselines.config import ExperimentConfig
from src.baselines.data_loader import (
    get_available_dates, download_period_data, load_daily_snapshot, cleanup_period_data,
)
from src.baselines.evaluation import compute_ranking_metrics
from src.baselines.experiment_logger import ExperimentLogger

logger = logging.getLogger(__name__)


def _build_adjacency(snapshots: Dict[str, pd.DataFrame]) -> Tuple[sparse.csr_matrix, np.ndarray]:
    """Build undirected adjacency matrix from multiple daily snapshots.

    Args:
        snapshots: Dict mapping dates to snapshot DataFrames.

    Returns:
        Tuple of (adjacency_matrix, node_array).
        adjacency_matrix: binary undirected CSR matrix.
        node_array: sorted unique node indices.
    """
    all_edges = set()
    for snap in snapshots.values():
        if snap is not None and len(snap) > 0:
            for s, d in zip(snap["src_idx"].values, snap["dst_idx"].values):
                all_edges.add((s, d))
                all_edges.add((d, s))

    if not all_edges:
        return sparse.csr_matrix((0, 0)), np.array([])

    edges_arr = np.array(list(all_edges), dtype=np.int64)
    nodes = np.unique(edges_arr)
    node_to_idx = {n: i for i, n in enumerate(nodes)}
    n = len(nodes)

    src = np.array([node_to_idx[e[0]] for e in all_edges])
    dst = np.array([node_to_idx[e[1]] for e in all_edges])
    data = np.ones(len(src), dtype=np.float32)

    adj = sparse.csr_matrix((data, (src, dst)), shape=(n, n))
    adj.data[:] = 1.0

    return adj, nodes


def compute_common_neighbors(adj: sparse.csr_matrix, src_idx: np.ndarray, dst_idx: np.ndarray) -> np.ndarray:
    """Compute Common Neighbors scores for given pairs.

    Args:
        adj: Binary undirected adjacency matrix.
        src_idx: Source node indices (local to adjacency).
        dst_idx: Destination node indices (local to adjacency).

    Returns:
        Array of CN scores.
    """
    scores = np.zeros(len(src_idx), dtype=np.float64)
    for i in range(len(src_idx)):
        s_neighbors = set(adj[src_idx[i]].indices)
        d_neighbors = set(adj[dst_idx[i]].indices)
        scores[i] = len(s_neighbors & d_neighbors)
    return scores


def compute_jaccard(adj: sparse.csr_matrix, src_idx: np.ndarray, dst_idx: np.ndarray) -> np.ndarray:
    """Compute Jaccard coefficient scores for given pairs."""
    scores = np.zeros(len(src_idx), dtype=np.float64)
    for i in range(len(src_idx)):
        s_neighbors = set(adj[src_idx[i]].indices)
        d_neighbors = set(adj[dst_idx[i]].indices)
        intersection = len(s_neighbors & d_neighbors)
        union = len(s_neighbors | d_neighbors)
        scores[i] = intersection / union if union > 0 else 0.0
    return scores


def compute_adamic_adar(adj: sparse.csr_matrix, src_idx: np.ndarray, dst_idx: np.ndarray) -> np.ndarray:
    """Compute Adamic-Adar scores for given pairs."""
    degrees = np.array(adj.sum(axis=1)).ravel()
    scores = np.zeros(len(src_idx), dtype=np.float64)
    for i in range(len(src_idx)):
        s_neighbors = set(adj[src_idx[i]].indices)
        d_neighbors = set(adj[dst_idx[i]].indices)
        common = s_neighbors & d_neighbors
        score = 0.0
        for cn in common:
            deg = degrees[cn]
            if deg > 1:
                score += 1.0 / np.log(deg)
        scores[i] = score
    return scores


def compute_preferential_attachment(adj: sparse.csr_matrix, src_idx: np.ndarray, dst_idx: np.ndarray) -> np.ndarray:
    """Compute Preferential Attachment scores for given pairs."""
    degrees = np.array(adj.sum(axis=1)).ravel()
    return degrees[src_idx] * degrees[dst_idx]


def _evaluate_heuristic_ranking(
    adj: sparse.csr_matrix,
    node_to_idx: Dict[int, int],
    nodes: np.ndarray,
    target_edges: Set[Tuple[int, int]],
    historical_neighbors: Dict[int, Set[int]],
    active_nodes: np.ndarray,
    heuristic_fn,
    n_negatives: int,
    seed: int,
) -> np.ndarray:
    """Evaluate a heuristic using per-source ranking protocol (TGB-style).

    For each positive edge (s, d_true), builds a candidate set of
    {d_true} ∪ {neg_1, ..., neg_q} and ranks d_true by heuristic score.

    Args:
        adj: Binary undirected adjacency matrix (local indices).
        node_to_idx: Mapping from global node ID to local index.
        nodes: Array of global node IDs.
        target_edges: Set of (src, dst) edges in the target day.
        historical_neighbors: Per-source historical neighbor sets.
        active_nodes: Array of all active node IDs.
        heuristic_fn: Function(adj, src_local, dst_local) -> scores.
        n_negatives: Number of negative candidates per query.
        seed: Random seed.

    Returns:
        Array of 1-based ranks for each query.
    """
    rng = np.random.RandomState(seed)
    valid_nodes = set(node_to_idx.keys())
    active_set = set(active_nodes)

    edges_by_source: Dict[int, List[int]] = {}
    for s, d in target_edges:
        if s in valid_nodes and d in valid_nodes:
            edges_by_source.setdefault(s, []).append(d)

    all_ranks = []

    for source, true_dsts in edges_by_source.items():
        target_set = set(true_dsts)
        hist = historical_neighbors.get(source, set())
        hist_candidates = list(hist - target_set - {source})

        for d_true in true_dsts:
            n_hist = min(n_negatives // 2, len(hist_candidates))
            n_rand = n_negatives - n_hist

            negatives = []
            if n_hist > 0:
                negatives.extend(rng.choice(hist_candidates, size=n_hist, replace=False).tolist())

            rand_pool = list(active_set - target_set - set(negatives) - {source})
            if len(rand_pool) >= n_rand:
                negatives.extend(rng.choice(rand_pool, size=n_rand, replace=False).tolist())
            else:
                negatives.extend(rand_pool)

            candidates = [d_true] + negatives
            candidates_in_adj = [c for c in candidates if c in valid_nodes]

            if len(candidates_in_adj) < 2:
                continue

            src_local = np.array([node_to_idx[source]] * len(candidates_in_adj))
            dst_local = np.array([node_to_idx[c] for c in candidates_in_adj])

            scores = heuristic_fn(adj, src_local, dst_local)

            true_score = scores[candidates_in_adj.index(d_true)]
            rank = int(np.sum(scores > true_score)) + 1
            all_ranks.append(rank)

    return np.array(all_ranks)


def run_heuristic_baselines(config: ExperimentConfig, token: str) -> None:
    """Run heuristic link prediction baselines.

    Args:
        config: Experiment configuration.
        token: Yandex.Disk OAuth token.
    """
    output_dir = os.path.join(
        config.output_dir or f"/tmp/baseline_results/{config.experiment_name}",
        config.sub_experiment or "heuristic",
    )
    exp_logger = ExperimentLogger(output_dir)

    if exp_logger.is_completed():
        logger.info("Experiment already completed: %s", output_dir)
        exp_logger.close()
        return

    exp_logger.log_config(config.to_dict())

    all_dates = get_available_dates(config.period_start, config.period_end)
    if len(all_dates) < config.window_size + 1:
        logger.error("Not enough dates for heuristic baselines: %d", len(all_dates))
        exp_logger.close()
        return

    prediction_dates = all_dates[config.window_size:]

    logger.info("Downloading daily snapshots...")
    download_period_data(
        all_dates, config.local_data_dir, token,
        need_node_features=False, need_snapshots=True,
    )

    heuristic_names = ["common_neighbors", "jaccard", "adamic_adar", "pref_attachment"]
    all_metrics = {h: [] for h in heuristic_names}

    for target_date in prediction_dates:
        target_idx = all_dates.index(target_date)
        window_start = max(0, target_idx - config.window_size)
        window_dates = all_dates[window_start:target_idx]

        window_snapshots = {}
        for d in window_dates:
            snap = load_daily_snapshot(d, config.local_data_dir)
            if snap is not None:
                window_snapshots[d] = snap

        if not window_snapshots:
            continue

        adj, nodes = _build_adjacency(window_snapshots)
        if len(nodes) < 2:
            continue
        node_to_idx = {n: i for i, n in enumerate(nodes)}

        target_snap = load_daily_snapshot(target_date, config.local_data_dir)
        if target_snap is None or len(target_snap) == 0:
            continue

        target_edges = set(zip(
            target_snap["src_idx"].values, target_snap["dst_idx"].values
        ))

        historical_neighbors: Dict[int, Set[int]] = {}
        for snap in window_snapshots.values():
            if snap is not None and len(snap) > 0:
                for s, d in zip(snap["src_idx"].values, snap["dst_idx"].values):
                    historical_neighbors.setdefault(s, set()).add(d)

        active_nodes = nodes

        seed = config.random_seed + hash(target_date) % 10000

        heuristic_fns = {
            "common_neighbors": compute_common_neighbors,
            "jaccard": compute_jaccard,
            "adamic_adar": compute_adamic_adar,
            "pref_attachment": compute_preferential_attachment,
        }

        for hname, hfn in heuristic_fns.items():
            ranks = _evaluate_heuristic_ranking(
                adj, node_to_idx, nodes, target_edges,
                historical_neighbors, active_nodes, hfn,
                config.n_negatives, seed,
            )
            if len(ranks) == 0:
                continue
            metrics = compute_ranking_metrics(ranks)
            metrics["date"] = target_date
            metrics["heuristic"] = hname
            exp_logger.log_metrics(metrics)
            all_metrics[hname].append(metrics)

        del adj, window_snapshots
        gc.collect()

        logger.info(
            "  %s: %s",
            target_date,
            {h: f"MRR={m[-1].get('mrr', 0):.4f}" for h, m in all_metrics.items() if m},
        )

    summary = {"config": config.to_dict(), "heuristics": {}}
    for hname, metrics_list in all_metrics.items():
        if not metrics_list:
            continue
        numeric_keys = [k for k in metrics_list[0] if isinstance(metrics_list[0][k], (int, float))]
        h_summary = {}
        for key in numeric_keys:
            values = [m[key] for m in metrics_list if key in m and not np.isnan(m.get(key, float("nan")))]
            if values:
                h_summary[f"mean_{key}"] = float(np.mean(values))
                h_summary[f"std_{key}"] = float(np.std(values))
        h_summary["n_days"] = len(metrics_list)
        summary["heuristics"][hname] = h_summary

    exp_logger.write_summary(summary)

    remote_dir = f"{config.yadisk_experiments_base}/{config.experiment_name}/{config.sub_experiment or 'heuristic'}"
    exp_logger.upload_to_yadisk(remote_dir, token)
    exp_logger.close()

    logger.info("=== Heuristic baselines complete ===")

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
    src_list, dst_list = [], []
    for snap in snapshots.values():
        if snap is not None and len(snap) > 0:
            s = snap["src_idx"].values
            d = snap["dst_idx"].values
            src_list.append(s)
            dst_list.append(d)
            src_list.append(d)
            dst_list.append(s)

    if not src_list:
        return sparse.csr_matrix((0, 0)), np.array([])

    all_src = np.concatenate(src_list)
    all_dst = np.concatenate(dst_list)

    nodes = np.unique(np.concatenate([all_src, all_dst]))
    n = len(nodes)

    local_src = np.searchsorted(nodes, all_src)
    local_dst = np.searchsorted(nodes, all_dst)

    pairs = np.stack([local_src, local_dst], axis=1)
    pairs = np.unique(pairs, axis=0)

    data = np.ones(len(pairs), dtype=np.float32)
    adj = sparse.csr_matrix((data, (pairs[:, 0], pairs[:, 1])), shape=(n, n))

    return adj, nodes


def compute_common_neighbors(adj: sparse.csr_matrix, src_idx: np.ndarray, dst_idx: np.ndarray) -> np.ndarray:
    """Compute Common Neighbors scores for given pairs (vectorized).

    Args:
        adj: Binary undirected adjacency matrix.
        src_idx: Source node indices (local to adjacency).
        dst_idx: Destination node indices (local to adjacency).

    Returns:
        Array of CN scores.
    """
    return np.array(adj[src_idx].multiply(adj[dst_idx]).sum(axis=1)).ravel()


def compute_jaccard(adj: sparse.csr_matrix, src_idx: np.ndarray, dst_idx: np.ndarray) -> np.ndarray:
    """Compute Jaccard coefficient scores for given pairs (vectorized)."""
    cn = np.array(adj[src_idx].multiply(adj[dst_idx]).sum(axis=1)).ravel()
    s_deg = np.array(adj[src_idx].sum(axis=1)).ravel()
    d_deg = np.array(adj[dst_idx].sum(axis=1)).ravel()
    union = s_deg + d_deg - cn
    scores = np.zeros_like(cn, dtype=np.float64)
    mask = union > 0
    scores[mask] = cn[mask] / union[mask]
    return scores


def compute_adamic_adar(adj: sparse.csr_matrix, src_idx: np.ndarray, dst_idx: np.ndarray) -> np.ndarray:
    """Compute Adamic-Adar scores for given pairs (vectorized)."""
    degrees = np.array(adj.sum(axis=1)).ravel()
    weights = np.zeros(adj.shape[0], dtype=np.float64)
    mask = degrees > 1
    weights[mask] = 1.0 / np.log(degrees[mask])
    common = adj[src_idx].multiply(adj[dst_idx])
    return np.array(common @ weights.reshape(-1, 1)).ravel()


def compute_preferential_attachment(adj: sparse.csr_matrix, src_idx: np.ndarray, dst_idx: np.ndarray) -> np.ndarray:
    """Compute Preferential Attachment scores for given pairs."""
    degrees = np.array(adj.sum(axis=1)).ravel()
    return degrees[src_idx] * degrees[dst_idx]


def _evaluate_all_heuristics_ranking(
    adj: sparse.csr_matrix,
    node_to_idx: Dict[int, int],
    nodes: np.ndarray,
    target_edges: Set[Tuple[int, int]],
    historical_neighbors: Dict[int, Set[int]],
    active_nodes: np.ndarray,
    n_negatives: int,
    seed: int,
) -> Dict[str, np.ndarray]:
    """Evaluate all heuristics in a single pass using per-source ranking (TGB-style).

    Computes CN, Jaccard, Adamic-Adar, and PA for each query in one pass,
    avoiding redundant candidate generation and adjacency lookups.

    Args:
        adj: Binary undirected adjacency matrix (local indices).
        node_to_idx: Mapping from global node ID to local index.
        nodes: Array of global node IDs.
        target_edges: Set of (src, dst) edges in the target day.
        historical_neighbors: Per-source historical neighbor sets.
        active_nodes: Array of all active node IDs.
        n_negatives: Number of negative candidates per query.
        seed: Random seed.

    Returns:
        Dict mapping heuristic name to array of 1-based ranks.
    """
    rng = np.random.RandomState(seed)
    valid_nodes = set(node_to_idx.keys())

    edges_by_source: Dict[int, List[int]] = {}
    for s, d in target_edges:
        if s in valid_nodes and d in valid_nodes:
            edges_by_source.setdefault(s, []).append(d)

    degrees = np.array(adj.sum(axis=1)).ravel()
    aa_weights = np.zeros(adj.shape[0], dtype=np.float64)
    aa_mask = degrees > 1
    aa_weights[aa_mask] = 1.0 / np.log(degrees[aa_mask])

    n_active = len(active_nodes)
    active_set = valid_nodes

    heuristic_names = ["common_neighbors", "jaccard", "adamic_adar", "pref_attachment"]
    all_ranks = {h: [] for h in heuristic_names}

    for source, true_dsts in edges_by_source.items():
        target_set = set(true_dsts)
        hist = historical_neighbors.get(source, set())
        hist_candidates = list(hist - target_set - {source})
        src_local_idx = node_to_idx[source]

        for d_true in true_dsts:
            n_hist = min(n_negatives // 2, len(hist_candidates))
            n_rand = n_negatives - n_hist

            negatives = []
            if n_hist > 0:
                negatives.extend(rng.choice(hist_candidates, size=n_hist, replace=False).tolist())

            if n_rand > 0 and n_active > 0:
                exclude = target_set | set(negatives) | {source, d_true}
                batch = rng.randint(0, n_active, size=n_rand * 3)
                candidates = active_nodes[batch]
                seen = exclude.copy()
                for node in candidates:
                    node_int = int(node)
                    if node_int in active_set and node_int not in seen:
                        negatives.append(node_int)
                        seen.add(node_int)
                        if len(negatives) >= n_hist + n_rand:
                            break

            candidates_global = [d_true] + [c for c in negatives if c in valid_nodes]
            if len(candidates_global) < 2:
                continue

            src_local = np.full(len(candidates_global), src_local_idx, dtype=np.int32)
            dst_local = np.array([node_to_idx[c] for c in candidates_global], dtype=np.int32)

            src_rows = adj[src_local]
            dst_rows = adj[dst_local]
            common = src_rows.multiply(dst_rows)

            cn_scores = np.array(common.sum(axis=1)).ravel()

            s_deg = degrees[src_local]
            d_deg = degrees[dst_local]
            union = s_deg + d_deg - cn_scores
            jaccard_scores = np.zeros_like(cn_scores, dtype=np.float64)
            jmask = union > 0
            jaccard_scores[jmask] = cn_scores[jmask] / union[jmask]

            aa_scores = np.array(common @ aa_weights.reshape(-1, 1)).ravel()

            pa_scores = s_deg * d_deg

            scores_map = {
                "common_neighbors": cn_scores,
                "jaccard": jaccard_scores,
                "adamic_adar": aa_scores,
                "pref_attachment": pa_scores,
            }

            for hname, scores in scores_map.items():
                true_score = scores[0]
                rank = int(np.sum(scores[1:] > true_score)) + 1
                all_ranks[hname].append(rank)

    return {h: np.array(r) for h, r in all_ranks.items()}


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
    """Evaluate a single heuristic using per-source ranking protocol (TGB-style).

    Kept for backward compatibility with tests.

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

    edges_by_source: Dict[int, List[int]] = {}
    for s, d in target_edges:
        if s in valid_nodes and d in valid_nodes:
            edges_by_source.setdefault(s, []).append(d)

    all_ranks = []
    n_active = len(active_nodes)

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

            exclude = target_set | set(negatives) | {source, d_true}
            if n_rand > 0 and n_active > 0:
                batch = rng.randint(0, n_active, size=n_rand * 3)
                candidates = active_nodes[batch]
                seen = exclude.copy()
                for node in candidates:
                    node_int = int(node)
                    if node_int in valid_nodes and node_int not in seen:
                        negatives.append(node_int)
                        seen.add(node_int)
                        if len(negatives) >= n_hist + n_rand:
                            break

            candidates_in_adj = [d_true] + [c for c in negatives if c in valid_nodes]
            if len(candidates_in_adj) < 2:
                continue

            src_local = np.array([node_to_idx[source]] * len(candidates_in_adj))
            dst_local = np.array([node_to_idx[c] for c in candidates_in_adj])

            scores = heuristic_fn(adj, src_local, dst_local)

            true_score = scores[0]
            rank = int(np.sum(scores[1:] > true_score)) + 1
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

    t_total = time.time()
    n_pred = len(prediction_dates)

    for day_i, target_date in enumerate(prediction_dates):
        t_day = time.time()
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

        ranks_by_heuristic = _evaluate_all_heuristics_ranking(
            adj, node_to_idx, nodes, target_edges,
            historical_neighbors, active_nodes,
            config.n_negatives, seed,
        )

        for hname in heuristic_names:
            ranks = ranks_by_heuristic.get(hname, np.array([]))
            if len(ranks) == 0:
                continue
            metrics = compute_ranking_metrics(ranks)
            metrics["date"] = target_date
            metrics["heuristic"] = hname
            exp_logger.log_metrics(metrics)
            all_metrics[hname].append(metrics)

        del adj, window_snapshots
        gc.collect()

        elapsed_day = time.time() - t_day
        elapsed_total = time.time() - t_total
        avg_per_day = elapsed_total / (day_i + 1)
        eta = avg_per_day * (n_pred - day_i - 1)

        logger.info(
            "  [%d/%d] %s (%.1fs): %s | ETA: %.0fm",
            day_i + 1, n_pred, target_date, elapsed_day,
            {h: f"MRR={m[-1].get('mrr', 0):.4f}" for h, m in all_metrics.items() if m},
            eta / 60,
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

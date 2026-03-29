"""Heuristic baselines for temporal link prediction on stream graph.

Heuristics: CN (Common Neighbors), Jaccard, AA (Adamic-Adar), PA (Preferential Attachment).
All computed from TRAIN adjacency only — no leakage from val/test.

Evaluation: TGB-style per-source ranking on val and test splits.
"""

import time

import numpy as np
from scipy import sparse

from scripts.compute_stream_adjacency import compute_cn, compute_aa
from sg_baselines.sampling import sample_negatives_for_eval
from src.baselines.evaluation import compute_ranking_metrics


def compute_jaccard(
    adj: sparse.csr_matrix,
    src_idx: np.ndarray,
    dst_idx: np.ndarray,
) -> np.ndarray:
    """Compute Jaccard coefficient for node pairs.

    Jaccard(u,v) = |N(u) ∩ N(v)| / |N(u) ∪ N(v)|.
    """
    cn = np.array(adj[src_idx].multiply(adj[dst_idx]).sum(axis=1), dtype=np.float64).ravel()
    deg_src = np.array(adj[src_idx].sum(axis=1), dtype=np.float64).ravel()
    deg_dst = np.array(adj[dst_idx].sum(axis=1), dtype=np.float64).ravel()
    union = deg_src + deg_dst - cn
    result = np.zeros_like(cn)
    mask = union > 0
    result[mask] = cn[mask] / union[mask]
    return result.astype(np.float32)


def compute_pa(
    adj: sparse.csr_matrix,
    src_idx: np.ndarray,
    dst_idx: np.ndarray,
) -> np.ndarray:
    """Compute Preferential Attachment score for node pairs.

    PA(u,v) = deg(u) * deg(v).
    """
    deg_src = np.array(adj[src_idx].sum(axis=1), dtype=np.float64).ravel()
    deg_dst = np.array(adj[dst_idx].sum(axis=1), dtype=np.float64).ravel()
    return (deg_src * deg_dst).astype(np.float32)


def _score_candidates(
    heuristic: str,
    adj: sparse.csr_matrix,
    src_local: np.ndarray,
    dst_local: np.ndarray,
) -> np.ndarray:
    """Score (src, dst) pairs with a given heuristic."""
    if heuristic == "cn":
        return compute_cn(adj, src_local, dst_local)
    elif heuristic == "aa":
        return compute_aa(adj, src_local, dst_local)
    elif heuristic == "jaccard":
        return compute_jaccard(adj, src_local, dst_local)
    elif heuristic == "pa":
        return compute_pa(adj, src_local, dst_local)
    else:
        raise ValueError(f"Unknown heuristic: {heuristic}")


def evaluate_heuristics(
    eval_edges: "pd.DataFrame",
    train_neighbors: dict[int, set[int]],
    active_nodes: np.ndarray,
    node_mapping: np.ndarray,
    adj_undirected: sparse.csr_matrix,
    heuristics: list[str],
    n_negatives: int,
    seed: int,
    split_name: str,
) -> dict[str, dict]:
    """Evaluate heuristic baselines on a split (val or test).

    Args:
        eval_edges: DataFrame with src_idx, dst_idx columns.
        train_neighbors: Per-source neighbor sets from train.
        active_nodes: Sorted array of active nodes from train.
        node_mapping: Local->global mapping for adjacency.
        adj_undirected: Undirected adjacency matrix (local indices).
        heuristics: List of heuristic names to evaluate.
        n_negatives: Number of negative candidates per query.
        seed: Random seed for negative sampling.
        split_name: "val" or "test" (for logging).

    Returns:
        Dict mapping heuristic name -> ranking metrics dict.
    """
    import pandas as pd

    src_all = eval_edges["src_idx"].values
    dst_all = eval_edges["dst_idx"].values

    unique_edges = pd.DataFrame({"src": src_all, "dst": dst_all}).drop_duplicates()
    src_unique = unique_edges["src"].values
    dst_unique = unique_edges["dst"].values
    n_queries = len(src_unique)
    print(f"  [{split_name}] {n_queries:,} unique positive edges for ranking")

    positives_per_src: dict[int, set[int]] = {}
    for s, d in zip(src_unique, dst_unique):
        positives_per_src.setdefault(int(s), set()).add(int(d))

    rng = np.random.RandomState(seed)

    results = {}
    for heuristic in heuristics:
        t0 = time.time()
        ranks = []

        for i in range(n_queries):
            s = int(src_unique[i])
            d_true = int(dst_unique[i])

            negatives = sample_negatives_for_eval(
                s, d_true, train_neighbors, positives_per_src.get(s, set()),
                active_nodes, n_negatives, rng,
            )

            candidates = np.concatenate([[d_true], negatives])
            src_rep = np.full(len(candidates), s, dtype=np.int64)

            src_local, dst_local, valid_mask = _map_to_local(
                src_rep, candidates, node_mapping
            )

            scores = np.zeros(len(candidates), dtype=np.float32)
            if valid_mask.any():
                scores[valid_mask] = _score_candidates(
                    heuristic, adj_undirected, src_local[valid_mask], dst_local[valid_mask]
                )

            rank = _compute_rank(scores)
            ranks.append(rank)

        ranks_arr = np.array(ranks, dtype=np.float64)
        metrics = compute_ranking_metrics(ranks_arr)
        elapsed = time.time() - t0
        print(f"    {heuristic}: MRR={metrics['mrr']:.4f}, "
              f"Hits@1={metrics['hits@1']:.4f}, "
              f"Hits@10={metrics['hits@10']:.4f} ({elapsed:.1f}s)")
        results[heuristic] = metrics

    return results


def _map_to_local(
    src_global: np.ndarray,
    dst_global: np.ndarray,
    node_mapping: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Map global indices to local indices. Returns (src_local, dst_local, valid_mask)."""
    src_pos = np.searchsorted(node_mapping, src_global)
    dst_pos = np.searchsorted(node_mapping, dst_global)

    n = len(node_mapping)
    src_valid = (src_pos < n) & (node_mapping[np.minimum(src_pos, n - 1)] == src_global)
    dst_valid = (dst_pos < n) & (node_mapping[np.minimum(dst_pos, n - 1)] == dst_global)
    valid = src_valid & dst_valid

    src_local = np.where(valid, src_pos, 0)
    dst_local = np.where(valid, dst_pos, 0)

    return src_local, dst_local, valid


def _compute_rank(scores: np.ndarray) -> int:
    """Compute 1-based rank of the first element (true destination).

    Higher scores are better. Ties broken conservatively (worst rank).
    """
    true_score = scores[0]
    rank = int(np.sum(scores > true_score)) + 1
    return rank

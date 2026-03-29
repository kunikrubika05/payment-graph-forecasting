"""Heuristic baselines for temporal link prediction on stream graph.

Heuristics: CN (Common Neighbors), Jaccard, AA (Adamic-Adar), PA (Preferential Attachment).
All computed from TRAIN adjacency only — no leakage from val/test.

Evaluation: TGB-style per-source ranking on val and test splits.
Batched implementation: all candidates scored in one sparse operation.
"""

import time

import numpy as np
import pandas as pd
from scipy import sparse
from tqdm import tqdm

from scripts.compute_stream_adjacency import compute_cn, compute_aa
from sg_baselines.sampling import sample_negatives_for_eval
from src.baselines.evaluation import compute_ranking_metrics


def compute_jaccard(
    adj: sparse.csr_matrix,
    src_idx: np.ndarray,
    dst_idx: np.ndarray,
) -> np.ndarray:
    """Compute Jaccard coefficient for node pairs."""
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
    """Compute Preferential Attachment score for node pairs."""
    deg_src = np.array(adj[src_idx].sum(axis=1), dtype=np.float64).ravel()
    deg_dst = np.array(adj[dst_idx].sum(axis=1), dtype=np.float64).ravel()
    return (deg_src * deg_dst).astype(np.float32)


def _score_batch(
    heuristic: str,
    adj: sparse.csr_matrix,
    src_local: np.ndarray,
    dst_local: np.ndarray,
) -> np.ndarray:
    """Score (src, dst) pairs with a given heuristic. Batched."""
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
    eval_edges: pd.DataFrame,
    train_neighbors: dict[int, set[int]],
    active_nodes: np.ndarray,
    node_mapping: np.ndarray,
    adj_undirected: sparse.csr_matrix,
    heuristics: list[str],
    n_negatives: int,
    seed: int,
    split_name: str,
    max_queries: int = 50_000,
) -> dict[str, dict]:
    """Evaluate heuristic baselines on a split (val or test).

    Uses batched scoring: collects all (src, candidate) pairs first,
    scores in one scipy call, then extracts ranks.

    Args:
        max_queries: Cap on number of evaluated edges. 50K gives stable MRR
            estimates (std < 0.002 for MRR ~0.5).
    """
    src_all = eval_edges["src_idx"].values
    dst_all = eval_edges["dst_idx"].values

    unique_edges = pd.DataFrame({"src": src_all, "dst": dst_all}).drop_duplicates()
    src_unique = unique_edges["src"].values.astype(np.int64)
    dst_unique = unique_edges["dst"].values.astype(np.int64)
    n_total = len(src_unique)

    if n_total > max_queries:
        rng_sub = np.random.RandomState(seed + 777)
        idx = rng_sub.choice(n_total, size=max_queries, replace=False)
        idx.sort()
        src_unique = src_unique[idx]
        dst_unique = dst_unique[idx]
        print(f"  [{split_name}] Subsampled {n_total:,} -> {max_queries:,} queries",
              flush=True)

    n_queries = len(src_unique)
    print(f"  [{split_name}] {n_queries:,} queries for ranking", flush=True)

    positives_per_src: dict[int, set[int]] = {}
    for s, d in zip(src_unique, dst_unique):
        positives_per_src.setdefault(int(s), set()).add(int(d))

    print(f"  [{split_name}] Sampling negatives...", flush=True)
    t0 = time.time()
    rng = np.random.RandomState(seed)

    all_src_flat = []
    all_dst_flat = []
    query_offsets = [0]

    for i in tqdm(range(n_queries), desc=f"  neg_sample_{split_name}", miniinterval=5.0):
        s = int(src_unique[i])
        d_true = int(dst_unique[i])
        negatives = sample_negatives_for_eval(
            s, d_true, train_neighbors, positives_per_src.get(s, set()),
            active_nodes, n_negatives, rng,
        )
        candidates = np.concatenate([[d_true], negatives])
        n_cand = len(candidates)
        all_src_flat.extend([s] * n_cand)
        all_dst_flat.extend(candidates.tolist())
        query_offsets.append(query_offsets[-1] + n_cand)

    all_src_arr = np.array(all_src_flat, dtype=np.int64)
    all_dst_arr = np.array(all_dst_flat, dtype=np.int64)
    total_pairs = len(all_src_arr)
    print(f"  [{split_name}] Sampled {total_pairs:,} total candidates "
          f"({time.time() - t0:.1f}s)", flush=True)

    src_pos = np.searchsorted(node_mapping, all_src_arr)
    dst_pos = np.searchsorted(node_mapping, all_dst_arr)
    n_map = len(node_mapping)
    src_valid = (src_pos < n_map) & (node_mapping[np.minimum(src_pos, n_map - 1)] == all_src_arr)
    dst_valid = (dst_pos < n_map) & (node_mapping[np.minimum(dst_pos, n_map - 1)] == all_dst_arr)
    valid_mask = src_valid & dst_valid
    src_local = np.where(valid_mask, src_pos, 0)
    dst_local = np.where(valid_mask, dst_pos, 0)

    results = {}
    for heuristic in heuristics:
        t0 = time.time()
        print(f"  [{split_name}] Scoring {heuristic} ({total_pairs:,} pairs)...",
              flush=True)

        scores = np.zeros(total_pairs, dtype=np.float32)
        if valid_mask.any():
            valid_scores = _score_batch(
                heuristic, adj_undirected,
                src_local[valid_mask], dst_local[valid_mask],
            )
            scores[valid_mask] = valid_scores

        ranks = np.empty(n_queries, dtype=np.float64)
        for q in range(n_queries):
            start = query_offsets[q]
            end = query_offsets[q + 1]
            q_scores = scores[start:end]
            true_score = q_scores[0]
            ranks[q] = float(np.sum(q_scores > true_score) + 1)

        metrics = compute_ranking_metrics(ranks)
        elapsed = time.time() - t0
        print(f"    {heuristic}: MRR={metrics['mrr']:.4f}, "
              f"Hits@1={metrics['hits@1']:.4f}, "
              f"Hits@10={metrics['hits@10']:.4f} ({elapsed:.1f}s)", flush=True)
        results[heuristic] = metrics

    return results


def _compute_rank(scores: np.ndarray) -> int:
    """Compute 1-based rank of the first element (true destination)."""
    true_score = scores[0]
    return int(np.sum(scores > true_score)) + 1


def _map_to_local(
    src_global: np.ndarray,
    dst_global: np.ndarray,
    node_mapping: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Map global indices to local indices."""
    src_pos = np.searchsorted(node_mapping, src_global)
    dst_pos = np.searchsorted(node_mapping, dst_global)
    n = len(node_mapping)
    src_valid = (src_pos < n) & (node_mapping[np.minimum(src_pos, n - 1)] == src_global)
    dst_valid = (dst_pos < n) & (node_mapping[np.minimum(dst_pos, n - 1)] == dst_global)
    valid = src_valid & dst_valid
    src_local = np.where(valid, src_pos, 0)
    dst_local = np.where(valid, dst_pos, 0)
    return src_local, dst_local, valid

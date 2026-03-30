"""TGB-style evaluation for HyperEvent temporal link prediction.

Matches the sg_baselines CORRECTNESS_CHECKLIST protocol:
    - Eval queries filtered: only edges where BOTH src and dst are in train
      node mapping (active_nodes). New-node edges are skipped to avoid
      rank=1 bias (all candidates score 0 for nodes with no train history).
    - 50 historical + 50 random negatives per positive edge.
    - Historical negatives exclude ALL positives of src from the full split
      (train+val+test), not just the current true_dst.
    - Random negatives sampled from active_nodes (train nodes only).
    - Conservative ties: rank = 1 + count(neg_scores > true_score).
    - Metrics: MRR, Hits@1, Hits@3, Hits@10.
    - Up to max_eval_edges queries evaluated (randomly subsampled if more).

Streaming adjacency: the table is updated after EVERY test edge (including
filtered-out and non-sampled ones) to preserve temporal correctness.
"""

import contextlib
import logging
import time
from typing import Dict, Optional

import numpy as np
import torch
from tqdm import tqdm

from src.models.HyperEvent.hyperevent import HyperEventModel
from src.models.HyperEvent.data_utils import TemporalEdgeData
from src.models.HyperEvent.hyperevent_train import (
    AdjacencyTable,
    compute_batch_relational_vectors,
    build_adj_from_mask,
)
from src.baselines.evaluation import compute_ranking_metrics

logger = logging.getLogger(__name__)


def _amp_autocast(enabled: bool, device_type: str):
    if enabled and device_type == "cuda":
        return torch.amp.autocast("cuda")
    return contextlib.nullcontext()


def _sample_hist_negatives(
    adj: AdjacencyTable,
    src: int,
    all_positives_src: set,
    n_hist: int,
    rng: np.random.Generator,
    active_nodes: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Sample historical negatives from src's adjacency table.

    Returns nodes that src has previously interacted with, excluding ALL known
    positives of src from the full split (train+val+test). If fewer than n_hist
    candidates exist, the remainder is padded with random nodes from active_nodes.

    Args:
        adj: Current adjacency table.
        src: Source node.
        all_positives_src: All destinations src connects to across all splits.
        n_hist: Number of historical negatives requested.
        rng: Random number generator.
        active_nodes: Sorted int32 array of train node indices for random padding.
            If None, draws from all nodes (range [0, adj.num_nodes)).

    Returns:
        1-D int32 array of length n_hist.
    """
    nb = adj.get_neighbors(src)
    if len(nb) > 0:
        candidates = nb[~np.isin(nb, list(all_positives_src))]
    else:
        candidates = np.empty(0, dtype=np.int32)
    candidates = np.unique(candidates)

    if len(candidates) >= n_hist:
        idx = rng.choice(len(candidates), size=n_hist, replace=False)
        return candidates[idx].astype(np.int32)

    result = candidates.tolist()
    needed = n_hist - len(result)

    if active_nodes is not None:
        rand_idx = rng.integers(0, len(active_nodes), size=needed * 3)
        randoms = active_nodes[rand_idx].astype(np.int32)
    else:
        randoms = rng.integers(0, adj.num_nodes, size=needed * 3).astype(np.int32)
    randoms = randoms[~np.isin(randoms, list(all_positives_src))]
    randoms = np.unique(randoms)[:needed]
    result.extend(randoms.tolist())

    while len(result) < n_hist:
        if active_nodes is not None:
            r = int(active_nodes[rng.integers(0, len(active_nodes))])
        else:
            r = int(rng.integers(0, adj.num_nodes))
        if r not in all_positives_src:
            result.append(r)

    return np.array(result[:n_hist], dtype=np.int32)


@torch.no_grad()
def evaluate_tgb_style(
    model: HyperEventModel,
    data: TemporalEdgeData,
    eval_mask: np.ndarray,
    history_mask: np.ndarray,
    device: torch.device,
    n_neighbor: int = 20,
    n_latest: int = 10,
    n_hist_neg: int = 50,
    n_random_neg: int = 50,
    use_amp: bool = True,
    seed: int = 42,
    active_nodes: Optional[np.ndarray] = None,
    all_positives_per_src: Optional[dict] = None,
    max_eval_edges: Optional[int] = 50_000,
) -> Dict[str, float]:
    """Full TGB-style evaluation matching the sg_baselines CORRECTNESS_CHECKLIST.

    For each positive edge (src, dst, t) in the evaluation set:
        1. Skip if src or dst not in active_nodes (no train history → rank=1 bias).
        2. Sample n_hist_neg historical negatives from adj[src], excluding all
           known positives of src across the entire dataset.
        3. Sample n_random_neg random negatives from active_nodes (train pool).
        4. Score all 1 + n_hist_neg + n_random_neg candidates.
        5. Conservative rank: 1 + count(neg_scores > true_score).

    The adjacency table is initialised from history_mask and updated after
    every test edge — including filtered and non-sampled ones — to maintain
    the correct temporal state (streaming evaluation).

    Args:
        model: Trained HyperEventModel.
        data: TemporalEdgeData.
        eval_mask: Boolean mask selecting evaluation edges.
        history_mask: Boolean mask for all edges preceding the evaluation set
            (train | val) used to warm-start the adjacency table.
        device: Torch device.
        n_neighbor: Adjacency table capacity per node.
        n_latest: Context events per query node.
        n_hist_neg: Historical negatives per query (default 50).
        n_random_neg: Random negatives per query (default 50).
        use_amp: Enable mixed precision on CUDA.
        seed: Random seed (use 42 + 400 for test, 42 + 300 for val final eval).
        active_nodes: Sorted int32 array of train node indices. Random negatives
            are drawn from this pool and new-node queries are filtered out.
            If None, falls back to all nodes (not recommended).
        all_positives_per_src: Dict mapping src → set of all dst in the full
            dataset (train+val+test). Used to exclude known positives from
            historical negatives. If None, only true_dst is excluded.
        max_eval_edges: Maximum number of queries to score. If the filtered eval
            set is larger, max_eval_edges queries are randomly selected; the
            adjacency table is still updated for all edges in order.
            None means evaluate all queries.

    Returns:
        Dict with mrr, hits@1, hits@3, hits@10, eval_time_sec, edges_per_sec,
        n_queries (scored), and n_filtered (new-node edges skipped).
    """
    model.eval()
    rng = np.random.default_rng(seed)
    amp_enabled = use_amp and device.type == "cuda"

    adj = build_adj_from_mask(data, history_mask, n_neighbor)
    all_eval_indices = np.where(eval_mask)[0]

    # Filter: skip edges where src or dst is not in train node mapping.
    # (For such edges every candidate scores 0 → rank=1 → biased MRR.)
    if active_nodes is not None:
        src_ok = np.isin(data.src[all_eval_indices], active_nodes)
        dst_ok = np.isin(data.dst[all_eval_indices], active_nodes)
        filtered_indices = all_eval_indices[src_ok & dst_ok]
    else:
        filtered_indices = all_eval_indices
    n_filtered = len(all_eval_indices) - len(filtered_indices)
    if n_filtered > 0:
        logger.info("Filtered %d new-node eval edges (not in train)", n_filtered)

    # Subsample metrics queries; adj still updated for ALL edges (streaming).
    if max_eval_edges is not None and len(filtered_indices) > max_eval_edges:
        chosen = rng.choice(len(filtered_indices), size=max_eval_edges, replace=False)
        sampled_set = set(filtered_indices[chosen].tolist())
        logger.info(
            "Subsampled %d/%d queries for metrics", max_eval_edges, len(filtered_indices)
        )
    else:
        sampled_set = set(filtered_indices.tolist())

    n_total = len(sampled_set)
    logger.info(
        "HyperEvent TGB-style eval: %d metric queries, %d negatives each "
        "(%d total test edges)",
        n_total, n_hist_neg + n_random_neg, len(all_eval_indices),
    )

    all_ranks: list = []
    start_time = time.time()

    for idx in tqdm(all_eval_indices, desc="Evaluating", total=len(all_eval_indices)):
        src_node = int(data.src[idx])
        true_dst = int(data.dst[idx])

        if idx in sampled_set:
            positives_src = (
                all_positives_per_src.get(src_node, set())
                if all_positives_per_src is not None
                else {true_dst}
            )

            hist_neg = _sample_hist_negatives(
                adj, src_node, positives_src, n_hist_neg, rng, active_nodes
            )

            if active_nodes is not None:
                rand_idx = rng.integers(0, len(active_nodes), size=n_random_neg * 2)
                rand_neg = active_nodes[rand_idx].astype(np.int32)
            else:
                rand_neg = rng.integers(
                    0, data.num_nodes, size=n_random_neg * 2
                ).astype(np.int32)
            rand_neg = rand_neg[rand_neg != true_dst][:n_random_neg]
            if len(rand_neg) < n_random_neg:
                extra = n_random_neg - len(rand_neg)
                if active_nodes is not None:
                    extra_idx = rng.integers(0, len(active_nodes), size=extra)
                    rand_neg = np.concatenate(
                        [rand_neg, active_nodes[extra_idx].astype(np.int32)]
                    )
                else:
                    rand_neg = np.concatenate([
                        rand_neg,
                        rng.integers(0, data.num_nodes, size=extra).astype(np.int32),
                    ])

            all_dst = np.concatenate([[true_dst], hist_neg, rand_neg]).astype(np.int32)
            C = len(all_dst)
            u_stars = np.full(C, src_node, dtype=np.int32)
            vecs, masks = compute_batch_relational_vectors(adj, u_stars, all_dst, n_latest)

            def _t(arr, dtype=torch.float32):
                return torch.tensor(arr, dtype=dtype, device=device)

            with _amp_autocast(amp_enabled, device.type):
                scores = model(_t(vecs), _t(masks, torch.bool)).cpu().float().numpy()

            true_score = scores[0]
            # Conservative ties: rank = 1 + count(neg_scores > true_score)
            rank = 1.0 + float((scores[1:] > true_score).sum())
            all_ranks.append(rank)

        # Always update adj for temporal correctness (streaming protocol).
        adj.add_edge(src_node, true_dst)
        adj.add_edge(true_dst, src_node)

    elapsed = time.time() - start_time
    ranks_arr = np.array(all_ranks, dtype=np.float64)
    metrics = compute_ranking_metrics(ranks_arr)
    metrics["eval_time_sec"] = elapsed
    metrics["edges_per_sec"] = n_total / max(elapsed, 1e-9)
    metrics["n_queries"] = n_total
    metrics["n_filtered"] = n_filtered

    logger.info(
        "HyperEvent eval: MRR=%.4f Hits@1=%.3f Hits@3=%.3f Hits@10=%.3f "
        "(%d queries, %d filtered, %.1fs)",
        metrics["mrr"], metrics["hits@1"], metrics["hits@3"],
        metrics["hits@10"], n_total, n_filtered, elapsed,
    )
    return metrics

"""TGB-style evaluation for HyperEvent temporal link prediction.

Matches the evaluation protocol used by EAGLE and GLFormer:
    - 50 historical + 50 random negatives per positive edge
    - Per-source ranking: rank true destination among all 101 candidates
    - Metrics: MRR, Hits@1, Hits@3, Hits@10

Historical negatives are sampled from the adjacency table of the source node
(nodes it has interacted with previously, excluding the true destination).
Random negatives are uniformly sampled from all nodes.

The adjacency table is updated after scoring each test edge so that
subsequent edges see the correct structural state (streaming evaluation).
"""

import contextlib
import logging
import time
from typing import Dict

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
        return torch.cuda.amp.autocast()
    return contextlib.nullcontext()


def _sample_hist_negatives(
    adj: AdjacencyTable,
    src: int,
    true_dst: int,
    n_hist: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample historical negatives from src's adjacency table.

    Returns nodes that src has previously interacted with, excluding true_dst.
    If fewer than n_hist candidates exist, the remainder is filled with random nodes.

    Args:
        adj: Current adjacency table.
        src: Source node.
        true_dst: True destination (excluded from negatives).
        n_hist: Number of historical negatives requested.
        rng: Random number generator.

    Returns:
        1-D int32 array of length n_hist.
    """
    nb = adj.get_neighbors(src)
    candidates = nb[nb != true_dst] if len(nb) > 0 else np.empty(0, dtype=np.int32)
    candidates = np.unique(candidates)

    if len(candidates) >= n_hist:
        idx = rng.choice(len(candidates), size=n_hist, replace=False)
        return candidates[idx].astype(np.int32)

    # Pad with random nodes
    result = candidates.tolist()
    needed = n_hist - len(result)
    randoms = rng.integers(0, adj.num_nodes, size=needed * 3).astype(np.int32)
    randoms = randoms[(randoms != true_dst)]
    randoms = np.unique(randoms)[:needed]
    result.extend(randoms.tolist())

    while len(result) < n_hist:
        r = int(rng.integers(0, adj.num_nodes))
        if r != true_dst:
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
) -> Dict[str, float]:
    """Full TGB-style evaluation matching the EAGLE/GLFormer baseline protocol.

    For each positive edge (src, dst, t):
        1. Sample n_hist_neg historical negatives from adj[src] (exc. true dst).
        2. Sample n_random_neg random negatives uniformly from all nodes.
        3. Score all 1 + n_hist_neg + n_random_neg candidates.
        4. Compute fractional rank of the true destination.

    The adjacency table is initialised from history_mask edges and updated
    chronologically as each test edge is processed (streaming protocol).

    Args:
        model: Trained HyperEventModel.
        data: TemporalEdgeData.
        eval_mask: Boolean mask selecting evaluation edges.
        history_mask: Boolean mask for all edges preceding the evaluation set
            (e.g. train | val masks) used to warm-start the adjacency table.
        device: Torch device.
        n_neighbor: Adjacency table capacity per node.
        n_latest: Context events per query node.
        n_hist_neg: Number of historical negatives per query.
        n_random_neg: Number of random negatives per query.
        use_amp: Enable mixed precision.
        seed: Random seed for negative sampling.

    Returns:
        Dict with mrr, hits@1, hits@3, hits@10, eval_time_sec,
        edges_per_sec, and n_queries.
    """
    model.eval()
    rng = np.random.default_rng(seed)
    amp_enabled = use_amp and device.type == "cuda"

    adj = build_adj_from_mask(data, history_mask, n_neighbor)

    eval_indices = np.where(eval_mask)[0]
    n_total = len(eval_indices)
    logger.info(
        "HyperEvent TGB-style eval: %d edges, %d negatives each",
        n_total, n_hist_neg + n_random_neg,
    )

    all_ranks = []
    start_time = time.time()

    for idx in tqdm(eval_indices, desc="Evaluating"):
        src_node = int(data.src[idx])
        true_dst = int(data.dst[idx])

        # Build candidate set: true_dst + hist_neg + random_neg
        hist_neg = _sample_hist_negatives(adj, src_node, true_dst, n_hist_neg, rng)
        rand_neg = rng.integers(0, data.num_nodes, size=n_random_neg * 2).astype(np.int32)
        rand_neg = rand_neg[rand_neg != true_dst][:n_random_neg]
        if len(rand_neg) < n_random_neg:
            extra = n_random_neg - len(rand_neg)
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
        rank = (
            1.0
            + float((scores[1:] > true_score).sum())
            + 0.5 * float((scores[1:] == true_score).sum())
        )
        all_ranks.append(rank)

        # Update adjacency table (streaming: true edge is now observed)
        adj.add_edge(src_node, true_dst)
        adj.add_edge(true_dst, src_node)

    elapsed = time.time() - start_time
    ranks_arr = np.array(all_ranks, dtype=np.float64)
    metrics = compute_ranking_metrics(ranks_arr)
    metrics["eval_time_sec"] = elapsed
    metrics["edges_per_sec"] = n_total / elapsed if elapsed > 0 else 0.0
    metrics["n_queries"] = n_total

    logger.info(
        "HyperEvent eval: MRR=%.4f Hits@1=%.3f Hits@3=%.3f Hits@10=%.3f (%.1fs)",
        metrics["mrr"], metrics["hits@1"], metrics["hits@3"],
        metrics["hits@10"], elapsed,
    )
    return metrics

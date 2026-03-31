"""TGB-style evaluation for GLFormer temporal link prediction.

Matches the baseline protocol from sg_baselines/sampling.py exactly:
    - 50 historical + 50 random negatives per positive edge
    - Historical: train neighbors of src, excluding ALL eval positives of src
    - Random: from active train nodes only (not full 312M node space)
    - Per-source ranking: rank true destination among 101 candidates
    - Rank = 1 + count(score > true_score), conservative (no fractional ties)
    - Metrics: MRR, Hits@1, Hits@3, Hits@10
    - 50K query subsample
    - Eval seeds: val = seed+200, test = seed+400

Evaluation runs on DIRECTED edges only (not undirected duplicates).
"""

import logging
import time
from collections import defaultdict
from typing import Dict, Optional

import numpy as np
import torch
from tqdm import tqdm

from payment_graph_forecasting.evaluation.ranking_loop import (
    choose_query_indices,
    evaluate_ranking_loop,
)
from payment_graph_forecasting.training.amp import autocast_context
from payment_graph_forecasting.training.amp import amp_enabled_for_device
from payment_graph_forecasting.evaluation.temporal_ranking import (
    conservative_rank_from_scores,
    score_candidate_contexts,
)
from payment_graph_forecasting.training.temporal_context import (
    sample_node_contexts,
    to_device_tensor,
)
from src.models.GLFormer.glformer import GLFormerTime
from src.models.GLFormer.data_utils import (
    TemporalEdgeData,
    TemporalCSR,
    sample_neighbors_batch,
)
from src.models.GLFormer.glformer_train import (
    _compute_cooccurrence,
    _compute_cn_from_adj,
)
from sg_baselines.sampling import sample_negatives_for_eval

logger = logging.getLogger(__name__)


def build_eval_positives_per_src(
    eval_src: np.ndarray,
    eval_dst: np.ndarray,
) -> dict[int, set[int]]:
    """Build per-source set of ALL positive destinations in eval split.

    Used to exclude eval positives from historical negatives, preventing
    target leakage in negative sampling.
    """
    positives: dict[int, set[int]] = defaultdict(set)
    for s, d in zip(eval_src, eval_dst):
        positives[int(s)].add(int(d))
    return dict(positives)


@torch.no_grad()
def evaluate_tgb_style(
    model: GLFormerTime,
    data: TemporalEdgeData,
    csr: TemporalCSR,
    eval_src: np.ndarray,
    eval_dst: np.ndarray,
    eval_ts: np.ndarray,
    train_neighbors: dict[int, set[int]],
    active_nodes: np.ndarray,
    device: torch.device,
    num_neighbors: int = 20,
    n_hist_neg: int = 50,
    n_random_neg: int = 50,
    use_amp: bool = True,
    seed: int = 42,
    max_edges: int = 50_000,
    adj=None,
    node_mapping=None,
) -> Dict[str, float]:
    """Full TGB-style evaluation matching the baseline protocol exactly.

    For each positive DIRECTED edge (src, dst, t):
        1. Generate negatives via sample_negatives_for_eval (sg_baselines).
        2. Score all 1 + n_negatives candidates using GLFormer.
        3. Compute conservative rank (no fractional ties).

    Args:
        model: Trained GLFormerTime model.
        data: Temporal edge data (for neighbor features).
        csr: Temporal CSR built from edges preceding the evaluation period
            (train-only for val eval, train+val for test eval).
        eval_src: Directed source node indices for eval split.
        eval_dst: Directed destination node indices for eval split.
        eval_ts: Timestamps for eval edges.
        train_neighbors: Per-source neighbor sets from DIRECTED train edges.
        active_nodes: Sorted array of active train node indices.
        device: Torch device.
        num_neighbors: K most-recent neighbors sampled per node.
        n_hist_neg: Number of historical negatives per query.
        n_random_neg: Number of random negatives per query.
        use_amp: Enable mixed precision.
        seed: Random seed for negative sampling.
        max_edges: Maximum queries to evaluate (subsampled if more).

    Returns:
        Dict with MRR, Hits@1, Hits@3, Hits@10, eval_time_sec,
        edges_per_sec, and n_queries.
    """
    model.eval()
    rng = np.random.RandomState(seed)
    amp_enabled = amp_enabled_for_device(use_amp, device)
    K = num_neighbors
    n_negatives = n_hist_neg + n_random_neg
    use_edge_feats = model.edge_feat_dim > 0
    use_node_feats = model.node_feat_dim > 0
    use_cooc = model.use_cooccurrence

    n_total_edges = len(eval_src)
    eval_positives_per_src = build_eval_positives_per_src(eval_src, eval_dst)

    chosen = choose_query_indices(n_total_edges, max_edges, rng=rng)

    n_queries = len(chosen)
    logger.info(
        "GLFormer TGB-style eval: %d/%d edges, %d negatives each (seed=%d)",
        n_queries, n_total_edges, n_negatives, seed,
    )

    def _score_rank(i: int) -> float:
        src_node = int(eval_src[i])
        true_dst = int(eval_dst[i])
        ts = float(eval_ts[i])

        src_positives = eval_positives_per_src.get(src_node, set())

        neg_nodes = sample_negatives_for_eval(
            src=src_node,
            dst_true=true_dst,
            train_neighbors=train_neighbors,
            eval_positives_of_src=src_positives,
            active_nodes=active_nodes,
            n_negatives=n_negatives,
            rng=rng,
        )
        all_dst = np.concatenate([[true_dst], neg_nodes]).astype(np.int32)
        C = len(all_dst)

        src_arr = np.array([src_node], dtype=np.int32)
        ts_arr = np.array([ts], dtype=np.float64)
        src_context = sample_node_contexts(
            csr=csr,
            data=data,
            sample_neighbors_fn=sample_neighbors_batch,
            nodes=src_arr,
            query_timestamps=ts_arr,
            num_neighbors=K,
            use_edge_feats=use_edge_feats,
            use_node_feats=use_node_feats,
            zero_pad_delta=True,
        )

        dst_ts_arr = np.full(C, ts, dtype=np.float64)
        dst_context = sample_node_contexts(
            csr=csr,
            data=data,
            sample_neighbors_fn=sample_neighbors_batch,
            nodes=all_dst,
            query_timestamps=dst_ts_arr,
            num_neighbors=K,
            use_edge_feats=use_edge_feats,
            use_node_feats=use_node_feats,
            zero_pad_delta=True,
        )

        cooc_counts = None
        if use_cooc:
            if adj is not None and node_mapping is not None:
                cooc_np = _compute_cn_from_adj(
                    adj, node_mapping,
                    np.full(C, src_node, dtype=np.int64),
                    all_dst.astype(np.int64),
                )
            else:
                src_nids_rep = np.repeat(src_context.neighbor_ids, C, axis=0)
                src_lens_rep = np.repeat(src_context.lengths, C)
                cooc_np = _compute_cooccurrence(
                    src_nids_rep, src_lens_rep, dst_context.neighbor_ids, dst_context.lengths
                )
            cooc_counts = torch.tensor(cooc_np, dtype=torch.float32, device=device)

        scores = score_candidate_contexts(
            model=model,
            device=device,
            src_context=src_context,
            dst_context=dst_context,
            amp_enabled=amp_enabled,
            cooc_counts=cooc_counts,
        )
        return conservative_rank_from_scores(scores)

    metrics, elapsed = evaluate_ranking_loop(
        chosen,
        score_rank_fn=lambda idx: _score_rank(idx),
    )

    logger.info(
        "GLFormer eval: MRR=%.4f Hits@1=%.3f Hits@3=%.3f Hits@10=%.3f "
        "(%d queries, %.1fs)",
        metrics["mrr"], metrics["hits@1"], metrics["hits@3"],
        metrics["hits@10"], n_queries, elapsed,
    )
    return metrics

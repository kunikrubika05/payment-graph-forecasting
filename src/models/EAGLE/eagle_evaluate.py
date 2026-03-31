"""TGB-style evaluation for EAGLE-Time temporal link prediction.

Matches sg_baselines protocol exactly (CORRECTNESS_CHECKLIST.md):
- Negatives from sample_negatives_for_eval (sg_baselines/sampling.py)
- Historical negatives exclude ALL positives of src in eval split
- Random negatives from active_nodes (train nodes only)
- Eval queries filtered: only edges with src AND dst_true in train nodes
- 50K query subsample
- Conservative rank: count(score > true_score) + 1
- Seeds: val=42+300, test=42+400
"""

import logging
import time
from typing import Dict, Optional, Set

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
from src.models.EAGLE.eagle import EAGLETime
from src.models.EAGLE.data_utils import (
    TemporalEdgeData,
    TemporalCSR,
    sample_neighbors_batch,
)
from sg_baselines.sampling import sample_negatives_for_eval

logger = logging.getLogger(__name__)


def _get_forward_mask(data: TemporalEdgeData, split_mask: np.ndarray) -> np.ndarray:
    """Get forward (original) edge indices from an undirected split.

    In undirected mode, each original edge appears twice (forward + reverse).
    We take only the first half of edges within each unique timestamp group
    to avoid double-counting.

    For directed data, returns all indices in the split.
    """
    indices = np.where(split_mask)[0]
    n = len(indices)
    if n == 0:
        return indices
    n_forward = n // 2
    return indices[:n_forward]


@torch.no_grad()
def evaluate_tgb_style(
    model: EAGLETime,
    data: TemporalEdgeData,
    csr: TemporalCSR,
    eval_mask: np.ndarray,
    device: torch.device,
    train_neighbors: Dict[int, Set[int]],
    active_nodes: np.ndarray,
    num_neighbors: int = 20,
    n_negatives: int = 100,
    use_amp: bool = True,
    seed: int = 42,
    max_queries: int = 50_000,
    is_undirected: bool = True,
) -> Dict[str, float]:
    """Full TGB-style evaluation matching sg_baselines protocol.

    For each positive edge (src, dst, t):
        1. Filter: skip if src or dst_true not in active_nodes (train nodes)
        2. Generate negatives via sg_baselines/sampling.py
        3. Score all 101 candidates
        4. Compute conservative rank of true destination

    Args:
        model: Trained EAGLE-Time model.
        data: Temporal edge data.
        csr: Temporal CSR (built from edges up to eval period).
        eval_mask: Boolean mask selecting evaluation edges.
        device: Torch device.
        train_neighbors: Per-source neighbor sets from train edges.
        active_nodes: Sorted array of train node indices (node_mapping).
        num_neighbors: K neighbors to sample.
        n_negatives: Number of negatives per query (default 100).
        use_amp: Enable mixed precision.
        seed: Random seed for negative sampling.
        max_queries: Maximum number of eval queries (default 50K).
        is_undirected: If True, take only forward edges from eval split.

    Returns:
        Dict with MRR, Hits@1, Hits@3, Hits@10, and metadata.
    """
    model.eval()
    rng = np.random.RandomState(seed)
    amp_enabled = amp_enabled_for_device(use_amp, device)
    K = num_neighbors
    use_edge_feats = model.edge_feat_dim > 0
    use_node_feats = model.node_feat_dim > 0

    if is_undirected:
        eval_indices = _get_forward_mask(data, eval_mask)
    else:
        eval_indices = np.where(eval_mask)[0]

    src_all = data.src[eval_indices].astype(np.int64)
    dst_all = data.dst[eval_indices].astype(np.int64)

    unique_df_arr = np.stack([src_all, dst_all], axis=1)
    _, unique_idx = np.unique(unique_df_arr, axis=0, return_index=True)
    unique_idx.sort()
    src_unique = src_all[unique_idx]
    dst_unique = dst_all[unique_idx]
    eval_edge_indices = eval_indices[unique_idx]

    train_node_set = set(active_nodes.tolist())
    keep = np.array([
        int(s) in train_node_set and int(d) in train_node_set
        for s, d in zip(src_unique, dst_unique)
    ], dtype=bool)

    n_before = len(src_unique)
    src_unique = src_unique[keep]
    dst_unique = dst_unique[keep]
    eval_edge_indices = eval_edge_indices[keep]
    n_after = len(src_unique)
    n_filtered = n_before - n_after

    logger.info(
        "Eval: filtered %d/%d queries with unknown nodes",
        n_filtered, n_before,
    )

    all_positives_per_src: Dict[int, Set[int]] = {}
    for s, d in zip(src_unique, dst_unique):
        all_positives_per_src.setdefault(int(s), set()).add(int(d))

    n_total = len(src_unique)
    chosen = choose_query_indices(
        n_total,
        max_queries,
        rng=np.random.RandomState(seed + 777),
    )
    if len(chosen) < n_total:
        src_unique = src_unique[chosen]
        dst_unique = dst_unique[chosen]
        eval_edge_indices = eval_edge_indices[chosen]
        logger.info("Subsampled %d -> %d queries", n_total, len(chosen))

    n_queries = len(src_unique)
    logger.info(
        "EAGLE TGB-style eval: %d queries, %d negatives per query",
        n_queries, n_negatives,
    )

    def _score_rank(q: int) -> float:
        src_node = int(src_unique[q])
        true_dst = int(dst_unique[q])
        edge_idx = eval_edge_indices[q]
        ts = data.timestamps[edge_idx]

        neg_nodes = sample_negatives_for_eval(
            src=src_node,
            dst_true=true_dst,
            train_neighbors=train_neighbors,
            eval_positives_of_src=all_positives_per_src.get(src_node, set()),
            active_nodes=active_nodes,
            n_negatives=n_negatives,
            rng=rng,
        )
        all_dst = np.concatenate([[true_dst], neg_nodes]).astype(np.int32)
        num_candidates = len(all_dst)

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
        )

        dst_ts_arr = np.full(num_candidates, ts, dtype=np.float64)
        dst_context = sample_node_contexts(
            csr=csr,
            data=data,
            sample_neighbors_fn=sample_neighbors_batch,
            nodes=all_dst,
            query_timestamps=dst_ts_arr,
            num_neighbors=K,
            use_edge_feats=use_edge_feats,
            use_node_feats=use_node_feats,
        )

        scores = score_candidate_contexts(
            model=model,
            device=device,
            src_context=src_context,
            dst_context=dst_context,
            amp_enabled=amp_enabled,
        )
        return conservative_rank_from_scores(scores)

    metrics, elapsed = evaluate_ranking_loop(
        np.arange(n_queries),
        score_rank_fn=lambda idx: _score_rank(idx),
    )
    metrics["n_filtered"] = n_filtered

    logger.info(
        "EAGLE eval: MRR=%.4f Hits@1=%.3f Hits@3=%.3f Hits@10=%.3f (%.1fs, %d queries)",
        metrics["mrr"],
        metrics["hits@1"],
        metrics["hits@3"],
        metrics["hits@10"],
        elapsed,
        n_queries,
    )

    return metrics

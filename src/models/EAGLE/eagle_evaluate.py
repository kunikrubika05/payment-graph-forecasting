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

import contextlib
import logging
import time
from typing import Dict, Optional, Set

import numpy as np
import torch
from tqdm import tqdm

from src.models.EAGLE.eagle import EAGLETime
from src.models.EAGLE.data_utils import (
    TemporalEdgeData,
    TemporalCSR,
    sample_neighbors_batch,
)
from src.models.data_utils import featurize_neighbors
from src.baselines.evaluation import compute_ranking_metrics
from sg_baselines.sampling import sample_negatives_for_eval

logger = logging.getLogger(__name__)


def _amp_autocast(enabled: bool, device_type: str):
    """Return AMP autocast context or no-op."""
    if enabled and device_type == "cuda":
        return torch.cuda.amp.autocast()
    return contextlib.nullcontext()


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
    amp_enabled = use_amp and device.type == "cuda"
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
    if n_total > max_queries:
        rng_sub = np.random.RandomState(seed + 777)
        idx = rng_sub.choice(n_total, size=max_queries, replace=False)
        idx.sort()
        src_unique = src_unique[idx]
        dst_unique = dst_unique[idx]
        eval_edge_indices = eval_edge_indices[idx]
        logger.info("Subsampled %d -> %d queries", n_total, max_queries)

    n_queries = len(src_unique)
    logger.info(
        "EAGLE TGB-style eval: %d queries, %d negatives per query",
        n_queries, n_negatives,
    )

    all_ranks = []
    start_time = time.time()

    for q in tqdm(range(n_queries), desc="Evaluating"):
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
        src_nids, src_nts, src_neids, src_lens = sample_neighbors_batch(
            csr, src_arr, ts_arr, K
        )
        src_dt = np.maximum(
            (ts_arr[:, None] - src_nts), 0.0
        ).astype(np.float32)

        dst_ts_arr = np.full(num_candidates, ts, dtype=np.float64)
        dst_nids, dst_nts, dst_neids, dst_lens = sample_neighbors_batch(
            csr, all_dst, dst_ts_arr, K
        )
        dst_dt = np.maximum(
            (dst_ts_arr[:, None] - dst_nts), 0.0
        ).astype(np.float32)

        src_ef = dst_ef = src_nf = dst_nf = None
        if use_edge_feats or use_node_feats:
            _, src_ef_raw, _ = featurize_neighbors(
                src_nids, src_neids, src_lens, src_nts, ts_arr,
                data.node_feats, data.edge_feats,
            )
            _, dst_ef_raw, _ = featurize_neighbors(
                dst_nids, dst_neids, dst_lens, dst_nts, dst_ts_arr,
                data.node_feats, data.edge_feats,
            )
            if use_edge_feats:
                src_ef = src_ef_raw.astype(np.float32)
                dst_ef = dst_ef_raw.astype(np.float32)
            if use_node_feats:
                src_nf = data.node_feats[[src_node]].astype(np.float32)
                dst_nf = data.node_feats[all_dst].astype(np.float32)

        def _t(arr, dtype=torch.float32):
            return torch.tensor(arr, dtype=dtype, device=device)

        with _amp_autocast(amp_enabled, device.type):
            h_src = model.encode_nodes(
                _t(src_dt), _t(src_lens, torch.int64),
                edge_feats=_t(src_ef) if src_ef is not None else None,
                node_feats=_t(src_nf) if src_nf is not None else None,
            )
            h_dst = model.encode_nodes(
                _t(dst_dt), _t(dst_lens, torch.int64),
                edge_feats=_t(dst_ef) if dst_ef is not None else None,
                node_feats=_t(dst_nf) if dst_nf is not None else None,
            )
            h_src_exp = h_src.expand(num_candidates, -1)
            scores = model.edge_predictor(
                h_src_exp, h_dst
            ).cpu().float().numpy()

        true_score = scores[0]
        rank = float(np.sum(scores[1:] > true_score) + 1)
        all_ranks.append(rank)

    elapsed = time.time() - start_time
    ranks_arr = np.array(all_ranks, dtype=np.float64)
    metrics = compute_ranking_metrics(ranks_arr)
    metrics["eval_time_sec"] = elapsed
    metrics["edges_per_sec"] = n_queries / elapsed if elapsed > 0 else 0
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

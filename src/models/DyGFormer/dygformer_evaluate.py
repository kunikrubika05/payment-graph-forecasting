"""TGB-style evaluation for DyGFormer temporal link prediction.

Matches the baseline protocol from sg_baselines/sampling.py exactly:
    - 50 historical + 50 random negatives per positive edge
    - Historical: train neighbors of src, excluding ALL eval positives of src
    - Random: from active train nodes only (not full 312M node space)
    - Per-source ranking: rank true destination among 101 candidates
    - Rank = 1 + count(score > true_score), conservative (no fractional ties)
    - Metrics: MRR, Hits@1, Hits@3, Hits@10
    - 50K query subsample
    - Eval seeds: val = seed+200, test = seed+400

Note: DyGFormer evaluation is significantly slower than EAGLE/GLFormer
because each candidate destination requires a separate encoder forward
pass (src and dst are encoded jointly through the Transformer).
"""

import contextlib
import logging
import time
from collections import defaultdict
from typing import Dict, Optional

import numpy as np
import torch
from tqdm import tqdm

from src.models.DyGFormer.dygformer import DyGFormerTime
from src.models.DyGFormer.data_utils import (
    TemporalEdgeData,
    TemporalCSR,
    sample_neighbors_batch,
)
from src.models.DyGFormer.dygformer_train import compute_neighbor_cooccurrence
from src.models.data_utils import featurize_neighbors
from src.baselines.evaluation import compute_ranking_metrics
from sg_baselines.sampling import sample_negatives_for_eval

logger = logging.getLogger(__name__)


def _amp_autocast(enabled: bool, device_type: str):
    """Return AMP autocast context or a no-op context manager."""
    if enabled and device_type == "cuda":
        return torch.amp.autocast("cuda")
    return contextlib.nullcontext()


def build_eval_positives_per_src(
    eval_src: np.ndarray,
    eval_dst: np.ndarray,
) -> dict[int, set[int]]:
    """Build per-source set of ALL positive destinations in eval split."""
    positives: dict[int, set[int]] = defaultdict(set)
    for s, d in zip(eval_src, eval_dst):
        positives[int(s)].add(int(d))
    return dict(positives)


@torch.no_grad()
def evaluate_tgb_style(
    model: DyGFormerTime,
    data: TemporalEdgeData,
    csr: TemporalCSR,
    eval_src: np.ndarray,
    eval_dst: np.ndarray,
    eval_ts: np.ndarray,
    train_neighbors: dict[int, set[int]],
    active_nodes: np.ndarray,
    device: torch.device,
    num_neighbors: int = 32,
    n_hist_neg: int = 50,
    n_random_neg: int = 50,
    use_amp: bool = True,
    seed: int = 42,
    max_edges: int = 50_000,
) -> Dict[str, float]:
    """Full TGB-style evaluation matching the baseline protocol exactly.

    For each positive DIRECTED edge (src, dst, t):
        1. Generate negatives via sample_negatives_for_eval (sg_baselines).
        2. Score all 1 + n_negatives candidates using DyGFormer.
           Each candidate requires a separate forward pass because
           DyGFormer encodes src and dst jointly.
        3. Compute conservative rank (no fractional ties).

    Args:
        model: Trained DyGFormerTime model.
        data: Temporal edge data.
        csr: Temporal CSR built from edges preceding the evaluation period.
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
        max_edges: Maximum queries to evaluate.

    Returns:
        Dict with MRR, Hits@1, Hits@3, Hits@10, eval_time_sec,
        edges_per_sec, and n_queries.
    """
    model.eval()
    rng = np.random.RandomState(seed)
    amp_enabled = use_amp and device.type == "cuda"
    K = num_neighbors
    n_negatives = n_hist_neg + n_random_neg
    use_edge_feats = model.edge_feat_dim > 0
    use_node_feats = model.node_feat_dim > 0

    n_total_edges = len(eval_src)
    eval_positives_per_src = build_eval_positives_per_src(eval_src, eval_dst)

    if n_total_edges > max_edges:
        chosen = rng.choice(n_total_edges, size=max_edges, replace=False)
        chosen.sort()
    else:
        chosen = np.arange(n_total_edges)

    n_queries = len(chosen)
    logger.info(
        "DyGFormer TGB-style eval: %d/%d edges, %d negatives each (seed=%d)",
        n_queries, n_total_edges, n_negatives, seed,
    )

    all_ranks = []
    start_time = time.time()

    for i in tqdm(chosen, desc="Evaluating"):
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
        src_nids, src_nts, src_neids, src_lens = sample_neighbors_batch(
            csr, src_arr, ts_arr, K
        )
        src_dt = np.maximum(ts_arr[:, None] - src_nts, 0.0).astype(np.float32)
        src_dt[0, src_lens[0]:] = 0.0

        src_ef = src_nf = None
        if use_edge_feats or use_node_feats:
            nf_raw, ef_raw, _ = featurize_neighbors(
                src_nids, src_neids, src_lens, src_nts, ts_arr,
                data.node_feats, data.edge_feats,
            )
            if use_edge_feats:
                src_ef = ef_raw.astype(np.float32)
            if use_node_feats:
                src_nf = nf_raw.astype(np.float32)

        scores = np.zeros(C, dtype=np.float32)

        for c_idx in range(C):
            dst_node = all_dst[c_idx]
            dst_arr = np.array([dst_node], dtype=np.int32)
            dst_ts_arr = np.array([ts], dtype=np.float64)
            dst_nids_c, dst_nts_c, dst_neids_c, dst_lens_c = sample_neighbors_batch(
                csr, dst_arr, dst_ts_arr, K
            )
            dst_dt_c = np.maximum(dst_ts_arr[:, None] - dst_nts_c, 0.0).astype(np.float32)
            dst_dt_c[0, dst_lens_c[0]:] = 0.0

            dst_ef_c = dst_nf_c = None
            if use_edge_feats or use_node_feats:
                nf_raw_c, ef_raw_c, _ = featurize_neighbors(
                    dst_nids_c, dst_neids_c, dst_lens_c, dst_nts_c, dst_ts_arr,
                    data.node_feats, data.edge_feats,
                )
                if use_edge_feats:
                    dst_ef_c = ef_raw_c.astype(np.float32)
                if use_node_feats:
                    dst_nf_c = nf_raw_c.astype(np.float32)

            src_cooc_c, dst_cooc_c = compute_neighbor_cooccurrence(
                src_nids, src_lens, dst_nids_c, dst_lens_c
            )

            def _t(arr, dtype=torch.float32):
                if arr is None:
                    return None
                return torch.tensor(arr, dtype=dtype, device=device)

            with _amp_autocast(amp_enabled, device.type):
                logit = model(
                    src_delta_times=_t(src_dt),
                    src_lengths=_t(src_lens, torch.int64),
                    dst_delta_times=_t(dst_dt_c),
                    dst_lengths=_t(dst_lens_c, torch.int64),
                    src_cooc_counts=_t(src_cooc_c),
                    dst_cooc_counts=_t(dst_cooc_c),
                    src_edge_feats=_t(src_ef),
                    dst_edge_feats=_t(dst_ef_c),
                    src_node_feats=_t(src_nf),
                    dst_node_feats=_t(dst_nf_c),
                )
                scores[c_idx] = logit.cpu().float().item()

        true_score = scores[0]
        rank = 1.0 + (scores[1:] > true_score).sum()
        all_ranks.append(float(rank))

    elapsed = time.time() - start_time
    ranks_arr = np.array(all_ranks, dtype=np.float64)
    metrics = compute_ranking_metrics(ranks_arr)
    metrics["eval_time_sec"] = elapsed
    metrics["edges_per_sec"] = n_queries / elapsed if elapsed > 0 else 0.0

    logger.info(
        "DyGFormer eval: MRR=%.4f Hits@1=%.3f Hits@3=%.3f Hits@10=%.3f "
        "(%d queries, %.1fs)",
        metrics["mrr"], metrics["hits@1"], metrics["hits@3"],
        metrics["hits@10"], n_queries, elapsed,
    )
    return metrics

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

import contextlib
import logging
import time
from collections import defaultdict
from typing import Dict, Optional

import numpy as np
import torch
from tqdm import tqdm

from src.models.GLFormer.glformer import GLFormerTime
from src.models.GLFormer.data_utils import (
    TemporalEdgeData,
    TemporalCSR,
    sample_neighbors_batch,
)
from src.models.GLFormer.glformer_train import _compute_cooccurrence
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
    amp_enabled = use_amp and device.type == "cuda"
    K = num_neighbors
    n_negatives = n_hist_neg + n_random_neg
    use_edge_feats = model.edge_feat_dim > 0
    use_node_feats = model.node_feat_dim > 0
    use_cooc = model.use_cooccurrence

    n_total_edges = len(eval_src)
    eval_positives_per_src = build_eval_positives_per_src(eval_src, eval_dst)

    if n_total_edges > max_edges:
        chosen = rng.choice(n_total_edges, size=max_edges, replace=False)
        chosen.sort()
    else:
        chosen = np.arange(n_total_edges)

    n_queries = len(chosen)
    logger.info(
        "GLFormer TGB-style eval: %d/%d edges, %d negatives each (seed=%d)",
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

        dst_ts_arr = np.full(C, ts, dtype=np.float64)
        dst_nids, dst_nts, dst_neids, dst_lens = sample_neighbors_batch(
            csr, all_dst, dst_ts_arr, K
        )
        dst_dt = np.maximum(dst_ts_arr[:, None] - dst_nts, 0.0).astype(np.float32)
        for b in range(C):
            dst_dt[b, dst_lens[b]:] = 0.0

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

        cooc_counts = None
        if use_cooc:
            src_nids_rep = np.repeat(src_nids, C, axis=0)
            src_lens_rep = np.repeat(src_lens, C)
            cooc_np = _compute_cooccurrence(
                src_nids_rep, src_lens_rep, dst_nids, dst_lens
            )
            cooc_counts = torch.tensor(cooc_np, dtype=torch.float32, device=device)

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
            h_src_exp = h_src.expand(C, -1)

            cooc_feat = None
            if model.cooc_encoder is not None and cooc_counts is not None:
                cooc_feat = model.cooc_encoder(cooc_counts)

            scores = model.edge_predictor(
                h_src_exp, h_dst, cooc_feat
            ).cpu().float().numpy()

        true_score = scores[0]
        rank = 1.0 + (scores[1:] > true_score).sum()
        all_ranks.append(float(rank))

    elapsed = time.time() - start_time
    ranks_arr = np.array(all_ranks, dtype=np.float64)
    metrics = compute_ranking_metrics(ranks_arr)
    metrics["eval_time_sec"] = elapsed
    metrics["edges_per_sec"] = n_queries / elapsed if elapsed > 0 else 0.0

    logger.info(
        "GLFormer eval: MRR=%.4f Hits@1=%.3f Hits@3=%.3f Hits@10=%.3f "
        "(%d queries, %.1fs)",
        metrics["mrr"], metrics["hits@1"], metrics["hits@3"],
        metrics["hits@10"], n_queries, elapsed,
    )
    return metrics

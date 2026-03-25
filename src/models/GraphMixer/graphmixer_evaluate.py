"""TGB-style evaluation for GraphMixer temporal link prediction.

Uses the same protocol as EAGLE and GLFormer:
    - 50 historical + 50 random negatives per positive edge
    - Per-source ranking: rank true destination among 101 candidates
    - Metrics: MRR, Hits@1, Hits@3, Hits@10
"""

import contextlib
import logging
import time
from typing import Dict

import numpy as np
import torch
from tqdm import tqdm

from src.models.GraphMixer.graphmixer import GraphMixerTime
from src.models.GraphMixer.data_utils import (
    TemporalEdgeData,
    TemporalCSR,
    sample_neighbors_batch,
    generate_negatives_for_eval,
)
from src.models.data_utils import featurize_neighbors
from src.baselines.evaluation import compute_ranking_metrics

logger = logging.getLogger(__name__)


def _amp_autocast(enabled: bool, device_type: str):
    """Return AMP autocast context or a no-op context manager."""
    if enabled and device_type == "cuda":
        return torch.cuda.amp.autocast()
    return contextlib.nullcontext()


@torch.no_grad()
def evaluate_tgb_style(
    model: GraphMixerTime,
    data: TemporalEdgeData,
    csr: TemporalCSR,
    eval_mask: np.ndarray,
    device: torch.device,
    num_neighbors: int = 20,
    n_hist_neg: int = 50,
    n_random_neg: int = 50,
    use_amp: bool = True,
    seed: int = 42,
) -> Dict[str, float]:
    """Full TGB-style evaluation matching the baseline protocol.

    For each positive edge (src, dst, t):
        1. Generate n_hist_neg historical + n_random_neg random negatives.
        2. Score all 1 + n_hist_neg + n_random_neg candidates.
        3. Compute fractional rank of the true destination.

    Historical negatives are nodes that src has interacted with previously
    (excluding the true dst). Random negatives are uniformly sampled from
    all nodes.

    Args:
        model: Trained GraphMixerTime model.
        data: Temporal edge data.
        csr: Temporal CSR built from edges preceding the evaluation period
            (train-only for val eval, train+val for test eval).
        eval_mask: Boolean mask selecting evaluation edges.
        device: Torch device.
        num_neighbors: K most-recent neighbors sampled per node.
        n_hist_neg: Number of historical negatives per query.
        n_random_neg: Number of random negatives per query.
        use_amp: Enable mixed precision.
        seed: Random seed for negative sampling.

    Returns:
        Dict with MRR, Hits@1, Hits@3, Hits@10, eval_time_sec,
        edges_per_sec, and n_queries.
    """
    model.eval()
    rng = np.random.default_rng(seed)
    amp_enabled = use_amp and device.type == "cuda"
    K = num_neighbors
    use_edge_feats = model.edge_feat_dim > 0
    use_node_feats = model.node_feat_dim > 0

    eval_indices = np.where(eval_mask)[0]
    n_total = len(eval_indices)
    logger.info(
        "GraphMixer TGB-style eval: %d edges, %d negatives each",
        n_total, n_hist_neg + n_random_neg,
    )

    all_ranks = []
    start_time = time.time()

    for idx in tqdm(eval_indices, desc="Evaluating"):
        src_node = data.src[idx]
        true_dst = data.dst[idx]
        ts = data.timestamps[idx]

        neg_nodes = generate_negatives_for_eval(
            src_node, true_dst, ts, csr, data.num_nodes,
            n_hist=n_hist_neg, n_random=n_random_neg, rng=rng,
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

        src_ef = dst_ef = src_nf = dst_nf = src_nnf = dst_nnf = None
        if use_edge_feats or use_node_feats:
            src_nnf_raw, src_ef_raw, _ = featurize_neighbors(
                src_nids, src_neids, src_lens, src_nts, ts_arr,
                data.node_feats, data.edge_feats,
            )
            dst_nnf_raw, dst_ef_raw, _ = featurize_neighbors(
                dst_nids, dst_neids, dst_lens, dst_nts, dst_ts_arr,
                data.node_feats, data.edge_feats,
            )
            if use_edge_feats:
                src_ef = src_ef_raw.astype(np.float32)
                dst_ef = dst_ef_raw.astype(np.float32)
            if use_node_feats:
                src_nf = data.node_feats[[src_node]].astype(np.float32)
                dst_nf = data.node_feats[all_dst].astype(np.float32)
                src_nnf = src_nnf_raw.astype(np.float32)
                dst_nnf = dst_nnf_raw.astype(np.float32)

        def _t(arr, dtype=torch.float32):
            return torch.tensor(arr, dtype=dtype, device=device)

        with _amp_autocast(amp_enabled, device.type):
            h_src = model.encode_nodes(
                _t(src_dt), _t(src_lens, torch.int64),
                edge_feats=_t(src_ef) if src_ef is not None else None,
                node_feats=_t(src_nf) if src_nf is not None else None,
                neighbor_node_feats=_t(src_nnf) if src_nnf is not None else None,
            )
            h_dst = model.encode_nodes(
                _t(dst_dt), _t(dst_lens, torch.int64),
                edge_feats=_t(dst_ef) if dst_ef is not None else None,
                node_feats=_t(dst_nf) if dst_nf is not None else None,
                neighbor_node_feats=_t(dst_nnf) if dst_nnf is not None else None,
            )
            h_src_exp = h_src.expand(C, -1)
            scores = model.edge_predictor(h_src_exp, h_dst).cpu().float().numpy()

        true_score = scores[0]
        rank = (
            1.0
            + (scores[1:] > true_score).sum()
            + 0.5 * (scores[1:] == true_score).sum()
        )
        all_ranks.append(float(rank))

    elapsed = time.time() - start_time
    ranks_arr = np.array(all_ranks, dtype=np.float64)
    metrics = compute_ranking_metrics(ranks_arr)
    metrics["eval_time_sec"] = elapsed
    metrics["edges_per_sec"] = n_total / elapsed if elapsed > 0 else 0.0
    metrics["n_queries"] = n_total

    logger.info(
        "GraphMixer eval: MRR=%.4f Hits@1=%.3f Hits@3=%.3f Hits@10=%.3f (%.1fs)",
        metrics["mrr"], metrics["hits@1"], metrics["hits@3"],
        metrics["hits@10"], elapsed,
    )
    return metrics

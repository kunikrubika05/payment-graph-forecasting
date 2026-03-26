"""TGB-style evaluation for GLFormer with CUDA-accelerated sampling.

Same protocol as GLFormer/glformer_evaluate.py (50 hist + 50 random negatives,
per-source ranking), but uses TemporalGraphSampler for neighbor lookups.
"""

import contextlib
import logging
import time
from typing import Dict

import numpy as np
import torch
from tqdm import tqdm

from src.models.GLFormer.glformer import GLFormerTime
from src.models.GLFormer_cuda.data_utils import TemporalEdgeData
from src.models.GLFormer_cuda.glformer_train import _compute_cooccurrence
from src.models.temporal_graph_sampler import TemporalGraphSampler
from src.models.EAGLE.data_utils import generate_negatives_for_eval
from src.baselines.evaluation import compute_ranking_metrics

logger = logging.getLogger(__name__)


def _amp_autocast(enabled: bool, device_type: str):
    """Return AMP autocast context or a no-op context manager."""
    if enabled and device_type == "cuda":
        return torch.cuda.amp.autocast()
    return contextlib.nullcontext()


@torch.no_grad()
def evaluate_tgb_style(
    model: GLFormerTime,
    data: TemporalEdgeData,
    sampler: TemporalGraphSampler,
    eval_mask: np.ndarray,
    device: torch.device,
    num_neighbors: int = 20,
    n_hist_neg: int = 50,
    n_random_neg: int = 50,
    use_amp: bool = True,
    seed: int = 42,
) -> Dict[str, float]:
    """Full TGB-style evaluation using CUDA-accelerated sampling.

    Same interface as GLFormer/glformer_evaluate.evaluate_tgb_style, but
    uses TemporalGraphSampler instead of TemporalCSR + sample_neighbors_batch.

    For historical negatives, we use EAGLE's generate_negatives_for_eval which
    requires a TemporalCSR. We build a lightweight Python-based CSR for this
    purpose only (negative generation is not the bottleneck).

    Args:
        sampler: TemporalGraphSampler for neighbor sampling/featurization.
        (all other args identical to GLFormer evaluate_tgb_style)

    Returns:
        Dict with MRR, Hits@1, Hits@3, Hits@10, eval_time_sec,
        edges_per_sec, n_queries.
    """
    from src.models.EAGLE.data_utils import build_temporal_csr

    model.eval()
    rng = np.random.default_rng(seed)
    amp_enabled = use_amp and device.type == "cuda"
    K = num_neighbors
    use_edge_feats = model.edge_feat_dim > 0
    use_node_feats = model.node_feat_dim > 0
    use_cooc = model.use_cooccurrence

    eval_indices = np.where(eval_mask)[0]
    n_total = len(eval_indices)
    logger.info(
        "GLFormer TGB-style eval (CUDA): %d edges, %d negatives each",
        n_total, n_hist_neg + n_random_neg,
    )

    neg_csr = build_temporal_csr(data, eval_mask | np.zeros_like(eval_mask, dtype=bool))
    all_before_mask = np.zeros(data.num_edges, dtype=bool)
    all_before_mask[:eval_indices[0]] = True
    neg_csr = build_temporal_csr(data, all_before_mask)

    all_ranks = []
    start_time = time.time()

    for idx in tqdm(eval_indices, desc="Evaluating (CUDA)"):
        src_node = data.src[idx]
        true_dst = data.dst[idx]
        ts = data.timestamps[idx]

        neg_nodes = generate_negatives_for_eval(
            src_node, true_dst, ts, neg_csr, data.num_nodes,
            n_hist=n_hist_neg, n_random=n_random_neg, rng=rng,
        )
        all_dst = np.concatenate([[true_dst], neg_nodes]).astype(np.int32)
        C = len(all_dst)

        src_arr = np.array([src_node], dtype=np.int32)
        ts_arr = np.array([ts], dtype=np.float64)

        src_nbr = sampler.sample_neighbors(src_arr, ts_arr, K)
        src_nbr_np = sampler.to_numpy(src_nbr)
        src_dt = np.maximum(ts_arr[:, None] - src_nbr_np.timestamps, 0.0).astype(np.float32)
        src_dt[0, src_nbr_np.lengths[0]:] = 0.0

        dst_ts_arr = np.full(C, ts, dtype=np.float64)
        dst_nbr = sampler.sample_neighbors(all_dst, dst_ts_arr, K)
        dst_nbr_np = sampler.to_numpy(dst_nbr)
        dst_dt = np.maximum(dst_ts_arr[:, None] - dst_nbr_np.timestamps, 0.0).astype(np.float32)
        for b in range(C):
            dst_dt[b, dst_nbr_np.lengths[b]:] = 0.0

        src_ef = dst_ef = src_nf = dst_nf = None
        if use_edge_feats or use_node_feats:
            src_feat = sampler.featurize(src_nbr, query_timestamps=ts_arr)
            src_feat_np = sampler.to_numpy_features(src_feat)
            dst_feat = sampler.featurize(dst_nbr, query_timestamps=dst_ts_arr)
            dst_feat_np = sampler.to_numpy_features(dst_feat)
            if use_edge_feats:
                src_ef = src_feat_np.edge_features.astype(np.float32)
                dst_ef = dst_feat_np.edge_features.astype(np.float32)
            if use_node_feats:
                src_nf = data.node_feats[[src_node]].astype(np.float32)
                dst_nf = data.node_feats[all_dst].astype(np.float32)

        cooc_counts = None
        if use_cooc:
            src_nids_rep = np.repeat(src_nbr_np.neighbor_ids, C, axis=0)
            src_lens_rep = np.repeat(src_nbr_np.lengths, C)
            cooc_np = _compute_cooccurrence(
                src_nids_rep, src_lens_rep,
                dst_nbr_np.neighbor_ids, dst_nbr_np.lengths,
            )
            cooc_counts = torch.tensor(cooc_np, dtype=torch.float32, device=device)

        def _t(arr, dtype=torch.float32):
            return torch.tensor(arr, dtype=dtype, device=device)

        with _amp_autocast(amp_enabled, device.type):
            h_src = model.encode_nodes(
                _t(src_dt), _t(src_nbr_np.lengths, torch.int64),
                edge_feats=_t(src_ef) if src_ef is not None else None,
                node_feats=_t(src_nf) if src_nf is not None else None,
            )
            h_dst = model.encode_nodes(
                _t(dst_dt), _t(dst_nbr_np.lengths, torch.int64),
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

    logger.info(
        "GLFormer eval (CUDA): MRR=%.4f Hits@1=%.3f Hits@3=%.3f Hits@10=%.3f (%.1fs)",
        metrics["mrr"], metrics["hits@1"], metrics["hits@3"],
        metrics["hits@10"], elapsed,
    )
    return metrics

"""TGB-style evaluation for temporal link prediction models.

Uses the same protocol as baselines: 50 historical + 50 random negatives,
per-source ranking, MRR/Hits@K metrics.
"""

import logging
import time
from typing import Dict, Optional

import numpy as np
import torch
from tqdm import tqdm

from src.models.graphmixer import GraphMixer
from src.models.data_utils import (
    TemporalEdgeData,
    TemporalCSR,
    sample_neighbors_batch,
    featurize_neighbors,
    generate_negatives_for_eval,
)
from src.baselines.evaluation import compute_ranking_metrics

logger = logging.getLogger(__name__)


@torch.no_grad()
def evaluate_tgb_style(
    model: GraphMixer,
    data: TemporalEdgeData,
    csr: TemporalCSR,
    eval_mask: np.ndarray,
    device: torch.device,
    num_neighbors: int = 20,
    n_hist_neg: int = 50,
    n_random_neg: int = 50,
    eval_batch_size: int = 32,
    seed: int = 42,
) -> Dict[str, float]:
    """Full TGB-style evaluation matching baseline protocol.

    For each positive edge (src, dst, t):
        1. Generate 50 historical + 50 random negatives
        2. Score all 101 candidates (1 positive + 100 negatives)
        3. Compute rank of true destination

    Args:
        model: Trained GraphMixer model.
        data: Temporal edge data.
        csr: Temporal CSR (built from all edges up to eval period).
        eval_mask: Boolean mask selecting evaluation edges.
        device: Torch device.
        num_neighbors: K neighbors to sample.
        n_hist_neg: Number of historical negatives per query.
        n_random_neg: Number of random negatives per query.
        eval_batch_size: Number of queries to process together.
        seed: Random seed.

    Returns:
        Dict with MRR, Hits@1, Hits@3, Hits@10, and metadata.
    """
    model.eval()
    rng = np.random.default_rng(seed)

    eval_indices = np.where(eval_mask)[0]
    n_total = len(eval_indices)
    n_negatives = n_hist_neg + n_random_neg

    logger.info("Starting TGB-style evaluation: %d edges, %d negatives per edge",
                n_total, n_negatives)

    all_ranks = []
    start_time = time.time()

    for batch_start in tqdm(range(0, n_total, eval_batch_size), desc="Evaluating"):
        batch_end = min(batch_start + eval_batch_size, n_total)
        batch_indices = eval_indices[batch_start:batch_end]

        for i, idx in enumerate(batch_indices):
            src_node = data.src[idx]
            true_dst = data.dst[idx]
            ts = data.timestamps[idx]

            neg_nodes = generate_negatives_for_eval(
                src_node, true_dst, ts, csr, data.num_nodes,
                n_hist=n_hist_neg, n_random=n_random_neg, rng=rng,
            )

            all_dst = np.concatenate([[true_dst], neg_nodes]).astype(np.int32)
            num_candidates = len(all_dst)

            src_arr = np.array([src_node], dtype=np.int32)
            ts_arr = np.array([ts], dtype=np.float64)

            src_nn, src_nts, src_neids, src_lens = sample_neighbors_batch(
                csr, src_arr, ts_arr, num_neighbors
            )
            src_nf = data.node_feats[src_arr]
            src_nnf, src_nef, src_nrt = featurize_neighbors(
                src_nn, src_neids, src_lens, src_nts, ts_arr,
                data.node_feats, data.edge_feats,
            )

            dst_nf = data.node_feats[all_dst]
            dst_ts_arr = np.full(num_candidates, ts, dtype=np.float64)
            dst_nn, dst_nts, dst_neids, dst_lens = sample_neighbors_batch(
                csr, all_dst, dst_ts_arr, num_neighbors
            )
            dst_nnf, dst_nef, dst_nrt = featurize_neighbors(
                dst_nn, dst_neids, dst_lens, dst_nts, dst_ts_arr,
                data.node_feats, data.edge_feats,
            )

            def _t(arr, dtype=torch.float32):
                return torch.tensor(arr, dtype=dtype, device=device)

            h_src = model.encode_node(
                _t(src_nf), _t(src_nnf), _t(src_nef), _t(src_nrt),
                _t(src_lens, torch.int64),
            )

            h_dst = model.encode_node(
                _t(dst_nf), _t(dst_nnf), _t(dst_nef), _t(dst_nrt),
                _t(dst_lens, torch.int64),
            )

            h_src_expanded = h_src.expand(num_candidates, -1)
            scores = model.link_classifier(h_src_expanded, h_dst).cpu().numpy()

            true_score = scores[0]
            rank = 1 + (scores[1:] > true_score).sum() + 0.5 * (scores[1:] == true_score).sum()
            all_ranks.append(float(rank))

    elapsed = time.time() - start_time
    ranks_arr = np.array(all_ranks, dtype=np.float64)
    metrics = compute_ranking_metrics(ranks_arr)
    metrics["eval_time_sec"] = elapsed
    metrics["edges_per_sec"] = n_total / elapsed if elapsed > 0 else 0

    logger.info(
        "Evaluation complete: MRR=%.4f, Hits@1=%.3f, Hits@3=%.3f, Hits@10=%.3f (%.1fs)",
        metrics["mrr"], metrics["hits@1"], metrics["hits@3"], metrics["hits@10"], elapsed,
    )

    return metrics

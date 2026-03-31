"""TGB-style evaluation for DyGFormer temporal link prediction."""

from __future__ import annotations

import logging
import time
from collections import defaultdict

import numpy as np
import torch

from payment_graph_forecasting.evaluation.ranking_loop import (
    choose_query_indices,
    evaluate_ranking_loop,
)
from payment_graph_forecasting.training.amp import amp_enabled_for_device
from src.models.DyGFormer.data_utils import (
    TemporalCSR,
    TemporalEdgeData,
)
from src.models.DyGFormer.dygformer import DyGFormerTime
from src.models.DyGFormer.dygformer_train import _sample_contexts, score_dygformer_candidates
from sg_baselines.sampling import sample_negatives_for_eval

logger = logging.getLogger(__name__)


def build_eval_positives_per_src(
    eval_src: np.ndarray,
    eval_dst: np.ndarray,
) -> dict[int, set[int]]:
    positives: dict[int, set[int]] = defaultdict(set)
    for src_node, dst_node in zip(eval_src, eval_dst):
        positives[int(src_node)].add(int(dst_node))
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
) -> dict[str, float]:
    """Run batched TGB-style ranking evaluation for DyGFormer."""

    model.eval()
    rng = np.random.RandomState(seed)
    amp_enabled = amp_enabled_for_device(use_amp, device)
    use_edge_feats = model.edge_feat_dim > 0
    use_node_feats = model.node_feat_dim > 0
    n_negatives = n_hist_neg + n_random_neg

    train_node_set = set(active_nodes.tolist())
    keep = np.array(
        [
            int(src_node) in train_node_set and int(dst_node) in train_node_set
            for src_node, dst_node in zip(eval_src, eval_dst)
        ],
        dtype=bool,
    )
    eval_src = eval_src[keep]
    eval_dst = eval_dst[keep]
    eval_ts = eval_ts[keep]
    n_filtered = int((~keep).sum())
    if n_filtered:
        logger.info("Eval: filtered %d/%d queries with unknown nodes", n_filtered, len(keep))

    positives_per_src = build_eval_positives_per_src(eval_src, eval_dst)
    chosen = choose_query_indices(len(eval_src), max_edges, rng=rng)
    logger.info(
        "DyGFormer TGB-style eval: %d/%d edges, %d negatives each (seed=%d)",
        len(chosen),
        len(eval_src),
        n_negatives,
        seed,
    )

    def _score_rank(query_idx: int) -> float:
        src_node = int(eval_src[query_idx])
        true_dst = int(eval_dst[query_idx])
        timestamp = float(eval_ts[query_idx])
        negatives = sample_negatives_for_eval(
            src=src_node,
            dst_true=true_dst,
            train_neighbors=train_neighbors,
            eval_positives_of_src=positives_per_src.get(src_node, set()),
            active_nodes=active_nodes,
            n_negatives=n_negatives,
            rng=rng,
        )
        all_dst = np.concatenate(([true_dst], negatives)).astype(np.int32)

        src_context = _sample_contexts(
            csr=csr,
            data=data,
            nodes=np.array([src_node], dtype=np.int32),
            timestamps=np.array([timestamp], dtype=np.float64),
            num_neighbors=num_neighbors,
            use_edge_feats=use_edge_feats,
            use_node_feats=use_node_feats,
        )
        dst_context = _sample_contexts(
            csr=csr,
            data=data,
            nodes=all_dst,
            timestamps=np.full(len(all_dst), timestamp, dtype=np.float64),
            num_neighbors=num_neighbors,
            use_edge_feats=use_edge_feats,
            use_node_feats=use_node_feats,
        )
        scores = score_dygformer_candidates(
            model=model,
            src_context=src_context,
            dst_context=dst_context,
            device=device,
            amp_enabled=amp_enabled,
        )
        true_score = scores[0]
        return float(1.0 + np.sum(scores[1:] > true_score))

    start_time = time.time()
    metrics, elapsed = evaluate_ranking_loop(
        chosen,
        score_rank_fn=lambda idx: _score_rank(idx),
    )
    metrics["n_filtered"] = n_filtered
    metrics["eval_time_sec"] = elapsed
    metrics["edges_per_sec"] = len(chosen) / elapsed if elapsed > 0 else 0.0

    logger.info(
        "DyGFormer eval: MRR=%.4f Hits@1=%.3f Hits@3=%.3f Hits@10=%.3f (%d queries, %.1fs)",
        metrics["mrr"],
        metrics["hits@1"],
        metrics["hits@3"],
        metrics["hits@10"],
        len(chosen),
        time.time() - start_time,
    )
    return metrics


__all__ = ["build_eval_positives_per_src", "evaluate_tgb_style"]

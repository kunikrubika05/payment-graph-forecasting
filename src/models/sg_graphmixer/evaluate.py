"""TGB-style evaluation for GraphMixer on stream graph.

Matches sg_baselines evaluation protocol EXACTLY:
- 100 negatives per query (50 historical + 50 random from train nodes)
- Conservative tie-breaking: rank = count(score > true_score) + 1
- Only edges with both src and dst in train node_mapping are evaluated
- Uses sample_negatives_for_eval from sg_baselines.sampling
- Seeds: val_early_stopping=seed+200, val_final=seed+300, test=seed+400
"""

import time
import logging
from typing import Dict
from collections import defaultdict

import numpy as np
import torch
from tqdm import tqdm

from src.models.graphmixer import GraphMixer
from src.models.data_utils import (
    TemporalEdgeData,
    TemporalCSR,
    sample_neighbors_batch,
    featurize_neighbors,
)
from src.baselines.evaluation import compute_ranking_metrics
from sg_baselines.sampling import sample_negatives_for_eval

logger = logging.getLogger(__name__)


@torch.no_grad()
def evaluate_tgb_style(
    model: GraphMixer,
    data: TemporalEdgeData,
    csr: TemporalCSR,
    eval_mask: np.ndarray,
    device: torch.device,
    num_neighbors: int,
    train_neighbors: dict[int, set[int]],
    active_nodes: np.ndarray,
    node_mapping: np.ndarray,
    n_negatives: int = 100,
    max_queries: int = 50_000,
    seed: int = 242,
) -> Dict[str, float]:
    """Evaluate GraphMixer with the exact sg_baselines TGB protocol.

    For each positive edge (src, dst_true):
        1. Filter: skip if src or dst_true not in train node_mapping
        2. Sample 100 negatives (50 hist + 50 random) from train nodes
        3. Score all 101 candidates
        4. Rank dst_true with conservative tie-breaking
        5. Compute MRR, Hits@K

    Args:
        model: Trained GraphMixer model.
        data: Temporal edge data (dense indices).
        csr: Temporal CSR (train edges only, undirected).
        eval_mask: Boolean mask selecting evaluation edges.
        device: Torch device.
        num_neighbors: K neighbors to sample.
        train_neighbors: Per-source neighbor sets (GLOBAL indices).
        active_nodes: Sorted GLOBAL indices of train nodes.
        node_mapping: Same as active_nodes.
        n_negatives: Number of negatives per query (100).
        max_queries: Max queries to evaluate.
        seed: Random seed for negative sampling.

    Returns:
        Dict with MRR, Hits@1, Hits@3, Hits@10, n_queries, eval_time_sec.
    """
    model.eval()
    rng = np.random.RandomState(seed)

    eval_indices = np.where(eval_mask)[0]
    global_to_dense = {int(g): i for i, g in enumerate(node_mapping)}
    node_set = set(node_mapping.tolist())

    src_global_all = np.array([
        int(data.reverse_node_map[data.src[idx]]) for idx in eval_indices
    ])
    dst_global_all = np.array([
        int(data.reverse_node_map[data.dst[idx]]) for idx in eval_indices
    ])

    valid = np.array([
        s in node_set and d in node_set
        for s, d in zip(src_global_all, dst_global_all)
    ])
    eval_indices = eval_indices[valid]
    src_global_all = src_global_all[valid]
    dst_global_all = dst_global_all[valid]

    unique_edges_set = set()
    unique_mask = np.zeros(len(eval_indices), dtype=bool)
    for i, (s, d) in enumerate(zip(src_global_all, dst_global_all)):
        key = (s, d)
        if key not in unique_edges_set:
            unique_edges_set.add(key)
            unique_mask[i] = True

    eval_indices = eval_indices[unique_mask]
    src_global_all = src_global_all[unique_mask]
    dst_global_all = dst_global_all[unique_mask]

    if len(eval_indices) > max_queries:
        subsample_idx = rng.choice(len(eval_indices), size=max_queries, replace=False)
        subsample_idx.sort()
        eval_indices = eval_indices[subsample_idx]
        src_global_all = src_global_all[subsample_idx]
        dst_global_all = dst_global_all[subsample_idx]

    all_positives_per_src: dict[int, set[int]] = defaultdict(set)
    for s, d in zip(src_global_all, dst_global_all):
        all_positives_per_src[int(s)].add(int(d))

    n_queries = len(eval_indices)
    start_time = time.time()
    all_ranks = []

    for qi in tqdm(range(n_queries), desc="Evaluating", leave=False):
        idx = eval_indices[qi]
        src_g = int(src_global_all[qi])
        dst_g = int(dst_global_all[qi])

        neg_global = sample_negatives_for_eval(
            src=src_g,
            dst_true=dst_g,
            train_neighbors=train_neighbors,
            eval_positives_of_src=all_positives_per_src[src_g],
            active_nodes=active_nodes,
            n_negatives=n_negatives,
            rng=rng,
        )

        all_dst_global = np.concatenate([[dst_g], neg_global])
        all_dst_dense = np.array([global_to_dense.get(int(g), 0) for g in all_dst_global],
                                 dtype=np.int32)

        src_dense = data.src[idx]
        ts = data.timestamps[idx]

        src_arr = np.array([src_dense], dtype=np.int32)
        ts_arr = np.array([ts], dtype=np.float64)

        src_nn, src_nts, src_neids, src_lens = sample_neighbors_batch(
            csr, src_arr, ts_arr, num_neighbors
        )
        src_nf = data.node_feats[src_arr]
        src_nnf, src_nef, src_nrt = featurize_neighbors(
            src_nn, src_neids, src_lens, src_nts, ts_arr,
            data.node_feats, data.edge_feats,
        )

        num_candidates = len(all_dst_dense)
        dst_nf = data.node_feats[all_dst_dense]
        dst_ts_arr = np.full(num_candidates, ts, dtype=np.float64)
        dst_nn, dst_nts, dst_neids, dst_lens = sample_neighbors_batch(
            csr, all_dst_dense, dst_ts_arr, num_neighbors
        )
        dst_nnf, dst_nef, dst_nrt = featurize_neighbors(
            dst_nn, dst_neids, dst_lens, dst_nts, dst_ts_arr,
            data.node_feats, data.edge_feats,
        )

        def _t(arr, dtype=torch.float32):
            return torch.tensor(np.ascontiguousarray(arr), dtype=dtype, device=device)

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
        rank = 1 + (scores[1:] > true_score).sum()
        all_ranks.append(float(rank))

    elapsed = time.time() - start_time
    ranks_arr = np.array(all_ranks, dtype=np.float64)
    metrics = compute_ranking_metrics(ranks_arr)
    metrics["eval_time_sec"] = elapsed
    metrics["n_queries"] = n_queries

    return metrics

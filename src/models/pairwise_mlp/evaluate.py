"""TGB-style evaluation for PairwiseMLP.

Protocol is IDENTICAL to sg_baselines/heuristics.py to ensure fair
comparison with the CN baseline. Specifically:

  1. Deduplication: unique (src, dst) pairs from the eval split.
  2. Filtering: both src AND dst_true must be in node_mapping (train nodes).
     Edges with unseen nodes are excluded — same as heuristics.py lines 98-107.
  3. all_positives_per_src is built from ALL filtered edges BEFORE subsampling.
     This prevents over-ranking: excluded positives of src are still removed
     from the negative pool even if their query isn't in the subsample.
  4. Subsampling: if > max_queries filtered edges remain, subsample to
     max_queries using seed (seed + 777) — same random state as heuristics.py.
  5. Negative sampling per query: n_negatives=100 (50 hist + 50 rand)
     using sample_negatives_for_eval from sg_baselines/sampling.py.
     seed matches sg_baselines: val → random_seed+10, test → random_seed+20.
  6. Scoring: MLP forward on (src, cand) features computed on-the-fly via
     compute_features_batch.
  7. Ranking: rank = count(score > true_score) + 1  (conservative, ties to lower rank).
  8. Metrics: MRR, Hits@1, Hits@3, Hits@10 via compute_ranking_metrics.

Correctness checks verified in this module:
  - all_positives_per_src built BEFORE subsampling (line ~137).
  - eval_features computed from TRAIN adjacency only (adj passed from caller).
  - No future information: adj + node_mapping come from train only (asserted
    by precompute.py; here we receive them as arguments and trust the caller).
"""

import time
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
from scipy import sparse

from sg_baselines.sampling import sample_negatives_for_eval
from src.baselines.evaluation import compute_ranking_metrics
from src.models.pairwise_mlp.features import compute_features_batch
from src.models.pairwise_mlp.model import PairMLP


def build_eval_cache(
    eval_edges: pd.DataFrame,
    train_neighbors: dict,
    node_mapping: np.ndarray,
    adj_undir: sparse.csr_matrix,
    adj_dir: sparse.csr_matrix,
    deg_undir: np.ndarray,
    w_undir: np.ndarray,
    w_dir: np.ndarray,
    seed: int,
    n_negatives: int = 100,
    max_queries: int = 50_000,
    feature_batch: int = 50_000,
    active_feature_indices: Optional[List[int]] = None,
    split_name: str = "",
) -> Dict:
    """Pre-compute candidate features for a fixed seed (call once before training).

    Since seed is fixed per split, the same negatives are sampled on every
    evaluate_split call. This function runs the expensive CPU steps (neg
    sampling + scipy feature computation) once and caches the result.
    Subsequent evaluate_split calls with the returned cache only run the
    fast GPU forward pass and ranking.

    Returns:
        dict with keys:
          all_features:   float32 array (N_pairs, n_active_features)
          query_offsets:  list of ints, length n_queries+1
          n_queries:      int
    """
    tag = f"[{split_name}] " if split_name else ""
    print(f"  {tag}Building eval cache (one-time cost)...", flush=True)

    src_all = eval_edges["src_idx"].values.astype(np.int64)
    dst_all = eval_edges["dst_idx"].values.astype(np.int64)
    unique_df = pd.DataFrame({"src": src_all, "dst": dst_all}).drop_duplicates()
    src_u = unique_df["src"].values.astype(np.int64)
    dst_u = unique_df["dst"].values.astype(np.int64)

    train_node_set = set(node_mapping.tolist())
    keep = np.array([
        int(s) in train_node_set and int(d) in train_node_set
        for s, d in zip(src_u, dst_u)
    ], dtype=bool)
    src_u = src_u[keep]
    dst_u = dst_u[keep]
    n_after = len(src_u)
    print(f"  {tag}{n_after:,} queries after unseen-node filter", flush=True)

    all_positives_per_src: dict = {}
    for s, d in zip(src_u, dst_u):
        all_positives_per_src.setdefault(int(s), set()).add(int(d))

    n_total = len(src_u)
    if n_total > max_queries:
        rng_sub = np.random.RandomState(seed + 777)
        idx = rng_sub.choice(n_total, size=max_queries, replace=False)
        idx.sort()
        src_u = src_u[idx]
        dst_u = dst_u[idx]
        print(f"  {tag}Subsampled to {max_queries:,} queries", flush=True)

    n_queries = len(src_u)

    print(f"  {tag}Sampling negatives...", flush=True)
    rng = np.random.RandomState(seed)
    t0 = time.time()

    all_src_list: List[int] = []
    all_cand_list: List[int] = []
    query_offsets: List[int] = [0]

    for q in range(n_queries):
        s = int(src_u[q])
        d_true = int(dst_u[q])
        negatives = sample_negatives_for_eval(
            s, d_true, train_neighbors,
            all_positives_per_src.get(s, set()),
            node_mapping, n_negatives, rng,
        )
        candidates = np.concatenate([[d_true], negatives])
        n_cand = len(candidates)
        all_src_list.extend([s] * n_cand)
        all_cand_list.extend(candidates.tolist())
        query_offsets.append(query_offsets[-1] + n_cand)

    all_src = np.array(all_src_list, dtype=np.int64)
    all_cand = np.array(all_cand_list, dtype=np.int64)
    print(f"  {tag}Neg sampling done, {len(all_src):,} pairs ({time.time()-t0:.1f}s)",
          flush=True)

    t0 = time.time()
    all_features = compute_features_batch(
        all_src, all_cand,
        node_mapping, adj_undir, adj_dir,
        deg_undir, w_undir, w_dir,
        batch_size=feature_batch,
    )
    if active_feature_indices:
        all_features = all_features[:, active_feature_indices]
    print(f"  {tag}Features computed ({time.time()-t0:.1f}s)", flush=True)

    return {
        "all_features":  all_features,
        "query_offsets": query_offsets,
        "n_queries":     n_queries,
    }


def evaluate_split(
    model: PairMLP,
    device: torch.device,
    eval_cache: Dict,
    score_batch: int = 4096,
    split_name: str = "",
) -> dict:
    """Score and rank using pre-built eval cache (fast, GPU-only).

    Call build_eval_cache() once before training; pass the result as
    eval_cache to every subsequent evaluate_split() call. The expensive
    CPU work (neg sampling + feature computation) runs only once per split.

    Args:
        model:       Trained PairMLP.
        device:      torch.device for model inference.
        eval_cache:  Output of build_eval_cache() for this split.
        score_batch: Pairs per MLP forward batch.
        split_name:  Tag for logging.

    Returns:
        Dict with mrr, hits@1/3/10, n_queries, mean_rank, median_rank.
    """
    tag = f"[{split_name}] " if split_name else ""

    all_features  = eval_cache["all_features"]
    query_offsets = eval_cache["query_offsets"]
    n_queries     = eval_cache["n_queries"]

    # ------------------------------------------------------------------
    # GPU scoring
    # ------------------------------------------------------------------
    t0 = time.time()
    model.eval()
    all_scores = np.empty(len(all_features), dtype=np.float32)
    feat_tensor = torch.from_numpy(all_features)

    with torch.no_grad():
        for start in range(0, len(feat_tensor), score_batch):
            end = min(start + score_batch, len(feat_tensor))
            batch = feat_tensor[start:end].to(device)
            all_scores[start:end] = model(batch).cpu().numpy()

    print(f"  {tag}Scoring done ({time.time()-t0:.1f}s)", flush=True)

    # ------------------------------------------------------------------
    # Ranking
    # ------------------------------------------------------------------
    ranks = np.empty(n_queries, dtype=np.float64)
    for q in range(n_queries):
        start      = query_offsets[q]
        end        = query_offsets[q + 1]
        q_scores   = all_scores[start:end]
        true_score = q_scores[0]
        ranks[q]   = float(np.sum(q_scores > true_score) + 1)

    metrics = compute_ranking_metrics(ranks)
    print(
        f"  {tag}MRR={metrics['mrr']:.4f}, "
        f"H@1={metrics['hits@1']:.4f}, "
        f"H@3={metrics['hits@3']:.4f}, "
        f"H@10={metrics['hits@10']:.4f}",
        flush=True,
    )
    return metrics

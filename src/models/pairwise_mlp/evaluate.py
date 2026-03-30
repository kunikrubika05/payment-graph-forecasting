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
from typing import Optional

import numpy as np
import pandas as pd
import torch
from scipy import sparse
from tqdm import tqdm

from sg_baselines.sampling import sample_negatives_for_eval
from src.baselines.evaluation import compute_ranking_metrics
from src.models.pairwise_mlp.config import PairMLPConfig, N_FEATURES
from src.models.pairwise_mlp.features import (
    compute_features_batch,
    global_to_local,
)
from src.models.pairwise_mlp.model import PairMLP


def evaluate_split(
    model: PairMLP,
    eval_edges: pd.DataFrame,
    train_neighbors: dict,
    node_mapping: np.ndarray,
    adj_undir: sparse.csr_matrix,
    adj_dir: sparse.csr_matrix,
    deg_undir: np.ndarray,
    w_undir: np.ndarray,
    w_dir: np.ndarray,
    seed: int,
    device: torch.device,
    n_negatives: int = 100,
    max_queries: int = 50_000,
    feature_batch: int = 50_000,
    score_batch: int = 4096,
    split_name: str = "",
    active_feature_indices: Optional[List[int]] = None,
) -> dict:
    """TGB-style evaluation on one split (val or test).

    Args:
        model:           Trained PairMLP in eval mode.
        eval_edges:      DataFrame with columns [src_idx, dst_idx, ...].
        train_neighbors: Dict src→set(dst) built from train edges only.
        node_mapping:    Sorted local→global index array from adjacency.
        adj_undir, adj_dir: CSR adjacency matrices (local indices, train only).
        deg_undir:       Precomputed undirected degrees (float64, local).
        w_undir, w_dir:  Precomputed AA weights (float64, local).
        seed:            Random seed matching sg_baselines convention:
                           val  → random_seed + 10
                           test → random_seed + 20
        device:          torch.device for model inference.
        n_negatives:            Negatives per query (default 100).
        max_queries:            Cap on evaluated queries (default 50K).
        feature_batch:          Pairs per scipy feature computation batch.
        score_batch:            Queries per MLP forward batch.
        split_name:             Tag for logging.
        active_feature_indices: Which columns to feed the model (None = all).
                                Must match the indices used during training.

    Returns:
        Dict with n_queries, mrr, mean_rank, median_rank, hits@1/3/10.
    """
    tag = f"[{split_name}] " if split_name else ""

    # ------------------------------------------------------------------
    # Step 1: deduplicate edges
    # ------------------------------------------------------------------
    src_all = eval_edges["src_idx"].values.astype(np.int64)
    dst_all = eval_edges["dst_idx"].values.astype(np.int64)
    unique_df = pd.DataFrame({"src": src_all, "dst": dst_all}).drop_duplicates()
    src_u = unique_df["src"].values.astype(np.int64)
    dst_u = unique_df["dst"].values.astype(np.int64)

    # ------------------------------------------------------------------
    # Step 2: filter — both src and dst_true must be in train node_mapping
    # IDENTICAL to heuristics.py lines 97-107.
    # ------------------------------------------------------------------
    train_node_set = set(node_mapping.tolist())
    keep = np.array([
        int(s) in train_node_set and int(d) in train_node_set
        for s, d in zip(src_u, dst_u)
    ], dtype=bool)
    n_before = len(src_u)
    src_u = src_u[keep]
    dst_u = dst_u[keep]
    n_after = len(src_u)
    print(f"  {tag}Filtered: {n_before:,} → {n_after:,} queries "
          f"({n_before - n_after:,} with unseen nodes removed)", flush=True)

    # ------------------------------------------------------------------
    # Step 3: build all_positives_per_src from ALL filtered edges
    # BEFORE subsampling — matches heuristics.py exactly.
    # ------------------------------------------------------------------
    all_positives_per_src: dict[int, set] = {}
    for s, d in zip(src_u, dst_u):
        all_positives_per_src.setdefault(int(s), set()).add(int(d))

    # ------------------------------------------------------------------
    # Step 4: subsample to max_queries if needed
    # Same seed offset (seed + 777) as heuristics.py.
    # ------------------------------------------------------------------
    n_total = len(src_u)
    if n_total > max_queries:
        rng_sub = np.random.RandomState(seed + 777)
        idx = rng_sub.choice(n_total, size=max_queries, replace=False)
        idx.sort()
        src_u = src_u[idx]
        dst_u = dst_u[idx]
        print(f"  {tag}Subsampled {n_total:,} → {max_queries:,} queries",
              flush=True)

    n_queries = len(src_u)
    print(f"  {tag}{n_queries:,} queries for ranking", flush=True)

    # ------------------------------------------------------------------
    # Step 5: sample negatives for each query (TGB-style)
    # seed = random_seed+10 (val) or random_seed+20 (test), same as sg_baselines.
    # Variable-length candidates per query (same as heuristics.py) to
    # handle edge cases where fewer than n_negatives can be sampled.
    # ------------------------------------------------------------------
    print(f"  {tag}Sampling negatives...", flush=True)
    rng = np.random.RandomState(seed)
    t0 = time.time()

    all_src_list   = []
    all_cand_list  = []
    query_offsets  = [0]  # start index of each query's candidates

    for q in range(n_queries):
        s      = int(src_u[q])
        d_true = int(dst_u[q])
        negatives = sample_negatives_for_eval(
            s, d_true,
            train_neighbors,
            all_positives_per_src.get(s, set()),
            node_mapping,           # active_nodes = train nodes only
            n_negatives,
            rng,
        )
        # Layout: [dst_true, neg_1, ..., neg_K]
        candidates = np.concatenate([[d_true], negatives])
        n_cand = len(candidates)
        all_src_list.extend([s] * n_cand)
        all_cand_list.extend(candidates.tolist())
        query_offsets.append(query_offsets[-1] + n_cand)

    all_src  = np.array(all_src_list,  dtype=np.int64)
    all_cand = np.array(all_cand_list, dtype=np.int64)
    print(f"  {tag}Neg sampling done, {len(all_src):,} pairs ({time.time()-t0:.1f}s)",
          flush=True)

    # ------------------------------------------------------------------
    # Step 6: compute features for all (src, cand) pairs
    # Uses train adjacency only (no leakage).
    # ------------------------------------------------------------------
    t0 = time.time()
    all_features = compute_features_batch(
        all_src, all_cand,
        node_mapping, adj_undir, adj_dir,
        deg_undir, w_undir, w_dir,
        batch_size=feature_batch,
    )
    # Apply feature column selection — must match what was used during training
    if active_feature_indices:
        all_features = all_features[:, active_feature_indices]
    print(f"  {tag}Features computed ({time.time()-t0:.1f}s)", flush=True)

    # ------------------------------------------------------------------
    # Step 7: score with MLP in batches
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
    # Step 8: compute ranks
    # rank = count(score > true_score) + 1  (conservative: ties broken low)
    # IDENTICAL to heuristics.py line 191.
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

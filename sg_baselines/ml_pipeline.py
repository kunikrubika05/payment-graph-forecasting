"""ML baseline pipeline for temporal link prediction on stream graph.

Models: LogReg, CatBoost, RandomForest.
Features: 34 per pair (15 src + 15 dst + 4 pair CN/AA).

Protocol (NO data leakage):
1. Train classifier on TRAIN edges (positive + negative pairs).
2. HP search: grid search, select by ranking MRR on VAL set (capped at 5K queries).
3. Retrain best HP on full TRAIN set.
4. Final evaluation on TEST set (TGB-style ranking, capped at 50K queries).
"""

import itertools
import time
from typing import Any

import numpy as np
import pandas as pd
from scipy import sparse
from tqdm import tqdm

from sg_baselines.config import ExperimentConfig, HP_GRIDS
from sg_baselines.features import build_pair_features, N_TOTAL_FEATURES
from sg_baselines.sampling import (
    sample_negatives_for_eval,
    sample_negatives_for_training,
)
from src.baselines.evaluation import compute_ranking_metrics


def prepare_training_data(
    train_edges: pd.DataFrame,
    node_idx: np.ndarray,
    node_features: np.ndarray,
    node_mapping: np.ndarray,
    adj_directed: sparse.csr_matrix,
    adj_undirected: sparse.csr_matrix,
    train_neighbors: dict[int, set[int]],
    active_nodes: np.ndarray,
    config: ExperimentConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """Prepare (X, y) training data with positive and negative samples."""
    src = train_edges["src_idx"].values
    dst = train_edges["dst_idx"].values

    unique_edges = pd.DataFrame({"src": src, "dst": dst}).drop_duplicates()
    src_pos = unique_edges["src"].values.astype(np.int64)
    dst_pos = unique_edges["dst"].values.astype(np.int64)
    n_unique = len(src_pos)

    max_pos = config.max_train_samples // (1 + config.negative_ratio)
    if n_unique > max_pos:
        rng = np.random.RandomState(config.random_seed)
        idx = rng.choice(n_unique, size=max_pos, replace=False)
        idx.sort()
        src_pos = src_pos[idx]
        dst_pos = dst_pos[idx]
        print(f"  Subsampled train positives: {n_unique:,} -> {max_pos:,}", flush=True)

    print(f"  Sampling negatives for {len(src_pos):,} positives...", flush=True)
    t0 = time.time()
    rng = np.random.RandomState(config.random_seed + 1)
    all_src, all_dst, all_labels = sample_negatives_for_training(
        src_pos, dst_pos, train_neighbors, active_nodes,
        config.negative_ratio, rng,
    )
    print(f"  Sampled {len(all_labels):,} pairs "
          f"(pos={np.sum(all_labels == 1):,}, neg={np.sum(all_labels == 0):,}) "
          f"in {time.time() - t0:.1f}s", flush=True)

    print(f"  Building pair features for {len(all_labels):,} pairs...", flush=True)
    t0 = time.time()

    batch_size = 200_000
    n_total = len(all_src)
    X = np.empty((n_total, N_TOTAL_FEATURES), dtype=np.float32)

    for start in tqdm(range(0, n_total, batch_size),
                      desc="  features"):
        end = min(start + batch_size, n_total)
        X[start:end] = build_pair_features(
            all_src[start:end], all_dst[start:end],
            node_idx, node_features,
            node_mapping, adj_directed, adj_undirected,
        )

    print(f"  Built features: {X.shape} ({time.time() - t0:.1f}s)", flush=True)
    return X, all_labels


def hp_search(
    model_name: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    val_edges: pd.DataFrame,
    node_idx: np.ndarray,
    node_features: np.ndarray,
    node_mapping: np.ndarray,
    adj_directed: sparse.csr_matrix,
    adj_undirected: sparse.csr_matrix,
    train_neighbors: dict[int, set[int]],
    active_nodes: np.ndarray,
    config: ExperimentConfig,
) -> tuple[dict, list[dict]]:
    """Grid search for best hyperparameters by val ranking MRR.

    Val eval capped at 5K queries per HP combo for speed.
    """
    grid = HP_GRIDS[model_name]
    keys = list(grid.keys())
    values = list(grid.values())
    all_combos = [dict(zip(keys, combo)) for combo in itertools.product(*values)]

    hp_max = config.hp_search_max_samples
    if len(X_train) > hp_max:
        rng_hp = np.random.RandomState(config.random_seed + 100)
        idx = rng_hp.choice(len(X_train), size=hp_max, replace=False)
        X_hp = X_train[idx]
        y_hp = y_train[idx]
        print(f"  HP search: subsampled train {len(X_train):,} -> {hp_max:,}", flush=True)
    else:
        X_hp = X_train
        y_hp = y_train

    print(f"  Pre-building val candidate features for HP search (5K queries)...",
          flush=True)
    val_data = _prebuild_eval_candidates(
        val_edges, node_idx, node_features, node_mapping,
        adj_directed, adj_undirected, train_neighbors, active_nodes,
        config, seed=config.random_seed + 200, max_queries=5000,
    )

    print(f"  HP search for {model_name}: {len(all_combos)} combinations", flush=True)
    results = []
    best_mrr = -1.0
    best_params = all_combos[0]

    for i, params in enumerate(all_combos):
        t0 = time.time()
        model = _create_model(model_name, params, config.random_seed)
        model.fit(X_hp, y_hp)

        mrr = _score_prebuilt(model, val_data)
        elapsed = time.time() - t0
        results.append({"params": params, "mrr": mrr})
        print(f"    [{i+1}/{len(all_combos)}] {params} -> val MRR={mrr:.4f} ({elapsed:.1f}s)",
              flush=True)

        if mrr > best_mrr:
            best_mrr = mrr
            best_params = params

    print(f"  Best HP: {best_params} (val MRR={best_mrr:.4f})", flush=True)
    return best_params, results


def train_and_evaluate(
    model_name: str,
    best_params: dict,
    X_train: np.ndarray,
    y_train: np.ndarray,
    val_edges: pd.DataFrame,
    test_edges: pd.DataFrame,
    node_idx: np.ndarray,
    node_features: np.ndarray,
    node_mapping: np.ndarray,
    adj_directed: sparse.csr_matrix,
    adj_undirected: sparse.csr_matrix,
    train_neighbors: dict[int, set[int]],
    active_nodes: np.ndarray,
    config: ExperimentConfig,
) -> dict[str, dict]:
    """Train model with best HP on full train, evaluate on val and test."""
    print(f"  Training {model_name} with best HP on full train set...", flush=True)
    t0 = time.time()
    model = _create_model(model_name, best_params, config.random_seed)
    model.fit(X_train, y_train)
    print(f"  Trained in {time.time() - t0:.1f}s", flush=True)

    results = {}
    for split_name, edges in [("val", val_edges), ("test", test_edges)]:
        seed = config.random_seed + (300 if split_name == "val" else 400)
        print(f"  [{split_name}] Building eval candidates (50K queries)...", flush=True)
        t0 = time.time()
        eval_data = _prebuild_eval_candidates(
            edges, node_idx, node_features, node_mapping,
            adj_directed, adj_undirected, train_neighbors, active_nodes,
            config, seed=seed, max_queries=50_000,
        )
        print(f"  [{split_name}] Built in {time.time() - t0:.1f}s", flush=True)

        t0 = time.time()
        ranks = _rank_prebuilt(model, eval_data)
        metrics = compute_ranking_metrics(ranks)
        elapsed = time.time() - t0
        print(f"  [{split_name}] {model_name}: MRR={metrics['mrr']:.4f}, "
              f"Hits@1={metrics['hits@1']:.4f}, "
              f"Hits@10={metrics['hits@10']:.4f} ({elapsed:.1f}s)", flush=True)
        results[split_name] = metrics

    return results


def _prebuild_eval_candidates(
    eval_edges: pd.DataFrame,
    node_idx: np.ndarray,
    node_features: np.ndarray,
    node_mapping: np.ndarray,
    adj_directed: sparse.csr_matrix,
    adj_undirected: sparse.csr_matrix,
    train_neighbors: dict[int, set[int]],
    active_nodes: np.ndarray,
    config: ExperimentConfig,
    seed: int,
    max_queries: int,
) -> dict:
    """Pre-build all candidate features for eval. Returns dict with X, offsets."""
    src_all = eval_edges["src_idx"].values
    dst_all = eval_edges["dst_idx"].values

    unique_edges = pd.DataFrame({"src": src_all, "dst": dst_all}).drop_duplicates()
    src_unique = unique_edges["src"].values.astype(np.int64)
    dst_unique = unique_edges["dst"].values.astype(np.int64)

    if len(src_unique) > max_queries:
        rng_sub = np.random.RandomState(seed + 777)
        idx = rng_sub.choice(len(src_unique), size=max_queries, replace=False)
        idx.sort()
        src_unique = src_unique[idx]
        dst_unique = dst_unique[idx]

    n_queries = len(src_unique)
    positives_per_src: dict[int, set[int]] = {}
    for s, d in zip(src_unique, dst_unique):
        positives_per_src.setdefault(int(s), set()).add(int(d))

    rng = np.random.RandomState(seed)
    all_src_flat = []
    all_dst_flat = []
    query_offsets = [0]

    for i in range(n_queries):
        s = int(src_unique[i])
        d_true = int(dst_unique[i])
        negatives = sample_negatives_for_eval(
            s, d_true, train_neighbors, positives_per_src.get(s, set()),
            active_nodes, config.n_negatives, rng,
        )
        candidates = np.concatenate([[d_true], negatives])
        all_src_flat.extend([s] * len(candidates))
        all_dst_flat.extend(candidates.tolist())
        query_offsets.append(query_offsets[-1] + len(candidates))

    all_src_arr = np.array(all_src_flat, dtype=np.int64)
    all_dst_arr = np.array(all_dst_flat, dtype=np.int64)
    total_pairs = len(all_src_arr)

    batch_size = 200_000
    X = np.empty((total_pairs, N_TOTAL_FEATURES), dtype=np.float32)
    for start in tqdm(range(0, total_pairs, batch_size),
                      desc=f"  eval_features({n_queries}q)"):
        end = min(start + batch_size, total_pairs)
        X[start:end] = build_pair_features(
            all_src_arr[start:end], all_dst_arr[start:end],
            node_idx, node_features,
            node_mapping, adj_directed, adj_undirected,
        )

    return {"X": X, "offsets": query_offsets, "n_queries": n_queries}


def _score_prebuilt(model, eval_data: dict) -> float:
    """Score pre-built candidates and return MRR."""
    ranks = _rank_prebuilt(model, eval_data)
    if len(ranks) == 0:
        return 0.0
    return float(np.mean(1.0 / ranks))


def _rank_prebuilt(model, eval_data: dict) -> np.ndarray:
    """Score pre-built candidates and return per-query ranks."""
    X = eval_data["X"]
    offsets = eval_data["offsets"]
    n_queries = eval_data["n_queries"]

    if hasattr(model, "predict_proba"):
        scores = model.predict_proba(X)[:, 1]
    else:
        scores = model.predict(X)

    ranks = np.empty(n_queries, dtype=np.float64)
    for q in range(n_queries):
        start = offsets[q]
        end = offsets[q + 1]
        q_scores = scores[start:end]
        true_score = q_scores[0]
        ranks[q] = float(np.sum(q_scores > true_score) + 1)

    return ranks


def _create_model(model_name: str, params: dict, seed: int) -> Any:
    """Create sklearn/catboost model with given hyperparameters."""
    if model_name == "logreg":
        from sklearn.linear_model import LogisticRegression
        return LogisticRegression(
            C=params["C"],
            penalty=params["penalty"],
            solver="liblinear",
            max_iter=1000,
            random_state=seed,
        )
    elif model_name == "catboost":
        from catboost import CatBoostClassifier
        return CatBoostClassifier(
            iterations=params["iterations"],
            depth=params["depth"],
            learning_rate=params["learning_rate"],
            random_seed=seed,
            verbose=0,
            thread_count=-1,
        )
    elif model_name == "rf":
        from sklearn.ensemble import RandomForestClassifier
        return RandomForestClassifier(
            n_estimators=params["n_estimators"],
            max_depth=params["max_depth"],
            min_samples_leaf=params["min_samples_leaf"],
            random_state=seed,
            n_jobs=-1,
        )
    else:
        raise ValueError(f"Unknown model: {model_name}")

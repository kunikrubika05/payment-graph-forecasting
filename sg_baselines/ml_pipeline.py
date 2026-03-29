"""ML baseline pipeline for temporal link prediction on stream graph.

Models: LogReg, CatBoost, RandomForest.
Features: 34 per pair (15 src + 15 dst + 4 pair CN/AA).

Protocol (NO data leakage):
1. Train classifier on TRAIN edges (positive + negative pairs).
2. HP search: grid search, select by ranking MRR on VAL set.
3. Retrain best HP on full TRAIN set.
4. Final evaluation on TEST set (TGB-style ranking).
"""

import itertools
import time
from typing import Any

import numpy as np
import pandas as pd
from scipy import sparse

from sg_baselines.config import ExperimentConfig, HP_GRIDS
from sg_baselines.features import build_pair_features
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
    """Prepare (X, y) training data with positive and negative samples.

    Subsamples positives if total would exceed max_train_samples.
    """
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
        print(f"  Subsampled train positives: {n_unique:,} -> {max_pos:,}")

    rng = np.random.RandomState(config.random_seed + 1)
    all_src, all_dst, all_labels = sample_negatives_for_training(
        src_pos, dst_pos, train_neighbors, active_nodes,
        config.negative_ratio, rng,
    )

    print(f"  Training samples: {len(all_labels):,} "
          f"(pos={np.sum(all_labels == 1):,}, neg={np.sum(all_labels == 0):,})")

    t0 = time.time()
    X = build_pair_features(
        all_src, all_dst, node_idx, node_features,
        node_mapping, adj_directed, adj_undirected,
    )
    print(f"  Built pair features: {X.shape} ({time.time() - t0:.1f}s)")

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

    HP are selected by MRR on val set (aligned with final metric).
    Val set is NEVER used for training — only for ranking evaluation.

    Returns:
        (best_params, list of {params, mrr} for all grid points).
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
        print(f"  HP search: subsampled train {len(X_train):,} -> {hp_max:,}")
    else:
        X_hp = X_train
        y_hp = y_train

    print(f"  HP search for {model_name}: {len(all_combos)} combinations")
    results = []
    best_mrr = -1.0
    best_params = all_combos[0]

    for i, params in enumerate(all_combos):
        t0 = time.time()
        model = _create_model(model_name, params, config.random_seed)
        model.fit(X_hp, y_hp)

        mrr = _evaluate_model_ranking(
            model, val_edges, node_idx, node_features,
            node_mapping, adj_directed, adj_undirected,
            train_neighbors, active_nodes, config,
            seed=config.random_seed + 200,
            max_queries=5000,
        )

        elapsed = time.time() - t0
        results.append({"params": params, "mrr": mrr})
        print(f"    [{i+1}/{len(all_combos)}] {params} -> val MRR={mrr:.4f} ({elapsed:.1f}s)")

        if mrr > best_mrr:
            best_mrr = mrr
            best_params = params

    print(f"  Best HP: {best_params} (val MRR={best_mrr:.4f})")
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
    """Train model with best HP on full train, evaluate on val and test.

    Returns:
        Dict with "val" and "test" ranking metrics.
    """
    print(f"  Training {model_name} with best HP on full train set...")
    t0 = time.time()
    model = _create_model(model_name, best_params, config.random_seed)
    model.fit(X_train, y_train)
    print(f"  Trained in {time.time() - t0:.1f}s")

    results = {}
    for split_name, edges in [("val", val_edges), ("test", test_edges)]:
        t0 = time.time()
        mrr = _evaluate_model_ranking(
            model, edges, node_idx, node_features,
            node_mapping, adj_directed, adj_undirected,
            train_neighbors, active_nodes, config,
            seed=config.random_seed + (300 if split_name == "val" else 400),
        )
        metrics = {"mrr": mrr}

        ranks = _evaluate_model_ranking_full(
            model, edges, node_idx, node_features,
            node_mapping, adj_directed, adj_undirected,
            train_neighbors, active_nodes, config,
            seed=config.random_seed + (300 if split_name == "val" else 400),
        )
        metrics = compute_ranking_metrics(ranks)
        elapsed = time.time() - t0
        print(f"  [{split_name}] {model_name}: MRR={metrics['mrr']:.4f}, "
              f"Hits@1={metrics['hits@1']:.4f}, "
              f"Hits@10={metrics['hits@10']:.4f} ({elapsed:.1f}s)")
        results[split_name] = metrics

    return results


def _evaluate_model_ranking(
    model,
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
    max_queries: int = 0,
) -> float:
    """Evaluate model ranking MRR on eval edges (for HP search)."""
    ranks = _evaluate_model_ranking_full(
        model, eval_edges, node_idx, node_features,
        node_mapping, adj_directed, adj_undirected,
        train_neighbors, active_nodes, config, seed, max_queries,
    )
    if len(ranks) == 0:
        return 0.0
    return float(np.mean(1.0 / ranks))


def _evaluate_model_ranking_full(
    model,
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
    max_queries: int = 0,
) -> np.ndarray:
    """Compute per-query ranks for all eval edges (TGB-style)."""
    src_all = eval_edges["src_idx"].values
    dst_all = eval_edges["dst_idx"].values

    unique_edges = pd.DataFrame({"src": src_all, "dst": dst_all}).drop_duplicates()
    src_unique = unique_edges["src"].values.astype(np.int64)
    dst_unique = unique_edges["dst"].values.astype(np.int64)

    if max_queries > 0 and len(src_unique) > max_queries:
        rng_sub = np.random.RandomState(seed + 999)
        idx = rng_sub.choice(len(src_unique), size=max_queries, replace=False)
        idx.sort()
        src_unique = src_unique[idx]
        dst_unique = dst_unique[idx]

    positives_per_src: dict[int, set[int]] = {}
    for s, d in zip(src_unique, dst_unique):
        positives_per_src.setdefault(int(s), set()).add(int(d))

    rng = np.random.RandomState(seed)
    ranks = []

    for i in range(len(src_unique)):
        s = int(src_unique[i])
        d_true = int(dst_unique[i])

        negatives = sample_negatives_for_eval(
            s, d_true, train_neighbors, positives_per_src.get(s, set()),
            active_nodes, config.n_negatives, rng,
        )

        candidates = np.concatenate([[d_true], negatives])
        src_rep = np.full(len(candidates), s, dtype=np.int64)

        X_cand = build_pair_features(
            src_rep, candidates, node_idx, node_features,
            node_mapping, adj_directed, adj_undirected,
        )

        if hasattr(model, "predict_proba"):
            scores = model.predict_proba(X_cand)[:, 1]
        else:
            scores = model.predict(X_cand)

        true_score = scores[0]
        rank = int(np.sum(scores > true_score)) + 1
        ranks.append(rank)

    return np.array(ranks, dtype=np.float64)


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

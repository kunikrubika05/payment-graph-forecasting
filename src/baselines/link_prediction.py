"""Link prediction baseline pipeline with TGB-style ranking evaluation.

Evaluation protocol (per TGB/DGB best practices):
- Per-source ranking: for each positive edge (s, d, t), fix source s,
  build candidate set {d_true} ∪ {neg_1, ..., neg_q}, rank candidates.
- Negatives: 50/50 mix of historical (edges from window absent in target day)
  and random (uniform from active nodes).
- Metrics: MRR, Hits@1, Hits@3, Hits@10 (per-query, averaged).
"""

import gc
import logging
import os
import time
from itertools import product as iter_product
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from src.baselines.config import (
    ExperimentConfig, NODE_FEATURE_COLUMNS,
    LOGREG_HP_GRID, CATBOOST_HP_GRID, RF_HP_GRID,
)
from src.baselines.data_loader import (
    get_available_dates, download_period_data, load_node_features,
    load_daily_snapshot, cleanup_period_data,
)
from src.baselines.feature_engineering import (
    aggregate_features_mean, aggregate_features_time_weighted,
    build_pair_features, compute_feature_correlations, get_feature_names,
)
from src.baselines.evaluation import compute_ranking_metrics
from src.baselines.experiment_logger import ExperimentLogger

logger = logging.getLogger(__name__)

EVAL_BATCH_SIZE = 5000


def _get_edges_set(snapshot: pd.DataFrame) -> Set[Tuple[int, int]]:
    """Extract set of directed edges from snapshot DataFrame."""
    if snapshot is None or len(snapshot) == 0:
        return set()
    return set(zip(snapshot["src_idx"].values, snapshot["dst_idx"].values))


def _get_active_nodes(snapshots: Dict[str, pd.DataFrame]) -> np.ndarray:
    """Get sorted array of all unique nodes from multiple snapshots."""
    nodes = set()
    for snap in snapshots.values():
        if snap is not None and len(snap) > 0:
            nodes.update(snap["src_idx"].values)
            nodes.update(snap["dst_idx"].values)
    return np.array(sorted(nodes))


def _get_source_neighbors(snapshots: Dict[str, pd.DataFrame]) -> Dict[int, Set[int]]:
    """Build per-source neighbor sets from window snapshots (historical edges)."""
    neighbors: Dict[int, Set[int]] = {}
    for snap in snapshots.values():
        if snap is None or len(snap) == 0:
            continue
        for s, d in zip(snap["src_idx"].values, snap["dst_idx"].values):
            if s not in neighbors:
                neighbors[s] = set()
            neighbors[s].add(d)
    return neighbors


def _sample_random_negatives(
    n_needed: int,
    active_nodes: np.ndarray,
    exclude: Set[int],
    rng: np.random.RandomState,
) -> List[int]:
    """Sample random negatives using randint + rejection (O(n_needed), not O(N))."""
    result = []
    n_active = len(active_nodes)
    if n_active == 0:
        return result

    max_attempts = n_needed * 10
    drawn = 0
    while len(result) < n_needed and drawn < max_attempts:
        batch_size = min((n_needed - len(result)) * 3, n_active)
        indices = rng.randint(0, n_active, size=batch_size)
        for idx in indices:
            node = active_nodes[idx]
            if node not in exclude:
                result.append(node)
                exclude.add(node)
                if len(result) >= n_needed:
                    break
        drawn += batch_size

    return result


def sample_negatives_per_source(
    source: int,
    true_dst: int,
    historical_neighbors: Set[int],
    target_edges_from_source: Set[int],
    active_nodes: np.ndarray,
    n_negatives: int,
    rng: np.random.RandomState,
) -> np.ndarray:
    """Sample negatives for one source node (TGB-style 50/50 mix).

    Args:
        source: Source node index.
        true_dst: True destination (excluded from negatives).
        historical_neighbors: Destinations this source connected to in the window.
        target_edges_from_source: Destinations this source connects to on target day.
        active_nodes: All active nodes in the window.
        n_negatives: Number of negative destinations to sample.
        rng: Random state.

    Returns:
        Array of negative destination node indices.
    """
    n_hist = n_negatives // 2
    n_rand = n_negatives - n_hist

    hist_candidates = historical_neighbors - target_edges_from_source - {source, true_dst}

    hist_negatives = []
    if hist_candidates:
        hist_arr = np.array(list(hist_candidates), dtype=np.int64)
        if len(hist_arr) >= n_hist:
            idx = rng.choice(len(hist_arr), n_hist, replace=False)
            hist_negatives = hist_arr[idx].tolist()
        else:
            hist_negatives = hist_arr.tolist()
            n_rand += n_hist - len(hist_negatives)

    exclude = target_edges_from_source | set(hist_negatives) | {source, true_dst}
    rand_negatives = _sample_random_negatives(n_rand, active_nodes, exclude, rng)

    all_negatives = hist_negatives + rand_negatives
    if not all_negatives:
        return np.array([], dtype=np.int64)
    return np.array(all_negatives, dtype=np.int64)


def evaluate_ranking_for_day(
    model,
    model_name: str,
    target_snapshot: pd.DataFrame,
    node_features_agg: pd.DataFrame,
    historical_neighbors: Dict[int, Set[int]],
    active_nodes: np.ndarray,
    config: ExperimentConfig,
    seed: int,
) -> Tuple[np.ndarray, int]:
    """Evaluate model on one day using per-source ranking protocol (batched).

    Processes edges in batches of EVAL_BATCH_SIZE to avoid O(N) per-edge overhead.
    All edges are evaluated (no subsampling), per TGB standard.

    Args:
        model: Trained model.
        model_name: Model type string.
        target_snapshot: Edge list for the target day.
        node_features_agg: Aggregated node features from the window.
        historical_neighbors: Per-source historical neighbor sets.
        active_nodes: Array of active nodes in the window.
        config: Experiment configuration.
        seed: Random seed.

    Returns:
        Tuple of (ranks_array, n_skipped).
    """
    rng = np.random.RandomState(seed)
    target_edges = _get_edges_set(target_snapshot)
    known_nodes = set(node_features_agg.index)

    queries = []
    n_skipped = 0

    src_to_dsts: Dict[int, Set[int]] = {}
    for s, d in target_edges:
        src_to_dsts.setdefault(s, set()).add(d)

    for source, destinations in src_to_dsts.items():
        if source not in known_nodes:
            n_skipped += len(destinations)
            continue

        hist_nbrs = historical_neighbors.get(source, set())
        hist_candidates = list(hist_nbrs - destinations - {source})

        for true_dst in destinations:
            if true_dst not in known_nodes:
                n_skipped += 1
                continue

            n_hist = min(config.n_negatives // 2, len(hist_candidates))
            n_rand = config.n_negatives - n_hist

            negatives = []
            if n_hist > 0:
                idx = rng.choice(len(hist_candidates), n_hist, replace=False)
                negatives = [hist_candidates[i] for i in idx]

            exclude = destinations | set(negatives) | {source, true_dst}
            rand_neg = _sample_random_negatives(n_rand, active_nodes, exclude, rng)
            negatives.extend(rand_neg)

            valid_neg = [d for d in negatives if d in known_nodes]
            if not valid_neg:
                n_skipped += 1
                continue

            queries.append((source, true_dst, valid_neg))

    if not queries:
        return np.array([], dtype=np.int64), n_skipped

    ranks = []
    n_candidates = 1 + config.n_negatives

    for batch_start in range(0, len(queries), EVAL_BATCH_SIZE):
        batch = queries[batch_start:batch_start + EVAL_BATCH_SIZE]

        all_src = []
        all_dst = []
        query_sizes = []

        for source, true_dst, neg_dsts in batch:
            candidates = [true_dst] + neg_dsts
            all_src.extend([source] * len(candidates))
            all_dst.extend(candidates)
            query_sizes.append(len(candidates))

        src_arr = np.array(all_src, dtype=np.int64)
        dst_arr = np.array(all_dst, dtype=np.int64)

        X, _ = build_pair_features(
            node_features_agg, src_arr, dst_arr, mode=config.feature_mode
        )

        scores = _predict_proba(model, model_name, X)

        offset = 0
        for q_size in query_sizes:
            q_scores = scores[offset:offset + q_size]
            true_score = q_scores[0]
            rank = 1 + int(np.sum(q_scores[1:] > true_score))
            ranks.append(rank)
            offset += q_size

    return np.array(ranks, dtype=np.int64), n_skipped


def prepare_training_samples(
    target_snapshot: pd.DataFrame,
    node_features_agg: pd.DataFrame,
    historical_neighbors: Dict[int, Set[int]],
    active_nodes: np.ndarray,
    config: ExperimentConfig,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Prepare training samples for one target day (binary classification).

    Vectorized per-source batch sampling: for each source, negatives are
    sampled once for all its edges (not per-edge), using 50/50 hist+random mix.

    Args:
        target_snapshot: Edge list for the target day.
        node_features_agg: Aggregated node features from the window.
        historical_neighbors: Per-source historical neighbor sets.
        active_nodes: Array of active nodes in the window.
        config: Experiment configuration.
        seed: Random seed.

    Returns:
        Tuple of (X, y).
    """
    rng = np.random.RandomState(seed)
    target_edges = _get_edges_set(target_snapshot)
    known_nodes = set(node_features_agg.index)

    src_to_dsts: Dict[int, List[int]] = {}
    for s, d in target_edges:
        if s in known_nodes and d in known_nodes:
            src_to_dsts.setdefault(s, []).append(d)

    all_src = []
    all_dst = []
    all_labels = []

    neg_per_positive = config.negative_ratio

    for source, destinations in src_to_dsts.items():
        dst_set = set(destinations)
        n_total_neg = len(destinations) * neg_per_positive

        hist_nbrs = historical_neighbors.get(source, set())
        hist_candidates = list((hist_nbrs - dst_set - {source}) & known_nodes)

        n_hist = min(n_total_neg // 2, len(hist_candidates))
        n_rand = n_total_neg - n_hist

        negatives = []
        if n_hist > 0:
            replace = len(hist_candidates) < n_hist
            idx = rng.choice(len(hist_candidates), n_hist, replace=replace)
            negatives = [hist_candidates[i] for i in idx]

        exclude = dst_set | set(negatives) | {source}
        rand_neg = _sample_random_negatives(n_rand, active_nodes, exclude, rng)
        negatives.extend(rand_neg)

        for true_dst in destinations:
            all_src.append(source)
            all_dst.append(true_dst)
            all_labels.append(1)

        for neg_d in negatives[:n_total_neg]:
            all_src.append(source)
            all_dst.append(neg_d)
            all_labels.append(0)

    if not all_src:
        n_feat = len(get_feature_names(config.feature_mode))
        return np.empty((0, n_feat)), np.array([])

    src_arr = np.array(all_src, dtype=np.int64)
    dst_arr = np.array(all_dst, dtype=np.int64)
    y = np.array(all_labels, dtype=np.float64)

    X, _ = build_pair_features(
        node_features_agg, src_arr, dst_arr, mode=config.feature_mode
    )

    return X, y


def _load_window_data(
    target_date: str,
    all_dates: List[str],
    config: ExperimentConfig,
) -> Tuple[Optional[pd.DataFrame], Dict[int, Set[int]], np.ndarray, Optional[pd.DataFrame]]:
    """Load and aggregate window data for a target date.

    Returns:
        Tuple of (node_feat_agg, historical_neighbors, active_nodes, target_snap).
        node_feat_agg and target_snap may be None if data is unavailable.
    """
    target_idx = all_dates.index(target_date)
    window_start = max(0, target_idx - config.window_size)
    window_dates = all_dates[window_start:target_idx]

    features_by_date = {}
    window_snapshots = {}
    for d in window_dates:
        nf = load_node_features(d, config.local_data_dir)
        if nf is not None:
            features_by_date[d] = nf
        snap = load_daily_snapshot(d, config.local_data_dir)
        if snap is not None:
            window_snapshots[d] = snap

    if not features_by_date:
        return None, {}, np.array([]), None

    if config.aggregation == "mean":
        node_feat_agg = aggregate_features_mean(features_by_date)
    else:
        node_feat_agg = aggregate_features_time_weighted(
            features_by_date, window_dates, config.decay_lambda
        )

    active_nodes = _get_active_nodes(window_snapshots)
    historical_neighbors = _get_source_neighbors(window_snapshots)

    target_snap = load_daily_snapshot(target_date, config.local_data_dir)

    return node_feat_agg, historical_neighbors, active_nodes, target_snap


def _collect_training_data(
    dates: List[str],
    all_dates: List[str],
    config: ExperimentConfig,
) -> Tuple[np.ndarray, np.ndarray]:
    """Collect training data across multiple days."""
    X_list = []
    y_list = []
    total_samples = 0

    for i, target_date in enumerate(dates):
        t0 = time.time()
        node_feat_agg, hist_nbrs, active_nodes, target_snap = _load_window_data(
            target_date, all_dates, config
        )
        if node_feat_agg is None or target_snap is None or len(target_snap) == 0:
            continue

        seed = config.random_seed + hash(target_date) % 10000
        X, y = prepare_training_samples(
            target_snap, node_feat_agg, hist_nbrs, active_nodes, config, seed
        )

        if len(y) == 0:
            continue

        X_list.append(X)
        y_list.append(y)
        total_samples += len(y)
        elapsed = time.time() - t0

        logger.info(
            "  [%d/%d] %s: %d samples (%.1f sec, cumulative=%d)",
            i + 1, len(dates), target_date, len(y), elapsed, total_samples,
        )

        del node_feat_agg, hist_nbrs
        gc.collect()

        if total_samples >= config.max_train_samples:
            break

    if not X_list:
        n_feat = len(get_feature_names(config.feature_mode))
        return np.empty((0, n_feat)), np.array([])

    X_all = np.vstack(X_list)
    y_all = np.concatenate(y_list)

    if len(y_all) > config.max_train_samples:
        rng = np.random.RandomState(config.random_seed)
        indices = rng.choice(len(y_all), size=config.max_train_samples, replace=False)
        X_all = X_all[indices]
        y_all = y_all[indices]
        logger.info("Subsampled to %d samples", config.max_train_samples)

    return X_all, y_all


def hp_search(
    model_name: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
) -> Tuple[dict, List[dict]]:
    """Run HP grid search. Uses PR-AUC on val set to select best params.

    Returns:
        Tuple of (best_params, all_results).
    """
    from sklearn.metrics import average_precision_score

    if model_name == "logreg":
        grid = LOGREG_HP_GRID
    elif model_name == "catboost":
        grid = CATBOOST_HP_GRID
    elif model_name == "rf":
        grid = RF_HP_GRID
    else:
        raise ValueError(f"Unknown model: {model_name}")

    param_combos = [
        dict(zip(grid.keys(), v)) for v in iter_product(*grid.values())
    ]

    results = []
    best_score = -1.0
    best_params = {}

    for params in param_combos:
        try:
            t0 = time.time()
            model = _create_model(model_name, params)
            model = _fit_model(model, model_name, X_train, y_train)
            train_time = time.time() - t0

            y_proba_val = _predict_proba(model, model_name, X_val)
            score = float(average_precision_score(y_val, y_proba_val))

            result = {
                "model": model_name,
                "params": params,
                "val_pr_auc": score,
                "train_time_sec": train_time,
            }
            results.append(result)

            if score > best_score:
                best_score = score
                best_params = params

            logger.info(
                "  %s %s -> val PR-AUC=%.4f (%.1fs)",
                model_name, params, score, train_time,
            )

            del model
            gc.collect()
        except Exception as e:
            logger.warning("  %s %s failed: %s", model_name, params, e)
            results.append({"model": model_name, "params": params, "error": str(e)})

    return best_params, results


def _create_model(model_name: str, params: dict):
    """Create a model instance with given hyperparameters."""
    if model_name == "logreg":
        l1_ratio = 1.0 if params["penalty"] == "l1" else 0.0
        solver = "saga" if params["penalty"] == "l1" else "lbfgs"
        return LogisticRegression(
            C=params["C"],
            l1_ratio=l1_ratio,
            solver=solver,
            class_weight="balanced",
            max_iter=300,
            random_state=42,
        )
    elif model_name == "catboost":
        from catboost import CatBoostClassifier
        return CatBoostClassifier(
            iterations=params["iterations"],
            depth=params["depth"],
            learning_rate=params["learning_rate"],
            auto_class_weights="Balanced",
            task_type="CPU",
            verbose=0,
            random_seed=42,
            thread_count=1,
        )
    elif model_name == "rf":
        return RandomForestClassifier(
            n_estimators=params["n_estimators"],
            max_depth=params["max_depth"],
            min_samples_leaf=params["min_samples_leaf"],
            class_weight="balanced_subsample",
            random_state=42,
            n_jobs=1,
        )
    raise ValueError(f"Unknown model: {model_name}")


def _fit_model(model, model_name: str, X: np.ndarray, y: np.ndarray):
    """Fit model on training data."""
    if model_name == "catboost":
        from catboost import Pool
        pool = Pool(X, y)
        model.fit(pool)
    else:
        model.fit(X, y)
    return model


def _predict_proba(model, model_name: str, X: np.ndarray) -> np.ndarray:
    """Get predicted probabilities for positive class."""
    return model.predict_proba(X)[:, 1]


def get_feature_importance(
    model, model_name: str, feature_names: List[str]
) -> dict:
    """Extract feature importance from trained model."""
    if model_name == "logreg":
        coefs = np.abs(model.coef_[0])
        importance = dict(zip(feature_names, coefs.tolist()))
    elif model_name == "catboost":
        imp = model.get_feature_importance()
        importance = dict(zip(feature_names, imp.tolist()))
    elif model_name == "rf":
        imp = model.feature_importances_
        importance = dict(zip(feature_names, imp.tolist()))
    else:
        importance = {}
    return {model_name: importance}


def run_link_prediction_mode_a(config: ExperimentConfig, token: str) -> None:
    """Run link prediction Mode A: train on train days, HP search on val, eval on test.

    Evaluation uses per-source ranking protocol (TGB-style).
    """
    output_dir = os.path.join(
        config.output_dir or f"/tmp/baseline_results/{config.experiment_name}",
        config.sub_experiment,
    )
    exp_logger = ExperimentLogger(output_dir)

    if exp_logger.is_completed():
        logger.info("Experiment already completed: %s", output_dir)
        exp_logger.close()
        return

    exp_logger.log_config(config.to_dict())

    all_dates = get_available_dates(config.period_start, config.period_end)
    if len(all_dates) < config.window_size + 3:
        logger.error("Not enough dates in period: %d", len(all_dates))
        exp_logger.close()
        return

    prediction_dates = all_dates[config.window_size:]
    n_pred = len(prediction_dates)
    n_train = int(n_pred * config.train_ratio)
    n_val = int(n_pred * config.val_ratio)

    train_dates = prediction_dates[:n_train]
    val_dates = prediction_dates[n_train:n_train + n_val]
    test_dates = prediction_dates[n_train + n_val:]

    logger.info(
        "Period %s: %d total dates, %d prediction dates "
        "(train=%d, val=%d, test=%d), window=%d",
        config.period_name, len(all_dates), n_pred,
        len(train_dates), len(val_dates), len(test_dates), config.window_size,
    )

    logger.info("Downloading data from Yandex.Disk...")
    download_period_data(all_dates, config.local_data_dir, token)

    logger.info("Collecting training samples (per-source negatives)...")
    X_train, y_train = _collect_training_data(train_dates, all_dates, config)
    if len(y_train) == 0:
        logger.error("No training samples collected, aborting")
        exp_logger.close()
        return

    pos_count = int(y_train.sum())
    logger.info("Training set: %d samples (%d pos, %d neg, ratio=%.3f)",
                len(y_train), pos_count, len(y_train) - pos_count,
                pos_count / len(y_train))

    logger.info("Collecting validation samples...")
    X_val, y_val = _collect_training_data(val_dates, all_dates, config)
    logger.info("Validation set: %d samples", len(y_val))

    feature_names = get_feature_names(config.feature_mode)
    if len(X_train) > 0:
        corr_df = compute_feature_correlations(
            X_train[:min(100000, len(X_train))], feature_names
        )
        exp_logger.log_feature_correlations(corr_df)
        exp_logger.log_high_correlations(corr_df)

    hp_max = config.hp_search_max_samples
    X_train_hp = X_train
    y_train_hp = y_train
    if len(X_train) > hp_max:
        rng_hp = np.random.RandomState(config.random_seed + 999)
        hp_idx = rng_hp.choice(len(X_train), hp_max, replace=False)
        X_train_hp = X_train[hp_idx]
        y_train_hp = y_train[hp_idx]
        logger.info("HP search subsample: %d / %d", hp_max, len(X_train))

    X_val_hp = X_val
    y_val_hp = y_val
    if len(X_val) > hp_max:
        rng_hp2 = np.random.RandomState(config.random_seed + 998)
        hp_idx2 = rng_hp2.choice(len(X_val), hp_max, replace=False)
        X_val_hp = X_val[hp_idx2]
        y_val_hp = y_val[hp_idx2]

    all_hp_results = []
    best_models = {}
    all_importance = {}

    for model_name in config.models:
        logger.info("=== HP search for %s ===", model_name)

        if len(y_val_hp) > 0:
            best_params, hp_results = hp_search(
                model_name, X_train_hp, y_train_hp, X_val_hp, y_val_hp
            )
        else:
            best_params = _default_params(model_name)
            hp_results = [{"model": model_name, "params": best_params, "note": "no val data"}]

        all_hp_results.extend(hp_results)

        logger.info("Training final %s on full %d samples with best params %s",
                     model_name, len(X_train), best_params)
        best_model = _create_model(model_name, best_params)
        best_model = _fit_model(best_model, model_name, X_train, y_train)

        best_models[model_name] = (best_model, best_params)
        importance = get_feature_importance(best_model, model_name, feature_names)
        all_importance.update(importance)

        ext = ".cbm" if model_name == "catboost" else ".pkl"
        exp_logger.save_model(best_model, f"best_{model_name}{ext}")

    del X_train_hp, y_train_hp, X_val_hp, y_val_hp
    exp_logger.log_hp_search(all_hp_results)
    exp_logger.log_feature_importance(all_importance)

    del X_train, y_train, X_val, y_val
    gc.collect()

    logger.info("=== Ranking evaluation on test days ===")
    all_test_ranks = {m: [] for m in best_models}

    for day_idx, target_date in enumerate(test_dates):
        t0 = time.time()
        node_feat_agg, hist_nbrs, active_nodes, target_snap = _load_window_data(
            target_date, all_dates, config
        )
        if node_feat_agg is None or target_snap is None or len(target_snap) == 0:
            continue

        seed = config.random_seed + hash(target_date) % 10000

        for model_name, (model, params) in best_models.items():
            ranks, n_skipped = evaluate_ranking_for_day(
                model, model_name, target_snap, node_feat_agg,
                hist_nbrs, active_nodes, config, seed,
            )

            if len(ranks) > 0:
                day_ranking = compute_ranking_metrics(ranks)
                day_ranking["model"] = model_name
                day_ranking["date"] = target_date
                day_ranking["n_skipped"] = n_skipped
                day_ranking["best_params"] = params
                exp_logger.log_metrics(day_ranking)
                all_test_ranks[model_name].extend(ranks.tolist())

                elapsed = time.time() - t0
                logger.info(
                    "  [%d/%d] %s %s: MRR=%.4f Hits@1=%.3f Hits@10=%.3f "
                    "(%d queries, %d skipped, %.1fs)",
                    day_idx + 1, len(test_dates), target_date, model_name,
                    day_ranking["mrr"], day_ranking["hits@1"], day_ranking["hits@10"],
                    day_ranking["n_queries"], n_skipped, elapsed,
                )

        del node_feat_agg, hist_nbrs
        gc.collect()

    summary = {"config": config.to_dict(), "models": {}}
    for model_name, ranks_list in all_test_ranks.items():
        if not ranks_list:
            continue
        all_ranks = np.array(ranks_list)
        model_summary = compute_ranking_metrics(all_ranks)
        model_summary["best_params"] = best_models[model_name][1]
        summary["models"][model_name] = model_summary
    exp_logger.write_summary(summary)

    remote_dir = f"{config.yadisk_experiments_base}/{config.experiment_name}/{config.sub_experiment}"
    exp_logger.upload_to_yadisk(remote_dir, token)
    exp_logger.close()

    logger.info("=== Experiment complete: %s/%s ===", config.experiment_name, config.sub_experiment)


def run_link_prediction_mode_b(config: ExperimentConfig, token: str) -> None:
    """Run link prediction Mode B: live-update with periodic retrain.

    Evaluation uses per-source ranking protocol.
    """
    output_dir = os.path.join(
        config.output_dir or f"/tmp/baseline_results/{config.experiment_name}",
        config.sub_experiment,
    )
    exp_logger = ExperimentLogger(output_dir)

    if exp_logger.is_completed():
        logger.info("Experiment already completed: %s", output_dir)
        exp_logger.close()
        return

    exp_logger.log_config(config.to_dict())

    all_dates = get_available_dates(config.period_start, config.period_end)
    if len(all_dates) < config.window_size + 3:
        logger.error("Not enough dates in period: %d", len(all_dates))
        exp_logger.close()
        return

    prediction_dates = all_dates[config.window_size:]
    n_pred = len(prediction_dates)
    n_warmup = max(3, int(n_pred * 0.2))

    warmup_dates = prediction_dates[:n_warmup]
    eval_dates = prediction_dates[n_warmup:]

    logger.info("Mode B: %d warmup days, %d eval days", len(warmup_dates), len(eval_dates))

    logger.info("Downloading data from Yandex.Disk...")
    download_period_data(all_dates, config.local_data_dir, token)

    logger.info("Collecting warmup samples...")
    X_cumulative, y_cumulative = _collect_training_data(
        warmup_dates, all_dates, config
    )
    if len(y_cumulative) == 0:
        logger.error("No warmup samples, aborting")
        exp_logger.close()
        return

    current_models = {}
    for model_name in config.models:
        params = _default_params(model_name)
        model = _create_model(model_name, params)
        model = _fit_model(model, model_name, X_cumulative, y_cumulative)
        current_models[model_name] = (model, params)

    all_test_ranks = {m: [] for m in config.models}
    days_since_retrain = 0

    for eval_idx, target_date in enumerate(eval_dates):
        t0 = time.time()
        node_feat_agg, hist_nbrs, active_nodes, target_snap = _load_window_data(
            target_date, all_dates, config
        )
        if node_feat_agg is None or target_snap is None or len(target_snap) == 0:
            continue

        seed = config.random_seed + hash(target_date) % 10000

        for model_name, (model, params) in current_models.items():
            ranks, n_skipped = evaluate_ranking_for_day(
                model, model_name, target_snap, node_feat_agg,
                hist_nbrs, active_nodes, config, seed,
            )
            if len(ranks) > 0:
                day_ranking = compute_ranking_metrics(ranks)
                day_ranking["model"] = model_name
                day_ranking["date"] = target_date
                day_ranking["n_skipped"] = n_skipped
                day_ranking["cumulative_train_size"] = len(y_cumulative)
                exp_logger.log_metrics(day_ranking)
                all_test_ranks[model_name].extend(ranks.tolist())

        X_day, y_day = prepare_training_samples(
            target_snap, node_feat_agg, hist_nbrs, active_nodes, config, seed
        )

        if len(y_day) > 0:
            X_cumulative = np.vstack([X_cumulative, X_day])
            y_cumulative = np.concatenate([y_cumulative, y_day])

            if len(y_cumulative) > config.max_train_samples:
                rng = np.random.RandomState(config.random_seed)
                keep = rng.choice(len(y_cumulative), config.max_train_samples, replace=False)
                X_cumulative = X_cumulative[keep]
                y_cumulative = y_cumulative[keep]

        days_since_retrain += 1
        should_retrain = (
            days_since_retrain >= config.retrain_interval
            or eval_idx == len(eval_dates) - 1
        )

        if should_retrain:
            for model_name in config.models:
                params = current_models[model_name][1]
                model = _create_model(model_name, params)
                model = _fit_model(model, model_name, X_cumulative, y_cumulative)
                current_models[model_name] = (model, params)
            days_since_retrain = 0

        elapsed = time.time() - t0
        del node_feat_agg, hist_nbrs
        gc.collect()

        logger.info(
            "  [%d/%d] %s (cumulative=%d, retrained=%s, %.1fs): %s",
            eval_idx + 1, len(eval_dates),
            target_date, len(y_cumulative),
            "yes" if should_retrain else "no", elapsed,
            {m: f"MRR={compute_ranking_metrics(np.array(r))['mrr']:.4f}"
             for m, r in all_test_ranks.items() if r},
        )

    for model_name, (model, params) in current_models.items():
        ext = ".cbm" if model_name == "catboost" else ".pkl"
        exp_logger.save_model(model, f"final_{model_name}{ext}")

    summary = {"config": config.to_dict(), "mode": "B", "models": {}}
    for model_name, ranks_list in all_test_ranks.items():
        if not ranks_list:
            continue
        all_ranks = np.array(ranks_list)
        model_summary = compute_ranking_metrics(all_ranks)
        model_summary["n_eval_days"] = len(eval_dates)
        summary["models"][model_name] = model_summary
    exp_logger.write_summary(summary)

    remote_dir = f"{config.yadisk_experiments_base}/{config.experiment_name}/{config.sub_experiment}"
    exp_logger.upload_to_yadisk(remote_dir, token)
    exp_logger.close()

    logger.info("=== Mode B complete: %s/%s ===", config.experiment_name, config.sub_experiment)


def run_link_prediction(config: ExperimentConfig, token: str) -> None:
    """Main entry point for link prediction experiment."""
    if config.mode == "A":
        run_link_prediction_mode_a(config, token)
    elif config.mode == "B":
        run_link_prediction_mode_b(config, token)
    else:
        raise ValueError(f"Unknown mode: {config.mode}")


def _default_params(model_name: str) -> dict:
    """Get default hyperparameters for a model."""
    if model_name == "logreg":
        return {"C": 1.0, "penalty": "l2"}
    elif model_name == "catboost":
        return {"iterations": 300, "depth": 6, "learning_rate": 0.05}
    elif model_name == "rf":
        return {"n_estimators": 200, "max_depth": 10, "min_samples_leaf": 5}
    return {}

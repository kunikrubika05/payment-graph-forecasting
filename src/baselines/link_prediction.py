"""Link prediction baseline pipeline: LogReg, CatBoost, RandomForest."""

import gc
import logging
import os
import time
from itertools import product as iter_product
from typing import Dict, List, Set, Tuple, Optional

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from src.baselines.config import (
    ExperimentConfig, NODE_FEATURE_COLUMNS,
    LOGREG_HP_GRID, CATBOOST_HP_GRID, RF_HP_GRID, K_VALUES,
)
from src.baselines.data_loader import (
    get_available_dates, download_period_data, load_node_features,
    load_daily_snapshot, cleanup_period_data,
)
from src.baselines.feature_engineering import (
    aggregate_features_mean, aggregate_features_time_weighted,
    build_pair_features, compute_feature_correlations, get_feature_names,
)
from src.baselines.evaluation import compute_classification_metrics
from src.baselines.experiment_logger import ExperimentLogger

logger = logging.getLogger(__name__)


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


def sample_negatives_random(
    positive_edges: Set[Tuple[int, int]],
    active_nodes: np.ndarray,
    n_samples: int,
    seed: int,
) -> np.ndarray:
    """Sample random negative edges (pairs not in positive set).

    Args:
        positive_edges: Set of (src, dst) tuples that are positive.
        active_nodes: Array of active node indices.
        n_samples: Number of negative samples to generate.
        seed: Random seed.

    Returns:
        Array of shape (n_actual_samples, 2) with (src, dst) pairs.
    """
    rng = np.random.RandomState(seed)
    n_nodes = len(active_nodes)
    if n_nodes < 2:
        return np.empty((0, 2), dtype=np.int64)

    negatives = []
    batch_size = min(n_samples * 3, 10_000_000)
    attempts = 0
    max_attempts = 10

    while len(negatives) < n_samples and attempts < max_attempts:
        src_idx = rng.randint(0, n_nodes, size=batch_size)
        dst_idx = rng.randint(0, n_nodes, size=batch_size)

        mask = src_idx != dst_idx
        src_idx = src_idx[mask]
        dst_idx = dst_idx[mask]

        src_nodes = active_nodes[src_idx]
        dst_nodes = active_nodes[dst_idx]

        for s, d in zip(src_nodes, dst_nodes):
            if (s, d) not in positive_edges:
                negatives.append((s, d))
                if len(negatives) >= n_samples:
                    break
        attempts += 1

    if not negatives:
        return np.empty((0, 2), dtype=np.int64)
    return np.array(negatives[:n_samples], dtype=np.int64)


def sample_negatives_historical(
    positive_edges: Set[Tuple[int, int]],
    historical_edges: Set[Tuple[int, int]],
    n_samples: int,
    seed: int,
) -> np.ndarray:
    """Sample historical negatives: edges from window that are absent in target day.

    Args:
        positive_edges: Edges in the target day.
        historical_edges: Edges from the window period.
        n_samples: Number of samples to generate.
        seed: Random seed.

    Returns:
        Array of shape (n_actual_samples, 2) with (src, dst) pairs.
    """
    candidates = historical_edges - positive_edges
    if not candidates:
        return np.empty((0, 2), dtype=np.int64)

    candidates_arr = np.array(list(candidates), dtype=np.int64)
    rng = np.random.RandomState(seed)

    if len(candidates_arr) <= n_samples:
        return candidates_arr

    indices = rng.choice(len(candidates_arr), size=n_samples, replace=False)
    return candidates_arr[indices]


def prepare_day_samples(
    target_snapshot: pd.DataFrame,
    node_features_agg: pd.DataFrame,
    historical_edges: Set[Tuple[int, int]],
    active_nodes: np.ndarray,
    config: ExperimentConfig,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Prepare feature matrix and labels for one target day.

    Args:
        target_snapshot: Edge list for the target day.
        node_features_agg: Aggregated node features from the window.
        historical_edges: Set of edges from the window period.
        active_nodes: Array of active nodes in the window.
        config: Experiment configuration.
        seed: Random seed.

    Returns:
        Tuple of (X, y, pairs_df).
        X: Feature matrix (n_samples, n_features).
        y: Binary labels.
        pairs_df: DataFrame with src, dst, label columns.
    """
    target_edges = _get_edges_set(target_snapshot)

    known_nodes = set(node_features_agg.index)
    valid_positive_edges = [
        (s, d) for s, d in target_edges
        if s in known_nodes and d in known_nodes
    ]

    if not valid_positive_edges:
        empty_x = np.empty((0, len(get_feature_names(config.feature_mode))))
        return empty_x, np.array([]), pd.DataFrame(columns=["src", "dst", "label"])

    pos_arr = np.array(valid_positive_edges, dtype=np.int64)
    n_pos = len(pos_arr)
    n_neg = n_pos * config.negative_ratio

    valid_edges_set = set(valid_positive_edges)

    if config.negative_strategy == "random":
        neg_arr = sample_negatives_random(
            target_edges, active_nodes, n_neg, seed
        )
    elif config.negative_strategy == "historical":
        neg_arr = sample_negatives_historical(
            target_edges, historical_edges, n_neg, seed
        )
    elif config.negative_strategy == "both":
        n_random = n_neg // 2
        n_hist = n_neg - n_random
        neg_random = sample_negatives_random(
            target_edges, active_nodes, n_random, seed
        )
        neg_hist = sample_negatives_historical(
            target_edges, historical_edges, n_hist, seed + 1
        )
        neg_arr = np.vstack([neg_random, neg_hist]) if len(neg_random) > 0 and len(neg_hist) > 0 else (
            neg_random if len(neg_random) > 0 else neg_hist
        )
    else:
        raise ValueError(f"Unknown negative strategy: {config.negative_strategy}")

    if len(neg_arr) == 0:
        neg_arr = sample_negatives_random(target_edges, active_nodes, n_neg, seed)

    all_src = np.concatenate([pos_arr[:, 0], neg_arr[:, 0]])
    all_dst = np.concatenate([pos_arr[:, 1], neg_arr[:, 1]])
    y = np.concatenate([np.ones(n_pos), np.zeros(len(neg_arr))])

    known_mask = np.array([
        s in known_nodes and d in known_nodes
        for s, d in zip(all_src, all_dst)
    ])
    all_src = all_src[known_mask]
    all_dst = all_dst[known_mask]
    y = y[known_mask]

    X, feature_names = build_pair_features(
        node_features_agg, all_src, all_dst, mode=config.feature_mode
    )

    pairs_df = pd.DataFrame({"src": all_src, "dst": all_dst, "label": y.astype(int)})
    return X, y, pairs_df


def hp_search(
    model_name: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
) -> Tuple[object, dict, List[dict]]:
    """Run hyperparameter grid search for a given model.

    Args:
        model_name: One of 'logreg', 'catboost', 'rf'.
        X_train: Training features.
        y_train: Training labels.
        X_val: Validation features.
        y_val: Validation labels.

    Returns:
        Tuple of (best_model, best_params, all_results).
    """
    if model_name == "logreg":
        grid = LOGREG_HP_GRID
        param_combos = [
            dict(zip(grid.keys(), v)) for v in iter_product(*grid.values())
        ]
    elif model_name == "catboost":
        grid = CATBOOST_HP_GRID
        param_combos = [
            dict(zip(grid.keys(), v)) for v in iter_product(*grid.values())
        ]
    elif model_name == "rf":
        grid = RF_HP_GRID
        param_combos = [
            dict(zip(grid.keys(), v)) for v in iter_product(*grid.values())
        ]
    else:
        raise ValueError(f"Unknown model: {model_name}")

    results = []
    best_score = -1.0
    best_model = None
    best_params = {}

    for params in param_combos:
        try:
            t0 = time.time()
            model = _create_model(model_name, params)
            model = _fit_model(model, model_name, X_train, y_train)
            train_time = time.time() - t0

            y_proba_val = _predict_proba(model, model_name, X_val)
            val_metrics = compute_classification_metrics(y_val, y_proba_val, K_VALUES)
            score = val_metrics.get("pr_auc", 0.0)

            result = {
                "model": model_name,
                "params": params,
                "val_pr_auc": score,
                "val_roc_auc": val_metrics.get("roc_auc", 0.0),
                "train_time_sec": train_time,
            }
            results.append(result)

            if score > best_score:
                best_score = score
                best_model = model
                best_params = params

            logger.info(
                "  %s %s -> val PR-AUC=%.4f, ROC-AUC=%.4f (%.1fs)",
                model_name, params, score, val_metrics.get("roc_auc", 0.0), train_time,
            )
        except Exception as e:
            logger.warning("  %s %s failed: %s", model_name, params, e)
            results.append({"model": model_name, "params": params, "error": str(e)})

    return best_model, best_params, results


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
    if model_name == "catboost":
        return model.predict_proba(X)[:, 1]
    return model.predict_proba(X)[:, 1]


def get_feature_importance(
    model, model_name: str, feature_names: List[str]
) -> dict:
    """Extract feature importance from trained model.

    Args:
        model: Trained model.
        model_name: Model type string.
        feature_names: List of feature names.

    Returns:
        Dict mapping model_name to {feature: importance} dict.
    """
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


def _collect_samples_for_days(
    dates: List[str],
    all_dates: List[str],
    config: ExperimentConfig,
    token: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """Collect and subsample training/validation data across multiple days.

    Args:
        dates: List of target dates to collect samples from.
        all_dates: Full ordered list of dates in the period.
        config: Experiment config.
        token: Yandex.Disk token.

    Returns:
        Tuple of (X_all, y_all) concatenated across all days.
    """
    X_list = []
    y_list = []
    total_samples = 0

    for target_date in dates:
        target_idx = all_dates.index(target_date)
        window_start = max(0, target_idx - config.window_size)
        window_dates = all_dates[window_start:target_idx]

        if not window_dates:
            continue

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
            continue

        if config.aggregation == "mean":
            node_feat_agg = aggregate_features_mean(features_by_date)
        else:
            node_feat_agg = aggregate_features_time_weighted(
                features_by_date, window_dates, config.decay_lambda
            )

        active_nodes = _get_active_nodes(window_snapshots)
        historical_edges = set()
        for snap in window_snapshots.values():
            historical_edges.update(_get_edges_set(snap))

        target_snap = load_daily_snapshot(target_date, config.local_data_dir)
        if target_snap is None or len(target_snap) == 0:
            continue

        seed = config.random_seed + hash(target_date) % 10000
        X, y, _ = prepare_day_samples(
            target_snap, node_feat_agg, historical_edges, active_nodes, config, seed
        )

        if len(y) == 0:
            continue

        X_list.append(X)
        y_list.append(y)
        total_samples += len(y)

        del features_by_date, window_snapshots, node_feat_agg
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


def run_link_prediction_mode_a(config: ExperimentConfig, token: str) -> None:
    """Run link prediction in Mode A: single model trained on all train days.

    Args:
        config: Experiment configuration.
        token: Yandex.Disk OAuth token.
    """
    from datetime import timedelta

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

    logger.info("Collecting training samples...")
    X_train, y_train = _collect_samples_for_days(
        train_dates, all_dates, config, token
    )
    if len(y_train) == 0:
        logger.error("No training samples collected, aborting")
        exp_logger.close()
        return

    logger.info("Training set: %d samples (%.1f%% positive)",
                len(y_train), y_train.mean() * 100)

    logger.info("Collecting validation samples...")
    X_val, y_val = _collect_samples_for_days(
        val_dates, all_dates, config, token
    )
    logger.info("Validation set: %d samples", len(y_val))

    feature_names = get_feature_names(config.feature_mode)
    if len(X_train) > 0:
        corr_df = compute_feature_correlations(
            X_train[:min(100000, len(X_train))], feature_names
        )
        exp_logger.log_feature_correlations(corr_df)
        exp_logger.log_high_correlations(corr_df)

    hp_max = config.hp_search_max_samples
    if len(X_train) > hp_max:
        rng_hp = np.random.RandomState(config.random_seed + 999)
        hp_idx = rng_hp.choice(len(X_train), hp_max, replace=False)
        X_train_hp = X_train[hp_idx]
        y_train_hp = y_train[hp_idx]
        logger.info("HP search subsample: %d / %d samples", hp_max, len(X_train))
    else:
        X_train_hp = X_train
        y_train_hp = y_train

    if len(X_val) > hp_max:
        rng_hp2 = np.random.RandomState(config.random_seed + 998)
        hp_idx2 = rng_hp2.choice(len(X_val), hp_max, replace=False)
        X_val_hp = X_val[hp_idx2]
        y_val_hp = y_val[hp_idx2]
    else:
        X_val_hp = X_val
        y_val_hp = y_val

    all_hp_results = []
    best_models = {}
    all_importance = {}

    for model_name in config.models:
        logger.info("=== HP search for %s ===", model_name)

        if len(y_val_hp) > 0:
            _, best_params, hp_results = hp_search(
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

    logger.info("=== Evaluating on test days ===")
    all_test_metrics = {m: [] for m in best_models}

    for target_date in test_dates:
        target_idx = all_dates.index(target_date)
        window_start = max(0, target_idx - config.window_size)
        window_dates = all_dates[window_start:target_idx]

        if not window_dates:
            continue

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
            continue

        if config.aggregation == "mean":
            node_feat_agg = aggregate_features_mean(features_by_date)
        else:
            node_feat_agg = aggregate_features_time_weighted(
                features_by_date, window_dates, config.decay_lambda
            )

        active_nodes = _get_active_nodes(window_snapshots)
        historical_edges = set()
        for snap in window_snapshots.values():
            historical_edges.update(_get_edges_set(snap))

        target_snap = load_daily_snapshot(target_date, config.local_data_dir)
        if target_snap is None or len(target_snap) == 0:
            continue

        seed = config.random_seed + hash(target_date) % 10000
        X_test, y_test, pairs_df = prepare_day_samples(
            target_snap, node_feat_agg, historical_edges, active_nodes, config, seed
        )

        if len(y_test) == 0:
            continue

        for model_name, (model, params) in best_models.items():
            y_proba = _predict_proba(model, model_name, X_test)
            metrics = compute_classification_metrics(y_test, y_proba, K_VALUES)
            metrics["date"] = target_date
            metrics["model"] = model_name
            metrics["best_params"] = params
            exp_logger.log_metrics(metrics)
            all_test_metrics[model_name].append(metrics)

            pairs_df_with_pred = pairs_df.copy()
            pairs_df_with_pred["pred_proba"] = y_proba
            pairs_df_with_pred["model"] = model_name
            exp_logger.save_predictions(pairs_df_with_pred, f"{target_date}_{model_name}")

        del features_by_date, window_snapshots, node_feat_agg
        gc.collect()

        logger.info(
            "  %s: %s",
            target_date,
            {m: f"ROC={met[-1].get('roc_auc', 0):.4f} PR={met[-1].get('pr_auc', 0):.4f}"
             for m, met in all_test_metrics.items() if met},
        )

    summary = {"config": config.to_dict(), "models": {}}
    for model_name, metrics_list in all_test_metrics.items():
        if not metrics_list:
            continue
        numeric_keys = [k for k in metrics_list[0] if isinstance(metrics_list[0][k], (int, float))]
        model_summary = {}
        for key in numeric_keys:
            values = [m[key] for m in metrics_list if key in m and not np.isnan(m.get(key, float("nan")))]
            if values:
                model_summary[f"mean_{key}"] = float(np.mean(values))
                model_summary[f"std_{key}"] = float(np.std(values))
        model_summary["n_test_days"] = len(metrics_list)
        model_summary["best_params"] = best_models[model_name][1]
        summary["models"][model_name] = model_summary
    exp_logger.write_summary(summary)

    remote_dir = f"{config.yadisk_experiments_base}/{config.experiment_name}/{config.sub_experiment}"
    exp_logger.upload_to_yadisk(remote_dir, token)
    exp_logger.close()

    logger.info("=== Experiment complete: %s/%s ===", config.experiment_name, config.sub_experiment)


def run_link_prediction_mode_b(config: ExperimentConfig, token: str) -> None:
    """Run link prediction in Mode B: live-update (retrain on each new day).

    Args:
        config: Experiment configuration.
        token: Yandex.Disk OAuth token.
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

    logger.info(
        "Mode B: %d warmup days, %d eval days", len(warmup_dates), len(eval_dates)
    )

    logger.info("Downloading data from Yandex.Disk...")
    download_period_data(all_dates, config.local_data_dir, token)

    logger.info("Collecting warmup samples...")
    X_cumulative, y_cumulative = _collect_samples_for_days(
        warmup_dates, all_dates, config, token
    )
    if len(y_cumulative) == 0:
        logger.error("No warmup samples, aborting")
        exp_logger.close()
        return

    feature_names = get_feature_names(config.feature_mode)
    all_test_metrics = {m: [] for m in config.models}
    current_models = {}

    for model_name in config.models:
        params = _default_params(model_name)
        model = _create_model(model_name, params)
        model = _fit_model(model, model_name, X_cumulative, y_cumulative)
        current_models[model_name] = (model, params)

    days_since_retrain = 0

    for eval_idx, target_date in enumerate(eval_dates):
        target_idx = all_dates.index(target_date)
        window_start = max(0, target_idx - config.window_size)
        window_dates = all_dates[window_start:target_idx]

        if not window_dates:
            continue

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
            continue

        if config.aggregation == "mean":
            node_feat_agg = aggregate_features_mean(features_by_date)
        else:
            node_feat_agg = aggregate_features_time_weighted(
                features_by_date, window_dates, config.decay_lambda
            )

        active_nodes = _get_active_nodes(window_snapshots)
        historical_edges = set()
        for snap in window_snapshots.values():
            historical_edges.update(_get_edges_set(snap))

        target_snap = load_daily_snapshot(target_date, config.local_data_dir)
        if target_snap is None or len(target_snap) == 0:
            continue

        seed = config.random_seed + hash(target_date) % 10000
        X_day, y_day, pairs_df = prepare_day_samples(
            target_snap, node_feat_agg, historical_edges, active_nodes, config, seed
        )

        if len(y_day) == 0:
            continue

        for model_name, (model, params) in current_models.items():
            y_proba = _predict_proba(model, model_name, X_day)
            metrics = compute_classification_metrics(y_day, y_proba, K_VALUES)
            metrics["date"] = target_date
            metrics["model"] = model_name
            metrics["cumulative_train_size"] = len(y_cumulative)
            exp_logger.log_metrics(metrics)
            all_test_metrics[model_name].append(metrics)

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

        del features_by_date, window_snapshots, node_feat_agg
        gc.collect()

        logger.info(
            "  %s (cumulative=%d, retrained=%s): %s",
            target_date, len(y_cumulative),
            "yes" if should_retrain else "no",
            {m: f"ROC={met[-1].get('roc_auc', 0):.4f}" for m, met in all_test_metrics.items() if met},
        )

    for model_name, (model, params) in current_models.items():
        ext = ".cbm" if model_name == "catboost" else ".pkl"
        exp_logger.save_model(model, f"final_{model_name}{ext}")

    summary = {"config": config.to_dict(), "mode": "B", "models": {}}
    for model_name, metrics_list in all_test_metrics.items():
        if not metrics_list:
            continue
        numeric_keys = [k for k in metrics_list[0] if isinstance(metrics_list[0][k], (int, float))]
        model_summary = {}
        for key in numeric_keys:
            values = [m[key] for m in metrics_list if key in m and not np.isnan(m.get(key, float("nan")))]
            if values:
                model_summary[f"mean_{key}"] = float(np.mean(values))
                model_summary[f"std_{key}"] = float(np.std(values))
        model_summary["n_eval_days"] = len(metrics_list)
        summary["models"][model_name] = model_summary
    exp_logger.write_summary(summary)

    remote_dir = f"{config.yadisk_experiments_base}/{config.experiment_name}/{config.sub_experiment}"
    exp_logger.upload_to_yadisk(remote_dir, token)
    exp_logger.close()

    logger.info("=== Mode B complete: %s/%s ===", config.experiment_name, config.sub_experiment)


def run_link_prediction(config: ExperimentConfig, token: str) -> None:
    """Main entry point for link prediction experiment.

    Args:
        config: Experiment configuration.
        token: Yandex.Disk OAuth token.
    """
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

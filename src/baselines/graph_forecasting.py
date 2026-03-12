"""Graph-level time series forecasting baselines: ARIMA, SARIMAX, Holt-Winters, naive."""

import logging
import os
import time
import warnings
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from src.baselines.config import ExperimentConfig, GRAPH_FORECAST_TARGETS
from src.baselines.data_loader import load_graph_features
from src.baselines.evaluation import compute_ts_metrics
from src.baselines.experiment_logger import ExperimentLogger

logger = logging.getLogger(__name__)

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


def _naive_persistence(train: np.ndarray, n_forecast: int) -> np.ndarray:
    """Naive persistence: predict last known value."""
    return np.full(n_forecast, train[-1])


def _naive_seasonal(train: np.ndarray, n_forecast: int, period: int = 7) -> np.ndarray:
    """Seasonal naive: repeat last known seasonal cycle."""
    if len(train) < period:
        return _naive_persistence(train, n_forecast)
    seasonal = train[-period:]
    reps = (n_forecast // period) + 1
    return np.tile(seasonal, reps)[:n_forecast]


def _naive_moving_average(train: np.ndarray, n_forecast: int, window: int = 7) -> np.ndarray:
    """Moving average baseline."""
    avg = np.mean(train[-window:])
    return np.full(n_forecast, avg)


def _run_arima_grid(
    train: np.ndarray, val: np.ndarray, target_name: str
) -> Tuple[dict, dict, List[dict]]:
    """Run ARIMA with grid search over (p, d, q) parameters.

    Returns:
        Tuple of (best_metrics, best_params, all_results).
    """
    from statsmodels.tsa.arima.model import ARIMA

    p_range = [0, 1, 2, 3, 5]
    d_range = [0, 1, 2]
    q_range = [0, 1, 2, 3]

    results = []
    best_score = float("inf")
    best_metrics = {}
    best_params = {}

    for p in p_range:
        for d in d_range:
            for q in q_range:
                if p == 0 and q == 0:
                    continue
                try:
                    t0 = time.time()
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        model = ARIMA(train, order=(p, d, q))
                        fitted = model.fit()
                    forecast = fitted.forecast(steps=len(val))
                    fit_time = time.time() - t0

                    metrics = compute_ts_metrics(val, forecast)
                    result = {
                        "model": "arima",
                        "target": target_name,
                        "params": {"p": p, "d": d, "q": q},
                        "fit_time_sec": fit_time,
                        **metrics,
                    }
                    results.append(result)

                    if metrics["mae"] < best_score:
                        best_score = metrics["mae"]
                        best_metrics = metrics
                        best_params = {"p": p, "d": d, "q": q}
                except Exception:
                    pass

    return best_metrics, best_params, results


def _run_sarimax_grid(
    train: np.ndarray, val: np.ndarray, target_name: str
) -> Tuple[dict, dict, List[dict]]:
    """Run SARIMAX with grid search over parameters.

    Returns:
        Tuple of (best_metrics, best_params, all_results).
    """
    from statsmodels.tsa.statespace.sarimax import SARIMAX

    configs = [
        {"order": (1, 1, 1), "seasonal_order": (1, 1, 1, 7)},
        {"order": (2, 1, 1), "seasonal_order": (1, 1, 1, 7)},
        {"order": (1, 1, 2), "seasonal_order": (1, 1, 1, 7)},
        {"order": (1, 1, 1), "seasonal_order": (0, 1, 1, 7)},
        {"order": (2, 1, 2), "seasonal_order": (1, 1, 1, 7)},
        {"order": (1, 0, 1), "seasonal_order": (1, 0, 1, 7)},
    ]

    results = []
    best_score = float("inf")
    best_metrics = {}
    best_params = {}

    for cfg in configs:
        try:
            t0 = time.time()
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model = SARIMAX(
                    train,
                    order=cfg["order"],
                    seasonal_order=cfg["seasonal_order"],
                    enforce_stationarity=False,
                    enforce_invertibility=False,
                )
                fitted = model.fit(disp=False, maxiter=200)
            forecast = fitted.forecast(steps=len(val))
            fit_time = time.time() - t0

            metrics = compute_ts_metrics(val, forecast)
            result = {
                "model": "sarimax",
                "target": target_name,
                "params": cfg,
                "fit_time_sec": fit_time,
                **metrics,
            }
            results.append(result)

            if metrics["mae"] < best_score:
                best_score = metrics["mae"]
                best_metrics = metrics
                best_params = cfg
        except Exception:
            pass

    return best_metrics, best_params, results


def _run_holt_winters(
    train: np.ndarray, val: np.ndarray, target_name: str
) -> Tuple[dict, dict, List[dict]]:
    """Run Exponential Smoothing (Holt-Winters) with grid search.

    Returns:
        Tuple of (best_metrics, best_params, all_results).
    """
    from statsmodels.tsa.holtwinters import ExponentialSmoothing

    configs = [
        {"trend": "add", "seasonal": None, "seasonal_periods": None},
        {"trend": "add", "seasonal": "add", "seasonal_periods": 7},
        {"trend": "mul", "seasonal": "add", "seasonal_periods": 7},
        {"trend": "add", "seasonal": "mul", "seasonal_periods": 7},
        {"trend": None, "seasonal": "add", "seasonal_periods": 7},
    ]

    results = []
    best_score = float("inf")
    best_metrics = {}
    best_params = {}

    for cfg in configs:
        try:
            t0 = time.time()
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model = ExponentialSmoothing(
                    train,
                    trend=cfg["trend"],
                    seasonal=cfg["seasonal"],
                    seasonal_periods=cfg["seasonal_periods"],
                )
                fitted = model.fit(optimized=True)
            forecast = fitted.forecast(steps=len(val))
            fit_time = time.time() - t0

            metrics = compute_ts_metrics(val, forecast)
            result = {
                "model": "holt_winters",
                "target": target_name,
                "params": cfg,
                "fit_time_sec": fit_time,
                **metrics,
            }
            results.append(result)

            if metrics["mae"] < best_score:
                best_score = metrics["mae"]
                best_metrics = metrics
                best_params = cfg
        except Exception:
            pass

    return best_metrics, best_params, results


def _run_prophet(
    train_series: pd.Series, val_series: pd.Series, target_name: str
) -> Tuple[dict, dict, List[dict]]:
    """Run Prophet forecasting.

    Returns:
        Tuple of (best_metrics, best_params, all_results).
    """
    try:
        from prophet import Prophet
    except ImportError:
        logger.warning("Prophet not installed, skipping")
        return {}, {}, []

    configs = [
        {"changepoint_prior_scale": 0.05},
        {"changepoint_prior_scale": 0.1},
        {"changepoint_prior_scale": 0.5},
        {"seasonality_mode": "multiplicative"},
    ]

    results = []
    best_score = float("inf")
    best_metrics = {}
    best_params = {}

    train_df = pd.DataFrame({
        "ds": train_series.index,
        "y": train_series.values,
    })

    for cfg in configs:
        try:
            t0 = time.time()
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model = Prophet(**cfg, daily_seasonality=True, weekly_seasonality=True)
                model.fit(train_df)
            future = model.make_future_dataframe(periods=len(val_series))
            forecast = model.predict(future)
            preds = forecast["yhat"].values[-len(val_series):]
            fit_time = time.time() - t0

            metrics = compute_ts_metrics(val_series.values, preds)
            result = {
                "model": "prophet",
                "target": target_name,
                "params": cfg,
                "fit_time_sec": fit_time,
                **metrics,
            }
            results.append(result)

            if metrics["mae"] < best_score:
                best_score = metrics["mae"]
                best_metrics = metrics
                best_params = cfg
        except Exception as e:
            logger.warning("Prophet config %s failed: %s", cfg, e)

    return best_metrics, best_params, results


def run_graph_forecasting(config: ExperimentConfig, token: str) -> None:
    """Run all graph-level forecasting models.

    Args:
        config: Experiment configuration.
        token: Yandex.Disk OAuth token.
    """
    output_dir = os.path.join(
        config.output_dir or f"/tmp/baseline_results/{config.experiment_name}",
        config.sub_experiment or "graph_forecasting",
    )
    exp_logger = ExperimentLogger(output_dir)

    if exp_logger.is_completed():
        logger.info("Experiment already completed: %s", output_dir)
        exp_logger.close()
        return

    exp_logger.log_config(config.to_dict())

    gf = load_graph_features()
    gf = gf.sort_values("date").reset_index(drop=True)
    gf = gf[gf["date"] != "2021-01-25"]

    start_mask = gf["date"] >= config.period_start
    end_mask = gf["date"] <= config.period_end
    gf_period = gf[start_mask & end_mask].reset_index(drop=True)

    n = len(gf_period)
    n_train = int(n * config.train_ratio)
    n_val = int(n * config.val_ratio)

    train_df = gf_period.iloc[:n_train]
    val_df = gf_period.iloc[n_train:n_train + n_val]
    test_df = gf_period.iloc[n_train + n_val:]

    logger.info(
        "Graph forecasting: %d days (train=%d, val=%d, test=%d)",
        n, len(train_df), len(val_df), len(test_df),
    )

    targets = config.target_variables or GRAPH_FORECAST_TARGETS
    all_hp_results = []
    summary = {"config": config.to_dict(), "targets": {}}

    for target in targets:
        if target not in gf_period.columns:
            logger.warning("Target %s not found in graph_features", target)
            continue

        logger.info("=== Target: %s ===", target)
        train_values = train_df[target].values.astype(float)
        val_values = val_df[target].values.astype(float)
        test_values = test_df[target].values.astype(float)

        if len(val_values) == 0 or len(test_values) == 0:
            logger.warning("Not enough data for target %s", target)
            continue

        full_train = np.concatenate([train_values, val_values])
        target_results = {}

        naive_models = {
            "persistence": _naive_persistence(train_values, len(val_values)),
            "seasonal_7": _naive_seasonal(train_values, len(val_values), 7),
            "moving_avg_7": _naive_moving_average(train_values, len(val_values), 7),
            "moving_avg_30": _naive_moving_average(train_values, len(val_values), 30),
        }

        for name, val_preds in naive_models.items():
            val_metrics = compute_ts_metrics(val_values, val_preds)
            test_preds_fn = {
                "persistence": lambda: _naive_persistence(full_train, len(test_values)),
                "seasonal_7": lambda: _naive_seasonal(full_train, len(test_values), 7),
                "moving_avg_7": lambda: _naive_moving_average(full_train, len(test_values), 7),
                "moving_avg_30": lambda: _naive_moving_average(full_train, len(test_values), 30),
            }
            test_preds = test_preds_fn[name]()
            test_metrics = compute_ts_metrics(test_values, test_preds)

            result = {
                "model": name, "target": target,
                "val_metrics": val_metrics, "test_metrics": test_metrics,
            }
            exp_logger.log_metrics({"target": target, "model": name, "split": "val", **val_metrics})
            exp_logger.log_metrics({"target": target, "model": name, "split": "test", **test_metrics})
            target_results[name] = test_metrics
            logger.info("  %s: val MAE=%.2f, test MAE=%.2f", name, val_metrics["mae"], test_metrics["mae"])

        logger.info("  Running ARIMA grid search...")
        arima_val_metrics, arima_params, arima_results = _run_arima_grid(
            train_values, val_values, target
        )
        all_hp_results.extend(arima_results)
        if arima_params:
            from statsmodels.tsa.arima.model import ARIMA
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model = ARIMA(full_train, order=(arima_params["p"], arima_params["d"], arima_params["q"]))
                fitted = model.fit()
            test_preds = fitted.forecast(steps=len(test_values))
            test_metrics = compute_ts_metrics(test_values, test_preds)
            exp_logger.log_metrics({"target": target, "model": "arima", "split": "test", "params": arima_params, **test_metrics})
            target_results["arima"] = {**test_metrics, "params": arima_params}
            logger.info("  ARIMA %s: test MAE=%.2f", arima_params, test_metrics["mae"])

        logger.info("  Running SARIMAX grid search...")
        sarimax_val_metrics, sarimax_params, sarimax_results = _run_sarimax_grid(
            train_values, val_values, target
        )
        all_hp_results.extend(sarimax_results)
        if sarimax_params:
            from statsmodels.tsa.statespace.sarimax import SARIMAX
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model = SARIMAX(
                    full_train,
                    order=sarimax_params["order"],
                    seasonal_order=sarimax_params["seasonal_order"],
                    enforce_stationarity=False,
                    enforce_invertibility=False,
                )
                fitted = model.fit(disp=False, maxiter=200)
            test_preds = fitted.forecast(steps=len(test_values))
            test_metrics = compute_ts_metrics(test_values, test_preds)
            exp_logger.log_metrics({"target": target, "model": "sarimax", "split": "test", "params": str(sarimax_params), **test_metrics})
            target_results["sarimax"] = {**test_metrics, "params": sarimax_params}
            logger.info("  SARIMAX %s: test MAE=%.2f", sarimax_params, test_metrics["mae"])

        logger.info("  Running Holt-Winters grid search...")
        hw_val_metrics, hw_params, hw_results = _run_holt_winters(
            train_values, val_values, target
        )
        all_hp_results.extend(hw_results)
        if hw_params:
            from statsmodels.tsa.holtwinters import ExponentialSmoothing
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model = ExponentialSmoothing(
                    full_train,
                    trend=hw_params["trend"],
                    seasonal=hw_params["seasonal"],
                    seasonal_periods=hw_params["seasonal_periods"],
                )
                fitted = model.fit(optimized=True)
            test_preds = fitted.forecast(steps=len(test_values))
            test_metrics = compute_ts_metrics(test_values, test_preds)
            exp_logger.log_metrics({"target": target, "model": "holt_winters", "split": "test", "params": str(hw_params), **test_metrics})
            target_results["holt_winters"] = {**test_metrics, "params": hw_params}
            logger.info("  Holt-Winters %s: test MAE=%.2f", hw_params, test_metrics["mae"])

        logger.info("  Running Prophet...")
        train_series = pd.Series(train_values, index=train_df["date"].values)
        val_series = pd.Series(val_values, index=val_df["date"].values)
        prophet_val, prophet_params, prophet_results = _run_prophet(
            train_series, val_series, target
        )
        all_hp_results.extend(prophet_results)
        if prophet_params:
            full_series = pd.Series(full_train, index=pd.concat([train_df, val_df])["date"].values)
            test_series = pd.Series(test_values, index=test_df["date"].values)
            prophet_test, _, _ = _run_prophet(full_series, test_series, target)
            if prophet_test:
                exp_logger.log_metrics({"target": target, "model": "prophet", "split": "test", **prophet_test})
                target_results["prophet"] = prophet_test
                logger.info("  Prophet: test MAE=%.2f", prophet_test.get("mae", float("nan")))

        summary["targets"][target] = target_results

    exp_logger.log_hp_search(all_hp_results)
    exp_logger.write_summary(summary)

    remote_dir = f"{config.yadisk_experiments_base}/{config.experiment_name}/{config.sub_experiment or 'graph_forecasting'}"
    exp_logger.upload_to_yadisk(remote_dir, token)
    exp_logger.close()

    logger.info("=== Graph forecasting complete ===")

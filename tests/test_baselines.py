"""Tests for baseline pipeline on tiny synthetic graphs."""

import json
import os
import shutil
import tempfile

import numpy as np
import pandas as pd
import pytest

from src.baselines.config import ExperimentConfig, NODE_FEATURE_COLUMNS
from src.baselines.evaluation import (
    compute_ranking_metrics,
    compute_ts_metrics,
)
from src.baselines.experiment_logger import ExperimentLogger
from src.baselines.feature_engineering import (
    aggregate_features_mean,
    aggregate_features_time_weighted,
    build_pair_features,
    compute_feature_correlations,
    get_feature_names,
)


def _make_node_features(node_ids, seed=42):
    """Create a synthetic node features DataFrame."""
    rng = np.random.RandomState(seed)
    n = len(node_ids)
    data = rng.rand(n, len(NODE_FEATURE_COLUMNS))
    df = pd.DataFrame(data, columns=NODE_FEATURE_COLUMNS)
    df.index = node_ids
    df.index.name = "node_idx"
    return df


def _make_snapshot(edges):
    """Create a synthetic daily snapshot DataFrame from edge list."""
    if not edges:
        return pd.DataFrame(columns=["src_idx", "dst_idx", "btc", "usd"])
    src = [e[0] for e in edges]
    dst = [e[1] for e in edges]
    n = len(edges)
    return pd.DataFrame({
        "src_idx": src,
        "dst_idx": dst,
        "btc": np.random.rand(n),
        "usd": np.random.rand(n) * 1000,
    })


class TestFeatureEngineering:

    def test_aggregate_mean_single_day(self):
        df = _make_node_features([0, 1, 2])
        result = aggregate_features_mean({"2020-01-01": df})
        assert len(result) == 3
        assert list(result.columns) == NODE_FEATURE_COLUMNS
        np.testing.assert_array_almost_equal(result.values, df.values)

    def test_aggregate_mean_multiple_days(self):
        df1 = _make_node_features([0, 1, 2], seed=1)
        df2 = _make_node_features([1, 2, 3], seed=2)
        result = aggregate_features_mean({"d1": df1, "d2": df2})
        assert len(result) == 4
        assert 0 in result.index
        assert 3 in result.index
        expected_node1 = (df1.loc[1].values + df2.loc[1].values) / 2
        np.testing.assert_array_almost_equal(result.loc[1].values, expected_node1)

    def test_aggregate_mean_empty(self):
        result = aggregate_features_mean({})
        assert len(result) == 0

    def test_aggregate_time_weighted(self):
        df1 = _make_node_features([0, 1], seed=1)
        df2 = _make_node_features([0, 1], seed=2)
        dates = ["2020-01-01", "2020-01-02"]
        result = aggregate_features_time_weighted(
            {"2020-01-01": df1, "2020-01-02": df2}, dates, decay_lambda=0.5
        )
        assert len(result) == 2
        w0 = np.exp(-0.5 * 1)
        w1 = np.exp(-0.5 * 0)
        expected = (df1.loc[0].values * w0 + df2.loc[0].values * w1) / (w0 + w1)
        np.testing.assert_array_almost_equal(result.loc[0].values, expected)

    def test_build_pair_features_base(self):
        df = _make_node_features([10, 20, 30])
        src = np.array([10, 20])
        dst = np.array([20, 30])
        X, names = build_pair_features(df, src, dst, mode="base")
        assert X.shape == (2, len(NODE_FEATURE_COLUMNS) * 2)
        assert len(names) == len(NODE_FEATURE_COLUMNS) * 2
        assert names[0].startswith("src_")

    def test_build_pair_features_extended(self):
        df = _make_node_features([10, 20, 30])
        src = np.array([10, 20])
        dst = np.array([20, 30])
        X, names = build_pair_features(df, src, dst, mode="extended")
        assert X.shape == (2, len(NODE_FEATURE_COLUMNS) * 4)
        assert any(n.startswith("diff_") for n in names)
        assert any(n.startswith("prod_") for n in names)

    def test_build_pair_features_no_nan(self):
        df = _make_node_features([0, 1, 2])
        src = np.array([0, 1])
        dst = np.array([1, 2])
        X, _ = build_pair_features(df, src, dst, mode="extended")
        assert not np.any(np.isnan(X))

    def test_get_feature_names(self):
        base_names = get_feature_names("base")
        ext_names = get_feature_names("extended")
        assert len(base_names) == len(NODE_FEATURE_COLUMNS) * 2
        assert len(ext_names) == len(NODE_FEATURE_COLUMNS) * 4

    def test_compute_correlations(self):
        rng = np.random.RandomState(42)
        X = rng.rand(100, 5)
        X[:, 4] = X[:, 0] * 2 + 0.01
        names = ["f0", "f1", "f2", "f3", "f4"]
        corr = compute_feature_correlations(X, names)
        assert corr.shape == (5, 5)
        assert abs(corr.loc["f0", "f4"]) > 0.9


class TestEvaluation:

    def test_ranking_metrics_basic(self):
        ranks = np.array([1, 3, 2, 5, 1])
        metrics = compute_ranking_metrics(ranks)
        assert metrics["n_queries"] == 5
        assert 0 < metrics["mrr"] <= 1.0
        assert 0 < metrics["hits@1"] <= 1.0
        assert metrics["hits@1"] == 2 / 5
        assert metrics["hits@3"] == 4 / 5
        assert metrics["hits@10"] == 1.0

    def test_ranking_metrics_perfect(self):
        ranks = np.array([1, 1, 1])
        metrics = compute_ranking_metrics(ranks)
        assert metrics["mrr"] == 1.0
        assert metrics["hits@1"] == 1.0
        assert metrics["hits@10"] == 1.0

    def test_ranking_metrics_empty(self):
        ranks = np.array([])
        metrics = compute_ranking_metrics(ranks)
        assert metrics["n_queries"] == 0
        assert np.isnan(metrics["mrr"])

    def test_ranking_metrics_custom_k(self):
        ranks = np.array([1, 5, 10, 50])
        metrics = compute_ranking_metrics(ranks, k_values=[1, 5, 20])
        assert "hits@1" in metrics
        assert "hits@5" in metrics
        assert "hits@20" in metrics
        assert metrics["hits@1"] == 0.25
        assert metrics["hits@5"] == 0.5
        assert metrics["hits@20"] == 0.75

    def test_ts_metrics(self):
        y_true = np.array([100.0, 200.0, 300.0, 400.0])
        y_pred = np.array([110.0, 190.0, 310.0, 380.0])
        metrics = compute_ts_metrics(y_true, y_pred)
        assert metrics["mae"] == pytest.approx(12.5)
        assert metrics["rmse"] > 0
        assert metrics["mape"] > 0
        assert metrics["smape"] > 0

    def test_ts_metrics_perfect(self):
        y = np.array([1.0, 2.0, 3.0])
        metrics = compute_ts_metrics(y, y)
        assert metrics["mae"] == 0.0
        assert metrics["rmse"] == 0.0


class TestExperimentLogger:

    def test_logger_creates_dirs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output = os.path.join(tmpdir, "test_exp")
            exp_logger = ExperimentLogger(output)
            assert os.path.exists(os.path.join(output, "model"))
            assert os.path.exists(os.path.join(output, "predictions"))
            exp_logger.close()

    def test_log_config(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output = os.path.join(tmpdir, "test_exp")
            exp_logger = ExperimentLogger(output)
            exp_logger.log_config({"model": "test", "lr": 0.01})
            config_path = os.path.join(output, "config.json")
            assert os.path.exists(config_path)
            with open(config_path) as f:
                data = json.load(f)
            assert data["model"] == "test"
            exp_logger.close()

    def test_log_metrics_jsonl(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output = os.path.join(tmpdir, "test_exp")
            exp_logger = ExperimentLogger(output)
            exp_logger.log_metrics({"mrr": 0.55, "date": "2020-01-01"})
            exp_logger.log_metrics({"mrr": 0.60, "date": "2020-01-02"})
            exp_logger.close()

            metrics_path = os.path.join(output, "metrics.jsonl")
            with open(metrics_path) as f:
                lines = f.readlines()
            assert len(lines) == 2
            first = json.loads(lines[0])
            assert first["mrr"] == 0.55

    def test_summary_and_completed(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output = os.path.join(tmpdir, "test_exp")
            exp_logger = ExperimentLogger(output)
            assert not exp_logger.is_completed()
            exp_logger.write_summary({"mean_mrr": 0.56})
            assert exp_logger.is_completed()
            exp_logger.close()

    def test_save_predictions(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output = os.path.join(tmpdir, "test_exp")
            exp_logger = ExperimentLogger(output)
            df = pd.DataFrame({
                "src": [0, 1], "dst": [1, 2],
                "true_label": [1, 0], "pred_proba": [0.9, 0.1],
            })
            exp_logger.save_predictions(df, "2020-01-01")
            pred_path = os.path.join(output, "predictions", "2020-01-01.parquet")
            assert os.path.exists(pred_path)
            loaded = pd.read_parquet(pred_path)
            assert len(loaded) == 2
            exp_logger.close()

    def test_feature_correlations(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output = os.path.join(tmpdir, "test_exp")
            exp_logger = ExperimentLogger(output)
            corr_df = pd.DataFrame(
                [[1.0, 0.96], [0.96, 1.0]],
                columns=["f1", "f2"], index=["f1", "f2"],
            )
            exp_logger.log_feature_correlations(corr_df)
            exp_logger.log_high_correlations(corr_df, threshold=0.95)
            assert os.path.exists(os.path.join(output, "feature_correlations.csv"))
            assert os.path.exists(os.path.join(output, "high_correlations.json"))
            with open(os.path.join(output, "high_correlations.json")) as f:
                pairs = json.load(f)
            assert len(pairs) == 1
            exp_logger.close()


class TestConfig:

    def test_config_defaults(self):
        config = ExperimentConfig(period_name="mid_2015q3")
        assert config.period_start == "2015-07-01"
        assert config.period_end == "2015-09-30"
        assert "w7" in config.sub_experiment
        assert "mean" in config.sub_experiment

    def test_config_serialization(self):
        config = ExperimentConfig(
            experiment_name="test",
            period_name="early_2012q1",
            window_size=14,
        )
        d = config.to_dict()
        config2 = ExperimentConfig.from_dict(d)
        assert config2.experiment_name == "test"
        assert config2.window_size == 14
        assert config2.period_start == "2012-01-01"

    def test_config_save_load(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "config.json")
            config = ExperimentConfig(period_name="early_2012q1")
            config.save(path)
            loaded = ExperimentConfig.load(path)
            assert loaded.period_name == "early_2012q1"

    def test_config_n_negatives(self):
        config = ExperimentConfig()
        assert config.n_negatives == 100
        assert config.negative_ratio == 5


class TestLinkPredictionHelpers:

    def test_sample_negatives_per_source(self):
        from src.baselines.link_prediction import sample_negatives_per_source

        rng = np.random.RandomState(42)
        active = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        hist_nbrs = {1, 2, 3, 4}
        target_dsts = {1, 5}

        neg = sample_negatives_per_source(
            source=0,
            true_dst=1,
            historical_neighbors=hist_nbrs,
            target_edges_from_source=target_dsts,
            active_nodes=active,
            n_negatives=6,
            rng=rng,
        )
        assert len(neg) > 0
        assert 1 not in neg
        assert 0 not in neg

    def test_sample_negatives_per_source_hist_random_mix(self):
        from src.baselines.link_prediction import sample_negatives_per_source

        rng = np.random.RandomState(42)
        active = np.arange(100)
        hist_nbrs = {10, 20, 30, 40, 50}
        target_dsts = {10}

        neg = sample_negatives_per_source(
            source=0,
            true_dst=10,
            historical_neighbors=hist_nbrs,
            target_edges_from_source=target_dsts,
            active_nodes=active,
            n_negatives=10,
            rng=rng,
        )
        assert len(neg) == 10
        hist_in_neg = set(neg) & (hist_nbrs - target_dsts)
        assert len(hist_in_neg) > 0

    def test_prepare_training_samples(self):
        from src.baselines.link_prediction import prepare_training_samples

        snapshot = _make_snapshot([(0, 1), (1, 2), (2, 0)])
        node_feats = _make_node_features([0, 1, 2, 3, 4])
        hist_nbrs = {0: {1, 3}, 1: {2, 4}, 2: {0}}
        active = np.array([0, 1, 2, 3, 4])
        config = ExperimentConfig(negative_ratio=2, feature_mode="extended")
        X, y = prepare_training_samples(
            snapshot, node_feats, hist_nbrs, active, config, seed=42
        )
        assert len(y) > 0
        assert y.sum() > 0
        assert (y == 0).sum() > 0
        assert X.shape[0] == len(y)
        assert X.shape[1] == len(NODE_FEATURE_COLUMNS) * 4

    def test_hp_search_logreg(self):
        from src.baselines.link_prediction import hp_search

        rng = np.random.RandomState(42)
        n = 200
        X_train = rng.rand(n, 10)
        y_train = (X_train[:, 0] > 0.5).astype(float)
        X_val = rng.rand(50, 10)
        y_val = (X_val[:, 0] > 0.5).astype(float)
        best_params, results = hp_search(
            "logreg", X_train, y_train, X_val, y_val
        )
        assert "C" in best_params
        assert len(results) > 0

    def test_evaluate_ranking(self):
        from src.baselines.link_prediction import evaluate_ranking_for_day

        from sklearn.linear_model import LogisticRegression
        rng = np.random.RandomState(42)
        X_train = rng.rand(100, len(NODE_FEATURE_COLUMNS) * 4)
        y_train = (rng.rand(100) > 0.5).astype(float)
        model = LogisticRegression(max_iter=100, random_state=42)
        model.fit(X_train, y_train)

        snapshot = _make_snapshot([(0, 1), (0, 2), (1, 2)])
        node_feats = _make_node_features([0, 1, 2, 3, 4, 5])
        hist_nbrs = {0: {1, 3, 4}, 1: {0, 2, 5}, 2: {0, 1}}
        active = np.array([0, 1, 2, 3, 4, 5])
        config = ExperimentConfig(n_negatives=4, feature_mode="extended")

        ranks, n_skipped = evaluate_ranking_for_day(
            model, "logreg", snapshot, node_feats,
            hist_nbrs, active, config, seed=42
        )
        assert len(ranks) > 0
        assert all(r >= 1 for r in ranks)
        assert all(r <= 5 for r in ranks)


class TestHeuristicHelpers:

    def test_build_adjacency(self):
        from src.baselines.heuristic_baselines import _build_adjacency

        snap = _make_snapshot([(0, 1), (1, 2), (2, 0)])
        adj, nodes = _build_adjacency({"d1": snap})
        assert len(nodes) == 3
        assert adj.shape == (3, 3)
        assert adj.nnz > 0

    def test_common_neighbors(self):
        from src.baselines.heuristic_baselines import (
            _build_adjacency, compute_common_neighbors,
        )
        snap = _make_snapshot([(0, 1), (0, 2), (1, 2)])
        adj, nodes = _build_adjacency({"d1": snap})
        node_to_idx = {n: i for i, n in enumerate(nodes)}
        src = np.array([node_to_idx[0]])
        dst = np.array([node_to_idx[1]])
        scores = compute_common_neighbors(adj, src, dst)
        assert scores[0] > 0

    def test_jaccard(self):
        from src.baselines.heuristic_baselines import (
            _build_adjacency, compute_jaccard,
        )
        snap = _make_snapshot([(0, 1), (0, 2), (1, 2)])
        adj, nodes = _build_adjacency({"d1": snap})
        node_to_idx = {n: i for i, n in enumerate(nodes)}
        src = np.array([node_to_idx[0]])
        dst = np.array([node_to_idx[1]])
        scores = compute_jaccard(adj, src, dst)
        assert 0 < scores[0] <= 1.0

    def test_adamic_adar(self):
        from src.baselines.heuristic_baselines import (
            _build_adjacency, compute_adamic_adar,
        )
        snap = _make_snapshot([(0, 1), (0, 2), (1, 2)])
        adj, nodes = _build_adjacency({"d1": snap})
        node_to_idx = {n: i for i, n in enumerate(nodes)}
        src = np.array([node_to_idx[0]])
        dst = np.array([node_to_idx[1]])
        scores = compute_adamic_adar(adj, src, dst)
        assert scores[0] >= 0

    def test_preferential_attachment(self):
        from src.baselines.heuristic_baselines import (
            _build_adjacency, compute_preferential_attachment,
        )
        snap = _make_snapshot([(0, 1), (0, 2), (1, 2)])
        adj, nodes = _build_adjacency({"d1": snap})
        node_to_idx = {n: i for i, n in enumerate(nodes)}
        src = np.array([node_to_idx[0]])
        dst = np.array([node_to_idx[1]])
        scores = compute_preferential_attachment(adj, src, dst)
        assert scores[0] > 0

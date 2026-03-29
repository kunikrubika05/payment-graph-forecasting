"""Tests for stream graph baselines pipeline.

Tests correctness, data leakage prevention, and evaluation protocol.
"""

import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from sg_baselines.config import (
    ExperimentConfig,
    PERIODS,
    TRAIN_RATIO,
    VAL_RATIO,
    get_experiment_configs,
)
from sg_baselines.data import build_train_neighbor_sets, split_stream_graph
from sg_baselines.features import (
    N_TOTAL_FEATURES,
    build_pair_features,
    get_feature_names,
    _lookup_node_features,
    _compute_pair_features,
)
from sg_baselines.heuristics import (
    _compute_rank,
    _map_to_local,
    compute_jaccard,
    compute_pa,
)
from sg_baselines.sampling import (
    sample_negatives_for_eval,
    sample_negatives_for_training,
    _sample_random,
)


def _make_stream_graph(n_edges=1000, n_nodes=100, seed=42):
    """Create a synthetic stream graph DataFrame."""
    rng = np.random.RandomState(seed)
    src = rng.randint(0, n_nodes, size=n_edges).astype(np.int64)
    dst = rng.randint(0, n_nodes, size=n_edges).astype(np.int64)
    mask = src != dst
    src, dst = src[mask], dst[mask]
    ts = np.sort(rng.randint(1_000_000, 2_000_000, size=len(src))).astype(np.int64)
    btc = rng.uniform(0.001, 1.0, size=len(src)).astype(np.float32)
    usd = btc * 50000
    return pd.DataFrame({
        "src_idx": src,
        "dst_idx": dst,
        "timestamp": ts,
        "btc": btc,
        "usd": usd.astype(np.float32),
    })


def _make_config(fraction=0.5):
    """Create a test experiment config."""
    return ExperimentConfig(
        period_name="test_period",
        fraction=fraction,
        label="test",
        train_ratio=0.70,
        val_ratio=0.15,
        n_negatives=10,
        negative_ratio=3,
        max_train_samples=500,
        hp_search_max_samples=200,
        random_seed=42,
    )


def _make_adjacency(n_nodes=50, n_edges=200, seed=42):
    """Create synthetic adjacency matrices and node mapping."""
    rng = np.random.RandomState(seed)
    node_mapping = np.sort(rng.choice(1000, size=n_nodes, replace=False)).astype(np.int64)
    src = rng.randint(0, n_nodes, size=n_edges)
    dst = rng.randint(0, n_nodes, size=n_edges)
    mask = src != dst
    src, dst = src[mask], dst[mask]
    edges = np.unique(np.column_stack([src, dst]), axis=0)
    ones = np.ones(len(edges), dtype=np.float32)
    adj_dir = sparse.csr_matrix((ones, (edges[:, 0], edges[:, 1])), shape=(n_nodes, n_nodes))
    edges_undir = np.unique(np.concatenate([edges, edges[:, ::-1]], axis=0), axis=0)
    ones_u = np.ones(len(edges_undir), dtype=np.float32)
    adj_undir = sparse.csr_matrix(
        (ones_u, (edges_undir[:, 0], edges_undir[:, 1])), shape=(n_nodes, n_nodes)
    )
    return node_mapping, adj_dir, adj_undir


def _make_node_features(node_mapping, n_features=15, seed=42):
    """Create synthetic node features."""
    rng = np.random.RandomState(seed)
    features = rng.randn(len(node_mapping), n_features).astype(np.float32)
    return node_mapping.copy(), features


class TestConfig:
    def test_experiment_configs_generated(self):
        configs = get_experiment_configs()
        assert len(configs) == 2
        names = {c.period_name for c in configs}
        assert names == {"period_10", "period_25"}

    def test_config_ratios_sum_to_one(self):
        config = _make_config()
        assert abs(config.train_ratio + config.val_ratio + config.test_ratio - 1.0) < 1e-9

    def test_config_to_dict(self):
        config = _make_config()
        d = config.to_dict()
        assert "period_name" in d
        assert "test_ratio" in d
        assert d["test_ratio"] == pytest.approx(0.15)

    def test_periods_match_labels(self):
        for name, params in PERIODS.items():
            assert "fraction" in params
            assert "label" in params


class TestSplitStreamGraph:
    def test_basic_split(self):
        df = _make_stream_graph(n_edges=1000)
        config = _make_config(fraction=0.5)
        train, val, test = split_stream_graph(df, config)
        total = len(train) + len(val) + len(test)
        assert total == int(len(df) * 0.5)

    def test_chronological_order(self):
        df = _make_stream_graph(n_edges=1000)
        config = _make_config(fraction=0.8)
        train, val, test = split_stream_graph(df, config)
        assert train["timestamp"].iloc[-1] <= val["timestamp"].iloc[0]
        assert val["timestamp"].iloc[-1] <= test["timestamp"].iloc[0]

    def test_no_overlap(self):
        df = _make_stream_graph(n_edges=1000)
        config = _make_config(fraction=0.5)
        train, val, test = split_stream_graph(df, config)
        train_idx = set(train.index)
        val_idx = set(val.index)
        test_idx = set(test.index)
        assert len(train_idx & val_idx) == 0
        assert len(train_idx & test_idx) == 0
        assert len(val_idx & test_idx) == 0

    def test_proportions(self):
        df = _make_stream_graph(n_edges=10000)
        config = _make_config(fraction=1.0)
        train, val, test = split_stream_graph(df, config)
        total = len(train) + len(val) + len(test)
        assert abs(len(train) / total - 0.70) < 0.02
        assert abs(len(val) / total - 0.15) < 0.02
        assert abs(len(test) / total - 0.15) < 0.02


class TestBuildTrainNeighbors:
    def test_basic(self):
        df = pd.DataFrame({
            "src_idx": [1, 1, 2, 3],
            "dst_idx": [10, 20, 10, 30],
            "timestamp": [1, 2, 3, 4],
        })
        neighbors = build_train_neighbor_sets(df)
        assert neighbors[1] == {10, 20}
        assert neighbors[2] == {10}
        assert neighbors[3] == {30}

    def test_no_reverse(self):
        df = pd.DataFrame({
            "src_idx": [1],
            "dst_idx": [2],
            "timestamp": [1],
        })
        neighbors = build_train_neighbor_sets(df)
        assert 1 in neighbors
        assert 2 not in neighbors


class TestFeatures:
    def test_feature_names_count(self):
        names = get_feature_names()
        assert len(names) == N_TOTAL_FEATURES
        assert len(names) == 34

    def test_build_pair_features_shape(self):
        node_mapping, adj_dir, adj_undir = _make_adjacency(n_nodes=50)
        node_idx, node_features = _make_node_features(node_mapping)
        src = node_mapping[:5]
        dst = node_mapping[5:10]
        X = build_pair_features(src, dst, node_idx, node_features, node_mapping, adj_dir, adj_undir)
        assert X.shape == (5, 34)
        assert X.dtype == np.float32

    def test_unknown_nodes_get_zeros(self):
        node_mapping, adj_dir, adj_undir = _make_adjacency(n_nodes=50)
        node_idx, node_features = _make_node_features(node_mapping)
        unknown_src = np.array([999999, 999998], dtype=np.int64)
        known_dst = node_mapping[:2]
        X = build_pair_features(unknown_src, known_dst, node_idx, node_features, node_mapping, adj_dir, adj_undir)
        assert X.shape == (2, 34)
        assert np.all(X[:, :15] == 0)

    def test_lookup_correct(self):
        node_idx = np.array([10, 20, 30], dtype=np.int64)
        features = np.array([[1, 2], [3, 4], [5, 6]], dtype=np.float32)
        result = _lookup_node_features(np.array([20, 10, 99], dtype=np.int64), node_idx, features)
        np.testing.assert_array_equal(result[0], [3, 4])
        np.testing.assert_array_equal(result[1], [1, 2])
        np.testing.assert_array_equal(result[2], [0, 0])


class TestSampling:
    def test_train_negatives_count(self):
        src = np.array([1, 1, 2], dtype=np.int64)
        dst = np.array([10, 20, 30], dtype=np.int64)
        neighbors = {1: {10, 20, 40, 50}, 2: {30, 60}}
        active = np.arange(100, dtype=np.int64)
        rng = np.random.RandomState(42)
        all_src, all_dst, all_labels = sample_negatives_for_training(
            src, dst, neighbors, active, negative_ratio=3, rng=rng,
        )
        n_pos = 3
        n_neg = 3 * 3
        assert np.sum(all_labels == 1) == n_pos
        assert np.sum(all_labels == 0) == n_neg

    def test_train_negatives_no_positive_overlap(self):
        src = np.array([1, 1], dtype=np.int64)
        dst = np.array([10, 20], dtype=np.int64)
        neighbors = {1: {10, 20, 30, 40}}
        active = np.arange(100, dtype=np.int64)
        rng = np.random.RandomState(42)
        all_src, all_dst, all_labels = sample_negatives_for_training(
            src, dst, neighbors, active, negative_ratio=5, rng=rng,
        )
        neg_mask = all_labels == 0
        neg_src = all_src[neg_mask]
        neg_dst = all_dst[neg_mask]
        for s, d in zip(neg_src, neg_dst):
            if s == 1:
                assert d not in {10, 20}, "Negative overlaps with positive!"

    def test_eval_negatives_count(self):
        rng = np.random.RandomState(42)
        active = np.arange(1000, dtype=np.int64)
        neighbors = {5: {10, 20, 30, 40, 50, 60, 70, 80, 90, 100}}
        negs = sample_negatives_for_eval(
            5, 10, neighbors, {10, 11}, active, n_negatives=20, rng=rng,
        )
        assert len(negs) == 20

    def test_eval_negatives_exclude_true(self):
        rng = np.random.RandomState(42)
        active = np.arange(1000, dtype=np.int64)
        neighbors = {5: set(range(10, 100))}
        negs = sample_negatives_for_eval(
            5, 10, neighbors, {10}, active, n_negatives=50, rng=rng,
        )
        assert 10 not in negs
        assert 5 not in negs

    def test_sample_random_excludes(self):
        active = np.arange(10, dtype=np.int64)
        exclude = {0, 1, 2, 3, 4}
        rng = np.random.RandomState(42)
        result = _sample_random(active, exclude.copy(), 3, rng)
        assert len(result) == 3
        for r in result:
            assert r not in {0, 1, 2, 3, 4}


class TestHeuristics:
    def _adj(self):
        edges = np.array([[0, 1], [0, 2], [1, 2], [1, 3], [2, 3]])
        n = 4
        src, dst = edges[:, 0], edges[:, 1]
        both = np.concatenate([edges, edges[:, ::-1]], axis=0)
        both = np.unique(both, axis=0)
        ones = np.ones(len(both), dtype=np.float32)
        return sparse.csr_matrix((ones, (both[:, 0], both[:, 1])), shape=(n, n))

    def test_jaccard_range(self):
        adj = self._adj()
        src = np.array([0, 0, 1])
        dst = np.array([1, 3, 3])
        scores = compute_jaccard(adj, src, dst)
        assert scores.dtype == np.float32
        assert np.all(scores >= 0)
        assert np.all(scores <= 1)

    def test_pa_positive(self):
        adj = self._adj()
        src = np.array([0, 1])
        dst = np.array([3, 3])
        scores = compute_pa(adj, src, dst)
        assert np.all(scores > 0)

    def test_compute_rank_best(self):
        scores = np.array([10.0, 5.0, 3.0, 1.0])
        assert _compute_rank(scores) == 1

    def test_compute_rank_worst(self):
        scores = np.array([1.0, 5.0, 3.0, 10.0])
        assert _compute_rank(scores) == 4

    def test_compute_rank_tie(self):
        scores = np.array([5.0, 5.0, 5.0])
        assert _compute_rank(scores) == 1

    def test_map_to_local_valid(self):
        mapping = np.array([10, 20, 30, 40], dtype=np.int64)
        src = np.array([10, 30], dtype=np.int64)
        dst = np.array([20, 40], dtype=np.int64)
        sl, dl, valid = _map_to_local(src, dst, mapping)
        assert np.all(valid)
        np.testing.assert_array_equal(sl, [0, 2])
        np.testing.assert_array_equal(dl, [1, 3])

    def test_map_to_local_unknown(self):
        mapping = np.array([10, 20, 30], dtype=np.int64)
        src = np.array([10, 999], dtype=np.int64)
        dst = np.array([20, 20], dtype=np.int64)
        sl, dl, valid = _map_to_local(src, dst, mapping)
        assert valid[0] == True
        assert valid[1] == False


class TestNoDataLeakage:
    """Tests that verify NO data leakage in the pipeline."""

    def test_train_val_test_timestamps_non_overlapping(self):
        df = _make_stream_graph(n_edges=5000)
        config = _make_config(fraction=0.8)
        train, val, test = split_stream_graph(df, config)
        assert train["timestamp"].max() <= val["timestamp"].min()
        assert val["timestamp"].max() <= test["timestamp"].min()

    def test_train_neighbors_only_from_train(self):
        df = _make_stream_graph(n_edges=1000)
        config = _make_config(fraction=0.5)
        train, val, test = split_stream_graph(df, config)
        neighbors = build_train_neighbor_sets(train)
        train_edges = set(zip(train["src_idx"].values, train["dst_idx"].values))
        for src, dsts in neighbors.items():
            for d in dsts:
                assert (src, d) in train_edges

    def test_eval_negatives_exclude_all_positives_of_source(self):
        rng = np.random.RandomState(42)
        active = np.arange(1000, dtype=np.int64)
        positives_of_src = {10, 20, 30}
        neighbors = {5: set(range(50, 100))}
        for dst_true in [10, 20, 30]:
            negs = sample_negatives_for_eval(
                5, dst_true, neighbors, positives_of_src,
                active, n_negatives=50, rng=np.random.RandomState(42),
            )
            for d in positives_of_src:
                assert d not in negs, f"Positive {d} found in negatives!"

    def test_features_unknown_nodes_zero(self):
        node_mapping, adj_dir, adj_undir = _make_adjacency(n_nodes=20)
        node_idx, node_features = _make_node_features(node_mapping)
        unknown = np.array([999999], dtype=np.int64)
        known = node_mapping[:1]
        X = build_pair_features(unknown, known, node_idx, node_features, node_mapping, adj_dir, adj_undir)
        assert np.all(X[0, :15] == 0)
        assert np.all(X[0, 30:] == 0)

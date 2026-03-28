"""Tests for scripts/compute_stream_node_features.py."""

import json
import os
import tempfile

import numpy as np
import pandas as pd
import pytest

from scripts.compute_stream_node_features import (
    FEATURE_COLUMNS,
    compute_node_features,
    process_period,
    _normalized_entropy,
)


def _make_stream_graph(n_edges=100, n_nodes=20, seed=42):
    """Create a synthetic stream graph DataFrame."""
    rng = np.random.RandomState(seed)
    src = rng.randint(0, n_nodes, size=n_edges).astype(np.int64)
    dst = rng.randint(0, n_nodes, size=n_edges).astype(np.int64)
    ts_base = 1590000000
    ts = np.sort(rng.randint(ts_base, ts_base + 86400 * 90, size=n_edges)).astype(np.int64)
    btc = rng.exponential(0.5, size=n_edges).astype(np.float32)
    usd = (btc * 9500).astype(np.float32)
    return pd.DataFrame({
        "src_idx": src,
        "dst_idx": dst,
        "timestamp": ts,
        "btc": btc,
        "usd": usd,
    })


class TestComputeNodeFeatures:
    """Tests for the core compute_node_features function."""

    def test_output_shape(self):
        df = _make_stream_graph(100, 20)
        feat = compute_node_features(
            df["src_idx"].values, df["dst_idx"].values,
            df["timestamp"].values, df["btc"].values, 20,
        )
        assert feat.shape == (20, 15)

    def test_output_dtype(self):
        df = _make_stream_graph(100, 20)
        feat = compute_node_features(
            df["src_idx"].values, df["dst_idx"].values,
            df["timestamp"].values, df["btc"].values, 20,
        )
        assert feat.dtype == np.float32

    def test_all_finite(self):
        df = _make_stream_graph(100, 20)
        feat = compute_node_features(
            df["src_idx"].values, df["dst_idx"].values,
            df["timestamp"].values, df["btc"].values, 20,
        )
        assert np.isfinite(feat).all()

    def test_inactive_nodes_defaults(self):
        """Nodes not in train edges should have default features.

        in_out_ratio=0.5 (spec default) and recency=1.0 (most distant).
        All other features should be 0.
        """
        num_nodes = 50
        src = np.array([0, 0, 1, 1, 2], dtype=np.int64)
        dst = np.array([1, 2, 2, 3, 3], dtype=np.int64)
        ts = np.array([100, 200, 300, 400, 500], dtype=np.int64)
        btc = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
        feat = compute_node_features(src, dst, ts, btc, num_nodes)
        for node_id in range(4, num_nodes):
            assert abs(feat[node_id, 2] - 0.5) < 1e-6, "in_out_ratio should be 0.5"
            assert abs(feat[node_id, 9] - 1.0) < 1e-6, "recency should be 1.0"
            non_default_cols = [0, 1, 3, 4, 5, 6, 7, 8, 10, 11, 12, 13, 14]
            for c in non_default_cols:
                assert feat[node_id, c] == 0, f"Node {node_id} col {c} should be 0"

    def test_num_nodes_larger_than_max_index(self):
        """num_nodes can be larger than max node index."""
        src = np.array([0, 1], dtype=np.int64)
        dst = np.array([1, 2], dtype=np.int64)
        ts = np.array([100, 200], dtype=np.int64)
        btc = np.array([1.0, 1.0], dtype=np.float32)
        feat = compute_node_features(src, dst, ts, btc, 1000)
        assert feat.shape == (1000, 15)
        assert np.isfinite(feat).all()


class TestDegreeFeatures:
    """Tests for degree-related features."""

    def test_log_in_degree_single_edge(self):
        """Node with 1 incoming edge: log_in_degree = log1p(1) = log(2)."""
        src = np.array([0], dtype=np.int64)
        dst = np.array([1], dtype=np.int64)
        ts = np.array([100], dtype=np.int64)
        btc = np.array([1.0], dtype=np.float32)
        feat = compute_node_features(src, dst, ts, btc, 3)
        assert abs(feat[1, 0] - np.float32(np.log(2))) < 1e-6

    def test_log_out_degree_single_edge(self):
        """Node with 1 outgoing edge: log_out_degree = log1p(1) = log(2)."""
        src = np.array([0], dtype=np.int64)
        dst = np.array([1], dtype=np.int64)
        ts = np.array([100], dtype=np.int64)
        btc = np.array([1.0], dtype=np.float32)
        feat = compute_node_features(src, dst, ts, btc, 3)
        assert abs(feat[0, 1] - np.float32(np.log(2))) < 1e-6

    def test_in_out_ratio_sender_only(self):
        """Node that only sends: in_out_ratio = 1.0 (all outgoing)."""
        src = np.array([0, 0, 0], dtype=np.int64)
        dst = np.array([1, 2, 3], dtype=np.int64)
        ts = np.array([100, 200, 300], dtype=np.int64)
        btc = np.array([1.0, 1.0, 1.0], dtype=np.float32)
        feat = compute_node_features(src, dst, ts, btc, 5)
        assert abs(feat[0, 2] - 1.0) < 1e-6

    def test_in_out_ratio_receiver_only(self):
        """Node that only receives: in_out_ratio = 0.0."""
        src = np.array([1, 2, 3], dtype=np.int64)
        dst = np.array([0, 0, 0], dtype=np.int64)
        ts = np.array([100, 200, 300], dtype=np.int64)
        btc = np.array([1.0, 1.0, 1.0], dtype=np.float32)
        feat = compute_node_features(src, dst, ts, btc, 5)
        assert abs(feat[0, 2] - 0.0) < 1e-6

    def test_in_out_ratio_inactive(self):
        """Inactive node: in_out_ratio = 0.5."""
        src = np.array([0], dtype=np.int64)
        dst = np.array([1], dtype=np.int64)
        ts = np.array([100], dtype=np.int64)
        btc = np.array([1.0], dtype=np.float32)
        feat = compute_node_features(src, dst, ts, btc, 5)
        assert abs(feat[4, 2] - 0.5) < 1e-6


class TestVolumeFeatures:
    """Tests for BTC volume features."""

    def test_total_btc_in(self):
        """Sum of incoming BTC."""
        src = np.array([0, 1], dtype=np.int64)
        dst = np.array([2, 2], dtype=np.int64)
        ts = np.array([100, 200], dtype=np.int64)
        btc = np.array([3.0, 5.0], dtype=np.float32)
        feat = compute_node_features(src, dst, ts, btc, 3)
        expected = np.float32(np.log1p(8.0))
        assert abs(feat[2, 5] - expected) < 1e-5

    def test_avg_btc_out(self):
        """Average outgoing BTC."""
        src = np.array([0, 0], dtype=np.int64)
        dst = np.array([1, 2], dtype=np.int64)
        ts = np.array([100, 200], dtype=np.int64)
        btc = np.array([4.0, 6.0], dtype=np.float32)
        feat = compute_node_features(src, dst, ts, btc, 3)
        expected = np.float32(np.log1p(5.0))
        assert abs(feat[0, 8] - expected) < 1e-5


class TestTemporalFeatures:
    """Tests for temporal pattern features."""

    def test_recency_last_event(self):
        """Node whose last event is at t_split: recency = 0."""
        src = np.array([0, 0], dtype=np.int64)
        dst = np.array([1, 1], dtype=np.int64)
        ts = np.array([100, 500], dtype=np.int64)
        btc = np.array([1.0, 1.0], dtype=np.float32)
        feat = compute_node_features(src, dst, ts, btc, 3)
        assert abs(feat[0, 9] - 0.0) < 1e-6
        assert abs(feat[1, 9] - 0.0) < 1e-6

    def test_burstiness_equal_intervals(self):
        """Equal inter-event intervals: std=0, burstiness ~ -1."""
        src = np.array([0, 0, 0, 0, 0], dtype=np.int64)
        dst = np.array([1, 1, 1, 1, 1], dtype=np.int64)
        ts = np.array([100, 200, 300, 400, 500], dtype=np.int64)
        btc = np.array([1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float32)
        feat = compute_node_features(src, dst, ts, btc, 3)
        assert feat[0, 12] < -0.99, f"Expected burstiness ~ -1, got {feat[0, 12]}"

    def test_burstiness_few_events(self):
        """Node with < 3 events (as src+dst combined): burstiness = 0 if < 2 intervals."""
        src = np.array([0], dtype=np.int64)
        dst = np.array([1], dtype=np.int64)
        ts = np.array([100], dtype=np.int64)
        btc = np.array([1.0], dtype=np.float32)
        feat = compute_node_features(src, dst, ts, btc, 3)
        assert abs(feat[0, 12] - 0.0) < 1e-6

    def test_activity_span_single_event(self):
        """Node with one event: activity_span = 0 (t_last == t_first)."""
        src = np.array([0, 1], dtype=np.int64)
        dst = np.array([2, 2], dtype=np.int64)
        ts = np.array([100, 500], dtype=np.int64)
        btc = np.array([1.0, 1.0], dtype=np.float32)
        feat = compute_node_features(src, dst, ts, btc, 5)
        assert abs(feat[0, 10] - 0.0) < 1e-6


class TestEntropyFeatures:
    """Tests for counterparty entropy features."""

    def test_single_counterparty_entropy_zero(self):
        """Node sending to 1 counterparty: entropy = 0."""
        src = np.array([0, 0, 0], dtype=np.int64)
        dst = np.array([1, 1, 1], dtype=np.int64)
        ts = np.array([100, 200, 300], dtype=np.int64)
        btc = np.array([1.0, 1.0, 1.0], dtype=np.float32)
        feat = compute_node_features(src, dst, ts, btc, 3)
        assert abs(feat[0, 13] - 0.0) < 1e-6

    def test_two_equal_counterparties_entropy_one(self):
        """Node sending equally to 2 counterparties: normalized entropy = 1.0."""
        src = np.array([0, 0, 0, 0], dtype=np.int64)
        dst = np.array([1, 2, 1, 2], dtype=np.int64)
        ts = np.array([100, 200, 300, 400], dtype=np.int64)
        btc = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32)
        feat = compute_node_features(src, dst, ts, btc, 3)
        assert abs(feat[0, 13] - 1.0) < 1e-5

    def test_in_entropy_two_equal_sources(self):
        """Node receiving equally from 2 sources: normalized in_entropy = 1.0."""
        src = np.array([1, 2, 1, 2], dtype=np.int64)
        dst = np.array([0, 0, 0, 0], dtype=np.int64)
        ts = np.array([100, 200, 300, 400], dtype=np.int64)
        btc = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32)
        feat = compute_node_features(src, dst, ts, btc, 3)
        assert abs(feat[0, 14] - 1.0) < 1e-5


class TestProcessPeriod:
    """Tests for process_period function."""

    def test_output_files_created(self):
        df = _make_stream_graph(200, 30)
        with tempfile.TemporaryDirectory() as tmpdir:
            meta = process_period(df, 0.50, 0.70, 30, tmpdir, "features_test")
            assert os.path.exists(os.path.join(tmpdir, "features_test.parquet"))
            assert os.path.exists(os.path.join(tmpdir, "features_test.json"))

    def test_metadata_fields(self):
        df = _make_stream_graph(200, 30)
        with tempfile.TemporaryDirectory() as tmpdir:
            meta = process_period(df, 0.50, 0.70, 30, tmpdir, "features_test")
            assert meta["period_fraction"] == 0.50
            assert meta["train_ratio"] == 0.70
            assert meta["num_edges_period"] == 100
            assert meta["num_edges_train"] == 70
            assert meta["num_nodes_total"] == 30
            assert meta["feature_columns"] == FEATURE_COLUMNS

    def test_parquet_shape(self):
        df = _make_stream_graph(200, 30)
        with tempfile.TemporaryDirectory() as tmpdir:
            process_period(df, 0.50, 0.70, 30, tmpdir, "features_test")
            result = pd.read_parquet(os.path.join(tmpdir, "features_test.parquet"))
            assert result.shape == (30, 15)
            assert list(result.columns) == FEATURE_COLUMNS

    def test_parquet_dtype_float32(self):
        df = _make_stream_graph(200, 30)
        with tempfile.TemporaryDirectory() as tmpdir:
            process_period(df, 0.50, 0.70, 30, tmpdir, "features_test")
            result = pd.read_parquet(os.path.join(tmpdir, "features_test.parquet"))
            for col in FEATURE_COLUMNS:
                assert result[col].dtype == np.float32

    def test_json_matches_metadata(self):
        df = _make_stream_graph(200, 30)
        with tempfile.TemporaryDirectory() as tmpdir:
            meta = process_period(df, 0.50, 0.70, 30, tmpdir, "features_test")
            with open(os.path.join(tmpdir, "features_test.json")) as f:
                saved = json.load(f)
            assert saved["num_edges_train"] == meta["num_edges_train"]
            assert saved["num_nodes_total"] == meta["num_nodes_total"]

    def test_train_does_not_include_val_test(self):
        """Verify train edges are strictly from the first portion."""
        df = _make_stream_graph(1000, 50, seed=123)
        period_end = int(1000 * 0.25)
        train_end = int(period_end * 0.70)
        train_ts_max = df.iloc[train_end - 1]["timestamp"]
        val_ts_min = df.iloc[train_end]["timestamp"]
        assert train_ts_max <= val_ts_min


class TestFeatures10SubsetOf25:
    """Verify that active nodes in features_10 are a subset of features_25."""

    def test_active_nodes_subset(self):
        df = _make_stream_graph(1000, 50, seed=99)
        num_nodes = 50

        period_10 = df.iloc[:int(1000 * 0.10)]
        train_10 = period_10.iloc[:int(len(period_10) * 0.70)]
        active_10 = set(train_10["src_idx"].unique()) | set(train_10["dst_idx"].unique())

        period_25 = df.iloc[:int(1000 * 0.25)]
        train_25 = period_25.iloc[:int(len(period_25) * 0.70)]
        active_25 = set(train_25["src_idx"].unique()) | set(train_25["dst_idx"].unique())

        assert active_10.issubset(active_25), (
            f"Nodes in 10% train but not in 25% train: {active_10 - active_25}"
        )


class TestEdgeCases:
    """Test edge cases and robustness."""

    def test_single_edge(self):
        src = np.array([0], dtype=np.int64)
        dst = np.array([1], dtype=np.int64)
        ts = np.array([100], dtype=np.int64)
        btc = np.array([1.0], dtype=np.float32)
        feat = compute_node_features(src, dst, ts, btc, 5)
        assert feat.shape == (5, 15)
        assert np.isfinite(feat).all()

    def test_self_loop(self):
        src = np.array([0, 0], dtype=np.int64)
        dst = np.array([0, 1], dtype=np.int64)
        ts = np.array([100, 200], dtype=np.int64)
        btc = np.array([1.0, 1.0], dtype=np.float32)
        feat = compute_node_features(src, dst, ts, btc, 3)
        assert feat.shape == (3, 15)
        assert np.isfinite(feat).all()

    def test_same_timestamp(self):
        src = np.array([0, 1, 2], dtype=np.int64)
        dst = np.array([1, 2, 0], dtype=np.int64)
        ts = np.array([100, 100, 100], dtype=np.int64)
        btc = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        feat = compute_node_features(src, dst, ts, btc, 3)
        assert feat.shape == (3, 15)
        assert np.isfinite(feat).all()

    def test_large_btc_values(self):
        src = np.array([0, 0], dtype=np.int64)
        dst = np.array([1, 1], dtype=np.int64)
        ts = np.array([100, 200], dtype=np.int64)
        btc = np.array([1e6, 1e6], dtype=np.float32)
        feat = compute_node_features(src, dst, ts, btc, 3)
        assert np.isfinite(feat).all()

    def test_zero_btc(self):
        src = np.array([0, 0], dtype=np.int64)
        dst = np.array([1, 2], dtype=np.int64)
        ts = np.array([100, 200], dtype=np.int64)
        btc = np.array([0.0, 0.0], dtype=np.float32)
        feat = compute_node_features(src, dst, ts, btc, 3)
        assert np.isfinite(feat).all()
        assert feat[0, 6] == 0.0  # log1p(0) = 0

    def test_feature_columns_count(self):
        assert len(FEATURE_COLUMNS) == 15

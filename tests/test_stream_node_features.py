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
    load_node_features,
    process_period,
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
        active, feat = compute_node_features(
            df["src_idx"].values, df["dst_idx"].values,
            df["timestamp"].values, df["btc"].values, 20,
        )
        assert feat.shape[1] == 15
        assert len(active) == feat.shape[0]

    def test_output_dtype(self):
        df = _make_stream_graph(100, 20)
        active, feat = compute_node_features(
            df["src_idx"].values, df["dst_idx"].values,
            df["timestamp"].values, df["btc"].values, 20,
        )
        assert feat.dtype == np.float32

    def test_all_finite(self):
        df = _make_stream_graph(100, 20)
        active, feat = compute_node_features(
            df["src_idx"].values, df["dst_idx"].values,
            df["timestamp"].values, df["btc"].values, 20,
        )
        assert np.isfinite(feat).all()

    def test_only_active_nodes_returned(self):
        """Only nodes appearing in edges are in the output."""
        src = np.array([0, 0, 1, 1, 2], dtype=np.int64)
        dst = np.array([1, 2, 2, 3, 3], dtype=np.int64)
        ts = np.array([100, 200, 300, 400, 500], dtype=np.int64)
        btc = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
        active, feat = compute_node_features(src, dst, ts, btc, 50)
        assert set(active) == {0, 1, 2, 3}
        assert feat.shape == (4, 15)

    def test_inactive_nodes_zero_via_load(self):
        """Inactive nodes get zero features when loaded into dense array."""
        src = np.array([0, 0, 1, 1, 2], dtype=np.int64)
        dst = np.array([1, 2, 2, 3, 3], dtype=np.int64)
        ts = np.array([100, 200, 300, 400, 500], dtype=np.int64)
        btc = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
        active, feat = compute_node_features(src, dst, ts, btc, 50)
        with tempfile.TemporaryDirectory() as tmpdir:
            feat_df = pd.DataFrame(feat, columns=FEATURE_COLUMNS)
            feat_df.insert(0, "node_idx", active)
            path = os.path.join(tmpdir, "test.parquet")
            feat_df.to_parquet(path, index=False)
            dense = load_node_features(path, 50)
            for node_id in range(4, 50):
                assert np.all(dense[node_id] == 0), f"Node {node_id} should be all zeros"

    def test_global_indices_preserved(self):
        """Active node indices match the global indices from src/dst."""
        src = np.array([10, 20, 30], dtype=np.int64)
        dst = np.array([40, 50, 60], dtype=np.int64)
        ts = np.array([100, 200, 300], dtype=np.int64)
        btc = np.array([1.0, 1.0, 1.0], dtype=np.float32)
        active, feat = compute_node_features(src, dst, ts, btc, 100)
        assert set(active) == {10, 20, 30, 40, 50, 60}
        assert feat.shape == (6, 15)


class TestDegreeFeatures:
    """Tests for degree-related features."""

    def _get_feat_for_node(self, active, feat, node):
        idx = np.where(active == node)[0][0]
        return feat[idx]

    def test_log_in_degree_single_edge(self):
        """Node with 1 incoming edge: log_in_degree = log1p(1) = log(2)."""
        src = np.array([0], dtype=np.int64)
        dst = np.array([1], dtype=np.int64)
        ts = np.array([100], dtype=np.int64)
        btc = np.array([1.0], dtype=np.float32)
        active, feat = compute_node_features(src, dst, ts, btc, 3)
        f1 = self._get_feat_for_node(active, feat, 1)
        assert abs(f1[0] - np.float32(np.log(2))) < 1e-6

    def test_log_out_degree_single_edge(self):
        """Node with 1 outgoing edge: log_out_degree = log1p(1) = log(2)."""
        src = np.array([0], dtype=np.int64)
        dst = np.array([1], dtype=np.int64)
        ts = np.array([100], dtype=np.int64)
        btc = np.array([1.0], dtype=np.float32)
        active, feat = compute_node_features(src, dst, ts, btc, 3)
        f0 = self._get_feat_for_node(active, feat, 0)
        assert abs(f0[1] - np.float32(np.log(2))) < 1e-6

    def test_in_out_ratio_sender_only(self):
        """Node that only sends: in_out_ratio = 1.0 (all outgoing)."""
        src = np.array([0, 0, 0], dtype=np.int64)
        dst = np.array([1, 2, 3], dtype=np.int64)
        ts = np.array([100, 200, 300], dtype=np.int64)
        btc = np.array([1.0, 1.0, 1.0], dtype=np.float32)
        active, feat = compute_node_features(src, dst, ts, btc, 5)
        f0 = self._get_feat_for_node(active, feat, 0)
        assert abs(f0[2] - 1.0) < 1e-6

    def test_in_out_ratio_receiver_only(self):
        """Node that only receives: in_out_ratio = 0.0."""
        src = np.array([1, 2, 3], dtype=np.int64)
        dst = np.array([0, 0, 0], dtype=np.int64)
        ts = np.array([100, 200, 300], dtype=np.int64)
        btc = np.array([1.0, 1.0, 1.0], dtype=np.float32)
        active, feat = compute_node_features(src, dst, ts, btc, 5)
        f0 = self._get_feat_for_node(active, feat, 0)
        assert abs(f0[2] - 0.0) < 1e-6


class TestVolumeFeatures:
    """Tests for BTC volume features."""

    def _get_feat_for_node(self, active, feat, node):
        idx = np.where(active == node)[0][0]
        return feat[idx]

    def test_total_btc_in(self):
        """Sum of incoming BTC."""
        src = np.array([0, 1], dtype=np.int64)
        dst = np.array([2, 2], dtype=np.int64)
        ts = np.array([100, 200], dtype=np.int64)
        btc = np.array([3.0, 5.0], dtype=np.float32)
        active, feat = compute_node_features(src, dst, ts, btc, 3)
        f2 = self._get_feat_for_node(active, feat, 2)
        expected = np.float32(np.log1p(8.0))
        assert abs(f2[5] - expected) < 1e-5

    def test_avg_btc_out(self):
        """Average outgoing BTC."""
        src = np.array([0, 0], dtype=np.int64)
        dst = np.array([1, 2], dtype=np.int64)
        ts = np.array([100, 200], dtype=np.int64)
        btc = np.array([4.0, 6.0], dtype=np.float32)
        active, feat = compute_node_features(src, dst, ts, btc, 3)
        f0 = self._get_feat_for_node(active, feat, 0)
        expected = np.float32(np.log1p(5.0))
        assert abs(f0[8] - expected) < 1e-5


class TestTemporalFeatures:
    """Tests for temporal pattern features."""

    def _get_feat_for_node(self, active, feat, node):
        idx = np.where(active == node)[0][0]
        return feat[idx]

    def test_recency_last_event(self):
        """Node whose last event is at t_split: recency = 0."""
        src = np.array([0, 0], dtype=np.int64)
        dst = np.array([1, 1], dtype=np.int64)
        ts = np.array([100, 500], dtype=np.int64)
        btc = np.array([1.0, 1.0], dtype=np.float32)
        active, feat = compute_node_features(src, dst, ts, btc, 3)
        f0 = self._get_feat_for_node(active, feat, 0)
        f1 = self._get_feat_for_node(active, feat, 1)
        assert abs(f0[9] - 0.0) < 1e-6
        assert abs(f1[9] - 0.0) < 1e-6

    def test_burstiness_equal_intervals(self):
        """Equal inter-event intervals: std=0, burstiness ~ -1."""
        src = np.array([0, 0, 0, 0, 0], dtype=np.int64)
        dst = np.array([1, 1, 1, 1, 1], dtype=np.int64)
        ts = np.array([100, 200, 300, 400, 500], dtype=np.int64)
        btc = np.array([1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float32)
        active, feat = compute_node_features(src, dst, ts, btc, 3)
        f0 = self._get_feat_for_node(active, feat, 0)
        assert f0[12] < -0.99, f"Expected burstiness ~ -1, got {f0[12]}"

    def test_burstiness_few_events(self):
        """Node with single event: burstiness = 0 (no intervals)."""
        src = np.array([0], dtype=np.int64)
        dst = np.array([1], dtype=np.int64)
        ts = np.array([100], dtype=np.int64)
        btc = np.array([1.0], dtype=np.float32)
        active, feat = compute_node_features(src, dst, ts, btc, 3)
        f0 = self._get_feat_for_node(active, feat, 0)
        assert abs(f0[12] - 0.0) < 1e-6

    def test_activity_span_single_event(self):
        """Node with one event: activity_span = 0 (t_last == t_first)."""
        src = np.array([0, 1], dtype=np.int64)
        dst = np.array([2, 2], dtype=np.int64)
        ts = np.array([100, 500], dtype=np.int64)
        btc = np.array([1.0, 1.0], dtype=np.float32)
        active, feat = compute_node_features(src, dst, ts, btc, 5)
        f0 = self._get_feat_for_node(active, feat, 0)
        assert abs(f0[10] - 0.0) < 1e-6


class TestEntropyFeatures:
    """Tests for counterparty entropy features."""

    def _get_feat_for_node(self, active, feat, node):
        idx = np.where(active == node)[0][0]
        return feat[idx]

    def test_single_counterparty_entropy_zero(self):
        """Node sending to 1 counterparty: entropy = 0."""
        src = np.array([0, 0, 0], dtype=np.int64)
        dst = np.array([1, 1, 1], dtype=np.int64)
        ts = np.array([100, 200, 300], dtype=np.int64)
        btc = np.array([1.0, 1.0, 1.0], dtype=np.float32)
        active, feat = compute_node_features(src, dst, ts, btc, 3)
        f0 = self._get_feat_for_node(active, feat, 0)
        assert abs(f0[13] - 0.0) < 1e-6

    def test_two_equal_counterparties_entropy_one(self):
        """Node sending equally to 2 counterparties: normalized entropy = 1.0."""
        src = np.array([0, 0, 0, 0], dtype=np.int64)
        dst = np.array([1, 2, 1, 2], dtype=np.int64)
        ts = np.array([100, 200, 300, 400], dtype=np.int64)
        btc = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32)
        active, feat = compute_node_features(src, dst, ts, btc, 3)
        f0 = self._get_feat_for_node(active, feat, 0)
        assert abs(f0[13] - 1.0) < 1e-5

    def test_in_entropy_two_equal_sources(self):
        """Node receiving equally from 2 sources: normalized in_entropy = 1.0."""
        src = np.array([1, 2, 1, 2], dtype=np.int64)
        dst = np.array([0, 0, 0, 0], dtype=np.int64)
        ts = np.array([100, 200, 300, 400], dtype=np.int64)
        btc = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32)
        active, feat = compute_node_features(src, dst, ts, btc, 3)
        f0 = self._get_feat_for_node(active, feat, 0)
        assert abs(f0[14] - 1.0) < 1e-5


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
            assert meta["num_nodes_global"] == 30
            assert meta["feature_columns"] == FEATURE_COLUMNS

    def test_parquet_has_node_idx_column(self):
        df = _make_stream_graph(200, 30)
        with tempfile.TemporaryDirectory() as tmpdir:
            process_period(df, 0.50, 0.70, 30, tmpdir, "features_test")
            result = pd.read_parquet(os.path.join(tmpdir, "features_test.parquet"))
            assert "node_idx" in result.columns
            assert result["node_idx"].dtype == np.int64
            assert list(result.columns) == ["node_idx"] + FEATURE_COLUMNS

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
            assert saved["num_nodes_global"] == meta["num_nodes_global"]

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

        period_10 = df.iloc[:int(1000 * 0.10)]
        train_10 = period_10.iloc[:int(len(period_10) * 0.70)]
        active_10 = set(train_10["src_idx"].unique()) | set(train_10["dst_idx"].unique())

        period_25 = df.iloc[:int(1000 * 0.25)]
        train_25 = period_25.iloc[:int(len(period_25) * 0.70)]
        active_25 = set(train_25["src_idx"].unique()) | set(train_25["dst_idx"].unique())

        assert active_10.issubset(active_25), (
            f"Nodes in 10% train but not in 25% train: {active_10 - active_25}"
        )


class TestLoadNodeFeatures:
    """Tests for the load_node_features helper."""

    def test_dense_array_shape(self):
        src = np.array([0, 5, 10], dtype=np.int64)
        dst = np.array([1, 6, 11], dtype=np.int64)
        ts = np.array([100, 200, 300], dtype=np.int64)
        btc = np.array([1.0, 1.0, 1.0], dtype=np.float32)
        active, feat = compute_node_features(src, dst, ts, btc, 20)
        with tempfile.TemporaryDirectory() as tmpdir:
            feat_df = pd.DataFrame(feat, columns=FEATURE_COLUMNS)
            feat_df.insert(0, "node_idx", active)
            path = os.path.join(tmpdir, "test.parquet")
            feat_df.to_parquet(path, index=False)
            dense = load_node_features(path, 20)
            assert dense.shape == (20, 15)
            assert dense.dtype == np.float32

    def test_inactive_nodes_zero_in_dense(self):
        src = np.array([0, 1], dtype=np.int64)
        dst = np.array([2, 3], dtype=np.int64)
        ts = np.array([100, 200], dtype=np.int64)
        btc = np.array([1.0, 1.0], dtype=np.float32)
        active, feat = compute_node_features(src, dst, ts, btc, 10)
        with tempfile.TemporaryDirectory() as tmpdir:
            feat_df = pd.DataFrame(feat, columns=FEATURE_COLUMNS)
            feat_df.insert(0, "node_idx", active)
            path = os.path.join(tmpdir, "test.parquet")
            feat_df.to_parquet(path, index=False)
            dense = load_node_features(path, 10)
            for node_id in range(4, 10):
                assert np.all(dense[node_id] == 0), f"Node {node_id} should be all zeros"

    def test_active_nodes_nonzero_in_dense(self):
        src = np.array([0, 0, 1], dtype=np.int64)
        dst = np.array([1, 2, 2], dtype=np.int64)
        ts = np.array([100, 200, 300], dtype=np.int64)
        btc = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        active, feat = compute_node_features(src, dst, ts, btc, 10)
        with tempfile.TemporaryDirectory() as tmpdir:
            feat_df = pd.DataFrame(feat, columns=FEATURE_COLUMNS)
            feat_df.insert(0, "node_idx", active)
            path = os.path.join(tmpdir, "test.parquet")
            feat_df.to_parquet(path, index=False)
            dense = load_node_features(path, 10)
            for node_id in [0, 1, 2]:
                assert not np.all(dense[node_id] == 0), f"Node {node_id} should have features"


class TestEdgeCases:
    """Test edge cases and robustness."""

    def test_single_edge(self):
        src = np.array([0], dtype=np.int64)
        dst = np.array([1], dtype=np.int64)
        ts = np.array([100], dtype=np.int64)
        btc = np.array([1.0], dtype=np.float32)
        active, feat = compute_node_features(src, dst, ts, btc, 5)
        assert feat.shape == (2, 15)
        assert np.isfinite(feat).all()

    def test_self_loop(self):
        src = np.array([0, 0], dtype=np.int64)
        dst = np.array([0, 1], dtype=np.int64)
        ts = np.array([100, 200], dtype=np.int64)
        btc = np.array([1.0, 1.0], dtype=np.float32)
        active, feat = compute_node_features(src, dst, ts, btc, 3)
        assert np.isfinite(feat).all()

    def test_same_timestamp(self):
        src = np.array([0, 1, 2], dtype=np.int64)
        dst = np.array([1, 2, 0], dtype=np.int64)
        ts = np.array([100, 100, 100], dtype=np.int64)
        btc = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        active, feat = compute_node_features(src, dst, ts, btc, 3)
        assert np.isfinite(feat).all()

    def test_large_btc_values(self):
        src = np.array([0, 0], dtype=np.int64)
        dst = np.array([1, 1], dtype=np.int64)
        ts = np.array([100, 200], dtype=np.int64)
        btc = np.array([1e6, 1e6], dtype=np.float32)
        active, feat = compute_node_features(src, dst, ts, btc, 3)
        assert np.isfinite(feat).all()

    def test_zero_btc(self):
        src = np.array([0, 0], dtype=np.int64)
        dst = np.array([1, 2], dtype=np.int64)
        ts = np.array([100, 200], dtype=np.int64)
        btc = np.array([0.0, 0.0], dtype=np.float32)
        active, feat = compute_node_features(src, dst, ts, btc, 3)
        assert np.isfinite(feat).all()

    def test_high_node_indices(self):
        """Works with sparse high node indices (like 330M in ORBITAAL)."""
        src = np.array([1000000, 2000000], dtype=np.int64)
        dst = np.array([3000000, 4000000], dtype=np.int64)
        ts = np.array([100, 200], dtype=np.int64)
        btc = np.array([1.0, 1.0], dtype=np.float32)
        active, feat = compute_node_features(src, dst, ts, btc, 5000000)
        assert feat.shape == (4, 15)
        assert set(active) == {1000000, 2000000, 3000000, 4000000}
        assert np.isfinite(feat).all()

    def test_feature_columns_count(self):
        assert len(FEATURE_COLUMNS) == 15

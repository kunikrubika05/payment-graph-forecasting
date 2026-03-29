"""Tests for scripts/compute_stream_adjacency.py."""

import os
import tempfile

import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from scripts.compute_stream_adjacency import (
    build_adjacency_matrices,
    compute_aa,
    compute_cn,
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
        "src_idx": src, "dst_idx": dst, "timestamp": ts, "btc": btc, "usd": usd,
    })


class TestBuildAdjacencyMatrices:
    """Tests for build_adjacency_matrices."""

    def test_output_types(self):
        src = np.array([0, 1, 2], dtype=np.int64)
        dst = np.array([1, 2, 0], dtype=np.int64)
        mapping, adj_dir, adj_undir = build_adjacency_matrices(src, dst, 10)
        assert isinstance(mapping, np.ndarray)
        assert isinstance(adj_dir, sparse.csr_matrix)
        assert isinstance(adj_undir, sparse.csr_matrix)

    def test_square_shape(self):
        src = np.array([0, 1, 2], dtype=np.int64)
        dst = np.array([1, 2, 0], dtype=np.int64)
        mapping, adj_dir, adj_undir = build_adjacency_matrices(src, dst, 10)
        n = len(mapping)
        assert adj_dir.shape == (n, n)
        assert adj_undir.shape == (n, n)

    def test_only_active_nodes(self):
        src = np.array([10, 20], dtype=np.int64)
        dst = np.array([30, 40], dtype=np.int64)
        mapping, adj_dir, adj_undir = build_adjacency_matrices(src, dst, 100)
        assert set(mapping) == {10, 20, 30, 40}
        assert adj_dir.shape == (4, 4)

    def test_binary_values(self):
        src = np.array([0, 0, 0, 1], dtype=np.int64)
        dst = np.array([1, 1, 1, 2], dtype=np.int64)
        mapping, adj_dir, adj_undir = build_adjacency_matrices(src, dst, 5)
        assert adj_dir.max() == 1.0
        assert adj_undir.max() == 1.0

    def test_duplicate_edges_deduped(self):
        src = np.array([0, 0, 0], dtype=np.int64)
        dst = np.array([1, 1, 1], dtype=np.int64)
        mapping, adj_dir, adj_undir = build_adjacency_matrices(src, dst, 5)
        assert adj_dir.nnz == 1

    def test_directed_not_symmetric(self):
        src = np.array([0], dtype=np.int64)
        dst = np.array([1], dtype=np.int64)
        mapping, adj_dir, adj_undir = build_adjacency_matrices(src, dst, 5)
        local_0 = np.searchsorted(mapping, 0)
        local_1 = np.searchsorted(mapping, 1)
        assert adj_dir[local_0, local_1] == 1.0
        assert adj_dir[local_1, local_0] == 0.0

    def test_undirected_symmetric(self):
        src = np.array([0], dtype=np.int64)
        dst = np.array([1], dtype=np.int64)
        mapping, adj_dir, adj_undir = build_adjacency_matrices(src, dst, 5)
        diff = adj_undir - adj_undir.T
        assert diff.nnz == 0

    def test_undirected_has_both_directions(self):
        src = np.array([0], dtype=np.int64)
        dst = np.array([1], dtype=np.int64)
        mapping, adj_dir, adj_undir = build_adjacency_matrices(src, dst, 5)
        local_0 = np.searchsorted(mapping, 0)
        local_1 = np.searchsorted(mapping, 1)
        assert adj_undir[local_0, local_1] == 1.0
        assert adj_undir[local_1, local_0] == 1.0

    def test_mapping_sorted(self):
        src = np.array([5, 2, 8], dtype=np.int64)
        dst = np.array([3, 7, 1], dtype=np.int64)
        mapping, _, _ = build_adjacency_matrices(src, dst, 10)
        assert np.all(mapping[:-1] <= mapping[1:])

    def test_global_indices_preserved(self):
        src = np.array([100, 200], dtype=np.int64)
        dst = np.array([300, 400], dtype=np.int64)
        mapping, adj_dir, adj_undir = build_adjacency_matrices(src, dst, 500)
        assert set(mapping) == {100, 200, 300, 400}


class TestComputeCN:
    """Tests for Common Neighbors computation."""

    def test_no_common_neighbors(self):
        src = np.array([0, 1], dtype=np.int64)
        dst = np.array([2, 3], dtype=np.int64)
        mapping, _, adj = build_adjacency_matrices(src, dst, 5)
        local_0 = np.searchsorted(mapping, 0)
        local_1 = np.searchsorted(mapping, 1)
        cn = compute_cn(adj, np.array([local_0]), np.array([local_1]))
        assert cn[0] == 0.0

    def test_one_common_neighbor(self):
        src = np.array([0, 1], dtype=np.int64)
        dst = np.array([2, 2], dtype=np.int64)
        mapping, _, adj = build_adjacency_matrices(src, dst, 5)
        local_0 = np.searchsorted(mapping, 0)
        local_1 = np.searchsorted(mapping, 1)
        cn = compute_cn(adj, np.array([local_0]), np.array([local_1]))
        assert cn[0] == 1.0

    def test_two_common_neighbors(self):
        src = np.array([0, 0, 1, 1], dtype=np.int64)
        dst = np.array([2, 3, 2, 3], dtype=np.int64)
        mapping, _, adj = build_adjacency_matrices(src, dst, 5)
        local_0 = np.searchsorted(mapping, 0)
        local_1 = np.searchsorted(mapping, 1)
        cn = compute_cn(adj, np.array([local_0]), np.array([local_1]))
        assert cn[0] == 2.0

    def test_cn_batch(self):
        src = np.array([0, 0, 1, 1, 2], dtype=np.int64)
        dst = np.array([3, 4, 3, 4, 3], dtype=np.int64)
        mapping, _, adj = build_adjacency_matrices(src, dst, 5)
        local = np.searchsorted(mapping, [0, 1, 2])
        cn = compute_cn(adj, np.array([local[0], local[0]]), np.array([local[1], local[2]]))
        assert cn[0] == 2.0
        assert cn[1] == 1.0

    def test_cn_unknown_node_zero(self):
        """Nodes not in adjacency get CN=0 by construction (not in matrix)."""
        src = np.array([0, 1], dtype=np.int64)
        dst = np.array([2, 2], dtype=np.int64)
        mapping, _, adj = build_adjacency_matrices(src, dst, 5)
        local_0 = np.searchsorted(mapping, 0)
        local_1 = np.searchsorted(mapping, 1)
        cn = compute_cn(adj, np.array([local_0]), np.array([local_1]))
        assert cn.dtype == np.float32


class TestComputeAA:
    """Tests for Adamic-Adar computation."""

    def test_aa_one_common_neighbor(self):
        """AA with one common neighbor of degree 2: 1/log(2)."""
        src = np.array([0, 1], dtype=np.int64)
        dst = np.array([2, 2], dtype=np.int64)
        mapping, _, adj = build_adjacency_matrices(src, dst, 5)
        local_0 = np.searchsorted(mapping, 0)
        local_1 = np.searchsorted(mapping, 1)
        aa = compute_aa(adj, np.array([local_0]), np.array([local_1]))
        expected = 1.0 / np.log(2)
        assert abs(aa[0] - expected) < 1e-5

    def test_aa_hub_less_weight(self):
        """Common neighbor with higher degree contributes less to AA."""
        src = np.array([0, 1, 2, 3, 4], dtype=np.int64)
        dst = np.array([5, 5, 5, 5, 5], dtype=np.int64)
        mapping, _, adj = build_adjacency_matrices(src, dst, 10)
        local_0 = np.searchsorted(mapping, 0)
        local_1 = np.searchsorted(mapping, 1)
        aa = compute_aa(adj, np.array([local_0]), np.array([local_1]))
        expected = 1.0 / np.log(5)
        assert abs(aa[0] - expected) < 1e-5

    def test_aa_no_common_neighbors(self):
        src = np.array([0, 1], dtype=np.int64)
        dst = np.array([2, 3], dtype=np.int64)
        mapping, _, adj = build_adjacency_matrices(src, dst, 5)
        local_0 = np.searchsorted(mapping, 0)
        local_1 = np.searchsorted(mapping, 1)
        aa = compute_aa(adj, np.array([local_0]), np.array([local_1]))
        assert aa[0] == 0.0

    def test_aa_dtype_float32(self):
        src = np.array([0, 1], dtype=np.int64)
        dst = np.array([2, 2], dtype=np.int64)
        mapping, _, adj = build_adjacency_matrices(src, dst, 5)
        local_0 = np.searchsorted(mapping, 0)
        local_1 = np.searchsorted(mapping, 1)
        aa = compute_aa(adj, np.array([local_0]), np.array([local_1]))
        assert aa.dtype == np.float32


class TestProcessPeriod:
    """Tests for process_period function."""

    def test_output_files_created(self):
        df = _make_stream_graph(200, 30)
        with tempfile.TemporaryDirectory() as tmpdir:
            meta = process_period(df, 0.50, 0.70, 30, tmpdir, "test")
            assert os.path.exists(os.path.join(tmpdir, "adj_test_directed.npz"))
            assert os.path.exists(os.path.join(tmpdir, "adj_test_undirected.npz"))
            assert os.path.exists(os.path.join(tmpdir, "node_mapping_test.npy"))
            assert os.path.exists(os.path.join(tmpdir, "adj_test.json"))

    def test_metadata_fields(self):
        df = _make_stream_graph(200, 30)
        with tempfile.TemporaryDirectory() as tmpdir:
            meta = process_period(df, 0.50, 0.70, 30, tmpdir, "test")
            assert meta["period_fraction"] == 0.50
            assert meta["train_ratio"] == 0.70
            assert meta["num_edges_period"] == 100
            assert meta["num_edges_train"] == 70
            assert meta["num_nodes_global"] == 30

    def test_can_reload_matrices(self):
        df = _make_stream_graph(200, 30)
        with tempfile.TemporaryDirectory() as tmpdir:
            process_period(df, 0.50, 0.70, 30, tmpdir, "test")
            adj_dir = sparse.load_npz(os.path.join(tmpdir, "adj_test_directed.npz"))
            adj_undir = sparse.load_npz(os.path.join(tmpdir, "adj_test_undirected.npz"))
            mapping = np.load(os.path.join(tmpdir, "node_mapping_test.npy"))
            assert adj_dir.shape[0] == len(mapping)
            assert adj_undir.shape[0] == len(mapping)

    def test_cn_aa_work_after_reload(self):
        df = _make_stream_graph(200, 30)
        with tempfile.TemporaryDirectory() as tmpdir:
            process_period(df, 0.50, 0.70, 30, tmpdir, "test")
            adj = sparse.load_npz(os.path.join(tmpdir, "adj_test_undirected.npz"))
            n = adj.shape[0]
            src_batch = np.array([0, 1, 2])
            dst_batch = np.array([3, 4, 5])
            cn = compute_cn(adj, src_batch, dst_batch)
            aa = compute_aa(adj, src_batch, dst_batch)
            assert len(cn) == 3
            assert len(aa) == 3
            assert np.isfinite(cn).all()
            assert np.isfinite(aa).all()

    def test_train_only_no_leakage(self):
        """Adjacency should only contain train edges, not val/test."""
        df = _make_stream_graph(1000, 50, seed=123)
        with tempfile.TemporaryDirectory() as tmpdir:
            meta = process_period(df, 0.25, 0.70, 50, tmpdir, "test")
            period_end = int(1000 * 0.25)
            train_end = int(period_end * 0.70)
            assert meta["num_edges_train"] == train_end


class TestMappingConsistency:
    """Tests for mapping consistency between adjacency and features."""

    def test_same_train_same_mapping(self):
        """Same train edges produce identical mappings in both scripts."""
        from scripts.compute_stream_node_features import compute_node_features

        df = _make_stream_graph(500, 30, seed=77)
        period = df.iloc[:int(500 * 0.25)]
        train = period.iloc[:int(len(period) * 0.70)]

        src = train["src_idx"].values.astype(np.int64)
        dst = train["dst_idx"].values.astype(np.int64)

        active_feat, _ = compute_node_features(
            src, dst, train["timestamp"].values.astype(np.int64),
            train["btc"].values.astype(np.float32), 30,
        )

        active_adj, _, _ = build_adjacency_matrices(src, dst, 30)

        assert np.array_equal(active_feat, active_adj)


class TestEdgeCases:
    """Test edge cases."""

    def test_single_edge(self):
        src = np.array([0], dtype=np.int64)
        dst = np.array([1], dtype=np.int64)
        mapping, adj_dir, adj_undir = build_adjacency_matrices(src, dst, 5)
        assert adj_dir.nnz == 1
        assert adj_undir.nnz == 2

    def test_self_loop(self):
        src = np.array([0, 0], dtype=np.int64)
        dst = np.array([0, 1], dtype=np.int64)
        mapping, adj_dir, adj_undir = build_adjacency_matrices(src, dst, 5)
        assert adj_dir.nnz == 2
        local_0 = np.searchsorted(mapping, 0)
        assert adj_dir[local_0, local_0] == 1.0

    def test_bidirectional_edge(self):
        src = np.array([0, 1], dtype=np.int64)
        dst = np.array([1, 0], dtype=np.int64)
        mapping, adj_dir, adj_undir = build_adjacency_matrices(src, dst, 5)
        assert adj_dir.nnz == 2
        assert adj_undir.nnz == 2

    def test_high_node_indices(self):
        src = np.array([1000000, 2000000], dtype=np.int64)
        dst = np.array([3000000, 4000000], dtype=np.int64)
        mapping, adj_dir, adj_undir = build_adjacency_matrices(src, dst, 5000000)
        assert adj_dir.shape == (4, 4)
        assert set(mapping) == {1000000, 2000000, 3000000, 4000000}

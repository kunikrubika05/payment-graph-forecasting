"""Tests for compute_features.py.

Tests cover:
- Correctness of individual feature functions on synthetic graphs
- Full pipeline on sample data
- Disk cleanup after batch upload (mocked)
- Resume capability (skip already processed days)
- Edge cases: empty graphs, single edge, disconnected components

Run: pytest tests/test_compute_features.py -v
"""

import shutil
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from compute_features import (
    build_adjacency,
    compute_pagerank,
    compute_clustering,
    compute_k_core,
    compute_triangle_counts,
    compute_node_features,
    compute_graph_features,
    gini_coefficient,
    process_single_day,
    run_pipeline,
    upload_batch_and_cleanup,
    format_eta,
)


def make_triangle_df():
    """Create a 3-node triangle: 0->1, 1->2, 2->0, all with btc=1.0."""
    return pd.DataFrame({
        "src_idx": [0, 1, 2],
        "dst_idx": [1, 2, 0],
        "btc": [1.0, 1.0, 1.0],
        "usd": [100.0, 100.0, 100.0],
    })


def make_star_df():
    """Create a star: center=0 sends to 1,2,3,4."""
    return pd.DataFrame({
        "src_idx": [0, 0, 0, 0],
        "dst_idx": [1, 2, 3, 4],
        "btc": [1.0, 2.0, 3.0, 4.0],
        "usd": [10.0, 20.0, 30.0, 40.0],
    })


def make_chain_df():
    """Create a chain: 0->1->2->3."""
    return pd.DataFrame({
        "src_idx": [0, 1, 2],
        "dst_idx": [1, 2, 3],
        "btc": [1.0, 2.0, 3.0],
        "usd": [10.0, 20.0, 30.0],
    })


def make_disconnected_df():
    """Create two disconnected components: {0->1} and {10->11}."""
    return pd.DataFrame({
        "src_idx": [0, 10],
        "dst_idx": [1, 11],
        "btc": [1.0, 2.0],
        "usd": [10.0, 20.0],
    })


class TestBuildAdjacency:
    def test_triangle(self):
        df = make_triangle_df()
        adj, uniq, src_c, dst_c = build_adjacency(
            df["src_idx"].values, df["dst_idx"].values, 3
        )
        assert adj.shape == (3, 3)
        assert adj.nnz == 3
        assert len(uniq) == 3

    def test_star(self):
        df = make_star_df()
        adj, uniq, src_c, dst_c = build_adjacency(
            df["src_idx"].values, df["dst_idx"].values, 5
        )
        assert adj.shape == (5, 5)
        assert adj.nnz == 4
        assert len(uniq) == 5

    def test_disconnected(self):
        df = make_disconnected_df()
        adj, uniq, src_c, dst_c = build_adjacency(
            df["src_idx"].values, df["dst_idx"].values, 12
        )
        assert len(uniq) == 4
        assert adj.shape == (4, 4)


class TestPageRank:
    def test_triangle_uniform(self):
        adj = sp.csr_matrix(np.array([
            [0, 1, 0],
            [0, 0, 1],
            [1, 0, 0],
        ], dtype=np.float32))
        pr = compute_pagerank(adj)
        np.testing.assert_allclose(pr, [1/3, 1/3, 1/3], atol=1e-4)

    def test_star_center_lower(self):
        adj = sp.csr_matrix(np.array([
            [0, 1, 1, 1, 1],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
        ], dtype=np.float32))
        pr = compute_pagerank(adj)
        assert pr[0] < pr[1]

    def test_empty(self):
        adj = sp.csr_matrix((0, 0), dtype=np.float32)
        pr = compute_pagerank(adj)
        assert len(pr) == 0

    def test_sums_to_one(self):
        adj = sp.csr_matrix(np.array([
            [0, 1, 0, 0],
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [0, 0, 1, 0],
        ], dtype=np.float32))
        pr = compute_pagerank(adj)
        np.testing.assert_allclose(pr.sum(), 1.0, atol=1e-6)


class TestClustering:
    def test_triangle_full(self):
        adj = sp.csr_matrix(np.array([
            [0, 1, 0],
            [0, 0, 1],
            [1, 0, 0],
        ], dtype=np.float32))
        cc = compute_clustering(adj)
        assert len(cc) == 3
        for c in cc:
            assert c == pytest.approx(1.0, abs=1e-6)

    def test_star_zero(self):
        adj = sp.csr_matrix(np.array([
            [0, 1, 1, 1, 1],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
        ], dtype=np.float32))
        cc = compute_clustering(adj)
        assert cc[0] == pytest.approx(0.0)

    def test_chain_zero(self):
        adj = sp.csr_matrix(np.array([
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
            [0, 0, 0, 0],
        ], dtype=np.float32))
        cc = compute_clustering(adj)
        for c in cc:
            assert c == pytest.approx(0.0)

    def test_empty(self):
        adj = sp.csr_matrix((0, 0), dtype=np.float32)
        cc = compute_clustering(adj)
        assert len(cc) == 0


class TestKCore:
    def test_triangle(self):
        adj = sp.csr_matrix(np.array([
            [0, 1, 0],
            [0, 0, 1],
            [1, 0, 0],
        ], dtype=np.float32))
        kc = compute_k_core(adj)
        assert len(kc) == 3
        for k in kc:
            assert k == 2

    def test_star(self):
        adj = sp.csr_matrix(np.array([
            [0, 1, 1, 1, 1],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
        ], dtype=np.float32))
        kc = compute_k_core(adj)
        for k in kc:
            assert k == 1

    def test_empty(self):
        adj = sp.csr_matrix((0, 0), dtype=np.float32)
        kc = compute_k_core(adj)
        assert len(kc) == 0


class TestTriangles:
    def test_triangle_one_each(self):
        adj = sp.csr_matrix(np.array([
            [0, 1, 0],
            [0, 0, 1],
            [1, 0, 0],
        ], dtype=np.float32))
        tri = compute_triangle_counts(adj)
        for t in tri:
            assert t == 1

    def test_no_triangles_chain(self):
        adj = sp.csr_matrix(np.array([
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
            [0, 0, 0, 0],
        ], dtype=np.float32))
        tri = compute_triangle_counts(adj)
        for t in tri:
            assert t == 0


class TestGiniCoefficient:
    def test_uniform(self):
        assert gini_coefficient(np.array([1.0, 1.0, 1.0, 1.0])) == pytest.approx(0.0, abs=1e-6)

    def test_maximum_inequality(self):
        g = gini_coefficient(np.array([0.0, 0.0, 0.0, 100.0]))
        assert g > 0.7

    def test_empty(self):
        assert gini_coefficient(np.array([])) == 0.0


class TestNodeFeatures:
    def test_triangle_degrees(self):
        df = make_triangle_df()
        adj, uniq, src_c, dst_c = build_adjacency(
            df["src_idx"].values, df["dst_idx"].values, 3
        )
        nf = compute_node_features(df, adj, uniq, src_c, dst_c)
        assert len(nf) == 3
        assert (nf["in_degree"] == 1).all()
        assert (nf["out_degree"] == 1).all()
        assert (nf["total_degree"] == 2).all()

    def test_star_center_degree(self):
        df = make_star_df()
        adj, uniq, src_c, dst_c = build_adjacency(
            df["src_idx"].values, df["dst_idx"].values, 5
        )
        nf = compute_node_features(df, adj, uniq, src_c, dst_c)
        center = nf[nf["node_idx"] == 0].iloc[0]
        assert center["out_degree"] == 4
        assert center["in_degree"] == 0
        assert center["weighted_out_btc"] == pytest.approx(10.0, abs=0.01)

    def test_has_all_columns(self):
        df = make_triangle_df()
        adj, uniq, src_c, dst_c = build_adjacency(
            df["src_idx"].values, df["dst_idx"].values, 3
        )
        nf = compute_node_features(df, adj, uniq, src_c, dst_c)
        expected_cols = [
            "node_idx", "in_degree", "out_degree", "total_degree",
            "weighted_in_btc", "weighted_out_btc", "weighted_in_usd", "weighted_out_usd",
            "balance_btc", "balance_usd",
            "avg_in_btc", "avg_out_btc", "median_in_btc", "median_out_btc",
            "max_in_btc", "max_out_btc", "min_in_btc", "min_out_btc",
            "std_in_btc", "std_out_btc",
            "unique_in_counterparties", "unique_out_counterparties",
            "pagerank", "clustering_coeff", "k_core", "triangle_count",
        ]
        for col in expected_cols:
            assert col in nf.columns, f"Missing column: {col}"


class TestGraphFeatures:
    def test_triangle_graph(self):
        df = make_triangle_df()
        adj, uniq, src_c, dst_c = build_adjacency(
            df["src_idx"].values, df["dst_idx"].values, 3
        )
        nf = compute_node_features(df, adj, uniq, src_c, dst_c)
        gf = compute_graph_features(df, "2024-01-01", adj, uniq, src_c, dst_c, nf)
        assert gf["num_nodes"] == 3
        assert gf["num_edges"] == 3
        assert gf["num_triangles"] == 1
        assert gf["reciprocity"] == pytest.approx(0.0)
        assert gf["avg_clustering"] == pytest.approx(1.0, abs=1e-6)
        assert gf["total_btc"] == pytest.approx(3.0)

    def test_empty_graph(self):
        df = pd.DataFrame({"src_idx": [], "dst_idx": [], "btc": [], "usd": []})
        adj = sp.csr_matrix((0, 0), dtype=np.float32)
        nf = compute_node_features(df, adj, np.array([]), np.array([]), np.array([]))
        gf = compute_graph_features(df, "2024-01-01", adj, np.array([]),
                                    np.array([]), np.array([]), nf)
        assert gf["num_nodes"] == 0
        assert gf["num_edges"] == 0

    def test_disconnected_components(self):
        df = make_disconnected_df()
        adj, uniq, src_c, dst_c = build_adjacency(
            df["src_idx"].values, df["dst_idx"].values, 12
        )
        nf = compute_node_features(df, adj, uniq, src_c, dst_c)
        gf = compute_graph_features(df, "2024-01-01", adj, uniq, src_c, dst_c, nf)
        assert gf["num_wcc"] == 2
        assert gf["largest_wcc_size"] == 2


class TestProcessSingleDay:
    def test_processes_parquet(self, tmp_path):
        df = make_triangle_df()
        parquet_path = tmp_path / "2024-01-01.parquet"
        df.to_parquet(parquet_path, index=False)

        date_str, gf, nf = process_single_day(parquet_path)
        assert date_str == "2024-01-01"
        assert gf["num_nodes"] == 3
        assert nf is not None
        assert len(nf) == 3

    def test_empty_parquet(self, tmp_path):
        df = pd.DataFrame({"src_idx": [], "dst_idx": [], "btc": [], "usd": []})
        parquet_path = tmp_path / "2024-01-02.parquet"
        df.to_parquet(parquet_path, index=False)

        date_str, gf, nf = process_single_day(parquet_path)
        assert gf["num_nodes"] == 0
        assert nf is None


class TestDiskCleanup:
    def test_upload_deletes_local_files(self, tmp_path):
        for i in range(5):
            p = tmp_path / f"2024-01-0{i+1}.parquet"
            make_triangle_df().to_parquet(p, index=False)

        files = list(tmp_path.glob("*.parquet"))
        assert len(files) == 5

        with patch("compute_features.upload_file_to_yadisk", return_value=True):
            deleted = upload_batch_and_cleanup(
                files, "remote/node_features", "fake_token"
            )

        assert deleted == 5
        remaining = list(tmp_path.glob("*.parquet"))
        assert len(remaining) == 0

    def test_failed_upload_keeps_files(self, tmp_path):
        p = tmp_path / "2024-01-01.parquet"
        make_triangle_df().to_parquet(p, index=False)

        with patch("compute_features.upload_file_to_yadisk", return_value=False):
            deleted = upload_batch_and_cleanup(
                [p], "remote/node_features", "fake_token"
            )

        assert deleted == 0
        assert p.exists()


class TestResume:
    def test_skips_already_processed(self, tmp_path):
        input_dir = tmp_path / "snapshots"
        input_dir.mkdir()
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        for date in ["2024-01-01", "2024-01-02", "2024-01-03"]:
            make_triangle_df().to_parquet(input_dir / f"{date}.parquet", index=False)

        run_pipeline(input_dir, output_dir, upload=False)
        csv1 = pd.read_csv(output_dir / "graph_features.csv")
        assert len(csv1) == 3

        make_star_df().to_parquet(input_dir / "2024-01-04.parquet", index=False)
        run_pipeline(input_dir, output_dir, upload=False)
        csv2 = pd.read_csv(output_dir / "graph_features.csv")
        assert len(csv2) == 4


class TestFullPipeline:
    def test_end_to_end(self, tmp_path):
        input_dir = tmp_path / "snapshots"
        input_dir.mkdir()
        output_dir = tmp_path / "output"

        make_triangle_df().to_parquet(input_dir / "2024-01-01.parquet", index=False)
        make_star_df().to_parquet(input_dir / "2024-01-02.parquet", index=False)

        run_pipeline(input_dir, output_dir, upload=False)

        assert (output_dir / "graph_features.csv").exists()
        assert (output_dir / "node_features" / "2024-01-01.parquet").exists()
        assert (output_dir / "node_features" / "2024-01-02.parquet").exists()

        gf = pd.read_csv(output_dir / "graph_features.csv")
        assert len(gf) == 2
        assert "num_nodes" in gf.columns
        assert "avg_clustering" in gf.columns
        assert "max_k_core" in gf.columns

        nf = pd.read_parquet(output_dir / "node_features" / "2024-01-01.parquet")
        assert "pagerank" in nf.columns
        assert "k_core" in nf.columns

    def test_disk_space_bounded(self, tmp_path):
        """Verify node features are written as individual files, not accumulated."""
        input_dir = tmp_path / "snapshots"
        input_dir.mkdir()
        output_dir = tmp_path / "output"

        for i in range(10):
            make_triangle_df().to_parquet(
                input_dir / f"2024-01-{i+1:02d}.parquet", index=False
            )

        run_pipeline(input_dir, output_dir, upload=False)

        node_files = list((output_dir / "node_features").glob("*.parquet"))
        assert len(node_files) == 10

        sizes = [f.stat().st_size for f in node_files]
        for s in sizes:
            assert s < 100_000


class TestFormatEta:
    def test_seconds(self):
        assert format_eta(45) == "45s"

    def test_minutes(self):
        assert format_eta(125) == "2m 05s"

    def test_hours(self):
        assert format_eta(7384) == "2h 03m 04s"

    def test_negative(self):
        assert format_eta(-1) == "?"

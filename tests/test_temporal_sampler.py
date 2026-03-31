"""Tests for TemporalGraphSampler with Python/C++/CUDA backend correctness.

Validates that all three backends produce identical results for:
  - Temporal neighbor sampling (sample_neighbors)
  - Feature gathering (featurize)
  - Negative sampling (sample_negatives)

Also tests edge cases, determinism, and API contracts.

Run:
    source venv/bin/activate
    PYTHONPATH=. pytest tests/test_temporal_sampler.py -v
"""

import importlib.machinery
import types

import numpy as np
import pytest
import torch

import src.models.temporal_graph_sampler as temporal_graph_sampler
from src.models.temporal_graph_sampler import (
    TemporalGraphSampler,
    NeighborBatch,
    FeatureBatch,
    Backend,
    resolve_backend,
)


def _make_graph(num_nodes=100, num_edges=500, seed=42):
    """Create a reproducible synthetic temporal graph."""
    rng = np.random.default_rng(seed)
    src = rng.integers(0, num_nodes, size=num_edges).astype(np.int32)
    dst = rng.integers(0, num_nodes, size=num_edges).astype(np.int32)
    timestamps = np.sort(rng.uniform(0, 100, size=num_edges)).astype(np.float64)
    edge_ids = np.arange(num_edges, dtype=np.int64)
    node_feats = rng.standard_normal((num_nodes, 4)).astype(np.float32)
    edge_feats = rng.standard_normal((num_edges, 2)).astype(np.float32)
    return {
        "num_nodes": num_nodes,
        "src": src,
        "dst": dst,
        "timestamps": timestamps,
        "edge_ids": edge_ids,
        "node_feats": node_feats,
        "edge_feats": edge_feats,
    }


def _make_sampler(graph, backend="python"):
    """Create a TemporalGraphSampler from graph dict."""
    return TemporalGraphSampler(
        num_nodes=graph["num_nodes"],
        src=graph["src"],
        dst=graph["dst"],
        timestamps=graph["timestamps"],
        edge_ids=graph["edge_ids"],
        node_feats=graph["node_feats"],
        edge_feats=graph["edge_feats"],
        backend=backend,
    )


@pytest.fixture
def graph():
    return _make_graph()


@pytest.fixture
def python_sampler(graph):
    return _make_sampler(graph, "python")


def _has_cpp():
    """Check if C++ extension is available."""
    try:
        from src.models.temporal_graph_sampler import _try_load_cpp
        return _try_load_cpp() is not None
    except Exception:
        return False


def _has_cuda():
    """Check if CUDA extension is available."""
    try:
        import torch
        if not torch.cuda.is_available():
            return False
        from src.models.temporal_graph_sampler import _try_load_cuda
        return _try_load_cuda() is not None
    except Exception:
        return False


def test_load_prebuilt_extension_uses_existing_binary(tmp_path, monkeypatch):
    build_dir = tmp_path / "build_cuda"
    build_dir.mkdir()
    suffix = importlib.machinery.EXTENSION_SUFFIXES[0]
    candidate = build_dir / f"temporal_sampling_cuda{suffix}"
    candidate.write_bytes(b"")

    loaded = {}

    class DummyLoader:
        def exec_module(self, module):
            module.marker = "loaded"
            loaded["path"] = module.__file__

    def fake_spec_from_file_location(name, location):
        return types.SimpleNamespace(
            loader=DummyLoader(),
            name=name,
            origin=str(location),
        )

    def fake_module_from_spec(spec):
        return types.SimpleNamespace(__file__=spec.origin, __name__=spec.name)

    monkeypatch.setattr(
        temporal_graph_sampler.importlib.util,
        "spec_from_file_location",
        fake_spec_from_file_location,
    )
    monkeypatch.setattr(
        temporal_graph_sampler.importlib.util,
        "module_from_spec",
        fake_module_from_spec,
    )

    module = temporal_graph_sampler._load_prebuilt_extension(
        "temporal_sampling_cuda", build_dir
    )

    assert module is not None
    assert module.marker == "loaded"
    assert loaded["path"] == str(candidate)


def test_try_load_cuda_falls_back_to_prebuilt_binary(monkeypatch, tmp_path):
    fallback_module = object()

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(temporal_graph_sampler.Path, "exists", lambda self: True)

    def fail_load(*args, **kwargs):
        raise RuntimeError("ninja missing")

    monkeypatch.setattr(
        temporal_graph_sampler,
        "_load_prebuilt_extension",
        lambda name, path: fallback_module,
    )
    monkeypatch.setattr("torch.utils.cpp_extension.load", fail_load)

    module = temporal_graph_sampler._try_load_cuda()
    assert module is fallback_module


class TestPythonBackend:
    """Tests for the pure Python backend (always available)."""

    def test_backend_name(self, python_sampler):
        assert python_sampler.backend == "python"

    def test_sample_neighbors_shape(self, python_sampler):
        nodes = np.array([0, 1, 2], dtype=np.int32)
        times = np.array([50.0, 50.0, 50.0], dtype=np.float64)
        nbr = python_sampler.sample_neighbors(nodes, times, num_neighbors=10)

        assert isinstance(nbr, NeighborBatch)
        assert nbr.neighbor_ids.shape == (3, 10)
        assert nbr.timestamps.shape == (3, 10)
        assert nbr.edge_ids.shape == (3, 10)
        assert nbr.lengths.shape == (3,)
        assert not nbr.on_gpu

    def test_sample_neighbors_dtypes(self, python_sampler):
        nodes = np.array([0], dtype=np.int32)
        times = np.array([50.0], dtype=np.float64)
        nbr = python_sampler.sample_neighbors(nodes, times, num_neighbors=5)

        assert nbr.neighbor_ids.dtype == np.int32
        assert nbr.timestamps.dtype == np.float64
        assert nbr.edge_ids.dtype == np.int64
        assert nbr.lengths.dtype == np.int32

    def test_padding_is_correct(self, python_sampler):
        """Unused slots should be -1 for ids and 0.0 for timestamps."""
        nodes = np.array([0], dtype=np.int32)
        times = np.array([50.0], dtype=np.float64)
        nbr = python_sampler.sample_neighbors(nodes, times, num_neighbors=100)

        length = int(nbr.lengths[0])
        if length < 100:
            assert np.all(nbr.neighbor_ids[0, length:] == -1)
            assert np.all(nbr.timestamps[0, length:] == 0.0)
            assert np.all(nbr.edge_ids[0, length:] == -1)

    def test_temporal_constraint(self, graph, python_sampler):
        """All returned timestamps must be < query_time."""
        nodes = np.array([0, 5, 10], dtype=np.int32)
        times = np.array([30.0, 60.0, 90.0], dtype=np.float64)
        nbr = python_sampler.sample_neighbors(nodes, times, num_neighbors=50)

        for i in range(3):
            length = int(nbr.lengths[i])
            if length > 0:
                valid_ts = nbr.timestamps[i, :length]
                assert np.all(valid_ts < times[i]), (
                    f"Query {i}: found timestamps >= query_time"
                )

    def test_returns_most_recent(self, python_sampler):
        """With K=1, should return the single most recent neighbor."""
        nodes = np.array([0], dtype=np.int32)
        times = np.array([50.0], dtype=np.float64)
        nbr_1 = python_sampler.sample_neighbors(nodes, times, num_neighbors=1)
        nbr_all = python_sampler.sample_neighbors(nodes, times, num_neighbors=500)

        length_all = int(nbr_all.lengths[0])
        if length_all > 0 and int(nbr_1.lengths[0]) > 0:
            assert nbr_1.neighbor_ids[0, 0] == nbr_all.neighbor_ids[0, length_all - 1]
            assert nbr_1.timestamps[0, 0] == nbr_all.timestamps[0, length_all - 1]

    def test_invalid_node(self, python_sampler):
        """Invalid node indices should return empty results."""
        nodes = np.array([-1, 999999], dtype=np.int32)
        times = np.array([50.0, 50.0], dtype=np.float64)
        nbr = python_sampler.sample_neighbors(nodes, times, num_neighbors=10)

        assert int(nbr.lengths[0]) == 0
        assert int(nbr.lengths[1]) == 0

    def test_zero_before_time(self, python_sampler):
        """Query at time 0 should return no neighbors (all timestamps >= 0)."""
        nodes = np.array([0], dtype=np.int32)
        times = np.array([0.0], dtype=np.float64)
        nbr = python_sampler.sample_neighbors(nodes, times, num_neighbors=10)
        assert int(nbr.lengths[0]) == 0

    def test_empty_batch(self, python_sampler):
        """Empty input batch should return empty output."""
        nodes = np.array([], dtype=np.int32)
        times = np.array([], dtype=np.float64)
        nbr = python_sampler.sample_neighbors(nodes, times, num_neighbors=10)
        assert nbr.neighbor_ids.shape == (0, 10)

    def test_featurize_shape(self, graph, python_sampler):
        nodes = np.array([0, 1], dtype=np.int32)
        times = np.array([50.0, 50.0], dtype=np.float64)
        nbr = python_sampler.sample_neighbors(nodes, times, num_neighbors=10)
        feat = python_sampler.featurize(nbr, query_timestamps=times)

        assert isinstance(feat, FeatureBatch)
        assert feat.node_features.shape == (2, 10, 4)
        assert feat.edge_features.shape == (2, 10, 2)
        assert feat.rel_timestamps.shape == (2, 10)
        assert not feat.on_gpu

    def test_featurize_rel_timestamps(self, graph, python_sampler):
        """Relative timestamps should be query_t - neighbor_t (>= 0)."""
        nodes = np.array([0], dtype=np.int32)
        times = np.array([80.0], dtype=np.float64)
        nbr = python_sampler.sample_neighbors(nodes, times, num_neighbors=20)
        feat = python_sampler.featurize(nbr, query_timestamps=times)

        length = int(nbr.lengths[0])
        if length > 0:
            rel_ts = feat.rel_timestamps[0, :length]
            assert np.all(rel_ts >= 0), "Relative timestamps must be non-negative"

    def test_featurize_correctness(self, graph, python_sampler):
        """Feature values should match direct indexing into feature arrays."""
        nodes = np.array([0], dtype=np.int32)
        times = np.array([80.0], dtype=np.float64)
        nbr = python_sampler.sample_neighbors(nodes, times, num_neighbors=5)
        feat = python_sampler.featurize(nbr, query_timestamps=times)

        length = int(nbr.lengths[0])
        for j in range(length):
            nid = int(nbr.neighbor_ids[0, j])
            eid = int(nbr.edge_ids[0, j])
            np.testing.assert_array_equal(
                feat.node_features[0, j],
                graph["node_feats"][nid],
            )
            if 0 <= eid < len(graph["edge_feats"]):
                np.testing.assert_array_equal(
                    feat.edge_features[0, j],
                    graph["edge_feats"][eid],
                )


class TestNegativeSampling:
    """Tests for negative sampling strategies."""

    def test_shape(self, python_sampler):
        src = np.array([0, 1], dtype=np.int32)
        dst = np.array([5, 10], dtype=np.int32)
        times = np.array([50.0, 50.0], dtype=np.float64)
        neg = python_sampler.sample_negatives(src, dst, times, n_negatives=20)
        assert neg.shape == (2, 20)

    def test_no_true_dst_in_negatives(self, python_sampler):
        src = np.array([0], dtype=np.int32)
        dst = np.array([5], dtype=np.int32)
        times = np.array([50.0], dtype=np.float64)
        neg = python_sampler.sample_negatives(src, dst, times, n_negatives=50)
        assert 5 not in neg[0], "true_dst should not appear in negatives"

    def test_no_self_loops(self, python_sampler):
        src = np.array([3], dtype=np.int32)
        dst = np.array([7], dtype=np.int32)
        times = np.array([50.0], dtype=np.float64)
        neg = python_sampler.sample_negatives(src, dst, times, n_negatives=50)
        assert 3 not in neg[0], "source should not appear in negatives"

    def test_random_strategy(self, python_sampler):
        src = np.array([0], dtype=np.int32)
        dst = np.array([5], dtype=np.int32)
        times = np.array([50.0], dtype=np.float64)
        neg = python_sampler.sample_negatives(
            src, dst, times, n_negatives=50, strategy="random"
        )
        assert neg.shape == (1, 50)
        assert len(set(neg[0])) == 50, "Random negatives should be unique"

    def test_mixed_strategy(self, python_sampler):
        src = np.array([0], dtype=np.int32)
        dst = np.array([5], dtype=np.int32)
        times = np.array([80.0], dtype=np.float64)
        neg = python_sampler.sample_negatives(
            src, dst, times, n_negatives=100, strategy="mixed", hist_ratio=0.5,
        )
        assert neg.shape == (1, 100)

    def test_deterministic_with_seed(self, python_sampler):
        src = np.array([0, 1], dtype=np.int32)
        dst = np.array([5, 10], dtype=np.int32)
        times = np.array([50.0, 50.0], dtype=np.float64)

        neg1 = python_sampler.sample_negatives(
            src, dst, times, n_negatives=50, rng=np.random.default_rng(123)
        )
        neg2 = python_sampler.sample_negatives(
            src, dst, times, n_negatives=50, rng=np.random.default_rng(123)
        )
        np.testing.assert_array_equal(neg1, neg2)


class TestCppBackendEquivalence:
    """Compare C++ backend results against Python reference."""

    @pytest.fixture
    def samplers(self, graph):
        py = _make_sampler(graph, "python")
        if not _has_cpp():
            pytest.skip("C++ extension not compiled")
        cpp = _make_sampler(graph, "cpp")
        return py, cpp

    def test_sample_neighbors_identical(self, samplers):
        py_sampler, cpp_sampler = samplers
        nodes = np.array([0, 5, 10, 20, 50], dtype=np.int32)
        times = np.array([30.0, 50.0, 70.0, 90.0, 100.0], dtype=np.float64)

        py_nbr = py_sampler.sample_neighbors(nodes, times, num_neighbors=20)
        cpp_nbr = cpp_sampler.sample_neighbors(nodes, times, num_neighbors=20)

        np.testing.assert_array_equal(py_nbr.lengths, cpp_nbr.lengths)
        for i in range(len(nodes)):
            length = int(py_nbr.lengths[i])
            np.testing.assert_array_equal(
                py_nbr.neighbor_ids[i, :length],
                cpp_nbr.neighbor_ids[i, :length],
            )
            np.testing.assert_array_equal(
                py_nbr.timestamps[i, :length],
                cpp_nbr.timestamps[i, :length],
            )
            np.testing.assert_array_equal(
                py_nbr.edge_ids[i, :length],
                cpp_nbr.edge_ids[i, :length],
            )

    def test_featurize_identical(self, graph, samplers):
        py_sampler, cpp_sampler = samplers
        nodes = np.array([0, 5, 10], dtype=np.int32)
        times = np.array([50.0, 50.0, 50.0], dtype=np.float64)

        py_nbr = py_sampler.sample_neighbors(nodes, times, num_neighbors=10)
        cpp_nbr = cpp_sampler.sample_neighbors(nodes, times, num_neighbors=10)

        py_feat = py_sampler.featurize(py_nbr, query_timestamps=times)
        cpp_feat = cpp_sampler.featurize(cpp_nbr, query_timestamps=times)

        for i in range(len(nodes)):
            length = int(py_nbr.lengths[i])
            np.testing.assert_array_almost_equal(
                py_feat.node_features[i, :length],
                cpp_feat.node_features[i, :length],
                decimal=6,
            )
            np.testing.assert_array_almost_equal(
                py_feat.edge_features[i, :length],
                cpp_feat.edge_features[i, :length],
                decimal=6,
            )
            np.testing.assert_array_almost_equal(
                py_feat.rel_timestamps[i, :length],
                cpp_feat.rel_timestamps[i, :length],
                decimal=10,
            )

    def test_various_k_values(self, samplers):
        py_sampler, cpp_sampler = samplers
        nodes = np.array([0, 1, 2], dtype=np.int32)
        times = np.array([50.0, 50.0, 50.0], dtype=np.float64)

        for K in [1, 5, 20, 100, 512]:
            py_nbr = py_sampler.sample_neighbors(nodes, times, num_neighbors=K)
            cpp_nbr = cpp_sampler.sample_neighbors(nodes, times, num_neighbors=K)
            np.testing.assert_array_equal(py_nbr.lengths, cpp_nbr.lengths)
            for i in range(len(nodes)):
                length = int(py_nbr.lengths[i])
                np.testing.assert_array_equal(
                    py_nbr.neighbor_ids[i, :length],
                    cpp_nbr.neighbor_ids[i, :length],
                )


class TestCudaBackendEquivalence:
    """Compare CUDA backend results against Python reference."""

    @pytest.fixture
    def samplers(self, graph):
        py = _make_sampler(graph, "python")
        if not _has_cuda():
            pytest.skip("CUDA extension not compiled or no GPU")
        cuda = _make_sampler(graph, "cuda")
        return py, cuda

    def _to_numpy(self, sampler, nbr):
        return sampler.to_numpy(nbr)

    def test_sample_neighbors_identical(self, samplers):
        py_sampler, cuda_sampler = samplers
        nodes = np.array([0, 5, 10, 20, 50], dtype=np.int32)
        times = np.array([30.0, 50.0, 70.0, 90.0, 100.0], dtype=np.float64)

        py_nbr = py_sampler.sample_neighbors(nodes, times, num_neighbors=20)
        cuda_nbr_raw = cuda_sampler.sample_neighbors(nodes, times, num_neighbors=20)
        cuda_nbr = self._to_numpy(cuda_sampler, cuda_nbr_raw)

        np.testing.assert_array_equal(py_nbr.lengths, cuda_nbr.lengths)
        for i in range(len(nodes)):
            length = int(py_nbr.lengths[i])
            np.testing.assert_array_equal(
                py_nbr.neighbor_ids[i, :length],
                cuda_nbr.neighbor_ids[i, :length],
            )
            np.testing.assert_array_equal(
                py_nbr.timestamps[i, :length],
                cuda_nbr.timestamps[i, :length],
            )
            np.testing.assert_array_equal(
                py_nbr.edge_ids[i, :length],
                cuda_nbr.edge_ids[i, :length],
            )

    def test_featurize_identical(self, graph, samplers):
        py_sampler, cuda_sampler = samplers
        nodes = np.array([0, 5, 10], dtype=np.int32)
        times = np.array([50.0, 50.0, 50.0], dtype=np.float64)

        py_nbr = py_sampler.sample_neighbors(nodes, times, num_neighbors=10)
        cuda_nbr = cuda_sampler.sample_neighbors(nodes, times, num_neighbors=10)

        py_feat = py_sampler.featurize(py_nbr, query_timestamps=times)
        cuda_feat_raw = cuda_sampler.featurize(cuda_nbr, query_timestamps=times)
        cuda_feat = cuda_sampler.to_numpy_features(cuda_feat_raw)

        for i in range(len(nodes)):
            length = int(py_nbr.lengths[i])
            np.testing.assert_array_almost_equal(
                py_feat.node_features[i, :length],
                cuda_feat.node_features[i, :length],
                decimal=5,
            )
            np.testing.assert_array_almost_equal(
                py_feat.edge_features[i, :length],
                cuda_feat.edge_features[i, :length],
                decimal=5,
            )
            np.testing.assert_array_almost_equal(
                py_feat.rel_timestamps[i, :length],
                cuda_feat.rel_timestamps[i, :length],
                decimal=10,
            )

    def test_large_batch(self, graph, samplers):
        """Stress test with large batch (1024 queries, K=512)."""
        py_sampler, cuda_sampler = samplers
        rng = np.random.default_rng(99)
        nodes = rng.integers(0, graph["num_nodes"], size=1024).astype(np.int32)
        times = rng.uniform(0, 100, size=1024).astype(np.float64)

        py_nbr = py_sampler.sample_neighbors(nodes, times, num_neighbors=512)
        cuda_nbr_raw = cuda_sampler.sample_neighbors(nodes, times, num_neighbors=512)
        cuda_nbr = self._to_numpy(cuda_sampler, cuda_nbr_raw)

        np.testing.assert_array_equal(py_nbr.lengths, cuda_nbr.lengths)
        for i in range(len(nodes)):
            length = int(py_nbr.lengths[i])
            if length > 0:
                np.testing.assert_array_equal(
                    py_nbr.neighbor_ids[i, :length],
                    cuda_nbr.neighbor_ids[i, :length],
                )

    def test_various_k_values(self, samplers):
        py_sampler, cuda_sampler = samplers
        nodes = np.array([0, 1, 2], dtype=np.int32)
        times = np.array([50.0, 50.0, 50.0], dtype=np.float64)

        for K in [1, 5, 20, 100, 512]:
            py_nbr = py_sampler.sample_neighbors(nodes, times, num_neighbors=K)
            cuda_nbr_raw = cuda_sampler.sample_neighbors(nodes, times, num_neighbors=K)
            cuda_nbr = self._to_numpy(cuda_sampler, cuda_nbr_raw)
            np.testing.assert_array_equal(py_nbr.lengths, cuda_nbr.lengths)


class TestAllBackendsConsistency:
    """Cross-validate all available backends produce the same output."""

    def test_all_backends_same_result(self, graph):
        """All available backends must produce identical neighbor sampling."""
        backends = ["python"]
        if _has_cpp():
            backends.append("cpp")
        if _has_cuda():
            backends.append("cuda")

        if len(backends) < 2:
            pytest.skip("Need at least 2 backends for cross-validation")

        samplers = {b: _make_sampler(graph, b) for b in backends}

        rng = np.random.default_rng(77)
        nodes = rng.integers(0, graph["num_nodes"], size=50).astype(np.int32)
        times = rng.uniform(0, 100, size=50).astype(np.float64)

        results = {}
        for b, s in samplers.items():
            nbr = s.sample_neighbors(nodes, times, num_neighbors=20)
            if nbr.on_gpu:
                nbr = s.to_numpy(nbr)
            results[b] = nbr

        ref = results["python"]
        for b in backends[1:]:
            np.testing.assert_array_equal(
                ref.lengths, results[b].lengths,
                err_msg=f"Lengths mismatch: python vs {b}",
            )
            for i in range(len(nodes)):
                length = int(ref.lengths[i])
                np.testing.assert_array_equal(
                    ref.neighbor_ids[i, :length],
                    results[b].neighbor_ids[i, :length],
                    err_msg=f"Neighbor IDs mismatch at query {i}: python vs {b}",
                )
                np.testing.assert_array_equal(
                    ref.timestamps[i, :length],
                    results[b].timestamps[i, :length],
                    err_msg=f"Timestamps mismatch at query {i}: python vs {b}",
                )


class TestResolveBackend:
    """Test backend resolution logic."""

    def test_explicit_python(self):
        assert resolve_backend("python") == Backend.PYTHON

    def test_explicit_cpp(self):
        if not _has_cpp():
            pytest.skip("C++ not available")
        assert resolve_backend("cpp") == Backend.CPP

    def test_auto_never_fails(self):
        """Auto should always resolve to something."""
        backend = resolve_backend("auto")
        assert backend in (Backend.PYTHON, Backend.CPP, Backend.CUDA)


class TestToNumpy:
    """Test GPU -> CPU conversion helpers."""

    def test_to_numpy_noop_on_cpu(self, python_sampler):
        nodes = np.array([0], dtype=np.int32)
        times = np.array([50.0], dtype=np.float64)
        nbr = python_sampler.sample_neighbors(nodes, times, num_neighbors=5)
        result = python_sampler.to_numpy(nbr)
        assert not result.on_gpu
        np.testing.assert_array_equal(nbr.neighbor_ids, result.neighbor_ids)

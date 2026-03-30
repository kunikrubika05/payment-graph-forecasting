"""Tests for CommonNeighbors: correctness across Python / C++ / CUDA backends.

Graph fixtures:
  triangle_star — small hand-crafted graph with known CN values.
  dense_random  — random dense graph for backend cross-validation.
"""

import numpy as np
import pytest
from scipy import sparse

from src.models.graph_metrics import CommonNeighbors, has_cpp, has_cuda


# ── helpers ─────────────────────────────────────────────────────────────────

def _make_adj(n: int, edges) -> sparse.csr_matrix:
    """Build a symmetric binary CSR from an undirected edge list."""
    us, vs = zip(*edges)
    r = list(us) + list(vs)
    c = list(vs) + list(us)
    data = np.ones(len(r), dtype=np.float32)
    A = sparse.csr_matrix((data, (r, c)), shape=(n, n))
    A.eliminate_zeros()
    A.sort_indices()
    return A


def _make_dense_adj(n: int, avg_deg: int, seed: int = 42) -> sparse.csr_matrix:
    """Random symmetric binary CSR with approximately avg_deg neighbors per node."""
    rng = np.random.default_rng(seed)
    num_edges = n * avg_deg // 2
    src = rng.integers(0, n, size=num_edges).astype(np.int32)
    dst = rng.integers(0, n, size=num_edges).astype(np.int32)
    mask = src != dst
    src, dst = src[mask], dst[mask]
    rows = np.concatenate([src, dst])
    cols = np.concatenate([dst, src])
    data = np.ones(len(rows), dtype=np.float32)
    A = sparse.csr_matrix((data, (rows, cols)), shape=(n, n))
    A.data[:] = 1.0
    A.sum_duplicates()
    A.sort_indices()
    return A


# triangle (0-1, 1-2, 0-2) + star (center=3, leaves=4,5,6) + isolated node 7
TRIANGLE_STAR_EDGES = [(0, 1), (1, 2), (0, 2), (3, 4), (3, 5), (3, 6)]
TRIANGLE_STAR_N = 8


@pytest.fixture(scope="module")
def ts_adj():
    return _make_adj(TRIANGLE_STAR_N, TRIANGLE_STAR_EDGES)


@pytest.fixture(scope="module")
def dense_adj():
    return _make_dense_adj(n=2000, avg_deg=100)


# ── Python backend — known values ────────────────────────────────────────────

def test_triangle_cn_python(ts_adj):
    cn = CommonNeighbors(ts_adj, backend="python")
    src = np.array([0, 0, 1, 0, 4, 4], dtype=np.int64)
    dst = np.array([1, 2, 2, 3, 5, 6], dtype=np.int64)
    result = cn.compute(src, dst)
    expected = np.array([1, 1, 1, 0, 1, 1], dtype=np.int32)
    np.testing.assert_array_equal(result, expected)


def test_isolated_node_python(ts_adj):
    cn = CommonNeighbors(ts_adj, backend="python")
    src = np.array([7, 7, 0], dtype=np.int64)
    dst = np.array([0, 3, 7], dtype=np.int64)
    result = cn.compute(src, dst)
    np.testing.assert_array_equal(result, [0, 0, 0])


def test_self_pair_python(ts_adj):
    cn = CommonNeighbors(ts_adj, backend="python")
    # CN(u, u) = degree(u): N(u) ∩ N(u) = N(u)
    src = np.array([0, 3], dtype=np.int64)
    dst = np.array([0, 3], dtype=np.int64)
    result = cn.compute(src, dst)
    # degree(0)=2, degree(3)=3
    np.testing.assert_array_equal(result, [2, 3])


def test_symmetric_python(ts_adj):
    cn = CommonNeighbors(ts_adj, backend="python")
    src = np.array([0, 1, 4], dtype=np.int64)
    dst = np.array([2, 0, 5], dtype=np.int64)
    fwd = cn.compute(src, dst)
    rev = cn.compute(dst, src)
    np.testing.assert_array_equal(fwd, rev)


def test_batch_size_one_python(ts_adj):
    cn = CommonNeighbors(ts_adj, backend="python")
    result = cn.compute(np.array([0], dtype=np.int64),
                        np.array([1], dtype=np.int64))
    assert result.shape == (1,)
    assert result[0] == 1


def test_star_leaves_python(ts_adj):
    cn = CommonNeighbors(ts_adj, backend="python")
    src = np.array([4, 4, 5], dtype=np.int64)
    dst = np.array([5, 6, 6], dtype=np.int64)
    result = cn.compute(src, dst)
    np.testing.assert_array_equal(result, [1, 1, 1])


# ── C++ backend ───────────────────────────────────────────────────────────────

@pytest.mark.skipif(not has_cpp(), reason="C++ extension not compiled")
def test_cpp_known_values(ts_adj):
    cn = CommonNeighbors(ts_adj, backend="cpp")
    src = np.array([0, 0, 1, 0, 4, 4], dtype=np.int64)
    dst = np.array([1, 2, 2, 3, 5, 6], dtype=np.int64)
    result = cn.compute(src, dst)
    expected = np.array([1, 1, 1, 0, 1, 1], dtype=np.int32)
    np.testing.assert_array_equal(result, expected)


@pytest.mark.skipif(not has_cpp(), reason="C++ extension not compiled")
def test_cpp_matches_python_dense(dense_adj):
    cn_py  = CommonNeighbors(dense_adj, backend="python")
    cn_cpp = CommonNeighbors(dense_adj, backend="cpp")
    rng = np.random.default_rng(7)
    src = rng.integers(0, dense_adj.shape[0], 200).astype(np.int64)
    dst = rng.integers(0, dense_adj.shape[0], 200).astype(np.int64)
    np.testing.assert_array_equal(cn_py.compute(src, dst),
                                  cn_cpp.compute(src, dst))


@pytest.mark.skipif(not has_cpp(), reason="C++ extension not compiled")
def test_cpp_isolated_and_self(ts_adj):
    cn = CommonNeighbors(ts_adj, backend="cpp")
    src = np.array([7, 0], dtype=np.int64)
    dst = np.array([0, 0], dtype=np.int64)
    result = cn.compute(src, dst)
    np.testing.assert_array_equal(result, [0, 2])


# ── CUDA backend ─────────────────────────────────────────────────────────────

@pytest.mark.skipif(not has_cuda(), reason="CUDA extension not compiled")
def test_cuda_known_values(ts_adj):
    cn = CommonNeighbors(ts_adj, backend="cuda")
    src = np.array([0, 0, 1, 0, 4, 4], dtype=np.int64)
    dst = np.array([1, 2, 2, 3, 5, 6], dtype=np.int64)
    result = cn.compute(src, dst)
    expected = np.array([1, 1, 1, 0, 1, 1], dtype=np.int32)
    np.testing.assert_array_equal(result, expected)


@pytest.mark.skipif(not has_cuda(), reason="CUDA extension not compiled")
def test_cuda_matches_python_dense(dense_adj):
    cn_py   = CommonNeighbors(dense_adj, backend="python")
    cn_cuda = CommonNeighbors(dense_adj, backend="cuda")
    rng = np.random.default_rng(11)
    src = rng.integers(0, dense_adj.shape[0], 500).astype(np.int64)
    dst = rng.integers(0, dense_adj.shape[0], 500).astype(np.int64)
    np.testing.assert_array_equal(cn_py.compute(src, dst),
                                  cn_cuda.compute(src, dst))


@pytest.mark.skipif(not has_cuda(), reason="CUDA extension not compiled")
def test_cuda_isolated_and_self(ts_adj):
    cn = CommonNeighbors(ts_adj, backend="cuda")
    src = np.array([7, 0], dtype=np.int64)
    dst = np.array([0, 0], dtype=np.int64)
    result = cn.compute(src, dst)
    np.testing.assert_array_equal(result, [0, 2])


@pytest.mark.skipif(not has_cuda(), reason="CUDA extension not compiled")
def test_cuda_large_batch(dense_adj):
    cn_py   = CommonNeighbors(dense_adj, backend="python")
    cn_cuda = CommonNeighbors(dense_adj, backend="cuda")
    rng = np.random.default_rng(99)
    src = rng.integers(0, dense_adj.shape[0], 2048).astype(np.int64)
    dst = rng.integers(0, dense_adj.shape[0], 2048).astype(np.int64)
    np.testing.assert_array_equal(cn_py.compute(src, dst),
                                  cn_cuda.compute(src, dst))


# ── all-backends cross-check ──────────────────────────────────────────────────

def test_all_available_backends_agree(dense_adj):
    """All compiled backends must produce identical results."""
    backends = ["python"]
    if has_cpp():
        backends.append("cpp")
    if has_cuda():
        backends.append("cuda")
    if len(backends) < 2:
        pytest.skip("Need at least 2 backends to compare")

    rng = np.random.default_rng(55)
    src = rng.integers(0, dense_adj.shape[0], 300).astype(np.int64)
    dst = rng.integers(0, dense_adj.shape[0], 300).astype(np.int64)

    results = {b: CommonNeighbors(dense_adj, backend=b).compute(src, dst)
               for b in backends}
    ref = results["python"]
    for b, res in results.items():
        np.testing.assert_array_equal(ref, res, err_msg=f"python vs {b} mismatch")


def test_output_dtype_is_int32(ts_adj):
    cn = CommonNeighbors(ts_adj, backend="python")
    result = cn.compute(np.array([0], dtype=np.int64),
                        np.array([1], dtype=np.int64))
    assert result.dtype == np.int32


def test_backend_property(ts_adj):
    cn = CommonNeighbors(ts_adj, backend="python")
    assert cn.backend == "python"

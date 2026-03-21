"""Tests for deep learning models (GraphMixer) for temporal link prediction."""

import numpy as np
import pytest
import torch

from src.models.graphmixer import (
    FixedTimeEncoding,
    FeatEncoder,
    FeedForward,
    MixerBlock,
    LinkEncoder,
    NodeEncoder,
    LinkClassifier,
    GraphMixer,
)
from src.models.data_utils import (
    TemporalEdgeData,
    TemporalCSR,
    sample_neighbors_batch,
    featurize_neighbors,
    generate_negatives_for_eval,
    chronological_split,
    _load_cpp_extension,
)


def _make_dummy_data(num_nodes=50, num_edges=200, edge_feat_dim=2, node_feat_dim=25):
    """Create a small synthetic TemporalEdgeData for testing."""
    rng = np.random.default_rng(42)
    src = rng.integers(0, num_nodes, size=num_edges).astype(np.int32)
    dst = rng.integers(0, num_nodes, size=num_edges).astype(np.int32)
    timestamps = np.sort(rng.uniform(0, 10, size=num_edges))
    edge_feats = rng.standard_normal((num_edges, edge_feat_dim)).astype(np.float32)
    node_feats = rng.standard_normal((num_nodes, node_feat_dim)).astype(np.float32)
    node_id_map = {i: i for i in range(num_nodes)}
    reverse_node_map = np.arange(num_nodes, dtype=np.int64)

    return TemporalEdgeData(
        src=src, dst=dst, timestamps=timestamps,
        edge_feats=edge_feats, node_feats=node_feats,
        node_id_map=node_id_map, reverse_node_map=reverse_node_map,
    )


class TestFixedTimeEncoding:
    """Tests for FixedTimeEncoding module."""

    def test_output_shape(self):
        enc = FixedTimeEncoding(dim=100)
        t = torch.tensor([1.0, 2.0, 3.0])
        out = enc(t)
        assert out.shape == (3, 100)

    def test_output_range(self):
        enc = FixedTimeEncoding(dim=50)
        t = torch.tensor([0.0, 100.0, 1e6])
        out = enc(t)
        assert out.min() >= -1.0
        assert out.max() <= 1.0

    def test_similar_timestamps_similar_encodings(self):
        enc = FixedTimeEncoding(dim=100)
        t1 = torch.tensor([100.0])
        t2 = torch.tensor([100.1])
        t3 = torch.tensor([1e6])
        out1 = enc(t1)
        out2 = enc(t2)
        out3 = enc(t3)
        sim_close = torch.cosine_similarity(out1, out2, dim=-1)
        sim_far = torch.cosine_similarity(out1, out3, dim=-1)
        assert sim_close > sim_far

    def test_zero_timestamp(self):
        enc = FixedTimeEncoding(dim=100)
        t = torch.tensor([0.0])
        out = enc(t)
        assert torch.allclose(out, torch.ones(1, 100))

    def test_non_trainable(self):
        enc = FixedTimeEncoding(dim=100)
        trainable = [p for p in enc.parameters() if p.requires_grad]
        assert len(trainable) == 0


class TestFeatEncoder:
    """Tests for FeatEncoder module."""

    def test_output_shape(self):
        enc = FeatEncoder(edge_feat_dim=2, time_dim=100, out_dim=64)
        feats = torch.randn(10, 2)
        ts = torch.randn(10)
        out = enc(feats, ts)
        assert out.shape == (10, 64)


class TestFeedForward:
    """Tests for FeedForward module."""

    def test_output_shape(self):
        ff = FeedForward(dim=64, expansion_factor=2)
        x = torch.randn(5, 64)
        out = ff(x)
        assert out.shape == (5, 64)


class TestMixerBlock:
    """Tests for MixerBlock module."""

    def test_output_shape(self):
        block = MixerBlock(num_tokens=20, hidden_dim=64)
        x = torch.randn(4, 20, 64)
        out = block(x)
        assert out.shape == (4, 20, 64)

    def test_residual_connection(self):
        block = MixerBlock(num_tokens=10, hidden_dim=32)
        x = torch.zeros(2, 10, 32)
        out = block(x)
        assert out.shape == (2, 10, 32)


class TestLinkEncoder:
    """Tests for LinkEncoder module."""

    def test_output_shape(self):
        enc = LinkEncoder(edge_feat_dim=2, num_neighbors=20, hidden_dim=64)
        feats = torch.randn(4, 20, 2)
        ts = torch.randn(4, 20)
        lengths = torch.tensor([5, 10, 20, 0])
        out = enc(feats, ts, lengths)
        assert out.shape == (4, 64)

    def test_zero_length_handled(self):
        enc = LinkEncoder(edge_feat_dim=2, num_neighbors=10, hidden_dim=32)
        feats = torch.zeros(2, 10, 2)
        ts = torch.zeros(2, 10)
        lengths = torch.tensor([0, 0])
        out = enc(feats, ts, lengths)
        assert out.shape == (2, 32)
        assert not torch.isnan(out).any()


class TestNodeEncoder:
    """Tests for NodeEncoder module."""

    def test_output_shape(self):
        enc = NodeEncoder(node_feat_dim=25, out_dim=64)
        nf = torch.randn(4, 25)
        nnf = torch.randn(4, 10, 25)
        lengths = torch.tensor([3, 5, 0, 10])
        out = enc(nf, nnf, lengths)
        assert out.shape == (4, 64)

    def test_zero_neighbors(self):
        enc = NodeEncoder(node_feat_dim=25, out_dim=64)
        nf = torch.randn(2, 25)
        nnf = torch.zeros(2, 10, 25)
        lengths = torch.tensor([0, 0])
        out = enc(nf, nnf, lengths)
        assert out.shape == (2, 64)
        assert not torch.isnan(out).any()


class TestLinkClassifier:
    """Tests for LinkClassifier module."""

    def test_pairwise(self):
        clf = LinkClassifier(input_dim=64, hidden_dim=32)
        h_src = torch.randn(4, 64)
        h_dst = torch.randn(4, 64)
        out = clf(h_src, h_dst)
        assert out.shape == (4,)

    def test_ranking_mode(self):
        clf = LinkClassifier(input_dim=64, hidden_dim=32)
        h_src = torch.randn(4, 64)
        h_dst = torch.randn(4, 10, 64)
        out = clf(h_src, h_dst)
        assert out.shape == (4, 10)


class TestGraphMixer:
    """Tests for the full GraphMixer model."""

    def _make_model(self):
        return GraphMixer(
            edge_feat_dim=2, node_feat_dim=25,
            hidden_dim=32, num_neighbors=10,
            num_mixer_layers=1, dropout=0.0,
        )

    def test_encode_node_shape(self):
        model = self._make_model()
        nf = torch.randn(4, 25)
        nnf = torch.randn(4, 10, 25)
        nef = torch.randn(4, 10, 2)
        nrt = torch.randn(4, 10)
        lengths = torch.tensor([3, 5, 0, 10])
        out = model.encode_node(nf, nnf, nef, nrt, lengths)
        assert out.shape == (4, 64)

    def test_forward_pairwise(self):
        model = self._make_model()
        batch = 4
        K = 10
        out = model(
            torch.randn(batch, 25), torch.randn(batch, K, 25),
            torch.randn(batch, K, 2), torch.randn(batch, K),
            torch.tensor([5, 3, 10, 0]),
            torch.randn(batch, 25), torch.randn(batch, K, 25),
            torch.randn(batch, K, 2), torch.randn(batch, K),
            torch.tensor([2, 7, 1, 10]),
        )
        assert out.shape == (batch,)

    def test_forward_ranking(self):
        model = self._make_model()
        batch = 2
        K = 10
        num_cand = 5
        out = model(
            torch.randn(batch, 25), torch.randn(batch, K, 25),
            torch.randn(batch, K, 2), torch.randn(batch, K),
            torch.tensor([5, 3]),
            torch.randn(batch, num_cand, 25),
            torch.randn(batch, num_cand, K, 25),
            torch.randn(batch, num_cand, K, 2),
            torch.randn(batch, num_cand, K),
            torch.tensor([[2, 7, 1, 10, 3], [5, 5, 5, 5, 5]]),
        )
        assert out.shape == (batch, num_cand)

    def test_backward(self):
        model = self._make_model()
        batch = 4
        K = 10
        out = model(
            torch.randn(batch, 25), torch.randn(batch, K, 25),
            torch.randn(batch, K, 2), torch.randn(batch, K),
            torch.tensor([5, 3, 10, 0]),
            torch.randn(batch, 25), torch.randn(batch, K, 25),
            torch.randn(batch, K, 2), torch.randn(batch, K),
            torch.tensor([2, 7, 1, 10]),
        )
        loss = out.sum()
        loss.backward()
        for p in model.parameters():
            if p.requires_grad:
                assert p.grad is not None

    def test_param_count(self):
        model = self._make_model()
        total = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        assert total > 0
        assert trainable > 0
        assert trainable == total


class TestTemporalEdgeData:
    """Tests for TemporalEdgeData container."""

    def test_creation(self):
        data = _make_dummy_data()
        assert data.num_nodes == 50
        assert data.num_edges == 200
        assert data.edge_feats.shape == (200, 2)
        assert data.node_feats.shape == (50, 25)

    def test_repr(self):
        data = _make_dummy_data()
        r = repr(data)
        assert "num_nodes=50" in r
        assert "num_edges=200" in r


class TestTemporalCSR:
    """Tests for TemporalCSR neighbor lookups."""

    def _make_csr(self):
        data = _make_dummy_data(num_nodes=10, num_edges=30)
        return TemporalCSR(
            data.num_nodes, data.src, data.dst,
            data.timestamps, np.arange(data.num_edges, dtype=np.int64),
        ), data

    def test_get_neighbors_returns_arrays(self):
        csr, data = self._make_csr()
        nids, nts, neids = csr.get_temporal_neighbors(0, before_time=5.0, k=5)
        assert isinstance(nids, np.ndarray)
        assert isinstance(nts, np.ndarray)
        assert len(nids) == len(nts) == len(neids)

    def test_k_limit(self):
        csr, _ = self._make_csr()
        nids, _, _ = csr.get_temporal_neighbors(0, before_time=100.0, k=3)
        assert len(nids) <= 3

    def test_before_time_filter(self):
        csr, _ = self._make_csr()
        nids, nts, _ = csr.get_temporal_neighbors(0, before_time=5.0, k=100)
        if len(nts) > 0:
            assert all(t < 5.0 for t in nts)

    def test_empty_node(self):
        src = np.array([0, 0], dtype=np.int32)
        dst = np.array([1, 2], dtype=np.int32)
        ts = np.array([1.0, 2.0])
        eids = np.array([0, 1], dtype=np.int64)
        csr = TemporalCSR(5, src, dst, ts, eids)
        nids, nts, neids = csr.get_temporal_neighbors(3, before_time=10.0, k=5)
        assert len(nids) == 0

    def test_no_future_leak(self):
        src = np.array([0, 0, 0], dtype=np.int32)
        dst = np.array([1, 2, 3], dtype=np.int32)
        ts = np.array([1.0, 5.0, 10.0])
        eids = np.array([0, 1, 2], dtype=np.int64)
        csr = TemporalCSR(4, src, dst, ts, eids)
        nids, nts, _ = csr.get_temporal_neighbors(0, before_time=6.0, k=10)
        assert 3 not in nids
        assert all(t < 6.0 for t in nts)


class TestSampleNeighborsBatch:
    """Tests for batch neighbor sampling."""

    def test_output_shapes(self):
        data = _make_dummy_data(num_nodes=10, num_edges=30)
        csr = TemporalCSR(
            data.num_nodes, data.src, data.dst,
            data.timestamps, np.arange(data.num_edges, dtype=np.int64),
        )
        nodes = np.array([0, 1, 2], dtype=np.int32)
        ts = np.array([5.0, 5.0, 5.0], dtype=np.float64)
        nn, nts, neids, lengths = sample_neighbors_batch(csr, nodes, ts, num_neighbors=10)
        assert nn.shape == (3, 10)
        assert nts.shape == (3, 10)
        assert neids.shape == (3, 10)
        assert lengths.shape == (3,)

    def test_padding(self):
        src = np.array([0], dtype=np.int32)
        dst = np.array([1], dtype=np.int32)
        ts = np.array([1.0])
        eids = np.array([0], dtype=np.int64)
        csr = TemporalCSR(5, src, dst, ts, eids)
        nodes = np.array([0], dtype=np.int32)
        ts_q = np.array([5.0], dtype=np.float64)
        nn, _, _, lengths = sample_neighbors_batch(csr, nodes, ts_q, num_neighbors=5)
        assert lengths[0] == 1
        assert nn[0, 0] == 1
        assert nn[0, 1] == -1


class TestGenerateNegatives:
    """Tests for negative sampling."""

    def test_output_length(self):
        data = _make_dummy_data(num_nodes=100, num_edges=500)
        csr = TemporalCSR(
            data.num_nodes, data.src, data.dst,
            data.timestamps, np.arange(data.num_edges, dtype=np.int64),
        )
        negs = generate_negatives_for_eval(
            src_node=0, true_dst=1, timestamp=5.0,
            csr=csr, num_nodes=100, n_hist=50, n_random=50,
        )
        assert len(negs) == 100

    def test_true_dst_excluded(self):
        data = _make_dummy_data(num_nodes=100, num_edges=500)
        csr = TemporalCSR(
            data.num_nodes, data.src, data.dst,
            data.timestamps, np.arange(data.num_edges, dtype=np.int64),
        )
        negs = generate_negatives_for_eval(
            src_node=0, true_dst=1, timestamp=5.0,
            csr=csr, num_nodes=100, n_hist=50, n_random=50,
        )
        assert 1 not in negs[:50]


class TestChronologicalSplit:
    """Tests for chronological data splitting."""

    def test_split_ratios(self):
        data = _make_dummy_data(num_edges=100)
        train, val, test = chronological_split(data, 0.6, 0.2)
        assert train.sum() == 60
        assert val.sum() == 20
        assert test.sum() == 20

    def test_no_overlap(self):
        data = _make_dummy_data(num_edges=100)
        train, val, test = chronological_split(data, 0.6, 0.2)
        assert not (train & val).any()
        assert not (train & test).any()
        assert not (val & test).any()

    def test_covers_all(self):
        data = _make_dummy_data(num_edges=100)
        train, val, test = chronological_split(data, 0.6, 0.2)
        assert (train | val | test).all()


class TestFeaturizeNeighbors:
    """Tests for featurize_neighbors function (C++ or Python)."""

    def test_output_shapes(self):
        data = _make_dummy_data(num_nodes=10, num_edges=30)
        csr = TemporalCSR(
            data.num_nodes, data.src, data.dst,
            data.timestamps, np.arange(data.num_edges, dtype=np.int64),
        )
        nodes = np.array([0, 1, 2], dtype=np.int32)
        ts = np.array([5.0, 5.0, 5.0], dtype=np.float64)
        nn, nts, neids, lengths = sample_neighbors_batch(csr, nodes, ts, num_neighbors=10)

        nnf, nef, nrt = featurize_neighbors(
            nn, neids, lengths, nts, ts,
            data.node_feats, data.edge_feats,
        )
        assert nnf.shape == (3, 10, data.node_feats.shape[1])
        assert nef.shape == (3, 10, data.edge_feats.shape[1])
        assert nrt.shape == (3, 10)

    def test_zero_length_produces_zeros(self):
        data = _make_dummy_data(num_nodes=10, num_edges=30)
        nn = np.full((2, 5), -1, dtype=np.int32)
        neids = np.full((2, 5), -1, dtype=np.int64)
        lengths = np.array([0, 0], dtype=np.int32)
        nts = np.zeros((2, 5), dtype=np.float64)
        qts = np.array([5.0, 5.0], dtype=np.float64)

        nnf, nef, nrt = featurize_neighbors(
            nn, neids, lengths, nts, qts,
            data.node_feats, data.edge_feats,
        )
        assert np.all(nnf == 0)
        assert np.all(nef == 0)
        assert np.all(nrt == 0)

    def test_relative_timestamps_correct(self):
        data = _make_dummy_data(num_nodes=10, num_edges=30)
        csr = TemporalCSR(
            data.num_nodes, data.src, data.dst,
            data.timestamps, np.arange(data.num_edges, dtype=np.int64),
        )
        nodes = np.array([0], dtype=np.int32)
        ts = np.array([8.0], dtype=np.float64)
        nn, nts, neids, lengths = sample_neighbors_batch(csr, nodes, ts, num_neighbors=5)

        _, _, nrt = featurize_neighbors(
            nn, neids, lengths, nts, ts,
            data.node_feats, data.edge_feats,
        )
        length = lengths[0]
        if length > 0:
            expected_nrt = ts[0] - nts[0, :length]
            np.testing.assert_allclose(nrt[0, :length], expected_nrt)
            assert np.all(nrt[0, :length] >= 0)

    def test_node_feats_copied_correctly(self):
        rng = np.random.default_rng(123)
        node_feats = rng.standard_normal((5, 3)).astype(np.float32)
        edge_feats = rng.standard_normal((10, 2)).astype(np.float32)

        nn = np.array([[1, 3, -1]], dtype=np.int32)
        neids = np.array([[0, 2, -1]], dtype=np.int64)
        lengths = np.array([2], dtype=np.int32)
        nts = np.array([[1.0, 2.0, 0.0]], dtype=np.float64)
        qts = np.array([5.0], dtype=np.float64)

        nnf, nef, nrt = featurize_neighbors(
            nn, neids, lengths, nts, qts, node_feats, edge_feats,
        )
        np.testing.assert_array_equal(nnf[0, 0], node_feats[1])
        np.testing.assert_array_equal(nnf[0, 1], node_feats[3])
        np.testing.assert_array_equal(nef[0, 0], edge_feats[0])
        np.testing.assert_array_equal(nef[0, 1], edge_feats[2])
        assert np.all(nnf[0, 2] == 0)


class TestCppExtension:
    """Tests verifying C++ extension correctness vs Python fallback."""

    def _make_python_csr(self, data):
        """Build a CSR using Python-only path."""
        num_nodes = data.num_nodes
        src, dst = data.src, data.dst
        timestamps = data.timestamps
        edge_ids = np.arange(data.num_edges, dtype=np.int64)

        sort_idx = np.lexsort((timestamps, src))
        src_sorted = src[sort_idx]
        dst_sorted = dst[sort_idx]
        ts_sorted = timestamps[sort_idx]
        eid_sorted = edge_ids[sort_idx]

        indptr = np.zeros(num_nodes + 1, dtype=np.int64)
        for s in src_sorted:
            indptr[s + 1] += 1
        np.cumsum(indptr, out=indptr)

        class PurePythonCSR:
            pass

        csr = PurePythonCSR()
        csr.num_nodes = num_nodes
        csr.indptr = indptr
        csr.neighbors = dst_sorted.astype(np.int32)
        csr.timestamps = ts_sorted
        csr.edge_ids = eid_sorted.astype(np.int64)
        return csr

    def _python_get_neighbors(self, csr, node, before_time, k):
        start = csr.indptr[node]
        end = csr.indptr[node + 1]
        if start == end:
            return np.array([], dtype=np.int32), np.array([], dtype=np.float64), np.array([], dtype=np.int64)
        ts_slice = csr.timestamps[start:end]
        valid_end = np.searchsorted(ts_slice, before_time, side="left")
        if valid_end == 0:
            return np.array([], dtype=np.int32), np.array([], dtype=np.float64), np.array([], dtype=np.int64)
        actual_start = max(start, start + valid_end - k)
        actual_end = start + valid_end
        return (
            csr.neighbors[actual_start:actual_end].copy(),
            csr.timestamps[actual_start:actual_end].copy(),
            csr.edge_ids[actual_start:actual_end].copy(),
        )

    def test_cpp_available(self):
        cpp = _load_cpp_extension()
        assert cpp is not None, "C++ extension should be compiled and available"

    def test_csr_neighbors_match(self):
        cpp = _load_cpp_extension()
        if cpp is None:
            pytest.skip("C++ extension not available")

        data = _make_dummy_data(num_nodes=20, num_edges=100)
        py_csr = self._make_python_csr(data)
        cpp_csr = cpp.TemporalCSR(
            data.num_nodes,
            np.ascontiguousarray(data.src, dtype=np.int32),
            np.ascontiguousarray(data.dst, dtype=np.int32),
            np.ascontiguousarray(data.timestamps, dtype=np.float64),
            np.arange(data.num_edges, dtype=np.int64),
        )

        for node in range(min(20, data.num_nodes)):
            for t in [0.0, 3.0, 5.0, 8.0, 100.0]:
                py_nids, py_nts, py_eids = self._python_get_neighbors(py_csr, node, t, k=10)
                cpp_nids, cpp_nts, cpp_eids = cpp_csr.get_temporal_neighbors(node, t, 10)
                np.testing.assert_array_equal(py_nids, cpp_nids)
                np.testing.assert_allclose(py_nts, cpp_nts)
                np.testing.assert_array_equal(py_eids, cpp_eids)

    def test_batch_sampling_match(self):
        cpp = _load_cpp_extension()
        if cpp is None:
            pytest.skip("C++ extension not available")

        data = _make_dummy_data(num_nodes=20, num_edges=100)
        eids = np.arange(data.num_edges, dtype=np.int64)

        csr_with_cpp = TemporalCSR(data.num_nodes, data.src, data.dst, data.timestamps, eids)
        assert csr_with_cpp._use_cpp

        nodes = np.array([0, 1, 5, 10], dtype=np.int32)
        ts = np.array([3.0, 5.0, 7.0, 9.0], dtype=np.float64)

        nn, nts, neids, lengths = sample_neighbors_batch(csr_with_cpp, nodes, ts, 10)
        assert nn.shape == (4, 10)
        assert lengths.shape == (4,)

        for i in range(4):
            single_nids, single_nts, single_eids = csr_with_cpp.get_temporal_neighbors(
                int(nodes[i]), float(ts[i]), 10
            )
            assert lengths[i] == len(single_nids)
            if lengths[i] > 0:
                np.testing.assert_array_equal(nn[i, :lengths[i]], single_nids)

    def test_featurize_match(self):
        cpp = _load_cpp_extension()
        if cpp is None:
            pytest.skip("C++ extension not available")

        rng = np.random.default_rng(42)
        node_feats = rng.standard_normal((20, 5)).astype(np.float32)
        edge_feats = rng.standard_normal((50, 2)).astype(np.float32)

        nn = np.array([[1, 3, -1, -1], [5, 10, 15, -1]], dtype=np.int32)
        neids = np.array([[0, 5, -1, -1], [10, 20, 30, -1]], dtype=np.int64)
        lengths = np.array([2, 3], dtype=np.int32)
        nts = np.array([[1.0, 2.0, 0.0, 0.0], [1.0, 3.0, 5.0, 0.0]], dtype=np.float64)
        qts = np.array([5.0, 8.0], dtype=np.float64)

        nnf, nef, nrt = cpp.featurize_neighbors(
            np.ascontiguousarray(nn), np.ascontiguousarray(neids),
            np.ascontiguousarray(lengths), np.ascontiguousarray(nts),
            np.ascontiguousarray(qts), np.ascontiguousarray(node_feats),
            np.ascontiguousarray(edge_feats),
        )

        np.testing.assert_array_equal(nnf[0, 0], node_feats[1])
        np.testing.assert_array_equal(nnf[0, 1], node_feats[3])
        np.testing.assert_array_equal(nef[0, 0], edge_feats[0])
        np.testing.assert_array_equal(nef[0, 1], edge_feats[5])
        np.testing.assert_allclose(nrt[0, 0], 4.0)
        np.testing.assert_allclose(nrt[0, 1], 3.0)
        assert np.all(nnf[0, 2:] == 0)
        assert np.all(nef[0, 2:] == 0)

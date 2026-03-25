"""Tests for EAGLE-Time model and TPPR for temporal link prediction."""

import numpy as np
import pytest
import torch

from src.models.eagle import (
    EAGLETimeEncoding,
    EAGLEFeedForward,
    EAGLEMixerBlock,
    EAGLETimeEncoder,
    EAGLEEdgePredictor,
    EAGLETime,
)
from src.models.tppr import TPPR, get_forward_edge_mask
from src.models.data_utils import (
    TemporalEdgeData,
    TemporalCSR,
    sample_neighbors_batch,
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


class TestEAGLETimeEncoding:
    """Tests for EAGLETimeEncoding module."""

    def test_output_shape(self):
        enc = EAGLETimeEncoding(dim=100)
        t = torch.tensor([1.0, 2.0, 3.0])
        out = enc(t)
        assert out.shape == (3, 100)

    def test_output_range(self):
        enc = EAGLETimeEncoding(dim=50)
        t = torch.tensor([0.0, 100.0, 1e6])
        out = enc(t)
        assert out.min() >= -1.0
        assert out.max() <= 1.0

    def test_zero_timestamp(self):
        enc = EAGLETimeEncoding(dim=100)
        t = torch.tensor([0.0])
        out = enc(t)
        assert torch.allclose(out, torch.ones(1, 100))

    def test_similar_timestamps_similar_encodings(self):
        enc = EAGLETimeEncoding(dim=100)
        t1 = torch.tensor([100.0])
        t2 = torch.tensor([100.1])
        t3 = torch.tensor([1e6])
        out1, out2, out3 = enc(t1), enc(t2), enc(t3)
        sim_close = torch.cosine_similarity(out1, out2, dim=-1)
        sim_far = torch.cosine_similarity(out1, out3, dim=-1)
        assert sim_close > sim_far

    def test_non_trainable(self):
        enc = EAGLETimeEncoding(dim=100)
        trainable = [p for p in enc.parameters() if p.requires_grad]
        assert len(trainable) == 0

    def test_batch_shape(self):
        enc = EAGLETimeEncoding(dim=64)
        t = torch.randn(4, 20)
        out = enc(t)
        assert out.shape == (4, 20, 64)

    def test_log_spaced_frequencies(self):
        enc = EAGLETimeEncoding(dim=10)
        expected = 1.0 / 10 ** np.linspace(0, 9, 10, dtype=np.float32)
        np.testing.assert_allclose(
            enc.omega.cpu().numpy(), expected, rtol=1e-5
        )


class TestEAGLEFeedForward:
    """Tests for EAGLEFeedForward module."""

    def test_output_shape(self):
        ff = EAGLEFeedForward(dim=64, expansion_factor=2.0)
        x = torch.randn(5, 64)
        out = ff(x)
        assert out.shape == (5, 64)

    def test_single_layer_mode(self):
        ff = EAGLEFeedForward(dim=32, single_layer=True)
        x = torch.randn(3, 32)
        out = ff(x)
        assert out.shape == (3, 32)

    def test_expansion_factor(self):
        ff = EAGLEFeedForward(dim=64, expansion_factor=0.5)
        x = torch.randn(2, 64)
        out = ff(x)
        assert out.shape == (2, 64)


class TestEAGLEMixerBlock:
    """Tests for EAGLEMixerBlock module."""

    def test_output_shape(self):
        block = EAGLEMixerBlock(num_tokens=20, hidden_dim=64)
        x = torch.randn(4, 20, 64)
        out = block(x)
        assert out.shape == (4, 20, 64)

    def test_residual_connection(self):
        block = EAGLEMixerBlock(num_tokens=10, hidden_dim=32)
        x = torch.zeros(2, 10, 32)
        out = block(x)
        assert out.shape == (2, 10, 32)

    def test_compressive_token_mixing(self):
        block = EAGLEMixerBlock(
            num_tokens=20, hidden_dim=64,
            token_expansion=0.5, channel_expansion=4.0,
        )
        x = torch.randn(2, 20, 64)
        out = block(x)
        assert out.shape == (2, 20, 64)


class TestEAGLETimeEncoder:
    """Tests for EAGLETimeEncoder module."""

    def test_output_shape(self):
        enc = EAGLETimeEncoder(
            hidden_dim=64, num_neighbors=20, num_mixer_layers=1,
        )
        dt = torch.randn(4, 20)
        lengths = torch.tensor([5, 10, 20, 0])
        out = enc(dt, lengths)
        assert out.shape == (4, 64)

    def test_zero_length_handled(self):
        enc = EAGLETimeEncoder(
            hidden_dim=32, num_neighbors=10, num_mixer_layers=1,
        )
        dt = torch.zeros(2, 10)
        lengths = torch.tensor([0, 0])
        out = enc(dt, lengths)
        assert out.shape == (2, 32)
        assert not torch.isnan(out).any()

    def test_multiple_mixer_layers(self):
        enc = EAGLETimeEncoder(
            hidden_dim=32, num_neighbors=10, num_mixer_layers=3,
        )
        dt = torch.randn(3, 10)
        lengths = torch.tensor([5, 10, 3])
        out = enc(dt, lengths)
        assert out.shape == (3, 32)

    def test_masking_zeroes_padding(self):
        enc = EAGLETimeEncoder(
            hidden_dim=32, num_neighbors=5, num_mixer_layers=1, dropout=0.0,
        )
        enc.eval()
        dt = torch.tensor([[1.0, 2.0, 0.0, 0.0, 0.0]])
        lengths_2 = torch.tensor([2])
        lengths_5 = torch.tensor([5])
        out_2 = enc(dt, lengths_2)
        out_5 = enc(dt, lengths_5)
        assert not torch.allclose(out_2, out_5, atol=1e-4)


class TestEAGLEEdgePredictor:
    """Tests for EAGLEEdgePredictor module."""

    def test_pairwise(self):
        pred = EAGLEEdgePredictor(input_dim=64, hidden_dim=32)
        h_src = torch.randn(4, 64)
        h_dst = torch.randn(4, 64)
        out = pred(h_src, h_dst)
        assert out.shape == (4,)

    def test_ranking_mode(self):
        pred = EAGLEEdgePredictor(input_dim=64, hidden_dim=32)
        h_src = torch.randn(4, 64)
        h_dst = torch.randn(4, 10, 64)
        out = pred(h_src, h_dst)
        assert out.shape == (4, 10)


class TestEAGLETime:
    """Tests for the full EAGLETime model."""

    def _make_model(self):
        return EAGLETime(
            hidden_dim=32, num_neighbors=10,
            num_mixer_layers=1, dropout=0.0,
        )

    def test_encode_nodes_shape(self):
        model = self._make_model()
        dt = torch.randn(4, 10)
        lengths = torch.tensor([3, 5, 0, 10])
        out = model.encode_nodes(dt, lengths)
        assert out.shape == (4, 32)

    def test_forward_pairwise(self):
        model = self._make_model()
        out = model(
            torch.randn(4, 10), torch.tensor([5, 3, 10, 0]),
            torch.randn(4, 10), torch.tensor([2, 7, 1, 10]),
        )
        assert out.shape == (4,)

    def test_forward_ranking(self):
        model = self._make_model()
        out = model(
            torch.randn(2, 10), torch.tensor([5, 3]),
            torch.randn(2, 5, 10), torch.tensor([[2, 7, 1, 10, 3], [5, 5, 5, 5, 5]]),
        )
        assert out.shape == (2, 5)

    def test_backward(self):
        model = self._make_model()
        out = model(
            torch.randn(4, 10), torch.tensor([5, 3, 10, 0]),
            torch.randn(4, 10), torch.tensor([2, 7, 1, 10]),
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

    def test_no_edge_or_node_features(self):
        model = self._make_model()
        out = model(
            torch.randn(2, 10), torch.tensor([5, 3]),
            torch.randn(2, 10), torch.tensor([5, 3]),
        )
        assert out.shape == (2,)


class TestTPPR:
    """Tests for Temporal Personalized PageRank."""

    def test_creation(self):
        tppr = TPPR(num_nodes=10, topk=50, alpha=0.9, beta=0.8)
        assert tppr.num_nodes == 10
        assert len(tppr.ppr) == 10

    def test_empty_similarity(self):
        tppr = TPPR(num_nodes=10)
        assert tppr.get_similarity(0, 1) == 0.0

    def test_update_edge(self):
        tppr = TPPR(num_nodes=10, topk=50)
        tppr.update_edge(0, 1)
        assert len(tppr.ppr[0]) > 0
        assert len(tppr.ppr[1]) > 0
        assert tppr.norms[0] > 0
        assert tppr.norms[1] > 0

    def test_similarity_after_update(self):
        tppr = TPPR(num_nodes=10, topk=50)
        tppr.update_edge(0, 1)
        sim = tppr.get_similarity(0, 1)
        assert sim > 0

    def test_similarity_symmetry(self):
        tppr = TPPR(num_nodes=10, topk=50)
        tppr.update_edge(0, 1)
        tppr.update_edge(1, 2)
        sim_01 = tppr.get_similarity(0, 1)
        sim_10 = tppr.get_similarity(1, 0)
        assert sim_01 == sim_10

    def test_connected_higher_than_disconnected(self):
        tppr = TPPR(num_nodes=10, topk=50)
        tppr.update_edge(0, 1)
        tppr.update_edge(0, 1)
        sim_connected = tppr.get_similarity(0, 1)
        sim_disconnected = tppr.get_similarity(0, 5)
        assert sim_connected > sim_disconnected

    def test_topk_truncation(self):
        tppr = TPPR(num_nodes=100, topk=5)
        for i in range(1, 50):
            tppr.update_edge(0, i)
        assert len(tppr.ppr[0]) <= 5

    def test_self_loop(self):
        tppr = TPPR(num_nodes=5)
        tppr.update_edge(0, 0)
        assert tppr.norms[0] > 0

    def test_process_edges(self):
        tppr = TPPR(num_nodes=10, topk=50)
        src = np.array([0, 1, 2, 0], dtype=np.int32)
        dst = np.array([1, 2, 3, 3], dtype=np.int32)
        tppr.process_edges(src, dst, desc="test")
        assert tppr.get_similarity(0, 1) > 0
        assert tppr.get_similarity(0, 3) > 0


class TestGetForwardEdgeMask:
    """Tests for get_forward_edge_mask."""

    def test_basic(self):
        data = _make_dummy_data(num_nodes=10, num_edges=20)
        mask = get_forward_edge_mask(data)
        assert mask.shape == (20,)
        assert mask.dtype == bool
        assert mask.sum() > 0

    def test_half_per_timestamp(self):
        src = np.array([0, 1, 2, 3], dtype=np.int32)
        dst = np.array([1, 0, 3, 2], dtype=np.int32)
        timestamps = np.array([1.0, 1.0, 2.0, 2.0])
        data = TemporalEdgeData(
            src=src, dst=dst, timestamps=timestamps,
            edge_feats=np.zeros((4, 2), dtype=np.float32),
            node_feats=np.zeros((4, 25), dtype=np.float32),
            node_id_map={i: i for i in range(4)},
            reverse_node_map=np.arange(4, dtype=np.int64),
        )
        mask = get_forward_edge_mask(data)
        assert mask.sum() == 2
        assert mask[0] and not mask[1]
        assert mask[2] and not mask[3]


class TestPrepareEagleBatch:
    """Tests for prepare_eagle_batch from eagle_train."""

    def test_output_keys_and_shapes(self):
        from src.models.eagle_train import prepare_eagle_batch

        data = _make_dummy_data(num_nodes=20, num_edges=100)
        csr = TemporalCSR(
            data.num_nodes, data.src, data.dst,
            data.timestamps, np.arange(data.num_edges, dtype=np.int64),
        )

        src = np.array([0, 1, 2], dtype=np.int32)
        dst = np.array([3, 4, 5], dtype=np.int32)
        ts = np.array([5.0, 5.0, 5.0], dtype=np.float64)
        neg = np.array([[6, 7], [8, 9], [10, 11]], dtype=np.int32)
        K = 10

        batch = prepare_eagle_batch(
            csr, data, src, dst, ts, neg, K, torch.device("cpu"),
        )

        assert "src_delta_times" in batch
        assert "src_lengths" in batch
        assert "pos_dst_delta_times" in batch
        assert "pos_dst_lengths" in batch
        assert "neg_dst_delta_times" in batch
        assert "neg_dst_lengths" in batch

        assert batch["src_delta_times"].shape == (3, K)
        assert batch["src_lengths"].shape == (3,)
        assert batch["pos_dst_delta_times"].shape == (3, K)
        assert batch["neg_dst_delta_times"].shape == (3, 2, K)
        assert batch["neg_dst_lengths"].shape == (3, 2)

    def test_delta_times_non_negative(self):
        from src.models.eagle_train import prepare_eagle_batch

        data = _make_dummy_data(num_nodes=20, num_edges=100)
        csr = TemporalCSR(
            data.num_nodes, data.src, data.dst,
            data.timestamps, np.arange(data.num_edges, dtype=np.int64),
        )

        src = np.array([0, 1], dtype=np.int32)
        dst = np.array([2, 3], dtype=np.int32)
        ts = np.array([5.0, 5.0], dtype=np.float64)
        neg = np.array([[4], [5]], dtype=np.int32)

        batch = prepare_eagle_batch(
            csr, data, src, dst, ts, neg, 10, torch.device("cpu"),
        )
        assert (batch["src_delta_times"] >= 0).all()
        assert (batch["pos_dst_delta_times"] >= 0).all()
        assert (batch["neg_dst_delta_times"] >= 0).all()


class TestEAGLETrainingIntegration:
    """Integration tests for EAGLE training components."""

    def test_train_epoch_runs(self):
        from src.models.eagle_train import train_epoch
        from src.models.data_utils import build_temporal_csr

        data = _make_dummy_data(num_nodes=20, num_edges=100)
        train_mask = data.timestamps < 7.0
        csr = build_temporal_csr(data, train_mask)
        train_indices = np.where(train_mask)[0]

        model = EAGLETime(hidden_dim=16, num_neighbors=5, num_mixer_layers=1)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        metrics = train_epoch(
            model, data, csr, train_indices, optimizer,
            torch.device("cpu"), batch_size=20, num_neighbors=5,
            use_amp=False, rng=np.random.default_rng(42),
        )
        assert "loss" in metrics
        assert not np.isnan(metrics["loss"])

    def test_validate_runs(self):
        from src.models.eagle_train import validate
        from src.models.data_utils import build_temporal_csr

        data = _make_dummy_data(num_nodes=20, num_edges=100)
        full_mask = np.ones(data.num_edges, dtype=bool)
        csr = build_temporal_csr(data, full_mask)
        val_indices = np.where(data.timestamps >= 7.0)[0][:10]

        model = EAGLETime(hidden_dim=16, num_neighbors=5, num_mixer_layers=1)
        model.eval()

        metrics = validate(
            model, data, csr, val_indices, torch.device("cpu"),
            num_neighbors=5, n_eval_negatives=10, max_eval_edges=5,
            use_amp=False,
        )
        assert "mrr" in metrics
        assert 0 <= metrics["mrr"] <= 1
        assert "hits@1" in metrics
        assert "hits@10" in metrics

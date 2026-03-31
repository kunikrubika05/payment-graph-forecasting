"""Unit tests for PairwiseMLP components.

Covers:
  - PairwiseDataset without node features (baseline behaviour)
  - PairwiseDataset with node features (shape, dtype, values)
  - make_scheduler: cosine, plateau, empty
  - build_eval_cache: node features concatenation, disk cache load/save
  - config.n_input_features with use_node_features
"""

import os
import tempfile

import numpy as np
import pytest
import torch

from src.models.pairwise_mlp.config import PairMLPConfig, N_FEATURES
from src.models.pairwise_mlp.dataset import PairwiseDataset, load_dataset, N_NODE_FEATURES
from src.models.pairwise_mlp.model import build_model
from src.models.pairwise_mlp.train import make_scheduler


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

N = 100
K = 4
N_NODES = 50


def make_pair_arrays(n=N, k=K):
    pos = np.random.rand(n, N_FEATURES).astype(np.float32)
    neg = np.random.rand(n, k, N_FEATURES).astype(np.float32)
    return pos, neg


def make_node_arrays(n_nodes=N_NODES, n_train=N, k=K):
    node_feat = np.random.rand(n_nodes, N_NODE_FEATURES).astype(np.float32)
    src_local = np.random.randint(0, n_nodes, size=n_train).astype(np.int64)
    dst_local = np.random.randint(0, n_nodes, size=n_train).astype(np.int64)
    neg_dst_local = np.random.randint(0, n_nodes, size=(n_train, k)).astype(np.int64)
    return node_feat, src_local, dst_local, neg_dst_local


# ---------------------------------------------------------------------------
# PairwiseDataset — without node features
# ---------------------------------------------------------------------------

class TestPairwiseDatasetBasic:
    def test_length(self):
        pos, neg = make_pair_arrays()
        ds = PairwiseDataset(pos, neg)
        assert len(ds) == N

    def test_item_shapes(self):
        pos, neg = make_pair_arrays()
        ds = PairwiseDataset(pos, neg)
        p, n = ds[0]
        assert p.shape == (N_FEATURES,)
        assert n.shape == (K, N_FEATURES)

    def test_item_dtype(self):
        pos, neg = make_pair_arrays()
        ds = PairwiseDataset(pos, neg)
        p, n = ds[0]
        assert p.dtype == torch.float32
        assert n.dtype == torch.float32

    def test_active_feature_indices(self):
        pos, neg = make_pair_arrays()
        ds = PairwiseDataset(pos, neg, active_feature_indices=[0, 2])
        p, n = ds[0]
        assert p.shape == (2,)
        assert n.shape == (K, 2)

    def test_neg_k_attribute(self):
        pos, neg = make_pair_arrays()
        ds = PairwiseDataset(pos, neg)
        assert ds.neg_k == K

    def test_values_match_input(self):
        pos, neg = make_pair_arrays()
        ds = PairwiseDataset(pos, neg)
        p, n = ds[5]
        np.testing.assert_allclose(p.numpy(), pos[5])
        np.testing.assert_allclose(n.numpy(), neg[5])


# ---------------------------------------------------------------------------
# PairwiseDataset — with node features
# ---------------------------------------------------------------------------

class TestPairwiseDatasetNodeFeatures:
    def test_item_shape_with_node_features(self):
        pos, neg = make_pair_arrays()
        nf, sl, dl, ndl = make_node_arrays()
        ds = PairwiseDataset(pos, neg, node_features=nf,
                             src_local=sl, dst_local=dl, neg_dst_local=ndl)
        p, n = ds[0]
        # pair(7) + node_src(15) + node_dst(15) = 37
        assert p.shape == (N_FEATURES + 2 * N_NODE_FEATURES,)
        assert n.shape == (K, N_FEATURES + 2 * N_NODE_FEATURES)

    def test_item_dtype_with_node_features(self):
        pos, neg = make_pair_arrays()
        nf, sl, dl, ndl = make_node_arrays()
        ds = PairwiseDataset(pos, neg, node_features=nf,
                             src_local=sl, dst_local=dl, neg_dst_local=ndl)
        p, n = ds[0]
        assert p.dtype == torch.float32
        assert n.dtype == torch.float32

    def test_pair_features_preserved_in_prefix(self):
        pos, neg = make_pair_arrays()
        nf, sl, dl, ndl = make_node_arrays()
        ds = PairwiseDataset(pos, neg, node_features=nf,
                             src_local=sl, dst_local=dl, neg_dst_local=ndl)
        p, n = ds[3]
        # First N_FEATURES dims must match the original pair features
        np.testing.assert_allclose(p[:N_FEATURES].numpy(), pos[3])
        np.testing.assert_allclose(n[:, :N_FEATURES].numpy(), neg[3])

    def test_src_node_features_appended(self):
        pos, neg = make_pair_arrays()
        nf, sl, dl, ndl = make_node_arrays()
        ds = PairwiseDataset(pos, neg, node_features=nf,
                             src_local=sl, dst_local=dl, neg_dst_local=ndl)
        i = 7
        p, n = ds[i]
        expected_src_feat = nf[sl[i]]
        np.testing.assert_allclose(
            p[N_FEATURES: N_FEATURES + N_NODE_FEATURES].numpy(), expected_src_feat
        )
        # All K negatives share same src features
        for k in range(K):
            np.testing.assert_allclose(
                n[k, N_FEATURES: N_FEATURES + N_NODE_FEATURES].numpy(), expected_src_feat
            )

    def test_dst_pos_node_features_appended(self):
        pos, neg = make_pair_arrays()
        nf, sl, dl, ndl = make_node_arrays()
        ds = PairwiseDataset(pos, neg, node_features=nf,
                             src_local=sl, dst_local=dl, neg_dst_local=ndl)
        i = 2
        p, _ = ds[i]
        expected_dst_feat = nf[dl[i]]
        np.testing.assert_allclose(
            p[N_FEATURES + N_NODE_FEATURES:].numpy(), expected_dst_feat
        )

    def test_neg_dst_node_features_appended(self):
        pos, neg = make_pair_arrays()
        nf, sl, dl, ndl = make_node_arrays()
        ds = PairwiseDataset(pos, neg, node_features=nf,
                             src_local=sl, dst_local=dl, neg_dst_local=ndl)
        i = 4
        _, n = ds[i]
        for k in range(K):
            expected = nf[ndl[i, k]]
            np.testing.assert_allclose(
                n[k, N_FEATURES + N_NODE_FEATURES:].numpy(), expected
            )

    def test_active_features_plus_node_features(self):
        pos, neg = make_pair_arrays()
        nf, sl, dl, ndl = make_node_arrays()
        ds = PairwiseDataset(pos, neg, active_feature_indices=[0, 1],
                             node_features=nf,
                             src_local=sl, dst_local=dl, neg_dst_local=ndl)
        p, n = ds[0]
        # 2 pair features + 15 src + 15 dst = 32
        assert p.shape == (2 + 2 * N_NODE_FEATURES,)
        assert n.shape == (K, 2 + 2 * N_NODE_FEATURES)

    def test_missing_index_arrays_raises(self):
        pos, neg = make_pair_arrays()
        nf = np.random.rand(N_NODES, N_NODE_FEATURES).astype(np.float32)
        with pytest.raises(AssertionError):
            PairwiseDataset(pos, neg, node_features=nf)  # missing src/dst/neg_dst

    def test_dataloader_batching(self):
        from torch.utils.data import DataLoader
        pos, neg = make_pair_arrays()
        nf, sl, dl, ndl = make_node_arrays()
        ds = PairwiseDataset(pos, neg, node_features=nf,
                             src_local=sl, dst_local=dl, neg_dst_local=ndl)
        loader = DataLoader(ds, batch_size=16)
        batch_p, batch_n = next(iter(loader))
        assert batch_p.shape == (16, N_FEATURES + 2 * N_NODE_FEATURES)
        assert batch_n.shape == (16, K, N_FEATURES + 2 * N_NODE_FEATURES)


# ---------------------------------------------------------------------------
# config.n_input_features
# ---------------------------------------------------------------------------

class TestConfigNInputFeatures:
    def test_all_pair_features_no_node(self):
        cfg = PairMLPConfig(use_node_features=False)
        assert cfg.n_input_features == N_FEATURES

    def test_subset_pair_features_no_node(self):
        cfg = PairMLPConfig(active_feature_indices=[0, 1, 2], use_node_features=False)
        assert cfg.n_input_features == 3

    def test_all_pair_features_with_node(self):
        cfg = PairMLPConfig(use_node_features=True)
        assert cfg.n_input_features == N_FEATURES + 30

    def test_subset_pair_features_with_node(self):
        cfg = PairMLPConfig(active_feature_indices=[0], use_node_features=True)
        assert cfg.n_input_features == 1 + 30


# ---------------------------------------------------------------------------
# make_scheduler
# ---------------------------------------------------------------------------

class TestMakeScheduler:
    def _optimizer(self):
        model = torch.nn.Linear(4, 1)
        return torch.optim.Adam(model.parameters(), lr=0.01)

    def test_no_scheduler(self):
        cfg = PairMLPConfig(scheduler="")
        opt = self._optimizer()
        sched = make_scheduler(opt, cfg)
        assert sched is None

    def test_cosine_scheduler(self):
        cfg = PairMLPConfig(scheduler="cosine", n_epochs=20, scheduler_min_lr=1e-7)
        opt = self._optimizer()
        sched = make_scheduler(opt, cfg)
        assert sched is not None
        assert isinstance(sched,
                          torch.optim.lr_scheduler.CosineAnnealingLR)

    def test_plateau_scheduler(self):
        cfg = PairMLPConfig(scheduler="plateau", scheduler_patience=2,
                            scheduler_factor=0.5)
        opt = self._optimizer()
        sched = make_scheduler(opt, cfg)
        assert sched is not None
        assert isinstance(sched,
                          torch.optim.lr_scheduler.ReduceLROnPlateau)

    def test_unknown_scheduler_raises(self):
        cfg = PairMLPConfig()
        cfg.scheduler = "cyclic"
        opt = self._optimizer()
        with pytest.raises(ValueError, match="Unknown scheduler"):
            make_scheduler(opt, cfg)

    def test_cosine_lr_decays(self):
        cfg = PairMLPConfig(scheduler="cosine", n_epochs=10,
                            scheduler_min_lr=0.0, lr=0.01)
        opt = self._optimizer()
        sched = make_scheduler(opt, cfg)
        initial_lr = opt.param_groups[0]["lr"]
        for _ in range(5):
            sched.step()
        mid_lr = opt.param_groups[0]["lr"]
        assert mid_lr < initial_lr

    def test_cosine_lr_reaches_min(self):
        min_lr = 1e-6
        cfg = PairMLPConfig(scheduler="cosine", n_epochs=10, scheduler_min_lr=min_lr)
        opt = self._optimizer()
        sched = make_scheduler(opt, cfg)
        for _ in range(10):
            sched.step()
        final_lr = opt.param_groups[0]["lr"]
        assert abs(final_lr - min_lr) < 1e-8


# ---------------------------------------------------------------------------
# build_eval_cache — disk cache save/load
# ---------------------------------------------------------------------------

class TestEvalCacheDisk:
    def _make_minimal_cache_inputs(self):
        """Create minimal inputs for build_eval_cache (no adjacency data needed
        for this test — we mock feature computation)."""
        return None  # placeholder, patched below

    def test_disk_cache_save_and_load(self):
        """build_eval_cache saves .npz and second call loads it without recomputing."""
        from scipy import sparse
        from src.models.pairwise_mlp.evaluate import build_eval_cache
        import pandas as pd

        n_nodes = 20
        node_mapping = np.arange(n_nodes, dtype=np.int64)
        adj = sparse.eye(n_nodes, format="csr", dtype=np.float32)
        deg = np.ones(n_nodes, dtype=np.float64)
        w = np.zeros(n_nodes, dtype=np.float64)

        # Build 50 simple val edges between existing nodes
        rng = np.random.RandomState(0)
        srcs = rng.randint(0, n_nodes, 50).astype(np.int64)
        dsts = rng.randint(0, n_nodes, 50).astype(np.int64)
        eval_edges = pd.DataFrame({"src_idx": srcs, "dst_idx": dsts,
                                   "timestamp": np.zeros(50, dtype=np.int64)})
        train_neighbors = {int(s): {int(d)} for s, d in zip(srcs, dsts)}

        with tempfile.TemporaryDirectory() as tmpdir:
            cache_path = os.path.join(tmpdir, "eval_cache_val.npz")

            # First call: builds and saves
            cache1 = build_eval_cache(
                eval_edges, train_neighbors, node_mapping,
                adj, adj, deg, w, w,
                seed=52, n_negatives=5, max_queries=20,
                cache_path=cache_path, split_name="val",
            )
            assert os.path.exists(cache_path)

            # Second call: loads from disk (would fail if recomputing with wrong data)
            cache2 = build_eval_cache(
                eval_edges, train_neighbors, node_mapping,
                adj, adj, deg, w, w,
                seed=52, n_negatives=5, max_queries=20,
                cache_path=cache_path, split_name="val",
            )

        assert cache1["n_queries"] == cache2["n_queries"]
        np.testing.assert_array_equal(cache1["all_features"], cache2["all_features"])
        assert cache1["query_offsets"] == cache2["query_offsets"]

    def test_node_features_appended_in_cache(self):
        """When node_features provided, final feature dim = n_pair + 30."""
        from scipy import sparse
        from src.models.pairwise_mlp.evaluate import build_eval_cache
        import pandas as pd

        n_nodes = 20
        node_mapping = np.arange(n_nodes, dtype=np.int64)
        adj = sparse.eye(n_nodes, format="csr", dtype=np.float32)
        deg = np.ones(n_nodes, dtype=np.float64)
        w = np.zeros(n_nodes, dtype=np.float64)
        node_features = np.random.rand(n_nodes, N_NODE_FEATURES).astype(np.float32)

        rng = np.random.RandomState(0)
        srcs = rng.randint(0, n_nodes, 30).astype(np.int64)
        dsts = rng.randint(0, n_nodes, 30).astype(np.int64)
        eval_edges = pd.DataFrame({"src_idx": srcs, "dst_idx": dsts,
                                   "timestamp": np.zeros(30, dtype=np.int64)})
        train_neighbors = {int(s): {int(d)} for s, d in zip(srcs, dsts)}

        cache = build_eval_cache(
            eval_edges, train_neighbors, node_mapping,
            adj, adj, deg, w, w,
            seed=52, n_negatives=5, max_queries=10,
            node_features=node_features, split_name="val",
        )
        # All 7 pair features + 15 src node + 15 dst node = 37
        assert cache["all_features"].shape[1] == N_FEATURES + 2 * N_NODE_FEATURES

    def test_active_features_with_node_features(self):
        """Selecting 1 pair feature + node features → dim = 1 + 30 = 31."""
        from scipy import sparse
        from src.models.pairwise_mlp.evaluate import build_eval_cache
        import pandas as pd

        n_nodes = 20
        node_mapping = np.arange(n_nodes, dtype=np.int64)
        adj = sparse.eye(n_nodes, format="csr", dtype=np.float32)
        deg = np.ones(n_nodes, dtype=np.float64)
        w = np.zeros(n_nodes, dtype=np.float64)
        node_features = np.random.rand(n_nodes, N_NODE_FEATURES).astype(np.float32)

        rng = np.random.RandomState(0)
        srcs = rng.randint(0, n_nodes, 30).astype(np.int64)
        dsts = rng.randint(0, n_nodes, 30).astype(np.int64)
        eval_edges = pd.DataFrame({"src_idx": srcs, "dst_idx": dsts,
                                   "timestamp": np.zeros(30, dtype=np.int64)})
        train_neighbors = {int(s): {int(d)} for s, d in zip(srcs, dsts)}

        cache = build_eval_cache(
            eval_edges, train_neighbors, node_mapping,
            adj, adj, deg, w, w,
            seed=52, n_negatives=5, max_queries=10,
            active_feature_indices=[0],
            node_features=node_features, split_name="val",
        )
        assert cache["all_features"].shape[1] == 1 + 2 * N_NODE_FEATURES


# ---------------------------------------------------------------------------
# load_dataset with precompute_dir (integration smoke test)
# ---------------------------------------------------------------------------

class TestLoadDataset:
    def test_load_without_node_features(self):
        pos, neg = make_pair_arrays()
        with tempfile.TemporaryDirectory() as tmpdir:
            np.save(os.path.join(tmpdir, "pos_features.npy"), pos)
            np.save(os.path.join(tmpdir, "neg_features.npy"), neg)
            ds = load_dataset(tmpdir)
        assert len(ds) == N
        p, n = ds[0]
        assert p.shape == (N_FEATURES,)

    def test_load_with_node_features(self):
        pos, neg = make_pair_arrays()
        nf, sl, dl, ndl = make_node_arrays()
        with tempfile.TemporaryDirectory() as tmpdir:
            np.save(os.path.join(tmpdir, "pos_features.npy"), pos)
            np.save(os.path.join(tmpdir, "neg_features.npy"), neg)
            ds = load_dataset(tmpdir, node_features=nf,
                              src_local=sl, dst_local=dl, neg_dst_local=ndl)
        p, n = ds[0]
        assert p.shape == (N_FEATURES + 2 * N_NODE_FEATURES,)
        assert n.shape == (K, N_FEATURES + 2 * N_NODE_FEATURES)

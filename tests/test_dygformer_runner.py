"""Tests for the migrated DyGFormer runner."""

from argparse import Namespace

import pytest
import torch
import numpy as np

from payment_graph_forecasting.experiments.runners.dygformer import (
    _build_eval_infrastructure,
    build_dygformer_arg_parser,
    run_dygformer_experiment,
)
from src.models.DyGFormer.dygformer import DyGFormerTime, DyGFormerTimeEncoding


def test_dygformer_arg_parser_supports_dry_run():
    parser = build_dygformer_arg_parser()
    args = parser.parse_args(["--parquet-path", "/tmp/stream.parquet", "--dry-run"])
    assert args.parquet_path == "/tmp/stream.parquet"
    assert args.sampling_backend == "auto"
    assert args.dry_run is True


def test_dygformer_runner_dry_run_returns_payload(tmp_path):
    args = Namespace(
        data_source="stream_graph",
        raw_path=None,
        raw_remote_path=None,
        parquet_path="/tmp/stream.parquet",
        parquet_remote_path=None,
        features_path=None,
        features_remote_path=None,
        node_mapping_path=None,
        node_mapping_remote_path=None,
        data_backend="yadisk",
        data_cache_dir=None,
        data_token_env="YADISK_TOKEN",
        sampling_backend="cuda",
        train_ratio=0.7,
        val_ratio=0.15,
        fraction=0.1,
        output=str(tmp_path),
        exp_name="dygformer_smoke",
        device="cpu",
        upload=False,
        remote_dir=None,
        upload_backend="yadisk",
        token_env="YADISK_TOKEN",
        epochs=1,
        batch_size=4,
        lr=1e-4,
        weight_decay=0.0,
        num_neighbors=12,
        patch_size=2,
        time_dim=16,
        aligned_dim=8,
        num_transformer_layers=1,
        num_attention_heads=2,
        cooc_dim=4,
        output_dim=16,
        dropout=0.1,
        patience=1,
        seed=42,
        no_amp=True,
        max_val_edges=10,
        max_test_edges=12,
        n_hist_neg=50,
        n_random_neg=50,
        neg_per_positive=3,
        edge_feat_dim=2,
        node_feat_dim=0,
        dry_run=True,
    )
    result = run_dygformer_experiment(args)
    assert result["mode"] == "dry_run"
    assert result["parquet_path"] == "/tmp/stream.parquet"
    assert result["patch_size"] == 2
    assert result["neg_per_positive"] == 3
    assert result["sampling_backend"] == "cuda"


def test_build_eval_infrastructure_uses_dense_ids_from_loaded_data():
    class DummyData:
        src = np.array([10, 0, 3, 7], dtype=np.int32)
        dst = np.array([20, 1, 4, 8], dtype=np.int32)
        timestamps = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64)

    train_mask = np.array([True, True, False, False])
    val_mask = np.array([False, False, True, False])
    test_mask = np.array([False, False, False, True])

    infra = _build_eval_infrastructure(DummyData, train_mask, val_mask, test_mask)

    assert sorted(infra["train_neighbors"].keys()) == [0, 10]
    assert np.array_equal(infra["active_nodes"], np.array([0, 1, 10, 20], dtype=np.int64))
    assert np.array_equal(infra["val_src"], np.array([3], dtype=np.int32))
    assert np.array_equal(infra["val_dst"], np.array([4], dtype=np.int32))
    assert np.array_equal(infra["test_src"], np.array([7], dtype=np.int32))
    assert np.array_equal(infra["test_dst"], np.array([8], dtype=np.int32))


def test_dygformer_forward_stays_finite():
    model = DyGFormerTime(
        time_dim=8,
        aligned_dim=4,
        num_neighbors=4,
        patch_size=2,
        num_transformer_layers=1,
        num_attention_heads=2,
        dropout=0.0,
        edge_feat_dim=2,
        node_feat_dim=0,
        cooc_dim=4,
        output_dim=8,
    )
    logits = model(
        src_delta_times=torch.tensor([[0.0, 1.0, 2.0, 0.0]]),
        src_lengths=torch.tensor([3]),
        dst_delta_times=torch.tensor([[0.0, 2.0, 3.0, 0.0]]),
        dst_lengths=torch.tensor([3]),
        src_cooc_counts=torch.tensor([[[1.0, 0.0], [1.0, 1.0], [1.0, 0.0], [0.0, 0.0]]]),
        dst_cooc_counts=torch.tensor([[[1.0, 1.0], [1.0, 0.0], [1.0, 0.0], [0.0, 0.0]]]),
        src_edge_feats=torch.tensor([[[1.0, 2.0], [2.0, 1.0], [0.5, 0.5], [0.0, 0.0]]]),
        dst_edge_feats=torch.tensor([[[0.5, 1.5], [1.5, 0.5], [0.1, 0.2], [0.0, 0.0]]]),
    )
    assert torch.isfinite(logits).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for AMP regression check")
def test_dygformer_time_encoding_stays_finite_under_cuda_autocast():
    encoder = DyGFormerTimeEncoding(100).cuda().eval()
    delta_times = torch.tensor([[0.0, 60.0, 3600.0, 86400.0, 1e6, 1e7]], device="cuda")

    with torch.autocast(device_type="cuda", dtype=torch.float16):
        encoded = encoder(delta_times)

    assert torch.isfinite(encoded).all()

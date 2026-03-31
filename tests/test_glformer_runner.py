"""Tests for the migrated GLFormer runner."""

from argparse import Namespace

import torch

from payment_graph_forecasting.experiments.runners.glformer import (
    build_glformer_arg_parser,
    run_glformer_experiment,
)
from src.models.GLFormer.glformer import GLFormerTime


def test_glformer_arg_parser_supports_dry_run():
    parser = build_glformer_arg_parser()
    args = parser.parse_args(["--parquet-path", "/tmp/stream.parquet", "--dry-run"])
    assert args.parquet_path == "/tmp/stream.parquet"
    assert args.dry_run is True


def test_glformer_runner_dry_run_returns_payload(tmp_path):
    args = Namespace(
        data_source="stream_graph",
        raw_path=None,
        raw_remote_path=None,
        parquet_path="/tmp/stream.parquet",
        parquet_remote_path=None,
        train_ratio=0.7,
        val_ratio=0.15,
        output=str(tmp_path),
        exp_name="glformer_smoke",
        device="cpu",
        upload=False,
        remote_dir=None,
        upload_backend="yadisk",
        token_env="YADISK_TOKEN",
        sampling_backend="cpp",
        epochs=1,
        batch_size=4,
        lr=1e-4,
        weight_decay=0.0,
        num_neighbors=12,
        hidden_dim=32,
        num_glformer_layers=1,
        channel_expansion=4.0,
        dropout=0.1,
        patience=1,
        seed=42,
        no_amp=True,
        max_val_edges=10,
        max_test_edges=12,
        edge_feat_dim=2,
        node_feat_dim=0,
        node_feats_path=None,
        node_feats_remote_path=None,
        use_cooccurrence=True,
        cooc_dim=8,
        adj_path=None,
        node_mapping_path=None,
        node_mapping_remote_path=None,
        data_backend="yadisk",
        data_cache_dir=None,
        data_token_env="YADISK_TOKEN",
        dry_run=True,
    )
    result = run_glformer_experiment(args)
    assert result["mode"] == "dry_run"
    assert result["parquet_path"] == "/tmp/stream.parquet"
    assert result["parquet_remote_path"] is None
    assert result["sampling_backend"] == "cpp"
    assert result["use_cooccurrence"] is True


def test_glformer_forward_stays_finite_with_large_edge_features():
    model = GLFormerTime(
        hidden_dim=16,
        num_neighbors=4,
        num_glformer_layers=1,
        channel_expansion=2.0,
        dropout=0.0,
        edge_feat_dim=2,
        node_feat_dim=0,
        use_cooccurrence=False,
    )
    logits = model(
        src_delta_times=torch.tensor([[0.0, 1.0, 2.0, 0.0]]),
        src_lengths=torch.tensor([3]),
        dst_delta_times=torch.tensor([[0.0, 2.0, 3.0, 0.0]]),
        dst_lengths=torch.tensor([3]),
        src_edge_feats=torch.tensor(
            [[[1e-2, 8.0e6], [1.0, 1.0e5], [5.0, 2.0e3], [float("nan"), float("inf")]]]
        ),
        dst_edge_feats=torch.tensor(
            [[[3e-2, 7.5e6], [2.0, 2.0e5], [7.0, 9.0e3], [float("-inf"), 0.0]]]
        ),
    )
    assert torch.isfinite(logits).all()

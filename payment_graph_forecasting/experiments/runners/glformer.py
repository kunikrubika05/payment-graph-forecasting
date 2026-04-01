"""GLFormer runner on top of the new package layout."""

from __future__ import annotations

import argparse
import logging
import os
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from payment_graph_forecasting.experiments.runner_utils import (
    attach_file_logger,
    configure_root_logging,
    describe_runtime,
    ensure_output_dir,
    maybe_upload_from_args,
    save_json,
    save_training_curves,
)
from payment_graph_forecasting.experiments.results import (
    build_dry_run_result,
    build_final_results,
)
from payment_graph_forecasting.evaluation.api import evaluate_glformer_model
from payment_graph_forecasting.infra.datasets import resolve_stream_graph_dataset
from payment_graph_forecasting.training.api import train_glformer_model
from src.models.stream_graph_data import TemporalCSR, build_temporal_csr, load_stream_graph_data

logger = configure_root_logging()


def build_glformer_arg_parser() -> argparse.ArgumentParser:
    """Build the GLFormer CLI parser."""

    parser = argparse.ArgumentParser(description="GLFormer temporal link prediction on stream graphs")
    parser.add_argument("--data-source", type=str, default="stream_graph")
    parser.add_argument("--raw-path", type=str, default=None)
    parser.add_argument("--raw-remote-path", type=str, default=None)
    parser.add_argument("--parquet-path", type=str, default=None)
    parser.add_argument("--parquet-remote-path", type=str, default=None)
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--output", type=str, default="/tmp/glformer_results")
    parser.add_argument("--exp-name", type=str, default=None)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--upload", action="store_true")
    parser.add_argument("--upload-backend", type=str, default="yadisk")
    parser.add_argument("--remote-dir", type=str, default=None)
    parser.add_argument("--token-env", type=str, default="YADISK_TOKEN")
    parser.add_argument("--sampling-backend", type=str, default="auto", choices=["auto", "cuda", "cpp", "python"])
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=200)
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--num-neighbors", type=int, default=20)
    parser.add_argument("--hidden-dim", type=int, default=100)
    parser.add_argument("--num-glformer-layers", type=int, default=2)
    parser.add_argument("--channel-expansion", type=float, default=4.0)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--max-val-edges", type=int, default=5000)
    parser.add_argument("--max-test-edges", type=int, default=None)
    parser.add_argument("--n-hist-neg", type=int, default=50)
    parser.add_argument("--n-random-neg", type=int, default=50)
    parser.add_argument("--edge-feat-dim", type=int, default=2)
    parser.add_argument("--node-feat-dim", type=int, default=0)
    parser.add_argument("--node-feats-path", type=str, default=None)
    parser.add_argument("--node-feats-remote-path", type=str, default=None)
    parser.add_argument("--use-cooccurrence", action="store_true")
    parser.add_argument("--cooc-dim", type=int, default=16)
    parser.add_argument("--adj-path", type=str, default=None)
    parser.add_argument("--node-mapping-path", type=str, default=None)
    parser.add_argument("--node-mapping-remote-path", type=str, default=None)
    parser.add_argument("--data-backend", type=str, default="yadisk")
    parser.add_argument("--data-cache-dir", type=str, default=None)
    parser.add_argument("--data-token-env", type=str, default="YADISK_TOKEN")
    parser.add_argument("--dry-run", action="store_true")
    return parser

def _build_data_summary(data, train_mask, val_mask, test_mask):
    return {
        "num_nodes": int(data.num_nodes),
        "num_edges": int(data.num_edges),
        "train_edges": int(train_mask.sum()),
        "val_edges": int(val_mask.sum()),
        "test_edges": int(test_mask.sum()),
        "timestamp_min": float(data.timestamps.min()),
        "timestamp_max": float(data.timestamps.max()),
    }


def _build_eval_infrastructure(parquet_path, train_ratio, val_ratio):
    df = pd.read_parquet(parquet_path)
    n_total = len(df)
    train_end = int(n_total * train_ratio)
    val_end = int(n_total * (train_ratio + val_ratio))

    train_df = df.iloc[:train_end]
    val_df = df.iloc[train_end:val_end]
    test_df = df.iloc[val_end:]

    train_src = train_df["src_idx"].values
    train_dst = train_df["dst_idx"].values
    train_neighbors: dict[int, set[int]] = defaultdict(set)
    for s, d in zip(train_src, train_dst):
        train_neighbors[int(s)].add(int(d))
    train_neighbors = dict(train_neighbors)

    active_nodes = np.unique(np.concatenate([train_src, train_dst])).astype(np.int64)

    return {
        "train_neighbors": train_neighbors,
        "active_nodes": active_nodes,
        "val_src": val_df["src_idx"].values.astype(np.int32),
        "val_dst": val_df["dst_idx"].values.astype(np.int32),
        "val_ts": val_df["timestamp"].values.astype(np.float64),
        "test_src": test_df["src_idx"].values.astype(np.int32),
        "test_dst": test_df["dst_idx"].values.astype(np.int32),
        "test_ts": test_df["timestamp"].values.astype(np.float64),
        "n_train": len(train_df),
        "n_val": len(val_df),
        "n_test": len(test_df),
    }


def run_glformer_experiment(args: argparse.Namespace):
    """Run a full GLFormer experiment using the new package runner."""

    parquet_ref = args.parquet_path or args.parquet_remote_path
    if parquet_ref is None and args.dry_run:
        parquet_ref = "unspecified"
    if parquet_ref is None:
        raise ValueError("GLFormer requires either --parquet-path or --parquet-remote-path")
    parquet_name = Path(parquet_ref).stem
    exp_name = args.exp_name if args.exp_name else f"glformer_{parquet_name}"
    output_dir = os.path.join(args.output, exp_name)
    n_hist_neg = getattr(args, "n_hist_neg", 50)
    n_random_neg = getattr(args, "n_random_neg", 50)

    if args.dry_run:
        ensure_output_dir(output_dir)
        return build_dry_run_result(
            experiment=exp_name,
            output_dir=output_dir,
            data_source=args.data_source,
            raw_path=args.raw_path,
            raw_remote_path=args.raw_remote_path,
            data_extra=getattr(args, "data_extra", {}),
            parquet_path=args.parquet_path,
            parquet_remote_path=args.parquet_remote_path,
            device=getattr(args, "device", "auto"),
            sampling_backend=getattr(args, "sampling_backend", "auto"),
            num_neighbors=args.num_neighbors,
            use_cooccurrence=bool(args.use_cooccurrence),
            node_feats_path=args.node_feats_path,
            node_feats_remote_path=args.node_feats_remote_path,
            node_mapping_path=args.node_mapping_path,
            node_mapping_remote_path=args.node_mapping_remote_path,
            upload=bool(getattr(args, "upload", False)),
            remote_dir=getattr(args, "remote_dir", None),
            n_hist_neg=n_hist_neg,
            n_random_neg=n_random_neg,
        )

    total_start = time.time()
    ensure_output_dir(output_dir)
    attach_file_logger(output_dir)

    runtime = describe_runtime(getattr(args, "device", "auto"), amp=not getattr(args, "no_amp", False))
    device = runtime.device

    resolved_data = resolve_stream_graph_dataset(
        type(
            "RunnerDataConfig",
            (),
            {
                "source": getattr(args, "data_source", "stream_graph"),
                "raw_path": getattr(args, "raw_path", None),
                "raw_remote_path": getattr(args, "raw_remote_path", None),
                "parquet_path": args.parquet_path,
                "parquet_remote_path": getattr(args, "parquet_remote_path", None),
                "features_path": args.node_feats_path,
                "features_remote_path": getattr(args, "node_feats_remote_path", None),
                "node_mapping_path": args.node_mapping_path,
                "node_mapping_remote_path": getattr(args, "node_mapping_remote_path", None),
                "download_backend": getattr(args, "data_backend", "yadisk"),
                "cache_dir": getattr(args, "data_cache_dir", None),
                "token_env": getattr(args, "data_token_env", "YADISK_TOKEN"),
                "extra": getattr(args, "data_extra", {}),
            },
        )()
    )
    parquet_path = resolved_data.parquet_path
    node_feats_path = resolved_data.features_path
    node_mapping_path = resolved_data.node_mapping_path

    data_start = time.time()
    data, train_mask, val_mask, test_mask = load_stream_graph_data(
        parquet_path,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        undirected=True,
    )
    eval_infra = _build_eval_infrastructure(parquet_path, args.train_ratio, args.val_ratio)
    data_time = time.time() - data_start

    if node_feats_path:
        from scripts.compute_stream_node_features import load_node_features as _load_nf

        node_feats = _load_nf(node_feats_path, data.num_nodes)
        data.node_feats = node_feats
        if args.node_feat_dim == 0:
            args.node_feat_dim = node_feats.shape[1]

    adj = None
    node_mapping = None
    sampling_backend = getattr(args, "sampling_backend", "auto")
    if args.use_cooccurrence and args.adj_path:
        from scipy import sparse as _sp

        adj = _sp.load_npz(args.adj_path)
        node_mapping = np.load(node_mapping_path)

    use_sampler_backend = sampling_backend != "auto"
    if use_sampler_backend and adj is not None:
        raise ValueError(
            "GLFormer sampler backends do not yet support adjacency-driven cooccurrence "
            "from adj_path/node_mapping_path in the unified runner"
        )

    save_json(os.path.join(output_dir, "data_summary.json"), _build_data_summary(data, train_mask, val_mask, test_mask))

    train_start = time.time()
    training_result = train_glformer_model(
        data=data,
        train_mask=train_mask,
        val_mask=val_mask,
        output_dir=output_dir,
        device=device,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        num_neighbors=args.num_neighbors,
        hidden_dim=args.hidden_dim,
        num_glformer_layers=args.num_glformer_layers,
        channel_expansion=args.channel_expansion,
        dropout=args.dropout,
        patience=args.patience,
        seed=args.seed,
        max_val_edges=args.max_val_edges,
        use_amp=not args.no_amp,
        edge_feat_dim=args.edge_feat_dim,
        node_feat_dim=args.node_feat_dim,
        use_cooccurrence=args.use_cooccurrence,
        cooc_dim=args.cooc_dim,
        adj=adj,
        node_mapping=node_mapping,
        sampling_backend=sampling_backend,
    )
    model = training_result.model
    history = training_result.history
    train_time = time.time() - train_start
    save_training_curves(output_dir, history)

    eval_start = time.time()
    if use_sampler_backend:
        # TODO(REFACTORING): route sampling_backend="auto" through this unified
        # sampler path once parity with the legacy TemporalCSR evaluation flow is verified.
        from src.models.GLFormer_cuda.data_utils import build_cuda_sampler

        train_sampler_for_val = build_cuda_sampler(data, train_mask, backend=sampling_backend)
        val_metrics = evaluate_glformer_model(
            model=model,
            data=data,
            sampler=train_sampler_for_val,
            eval_mask=val_mask,
            device=device,
            num_neighbors=args.num_neighbors,
            n_hist_neg=n_hist_neg,
            n_random_neg=n_random_neg,
            use_amp=not args.no_amp,
            seed=args.seed + 200,
            max_edges=50_000,
        ).metrics
        all_before_test = train_mask | val_mask
        test_sampler = build_cuda_sampler(data, all_before_test, backend="cpp")
        test_metrics = evaluate_glformer_model(
            model=model,
            data=data,
            sampler=test_sampler,
            eval_mask=test_mask,
            device=device,
            num_neighbors=args.num_neighbors,
            n_hist_neg=n_hist_neg,
            n_random_neg=n_random_neg,
            use_amp=not args.no_amp,
            seed=args.seed + 400,
            max_edges=args.max_test_edges if args.max_test_edges else 50_000,
        ).metrics
    else:
        train_csr_for_val = build_temporal_csr(data, train_mask)
        val_metrics = evaluate_glformer_model(
            model=model,
            data=data,
            csr=train_csr_for_val,
            eval_src=eval_infra["val_src"],
            eval_dst=eval_infra["val_dst"],
            eval_ts=eval_infra["val_ts"],
            train_neighbors=eval_infra["train_neighbors"],
            active_nodes=eval_infra["active_nodes"],
            device=device,
            num_neighbors=args.num_neighbors,
            n_hist_neg=n_hist_neg,
            n_random_neg=n_random_neg,
            use_amp=not args.no_amp,
            seed=args.seed + 200,
            max_edges=50_000,
            adj=adj,
            node_mapping=node_mapping,
        ).metrics
        all_before_test = train_mask | val_mask
        test_csr = build_temporal_csr(data, all_before_test)
        test_metrics = evaluate_glformer_model(
            model=model,
            data=data,
            csr=test_csr,
            eval_src=eval_infra["test_src"],
            eval_dst=eval_infra["test_dst"],
            eval_ts=eval_infra["test_ts"],
            train_neighbors=eval_infra["train_neighbors"],
            active_nodes=eval_infra["active_nodes"],
            device=device,
            num_neighbors=args.num_neighbors,
            n_hist_neg=n_hist_neg,
            n_random_neg=n_random_neg,
            use_amp=not args.no_amp,
            seed=args.seed + 400,
            max_edges=args.max_test_edges if args.max_test_edges else 50_000,
            adj=adj,
            node_mapping=node_mapping,
        ).metrics
    eval_time = time.time() - eval_start

    final_results = build_final_results(
        experiment=exp_name,
        model="GLFormerTime",
        history=history,
        timing={
            "data_prep_sec": data_time,
            "training_sec": train_time,
            "evaluation_sec": eval_time,
            "total_sec": time.time() - total_start,
        },
        args=vars(args),
        device_info={
            "device": runtime.resolved_device,
            "requested_device": runtime.requested_device,
            "cuda_available": runtime.cuda_available,
            "amp_enabled": runtime.amp_enabled,
            "gpu_name": runtime.gpu_name,
        },
        extra={
            "parquet_path": parquet_path,
            "sampling_backend": sampling_backend,
            "val_metrics": val_metrics,
            "test_metrics": test_metrics,
        },
    )
    save_json(os.path.join(output_dir, "final_results.json"), final_results)

    if maybe_upload_from_args(output_dir, args, experiment_name=exp_name, logger=logger):
        logger.info("Uploaded results for %s", exp_name)

    final_results["timing"]["total_sec"] = time.time() - total_start

    return final_results


def main(argv: list[str] | None = None) -> int:
    parser = build_glformer_arg_parser()
    args = parser.parse_args(argv)
    run_glformer_experiment(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

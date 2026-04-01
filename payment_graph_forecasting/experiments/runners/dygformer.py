"""DyGFormer runner on top of the new package layout."""

from __future__ import annotations

import argparse
import os
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from payment_graph_forecasting.evaluation.api import evaluate_dygformer_model
from payment_graph_forecasting.experiments.results import (
    build_dry_run_result,
    build_final_results,
)
from payment_graph_forecasting.experiments.runner_utils import (
    attach_file_logger,
    configure_root_logging,
    describe_runtime,
    ensure_output_dir,
    maybe_upload_from_args,
    save_json,
    save_training_curves,
)
from payment_graph_forecasting.infra.datasets import resolve_stream_graph_dataset
from payment_graph_forecasting.training.api import train_dygformer_model
from src.models.stream_graph_data import build_temporal_csr, load_stream_graph_data

logger = configure_root_logging()


def build_dygformer_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="DyGFormer temporal link prediction on stream graphs")
    parser.add_argument("--data-source", type=str, default="stream_graph")
    parser.add_argument("--raw-path", type=str, default=None)
    parser.add_argument("--raw-remote-path", type=str, default=None)
    parser.add_argument("--parquet-path", type=str, default=None)
    parser.add_argument("--parquet-remote-path", type=str, default=None)
    parser.add_argument("--features-path", type=str, default=None)
    parser.add_argument("--features-remote-path", type=str, default=None)
    parser.add_argument("--node-mapping-path", type=str, default=None)
    parser.add_argument("--node-mapping-remote-path", type=str, default=None)
    parser.add_argument("--data-backend", type=str, default="yadisk")
    parser.add_argument("--data-cache-dir", type=str, default=None)
    parser.add_argument("--data-token-env", type=str, default="YADISK_TOKEN")
    parser.add_argument("--sampling-backend", type=str, default="auto")
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--fraction", type=float, default=None)
    parser.add_argument("--output", type=str, default="/tmp/dygformer_results")
    parser.add_argument("--exp-name", type=str, default=None)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--upload", action="store_true")
    parser.add_argument("--upload-backend", type=str, default="yadisk")
    parser.add_argument("--remote-dir", type=str, default=None)
    parser.add_argument("--token-env", type=str, default="YADISK_TOKEN")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--num-neighbors", type=int, default=32)
    parser.add_argument("--patch-size", type=int, default=1)
    parser.add_argument("--time-dim", type=int, default=100)
    parser.add_argument("--aligned-dim", type=int, default=50)
    parser.add_argument("--num-transformer-layers", type=int, default=2)
    parser.add_argument("--num-attention-heads", type=int, default=2)
    parser.add_argument("--cooc-dim", type=int, default=50)
    parser.add_argument("--output-dim", type=int, default=172)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--max-val-edges", type=int, default=5000)
    parser.add_argument("--max-test-edges", type=int, default=None)
    parser.add_argument("--n-hist-neg", type=int, default=50)
    parser.add_argument("--n-random-neg", type=int, default=50)
    parser.add_argument("--neg-per-positive", type=int, default=5)
    parser.add_argument("--edge-feat-dim", type=int, default=2)
    parser.add_argument("--node-feat-dim", type=int, default=0)
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


def _build_eval_infrastructure(
    data,
    train_mask,
    val_mask,
    test_mask,
) -> dict[str, object]:
    train_src = data.src[train_mask]
    train_dst = data.dst[train_mask]
    train_neighbors: dict[int, set[int]] = defaultdict(set)
    for src_node, dst_node in zip(train_src, train_dst):
        train_neighbors[int(src_node)].add(int(dst_node))

    active_nodes = np.unique(
        np.concatenate([train_src, train_dst])
    ).astype(np.int64)

    return {
        "train_neighbors": dict(train_neighbors),
        "active_nodes": active_nodes,
        "val_src": data.src[val_mask].astype(np.int32, copy=False),
        "val_dst": data.dst[val_mask].astype(np.int32, copy=False),
        "val_ts": data.timestamps[val_mask].astype(np.float64, copy=False),
        "test_src": data.src[test_mask].astype(np.int32, copy=False),
        "test_dst": data.dst[test_mask].astype(np.int32, copy=False),
        "test_ts": data.timestamps[test_mask].astype(np.float64, copy=False),
    }


def run_dygformer_experiment(args: argparse.Namespace):
    parquet_ref = args.parquet_path or args.parquet_remote_path
    if parquet_ref is None and args.dry_run:
        parquet_ref = "unspecified"
    if parquet_ref is None:
        raise ValueError("DyGFormer requires either --parquet-path or --parquet-remote-path")

    parquet_name = Path(parquet_ref).stem
    exp_name = getattr(args, "exp_name", None) or f"dygformer_{parquet_name}"
    output_dir = os.path.join(args.output, exp_name)

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
            features_path=args.features_path,
            features_remote_path=args.features_remote_path,
            node_mapping_path=args.node_mapping_path,
            node_mapping_remote_path=args.node_mapping_remote_path,
            fraction=args.fraction,
            sampling_backend=getattr(args, "sampling_backend", "auto"),
            num_neighbors=args.num_neighbors,
            patch_size=args.patch_size,
            n_hist_neg=args.n_hist_neg,
            n_random_neg=args.n_random_neg,
            neg_per_positive=args.neg_per_positive,
            device=getattr(args, "device", "auto"),
            upload=bool(getattr(args, "upload", False)),
            remote_dir=getattr(args, "remote_dir", None),
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
                "features_path": args.features_path,
                "features_remote_path": getattr(args, "features_remote_path", None),
                "node_mapping_path": args.node_mapping_path,
                "node_mapping_remote_path": getattr(args, "node_mapping_remote_path", None),
                "download_backend": getattr(args, "data_backend", "yadisk"),
                "cache_dir": getattr(args, "data_cache_dir", None),
                "token_env": getattr(args, "data_token_env", "YADISK_TOKEN"),
                "fraction": getattr(args, "fraction", None),
                "extra": getattr(args, "data_extra", {}),
            },
        )()
    )
    parquet_path = resolved_data.parquet_path
    features_path = resolved_data.features_path

    data_start = time.time()
    data, train_mask, val_mask, test_mask = load_stream_graph_data(
        parquet_path,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        undirected=True,
        fraction=args.fraction,
        features_path=features_path,
    )
    eval_infra = _build_eval_infrastructure(data, train_mask, val_mask, test_mask)
    data_time = time.time() - data_start
    save_json(os.path.join(output_dir, "data_summary.json"), _build_data_summary(data, train_mask, val_mask, test_mask))

    if features_path and args.node_feat_dim == 0:
        args.node_feat_dim = int(data.node_feats.shape[1])

    train_start = time.time()
    training_result = train_dygformer_model(
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
        patch_size=args.patch_size,
        time_dim=args.time_dim,
        aligned_dim=args.aligned_dim,
        num_transformer_layers=args.num_transformer_layers,
        num_attention_heads=args.num_attention_heads,
        cooc_dim=args.cooc_dim,
        output_dim=args.output_dim,
        dropout=getattr(args, "dropout", 0.1),
        patience=args.patience,
        seed=args.seed,
        max_val_edges=args.max_val_edges,
        use_amp=not args.no_amp,
        edge_feat_dim=args.edge_feat_dim,
        node_feat_dim=args.node_feat_dim,
        neg_per_positive=args.neg_per_positive,
        sampling_backend=getattr(args, "sampling_backend", "auto"),
    )
    model = training_result.model
    history = training_result.history
    train_time = time.time() - train_start
    save_training_curves(output_dir, history)

    eval_start = time.time()
    val_csr = build_temporal_csr(data, train_mask)
    test_csr = build_temporal_csr(data, train_mask | val_mask)
    val_metrics = evaluate_dygformer_model(
        model=model,
        data=data,
        csr=val_csr,
        eval_src=eval_infra["val_src"],
        eval_dst=eval_infra["val_dst"],
        eval_ts=eval_infra["val_ts"],
        train_neighbors=eval_infra["train_neighbors"],
        active_nodes=eval_infra["active_nodes"],
        device=device,
        num_neighbors=args.num_neighbors,
        n_hist_neg=args.n_hist_neg,
        n_random_neg=args.n_random_neg,
        use_amp=not args.no_amp,
        seed=args.seed + 200,
        max_edges=args.max_val_edges,
    ).metrics
    test_metrics = evaluate_dygformer_model(
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
        n_hist_neg=args.n_hist_neg,
        n_random_neg=args.n_random_neg,
        use_amp=not args.no_amp,
        seed=args.seed + 400,
        max_edges=args.max_test_edges if args.max_test_edges is not None else 50_000,
    ).metrics
    eval_time = time.time() - eval_start

    final_results = build_final_results(
        experiment=exp_name,
        model="DyGFormer",
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
            "features_path": features_path,
            "val_metrics": val_metrics,
            "test_metrics": test_metrics,
        },
    )
    save_json(os.path.join(output_dir, "final_results.json"), final_results)

    if maybe_upload_from_args(output_dir, args, experiment_name=exp_name, logger=logger):
        logger.info("Uploaded results for %s", exp_name)

    return final_results


def main(argv: list[str] | None = None) -> int:
    parser = build_dygformer_arg_parser()
    args = parser.parse_args(argv)
    run_dygformer_experiment(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

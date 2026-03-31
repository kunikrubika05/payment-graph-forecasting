"""HyperEvent runner on top of the new package layout."""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

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
from payment_graph_forecasting.evaluation.api import evaluate_hyperevent_model
from payment_graph_forecasting.infra.datasets import resolve_stream_graph_dataset
from payment_graph_forecasting.training.api import train_hyperevent_model
from src.models.HyperEvent.data_utils import load_stream_graph_data

logger = configure_root_logging()


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


def build_hyperevent_arg_parser() -> argparse.ArgumentParser:
    """Build the HyperEvent CLI parser."""

    parser = argparse.ArgumentParser(description="HyperEvent temporal link prediction on stream graphs")
    parser.add_argument("--data-source", type=str, default="stream_graph")
    parser.add_argument("--raw-path", type=str, default=None)
    parser.add_argument("--raw-remote-path", type=str, default=None)
    parser.add_argument("--parquet-path", type=str, default=None)
    parser.add_argument("--parquet-remote-path", type=str, default=None)
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--output", type=str, default="/tmp/hyperevent_results")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--data-backend", type=str, default="yadisk")
    parser.add_argument("--data-cache-dir", type=str, default=None)
    parser.add_argument("--data-token-env", type=str, default="YADISK_TOKEN")
    parser.add_argument("--upload", action="store_true")
    parser.add_argument("--upload-backend", type=str, default="yadisk")
    parser.add_argument("--remote-dir", type=str, default=None)
    parser.add_argument("--token-env", type=str, default="YADISK_TOKEN")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=200)
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--n-neighbor", type=int, default=20)
    parser.add_argument("--n-latest", type=int, default=10)
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--n-layers", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--max-val-edges", type=int, default=5000)
    parser.add_argument("--n-hist-neg", type=int, default=50)
    parser.add_argument("--n-random-neg", type=int, default=50)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def run_hyperevent_experiment(args: argparse.Namespace):
    """Run a full HyperEvent experiment using the new package runner."""

    parquet_ref = args.parquet_path or args.parquet_remote_path
    if parquet_ref is None:
        raise ValueError("HyperEvent requires either --parquet-path or --parquet-remote-path")
    parquet_name = Path(parquet_ref).stem
    exp_name = f"hyperevent_{parquet_name}"
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
            upload=bool(getattr(args, "upload", False)),
            remote_dir=getattr(args, "remote_dir", None),
            n_neighbor=args.n_neighbor,
            n_latest=args.n_latest,
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
                "features_path": None,
                "features_remote_path": None,
                "node_mapping_path": None,
                "node_mapping_remote_path": None,
                "download_backend": getattr(args, "data_backend", "yadisk"),
                "cache_dir": getattr(args, "data_cache_dir", None),
                "token_env": getattr(args, "data_token_env", "YADISK_TOKEN"),
                "extra": getattr(args, "data_extra", {}),
            },
        )()
    )
    parquet_path = resolved_data.parquet_path

    data_start = time.time()
    data, train_mask, val_mask, test_mask = load_stream_graph_data(
        parquet_path,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        undirected=True,
    )
    data_time = time.time() - data_start
    save_json(os.path.join(output_dir, "data_summary.json"), _build_data_summary(data, train_mask, val_mask, test_mask))

    train_start = time.time()
    training_result = train_hyperevent_model(
        data=data,
        train_mask=train_mask,
        val_mask=val_mask,
        output_dir=output_dir,
        device=device,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        n_neighbor=args.n_neighbor,
        n_latest=args.n_latest,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        dropout=args.dropout,
        patience=args.patience,
        seed=args.seed,
        max_val_edges=args.max_val_edges,
        use_amp=not args.no_amp,
    )
    model = training_result.model
    history = training_result.history
    train_time = time.time() - train_start
    save_training_curves(output_dir, history)

    eval_start = time.time()
    history_mask = train_mask | val_mask
    test_metrics = evaluate_hyperevent_model(
        model=model,
        data=data,
        eval_mask=test_mask,
        history_mask=history_mask,
        device=device,
        n_neighbor=args.n_neighbor,
        n_latest=args.n_latest,
        n_hist_neg=n_hist_neg,
        n_random_neg=n_random_neg,
        use_amp=not args.no_amp,
        seed=args.seed,
    ).metrics
    eval_time = time.time() - eval_start

    final_results = build_final_results(
        experiment=exp_name,
        model="HyperEvent",
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
            "test_metrics": test_metrics,
        },
    )
    save_json(os.path.join(output_dir, "final_results.json"), final_results)

    if maybe_upload_from_args(output_dir, args, experiment_name=exp_name, logger=logger):
        logger.info("Uploaded results for %s", exp_name)

    return final_results


def main(argv: list[str] | None = None) -> int:
    parser = build_hyperevent_arg_parser()
    args = parser.parse_args(argv)
    run_hyperevent_experiment(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

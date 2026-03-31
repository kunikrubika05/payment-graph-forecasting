"""sg-baselines-aligned GraphMixer runner on top of the new package layout."""

from __future__ import annotations

import argparse
import os
import time

import numpy as np

from payment_graph_forecasting.experiments.runner_utils import (
    attach_file_logger,
    configure_root_logging,
    describe_runtime,
    ensure_output_dir,
    maybe_upload_from_args,
    save_json,
)
from payment_graph_forecasting.experiments.results import (
    build_dry_run_result,
    build_final_results,
)
from payment_graph_forecasting.evaluation.api import evaluate_sg_graphmixer_model
from payment_graph_forecasting.training.api import train_sg_graphmixer_model
from sg_baselines.config import ExperimentConfig, PERIODS
from sg_baselines.data import (
    build_train_neighbor_sets,
    load_adjacency,
    load_node_features_sparse,
    load_stream_graph,
    split_stream_graph,
)
from src.models.data_utils import TemporalCSR
from src.models.sg_graphmixer.data_utils import build_stream_graph_data

logger = configure_root_logging()


def build_sg_graphmixer_arg_parser() -> argparse.ArgumentParser:
    """Build the sg-GraphMixer CLI parser."""

    parser = argparse.ArgumentParser(description="GraphMixer on sg_baselines stream graph")
    parser.add_argument("--period", type=str, default="period_10", choices=list(PERIODS.keys()))
    parser.add_argument("--output", type=str, default="/tmp/sg_graphmixer_results")
    parser.add_argument("--data-dir", type=str, default="/tmp/sg_baselines_data")
    parser.add_argument("--upload", action="store_true")
    parser.add_argument("--upload-backend", type=str, default="yadisk")
    parser.add_argument("--remote-dir", type=str, default=None)
    parser.add_argument("--token-env", type=str, default="YADISK_TOKEN")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--hidden-dim", type=int, default=200)
    parser.add_argument("--num-neighbors", type=int, default=30)
    parser.add_argument("--num-mixer-layers", type=int, default=1)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=3e-6)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--batch-size", type=int, default=4000)
    parser.add_argument("--num-epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--max-val-queries", type=int, default=10_000)
    parser.add_argument("--max-test-queries", type=int, default=50_000)
    parser.add_argument("--n-negatives", type=int, default=100)
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def run_sg_graphmixer_experiment(args: argparse.Namespace):
    """Run the sg-baselines-aligned GraphMixer experiment."""

    period_cfg = PERIODS[args.period]
    config = ExperimentConfig(
        period_name=args.period,
        fraction=float(period_cfg["fraction"]),
        label=str(period_cfg["label"]),
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        local_data_dir=args.data_dir,
        output_dir=args.output,
        upload=args.upload,
        random_seed=args.seed,
        n_negatives=args.n_negatives,
    )
    exp_name = f"exp_sg_graphmixer_{config.label}"
    output_dir = os.path.join(args.output, exp_name)

    if args.dry_run:
        ensure_output_dir(output_dir)
        return build_dry_run_result(
            experiment=exp_name,
            output_dir=output_dir,
            period=args.period,
            fraction=config.fraction,
            device=getattr(args, "device", "auto"),
            upload=bool(getattr(args, "upload", False)),
            remote_dir=getattr(args, "remote_dir", None),
            n_negatives=args.n_negatives,
            max_val_queries=args.max_val_queries,
            max_test_queries=args.max_test_queries,
        )

    total_start = time.time()
    ensure_output_dir(output_dir)
    attach_file_logger(output_dir)

    runtime = describe_runtime(getattr(args, "device", "auto"), amp=not getattr(args, "no_amp", False))
    device = runtime.device
    data_start = time.time()
    token = os.environ.get(getattr(args, "token_env", "YADISK_TOKEN"), "")
    df = load_stream_graph(config, token)
    train_edges, val_edges, test_edges = split_stream_graph(df, config)
    del df

    node_idx, features_df = load_node_features_sparse(config, token)
    node_features = features_df.values.astype("float32")
    node_mapping, _adj_directed, _adj_undirected = load_adjacency(config, token)
    train_neighbors = build_train_neighbor_sets(train_edges)
    active_nodes = node_mapping
    data, train_mask, val_mask, test_mask = build_stream_graph_data(
        train_edges,
        val_edges,
        test_edges,
        node_mapping,
        node_features,
        undirected=True,
    )
    data_time = time.time() - data_start

    save_json(
        os.path.join(output_dir, "data_summary.json"),
        {
            "num_nodes": int(data.num_nodes),
            "num_edges": int(data.num_edges),
            "train_edges": int(train_mask.sum()),
            "val_edges": int(val_mask.sum()),
            "test_edges": int(test_mask.sum()),
            "timestamp_min": float(data.timestamps.min()),
            "timestamp_max": float(data.timestamps.max()),
            "active_nodes": int(len(active_nodes)),
            "fraction": float(config.fraction),
        },
    )

    train_start = time.time()
    training_result = train_sg_graphmixer_model(
        data=data,
        train_mask=train_mask,
        val_mask=val_mask,
        train_neighbors=train_neighbors,
        active_nodes=active_nodes,
        node_mapping=node_mapping,
        output_dir=output_dir,
        device=device,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        num_neighbors=args.num_neighbors,
        hidden_dim=args.hidden_dim,
        num_mixer_layers=args.num_mixer_layers,
        dropout=args.dropout,
        patience=args.patience,
        seed=args.seed,
        max_val_queries=args.max_val_queries,
        n_negatives=args.n_negatives,
    )
    model = training_result.model
    history = training_result.history
    train_time = time.time() - train_start

    train_csr = TemporalCSR(
        data.num_nodes,
        data.src[train_mask],
        data.dst[train_mask],
        data.timestamps[train_mask],
        np.where(train_mask)[0].astype(np.int64),
    )

    eval_start = time.time()
    val_metrics = evaluate_sg_graphmixer_model(
        model=model,
        data=data,
        csr=train_csr,
        eval_mask=val_mask,
        device=device,
        num_neighbors=args.num_neighbors,
        train_neighbors=train_neighbors,
        active_nodes=active_nodes,
        node_mapping=node_mapping,
        n_negatives=args.n_negatives,
        max_queries=args.max_test_queries,
        seed=args.seed + 300,
    ).metrics
    test_metrics = evaluate_sg_graphmixer_model(
        model=model,
        data=data,
        csr=train_csr,
        eval_mask=test_mask,
        device=device,
        num_neighbors=args.num_neighbors,
        train_neighbors=train_neighbors,
        active_nodes=active_nodes,
        node_mapping=node_mapping,
        n_negatives=args.n_negatives,
        max_queries=args.max_test_queries,
        seed=args.seed + 400,
    ).metrics
    eval_time = time.time() - eval_start

    final_results = build_final_results(
        experiment=exp_name,
        model="SG-GraphMixer",
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
            "period": args.period,
            "fraction": config.fraction,
            "val_metrics": val_metrics,
            "test_metrics": test_metrics,
            "active_nodes": int(len(active_nodes)),
        },
    )
    save_json(os.path.join(output_dir, "final_results.json"), final_results)

    if maybe_upload_from_args(output_dir, args, experiment_name=exp_name, logger=logger):
        logger.info("Uploaded results for %s", exp_name)

    return final_results


def main(argv: list[str] | None = None) -> int:
    parser = build_sg_graphmixer_arg_parser()
    args = parser.parse_args(argv)
    run_sg_graphmixer_experiment(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

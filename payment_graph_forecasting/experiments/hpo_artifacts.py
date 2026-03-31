"""Helpers for package-facing HPO output artifacts."""

from __future__ import annotations

from argparse import Namespace
from pathlib import Path
from typing import Any, Mapping

from payment_graph_forecasting.config.base import (
    DataConfig,
    ExperimentMetadata,
    ExperimentSpec,
    RuntimeConfig,
    SamplingConfig,
    TrainingConfig,
    UploadConfig,
)
from payment_graph_forecasting.config.yaml_io import save_experiment_spec


def _parquet_stem(path: str) -> str:
    return Path(path).stem


def _runtime_config(*, output_dir: str, use_amp: bool) -> RuntimeConfig:
    return RuntimeConfig(
        device="auto",
        amp=use_amp,
        dry_run=False,
        output_dir=output_dir,
    )


def _base_upload_config() -> UploadConfig:
    return UploadConfig(enabled=False, backend="yadisk", remote_dir=None, token_env="YADISK_TOKEN")


def build_best_hpo_spec(
    model_name: str,
    args: Namespace,
    best_params: Mapping[str, Any],
) -> ExperimentSpec:
    """Build a package-facing training spec from legacy HPO outputs."""

    parquet_path = str(args.parquet_path)
    experiment_name = f"{model_name}_{_parquet_stem(parquet_path)}_hpo_best"
    amp_enabled = not bool(getattr(args, "no_amp", False))

    if model_name == "eagle":
        return ExperimentSpec(
            experiment=ExperimentMetadata(name=experiment_name, model="eagle", tags=["hpo", "generated"]),
            data=DataConfig(
                source="stream_graph",
                parquet_path=parquet_path,
                features_path=getattr(args, "features_path", None),
                node_mapping_path=getattr(args, "node_mapping_path", None),
                fraction=getattr(args, "fraction", None),
                train_ratio=float(getattr(args, "train_ratio", 0.7)),
                val_ratio=float(getattr(args, "val_ratio", 0.15)),
                undirected=True,
            ),
            sampling=SamplingConfig(
                strategy="tgb_mixed",
                num_neighbors=int(best_params["num_neighbors"]),
                n_random_neg=50,
                n_hist_neg=50,
                backend="auto",
            ),
            training=TrainingConfig(
                epochs=100,
                batch_size=int(best_params["batch_size"]),
                lr=float(best_params["lr"]),
                patience=10,
                seed=int(getattr(args, "seed", 42)),
                weight_decay=float(best_params["weight_decay"]),
                hidden_dim=int(best_params["hidden_dim"]),
                dropout=float(best_params["dropout"]),
            ),
            runtime=_runtime_config(output_dir="/tmp/eagle_results", use_amp=amp_enabled),
            upload=_base_upload_config(),
            model={
                "num_mixer_layers": int(best_params["num_mixer_layers"]),
                "token_expansion": float(best_params["token_expansion"]),
                "channel_expansion": float(best_params["channel_expansion"]),
                "edge_feat_dim": int(getattr(args, "edge_feat_dim", 0)),
                "node_feat_dim": int(getattr(args, "node_feat_dim", 0)),
                "max_val_edges": int(getattr(args, "max_val_edges", 3000)),
                "max_test_edges": 50_000,
            },
        )

    if model_name == "glformer":
        return ExperimentSpec(
            experiment=ExperimentMetadata(name=experiment_name, model="glformer", tags=["hpo", "generated"]),
            data=DataConfig(
                source="stream_graph",
                parquet_path=parquet_path,
                features_path=getattr(args, "node_feats_path", None),
                train_ratio=float(getattr(args, "train_ratio", 0.7)),
                val_ratio=float(getattr(args, "val_ratio", 0.15)),
                undirected=True,
            ),
            sampling=SamplingConfig(
                strategy="tgb_mixed",
                num_neighbors=20,
                n_random_neg=50,
                n_hist_neg=50,
                backend="auto",
            ),
            training=TrainingConfig(
                epochs=100,
                batch_size=4000,
                lr=0.0001,
                patience=20,
                seed=int(getattr(args, "seed", 42)),
                weight_decay=1e-5,
                hidden_dim=int(best_params["hidden_dim"]),
                dropout=0.1,
            ),
            runtime=_runtime_config(output_dir="/tmp/glformer_results", use_amp=amp_enabled),
            upload=_base_upload_config(),
            model={
                "num_glformer_layers": int(best_params["num_glformer_layers"]),
                "channel_expansion": 4.0,
                "edge_feat_dim": int(getattr(args, "edge_feat_dim", 2)),
                "node_feat_dim": int(getattr(args, "node_feat_dim", 0)),
                "use_cooccurrence": bool(getattr(args, "use_cooccurrence", False)),
                "cooc_dim": int(getattr(args, "cooc_dim", 16)),
                "max_val_edges": int(getattr(args, "max_val_edges", 3000)),
            },
        )

    if model_name == "hyperevent":
        return ExperimentSpec(
            experiment=ExperimentMetadata(name=experiment_name, model="hyperevent", tags=["hpo", "generated"]),
            data=DataConfig(
                source="stream_graph",
                parquet_path=parquet_path,
                train_ratio=float(getattr(args, "train_ratio", 0.7)),
                val_ratio=float(getattr(args, "val_ratio", 0.15)),
                undirected=True,
            ),
            sampling=SamplingConfig(
                strategy="tgb_mixed",
                num_neighbors=int(best_params["n_neighbor"]),
                n_random_neg=50,
                n_hist_neg=50,
                backend="auto",
            ),
            training=TrainingConfig(
                epochs=50,
                batch_size=int(best_params["batch_size"]),
                lr=float(best_params["lr"]),
                patience=20,
                seed=int(getattr(args, "seed", 42)),
                weight_decay=float(best_params["weight_decay"]),
                hidden_dim=int(best_params["d_model"]),
                dropout=float(best_params["dropout"]),
            ),
            runtime=_runtime_config(output_dir="/tmp/hyperevent_results", use_amp=amp_enabled),
            upload=_base_upload_config(),
            model={
                "n_neighbor": int(best_params["n_neighbor"]),
                "n_latest": int(best_params["n_latest"]),
                "d_model": int(best_params["d_model"]),
                "n_heads": int(best_params["n_heads"]),
                "n_layers": int(best_params["n_layers"]),
                "max_val_edges": int(getattr(args, "max_val_edges", 3000)),
            },
        )

    raise KeyError(f"Unsupported package HPO artifact model: {model_name}")


def write_best_training_artifacts(
    model_name: str,
    args: Namespace,
    best_params: Mapping[str, Any],
    *,
    output_dir: str | Path,
) -> dict[str, str]:
    """Write package-facing best-spec and command artifacts for an HPO run."""

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    spec = build_best_hpo_spec(model_name, args, best_params)
    spec_path = save_experiment_spec(spec, output_path / "best_experiment.yaml")
    command = (
        f"PYTHONPATH=. python -m payment_graph_forecasting.experiments.launcher --config {spec_path}"
    )
    command_path = output_path / "best_train_command.sh"
    command_path.write_text("#!/bin/bash\n" + command + "\n")
    command_path.chmod(0o755)
    return {
        "spec_path": str(spec_path),
        "command_path": str(command_path),
        "command": command,
    }

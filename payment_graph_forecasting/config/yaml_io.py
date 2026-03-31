"""YAML loading helpers with a small built-in fallback parser.

The project should normally use ``PyYAML``. The fallback parser only supports
the subset used by our experiment specs and test fixtures.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from payment_graph_forecasting.config.base import (
    DataConfig,
    ExperimentMetadata,
    ExperimentSpec,
    RuntimeConfig,
    SamplingConfig,
    TrainingConfig,
    UploadConfig,
)


def _parse_scalar(raw: str) -> Any:
    value = raw.strip()
    if value == "":
        return {}
    if value in {"true", "True"}:
        return True
    if value in {"false", "False"}:
        return False
    if value in {"null", "None", "~"}:
        return None
    if value.startswith('"') and value.endswith('"'):
        return value[1:-1]
    if value.startswith("'") and value.endswith("'"):
        return value[1:-1]
    if value.startswith("[") and value.endswith("]"):
        inner = value[1:-1].strip()
        if not inner:
            return []
        return [_parse_scalar(part.strip()) for part in inner.split(",")]
    try:
        if any(ch in value for ch in (".", "e", "E")):
            return float(value)
        return int(value)
    except ValueError:
        return value


def _simple_yaml_load(text: str) -> dict[str, Any]:
    root: dict[str, Any] = {}
    stack: list[tuple[int, dict[str, Any]]] = [(-1, root)]

    for raw_line in text.splitlines():
        if not raw_line.strip() or raw_line.lstrip().startswith("#"):
            continue
        indent = len(raw_line) - len(raw_line.lstrip(" "))
        line = raw_line.strip()
        if ":" not in line:
            raise ValueError(f"Unsupported YAML line: {raw_line!r}")

        key, raw_value = line.split(":", 1)
        key = key.strip()
        value = raw_value.strip()

        while indent <= stack[-1][0]:
            stack.pop()
        current = stack[-1][1]

        if value == "":
            new_dict: dict[str, Any] = {}
            current[key] = new_dict
            stack.append((indent, new_dict))
        else:
            current[key] = _parse_scalar(value)

    return root


def _load_yaml_dict(text: str) -> dict[str, Any]:
    try:
        import yaml  # type: ignore
    except ModuleNotFoundError:
        return _simple_yaml_load(text)
    loaded = yaml.safe_load(text)
    return loaded or {}


def _split_known_fields(raw: dict[str, Any], known: set[str]) -> tuple[dict[str, Any], dict[str, Any]]:
    known_fields = {key: raw[key] for key in known if key in raw}
    extra_fields = {key: value for key, value in raw.items() if key not in known}
    return known_fields, extra_fields


def load_experiment_spec(path: str | Path) -> ExperimentSpec:
    """Load an experiment YAML file into typed config objects."""

    raw = _load_yaml_dict(Path(path).read_text())

    exp_raw = raw.get("experiment", {})
    exp_known, exp_extra = _split_known_fields(exp_raw, {"name", "model", "tags"})
    experiment = ExperimentMetadata(
        name=exp_known["name"],
        model=exp_known["model"],
        tags=list(exp_known.get("tags", [])),
        extra=exp_extra,
    )

    data_raw = raw.get("data", {})
    data_known, data_extra = _split_known_fields(
        data_raw,
        {
            "source",
            "raw_path",
            "raw_remote_path",
            "parquet_path",
            "parquet_remote_path",
            "features_path",
            "features_remote_path",
            "node_mapping_path",
            "node_mapping_remote_path",
            "period",
            "fraction",
            "train_ratio",
            "val_ratio",
            "window",
            "undirected",
            "data_dir",
            "download_backend",
            "cache_dir",
            "token_env",
        },
    )
    data = DataConfig(**data_known, extra=data_extra)

    sampling_raw = raw.get("sampling", {})
    sampling_known, sampling_extra = _split_known_fields(
        sampling_raw,
        {"strategy", "num_neighbors", "n_random_neg", "n_hist_neg", "backend"},
    )
    sampling = SamplingConfig(**sampling_known, extra=sampling_extra)

    training_raw = raw.get("training", {})
    training_known, training_extra = _split_known_fields(
        training_raw,
        {"epochs", "batch_size", "lr", "patience", "seed", "weight_decay", "hidden_dim", "dropout"},
    )
    training = TrainingConfig(**training_known, extra=training_extra)

    runtime_raw = raw.get("runtime", {})
    runtime_known, runtime_extra = _split_known_fields(
        runtime_raw,
        {"device", "amp", "dry_run", "output_dir"},
    )
    runtime = RuntimeConfig(**runtime_known, extra=runtime_extra)

    upload_raw = raw.get("upload", {})
    upload_known, upload_extra = _split_known_fields(
        upload_raw,
        {"enabled", "backend", "remote_dir", "token_env"},
    )
    upload = UploadConfig(**upload_known, extra=upload_extra)

    model_raw = raw.get("model", {})

    return ExperimentSpec(
        experiment=experiment,
        data=data,
        sampling=sampling,
        training=training,
        runtime=runtime,
        upload=upload,
        model=model_raw,
    )


def _spec_to_dict(spec: ExperimentSpec) -> dict[str, Any]:
    return {
        "experiment": {
            "name": spec.experiment.name,
            "model": spec.experiment.model,
            "tags": list(spec.experiment.tags),
            **spec.experiment.extra,
        },
        "data": {
            "source": spec.data.source,
            "raw_path": spec.data.raw_path,
            "raw_remote_path": spec.data.raw_remote_path,
            "parquet_path": spec.data.parquet_path,
            "parquet_remote_path": spec.data.parquet_remote_path,
            "features_path": spec.data.features_path,
            "features_remote_path": spec.data.features_remote_path,
            "node_mapping_path": spec.data.node_mapping_path,
            "node_mapping_remote_path": spec.data.node_mapping_remote_path,
            "period": spec.data.period,
            "fraction": spec.data.fraction,
            "train_ratio": spec.data.train_ratio,
            "val_ratio": spec.data.val_ratio,
            "window": spec.data.window,
            "undirected": spec.data.undirected,
            "data_dir": spec.data.data_dir,
            "download_backend": spec.data.download_backend,
            "cache_dir": spec.data.cache_dir,
            "token_env": spec.data.token_env,
            **spec.data.extra,
        },
        "sampling": {
            "strategy": spec.sampling.strategy,
            "num_neighbors": spec.sampling.num_neighbors,
            "n_random_neg": spec.sampling.n_random_neg,
            "n_hist_neg": spec.sampling.n_hist_neg,
            "backend": spec.sampling.backend,
            **spec.sampling.extra,
        },
        "training": {
            "epochs": spec.training.epochs,
            "batch_size": spec.training.batch_size,
            "lr": spec.training.lr,
            "patience": spec.training.patience,
            "seed": spec.training.seed,
            "weight_decay": spec.training.weight_decay,
            "hidden_dim": spec.training.hidden_dim,
            "dropout": spec.training.dropout,
            **spec.training.extra,
        },
        "runtime": {
            "device": spec.runtime.device,
            "amp": spec.runtime.amp,
            "dry_run": spec.runtime.dry_run,
            "output_dir": spec.runtime.output_dir,
            **spec.runtime.extra,
        },
        "upload": {
            "enabled": spec.upload.enabled,
            "backend": spec.upload.backend,
            "remote_dir": spec.upload.remote_dir,
            "token_env": spec.upload.token_env,
            **spec.upload.extra,
        },
        "model": dict(spec.model),
    }


def _yaml_scalar(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, str):
        if value == "" or any(ch in value for ch in [":", "#", "[", "]", "{", "}", ","]) or value != value.strip():
            escaped = value.replace("\\", "\\\\").replace('"', '\\"')
            return f'"{escaped}"'
        return value
    if isinstance(value, list):
        return "[" + ", ".join(_yaml_scalar(item) for item in value) + "]"
    return str(value)


def _dump_yaml_lines(data: dict[str, Any], *, indent: int = 0) -> list[str]:
    lines: list[str] = []
    prefix = " " * indent
    for key, value in data.items():
        if isinstance(value, dict):
            lines.append(f"{prefix}{key}:")
            lines.extend(_dump_yaml_lines(value, indent=indent + 2))
        else:
            lines.append(f"{prefix}{key}: {_yaml_scalar(value)}")
    return lines


def dump_experiment_spec(spec: ExperimentSpec) -> str:
    """Serialize an experiment spec into the YAML subset used by the repo."""

    return "\n".join(_dump_yaml_lines(_spec_to_dict(spec))) + "\n"


def save_experiment_spec(spec: ExperimentSpec, path: str | Path) -> Path:
    """Write an experiment spec YAML file to disk."""

    output_path = Path(path)
    output_path.write_text(dump_experiment_spec(spec))
    return output_path

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
            "parquet_path",
            "features_path",
            "node_mapping_path",
            "period",
            "fraction",
            "train_ratio",
            "val_ratio",
            "window",
            "undirected",
            "data_dir",
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

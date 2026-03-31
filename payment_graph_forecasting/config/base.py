"""Typed configuration objects for YAML experiment specs."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class ExperimentMetadata:
    """Top-level experiment metadata."""

    name: str
    model: str
    tags: list[str] = field(default_factory=list)
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class DataConfig:
    """Common data-related options shared across model families."""

    source: str = "stream_graph"
    raw_path: str | None = None
    raw_remote_path: str | None = None
    parquet_path: str | None = None
    parquet_remote_path: str | None = None
    features_path: str | None = None
    features_remote_path: str | None = None
    node_mapping_path: str | None = None
    node_mapping_remote_path: str | None = None
    period: str | None = None
    fraction: float | None = None
    train_ratio: float = 0.70
    val_ratio: float = 0.15
    window: int | None = None
    undirected: bool = True
    data_dir: str | None = None
    download_backend: str = "yadisk"
    cache_dir: str | None = None
    token_env: str = "YADISK_TOKEN"
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class SamplingConfig:
    """Unified negative sampling and neighborhood lookup configuration."""

    strategy: str = "tgb_mixed"
    num_neighbors: int = 20
    n_random_neg: int = 50
    n_hist_neg: int = 50
    backend: str = "auto"
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class TrainingConfig:
    """Core training parameters with model-specific extensions in ``extra``."""

    epochs: int = 100
    batch_size: int = 256
    lr: float = 1e-3
    patience: int = 20
    seed: int = 42
    weight_decay: float = 0.0
    hidden_dim: int | None = None
    dropout: float | None = None
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class RuntimeConfig:
    """Execution environment options."""

    device: str = "auto"
    amp: bool = True
    dry_run: bool = False
    output_dir: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class UploadConfig:
    """Artifact upload options."""

    enabled: bool = False
    backend: str = "yadisk"
    remote_dir: str | None = None
    token_env: str = "YADISK_TOKEN"
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ExperimentSpec:
    """Full typed experiment specification."""

    experiment: ExperimentMetadata
    data: DataConfig = field(default_factory=DataConfig)
    sampling: SamplingConfig = field(default_factory=SamplingConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    upload: UploadConfig = field(default_factory=UploadConfig)
    model: dict[str, Any] = field(default_factory=dict)

    @property
    def model_name(self) -> str:
        return self.experiment.model

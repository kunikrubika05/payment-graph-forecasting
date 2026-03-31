"""Dataset resolution helpers for library-facing experiment runners."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from payment_graph_forecasting.config.base import DataConfig
from payment_graph_forecasting.infra.data_access import (
    DEFAULT_DATA_CACHE_DIR,
    resolve_data_path,
)
from scripts.convert_jodie_csv_to_stream_graph import convert_jodie_csv_to_stream_graph


SUPPORTED_STREAM_GRAPH_SOURCES = {
    "stream_graph",
    "orbitaal_stream_graph",
    "jodie_csv",
    "jodie_wikipedia",
}


@dataclass(slots=True)
class ResolvedStreamGraphDataset:
    """Resolved local inputs for a stream-graph-style training run."""

    source: str
    parquet_path: str
    features_path: str | None = None
    node_mapping_path: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


def _resolve_standard_stream_graph(data: DataConfig) -> ResolvedStreamGraphDataset:
    parquet_path = resolve_data_path(
        data.parquet_path,
        remote_path=data.parquet_remote_path,
        backend=data.download_backend,
        cache_dir=data.cache_dir,
        token_env=data.token_env,
        artifact_name="stream graph parquet",
    )
    if parquet_path is None:
        raise ValueError("stream_graph source requires parquet_path or parquet_remote_path")

    return ResolvedStreamGraphDataset(
        source=data.source,
        parquet_path=parquet_path,
        features_path=resolve_data_path(
            data.features_path,
            remote_path=data.features_remote_path,
            backend=data.download_backend,
            cache_dir=data.cache_dir,
            token_env=data.token_env,
            artifact_name="node features parquet",
        ),
        node_mapping_path=resolve_data_path(
            data.node_mapping_path,
            remote_path=data.node_mapping_remote_path,
            backend=data.download_backend,
            cache_dir=data.cache_dir,
            token_env=data.token_env,
            artifact_name="node mapping file",
        ),
        metadata={"resolved_from": "stream_graph"},
    )


def _resolve_jodie_csv(data: DataConfig) -> ResolvedStreamGraphDataset:
    raw_csv_path = resolve_data_path(
        data.raw_path,
        remote_path=data.raw_remote_path,
        backend=data.download_backend,
        cache_dir=data.cache_dir,
        token_env=data.token_env,
        artifact_name="JODIE CSV dataset",
    )
    if raw_csv_path is None:
        raise ValueError("jodie_csv source requires raw_path or raw_remote_path")

    cache_dir = Path(data.cache_dir or DEFAULT_DATA_CACHE_DIR)
    cache_dir.mkdir(parents=True, exist_ok=True)
    parquet_path = data.parquet_path
    if parquet_path is None:
        parquet_path = str(cache_dir / f"{Path(raw_csv_path).stem}_stream_graph.parquet")

    convert_jodie_csv_to_stream_graph(
        raw_csv_path,
        parquet_path,
        skip_header=not bool(data.extra.get("no_skip_header", False)),
        feature_mode=str(data.extra.get("feature_mode", "first2")),
    )

    return ResolvedStreamGraphDataset(
        source=data.source,
        parquet_path=parquet_path,
        metadata={
            "resolved_from": "jodie_csv",
            "raw_csv_path": raw_csv_path,
        },
    )


def resolve_stream_graph_dataset(data: DataConfig) -> ResolvedStreamGraphDataset:
    """Resolve a typed data config into local stream-graph inputs."""

    if data.source in {"stream_graph", "orbitaal_stream_graph"}:
        return _resolve_standard_stream_graph(data)
    if data.source in {"jodie_csv", "jodie_wikipedia"}:
        return _resolve_jodie_csv(data)
    raise ValueError(
        f"Unsupported stream-graph data source '{data.source}'. "
        f"Supported sources: {sorted(SUPPORTED_STREAM_GRAPH_SOURCES)}"
    )


__all__ = [
    "ResolvedStreamGraphDataset",
    "SUPPORTED_STREAM_GRAPH_SOURCES",
    "resolve_stream_graph_dataset",
]

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
from scripts.convert_edge_table_to_stream_graph import convert_edge_table_to_stream_graph
from scripts.convert_jodie_csv_to_stream_graph import convert_jodie_csv_to_stream_graph


SUPPORTED_STREAM_GRAPH_SOURCES = {
    "stream_graph",
    "orbitaal_stream_graph",
    "jodie_csv",
    "jodie_wikipedia",
    "edge_table_csv",
    "edge_table_parquet",
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


def _resolve_edge_table(data: DataConfig, *, input_format: str) -> ResolvedStreamGraphDataset:
    raw_path = resolve_data_path(
        data.raw_path,
        remote_path=data.raw_remote_path,
        backend=data.download_backend,
        cache_dir=data.cache_dir,
        token_env=data.token_env,
        artifact_name=f"{input_format} edge table dataset",
    )
    if raw_path is None:
        raise ValueError(f"{data.source} source requires raw_path or raw_remote_path")

    cache_dir = Path(data.cache_dir or DEFAULT_DATA_CACHE_DIR)
    cache_dir.mkdir(parents=True, exist_ok=True)
    parquet_path = data.parquet_path or str(cache_dir / f"{Path(raw_path).stem}_stream_graph.parquet")
    extra = data.extra.get("extra", data.extra) if isinstance(data.extra.get("extra"), dict) else data.extra
    convert_edge_table_to_stream_graph(
        input_path=raw_path,
        output_parquet=parquet_path,
        input_format=input_format,
        src_col=str(extra.get("src_col", "src")),
        dst_col=str(extra.get("dst_col", "dst")),
        timestamp_col=str(extra.get("timestamp_col", "timestamp")),
        btc_col=extra.get("btc_col"),
        usd_col=extra.get("usd_col"),
        delimiter=str(extra.get("delimiter", ",")),
        has_header=not bool(extra.get("no_header", False)),
        default_btc=float(extra.get("default_btc", 1.0)),
        default_usd=float(extra.get("default_usd", 0.0)),
        remap_nodes=bool(extra.get("remap_nodes", False)),
        bipartite=bool(extra.get("bipartite", False)),
    )

    return ResolvedStreamGraphDataset(
        source=data.source,
        parquet_path=parquet_path,
        metadata={
            "resolved_from": data.source,
            "raw_path": raw_path,
        },
    )


def resolve_stream_graph_dataset(data: DataConfig) -> ResolvedStreamGraphDataset:
    """Resolve a typed data config into local stream-graph inputs."""

    if data.source in {"stream_graph", "orbitaal_stream_graph"}:
        return _resolve_standard_stream_graph(data)
    if data.source in {"jodie_csv", "jodie_wikipedia"}:
        return _resolve_jodie_csv(data)
    if data.source == "edge_table_csv":
        return _resolve_edge_table(data, input_format="csv")
    if data.source == "edge_table_parquet":
        return _resolve_edge_table(data, input_format="parquet")
    raise ValueError(
        f"Unsupported stream-graph data source '{data.source}'. "
        f"Supported sources: {sorted(SUPPORTED_STREAM_GRAPH_SOURCES)}"
    )


__all__ = [
    "ResolvedStreamGraphDataset",
    "SUPPORTED_STREAM_GRAPH_SOURCES",
    "resolve_stream_graph_dataset",
]

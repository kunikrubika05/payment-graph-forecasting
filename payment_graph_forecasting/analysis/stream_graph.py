"""Package-facing stream-graph analysis helpers."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import pandas as pd

from payment_graph_forecasting.data.stream_graph import StreamGraphDataset


@dataclass(frozen=True, slots=True)
class StreamGraphAnalysisReport:
    """Lightweight stream-graph report."""

    parquet_path: str
    label: str
    selection_description: str
    source_total_edges: int
    num_edges: int
    num_nodes: int
    unique_sources: int
    unique_destinations: int
    timestamp_min: int | None
    timestamp_max: int | None
    total_btc: float
    total_usd: float
    mean_btc: float
    median_btc: float
    max_btc: float
    unique_directed_edges: int
    repeated_pair_events: int
    self_loops: int
    mean_out_degree: float
    median_out_degree: float
    max_out_degree: int
    mean_in_degree: float
    median_in_degree: float
    max_in_degree: int
    density_directed: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_text(self) -> str:
        return format_stream_graph_report(self)


def analyze_stream_graph(dataset: StreamGraphDataset) -> StreamGraphAnalysisReport:
    """Compute lightweight metrics for a stream-graph dataset selection."""

    df = dataset.read_table().to_pandas()
    if df.empty:
        unique_pairs = pd.DataFrame(columns=["src_idx", "dst_idx"])
        out_degree = pd.Series(dtype="int64")
        in_degree = pd.Series(dtype="int64")
        num_nodes = 0
        unique_sources = 0
        unique_destinations = 0
        timestamp_min = None
        timestamp_max = None
        total_btc = 0.0
        total_usd = 0.0
        mean_btc = 0.0
        median_btc = 0.0
        max_btc = 0.0
        self_loops = 0
    else:
        unique_pairs = df[["src_idx", "dst_idx"]].drop_duplicates(ignore_index=True)
        out_degree = unique_pairs.groupby("src_idx", sort=False).size()
        in_degree = unique_pairs.groupby("dst_idx", sort=False).size()
        unique_sources = int(df["src_idx"].nunique())
        unique_destinations = int(df["dst_idx"].nunique())
        num_nodes = int(pd.concat([df["src_idx"], df["dst_idx"]], ignore_index=True).nunique())
        timestamp_min = int(df["timestamp"].min())
        timestamp_max = int(df["timestamp"].max())
        total_btc = float(df["btc"].sum())
        total_usd = float(df["usd"].sum())
        mean_btc = float(df["btc"].mean())
        median_btc = float(df["btc"].median())
        max_btc = float(df["btc"].max())
        self_loops = int((df["src_idx"] == df["dst_idx"]).sum())

    unique_directed_edges = int(len(unique_pairs))
    num_edges = int(len(df))
    repeated_pair_events = num_edges - unique_directed_edges
    mean_out_degree = float(out_degree.mean()) if not out_degree.empty else 0.0
    median_out_degree = float(out_degree.median()) if not out_degree.empty else 0.0
    max_out_degree = int(out_degree.max()) if not out_degree.empty else 0
    mean_in_degree = float(in_degree.mean()) if not in_degree.empty else 0.0
    median_in_degree = float(in_degree.median()) if not in_degree.empty else 0.0
    max_in_degree = int(in_degree.max()) if not in_degree.empty else 0
    density_directed = (
        float(unique_directed_edges / (num_nodes * (num_nodes - 1)))
        if num_nodes > 1
        else 0.0
    )

    return StreamGraphAnalysisReport(
        parquet_path=dataset.parquet_path,
        label=dataset.resolved_label,
        selection_description=dataset.selection.describe(
            source_total_edges=dataset.describe().source_total_edges
        ),
        source_total_edges=dataset.describe().source_total_edges,
        num_edges=num_edges,
        num_nodes=num_nodes,
        unique_sources=unique_sources,
        unique_destinations=unique_destinations,
        timestamp_min=timestamp_min,
        timestamp_max=timestamp_max,
        total_btc=total_btc,
        total_usd=total_usd,
        mean_btc=mean_btc,
        median_btc=median_btc,
        max_btc=max_btc,
        unique_directed_edges=unique_directed_edges,
        repeated_pair_events=repeated_pair_events,
        self_loops=self_loops,
        mean_out_degree=mean_out_degree,
        median_out_degree=median_out_degree,
        max_out_degree=max_out_degree,
        mean_in_degree=mean_in_degree,
        median_in_degree=median_in_degree,
        max_in_degree=max_in_degree,
        density_directed=density_directed,
    )


def format_stream_graph_report(report: StreamGraphAnalysisReport) -> str:
    """Render a human-readable report for terminal output."""

    return "\n".join(
        [
            "Stream Graph Analysis Report",
            f"Label: {report.label}",
            f"Source parquet: {report.parquet_path}",
            f"Selection: {report.selection_description}",
            f"Selected edges: {report.num_edges:,}",
            f"Source edges: {report.source_total_edges:,}",
            "",
            "Structure",
            f"  num_nodes: {report.num_nodes:,}",
            f"  unique_sources: {report.unique_sources:,}",
            f"  unique_destinations: {report.unique_destinations:,}",
            f"  unique_directed_edges: {report.unique_directed_edges:,}",
            f"  repeated_pair_events: {report.repeated_pair_events:,}",
            f"  self_loops: {report.self_loops:,}",
            f"  density_directed: {report.density_directed:.6e}",
            "",
            "Time",
            f"  timestamp_min: {report.timestamp_min}",
            f"  timestamp_max: {report.timestamp_max}",
            "",
            "Value",
            f"  total_btc: {report.total_btc:.6f}",
            f"  total_usd: {report.total_usd:.6f}",
            f"  mean_btc: {report.mean_btc:.6f}",
            f"  median_btc: {report.median_btc:.6f}",
            f"  max_btc: {report.max_btc:.6f}",
            "",
            "Degree",
            f"  mean_out_degree: {report.mean_out_degree:.6f}",
            f"  median_out_degree: {report.median_out_degree:.6f}",
            f"  max_out_degree: {report.max_out_degree:,}",
            f"  mean_in_degree: {report.mean_in_degree:.6f}",
            f"  median_in_degree: {report.median_in_degree:.6f}",
            f"  max_in_degree: {report.max_in_degree:,}",
        ]
    )


__all__ = [
    "StreamGraphAnalysisReport",
    "analyze_stream_graph",
    "format_stream_graph_report",
]

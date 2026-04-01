"""Package-facing stream-graph access and slicing helpers."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq

from payment_graph_forecasting.infra.data_access import resolve_data_path

STREAM_GRAPH_COLUMNS = ("src_idx", "dst_idx", "timestamp", "btc", "usd")


def _date_to_unix_start(date_str: str) -> int:
    dt = datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    return int(dt.timestamp())


def _date_to_unix_end(date_str: str) -> int:
    dt = datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc) + timedelta(days=1)
    return int(dt.timestamp()) - 1


@dataclass(frozen=True, slots=True)
class StreamGraphSelection:
    """Selection over a chronologically sorted stream-graph parquet."""

    kind: str = "full"
    fraction: float | None = None
    start_date: str | None = None
    end_date: str | None = None

    @classmethod
    def full(cls) -> "StreamGraphSelection":
        return cls(kind="full")

    @classmethod
    def period_fraction(cls, fraction: float) -> "StreamGraphSelection":
        if not 0.0 < fraction <= 1.0:
            raise ValueError("fraction must be in the interval (0, 1]")
        return cls(kind="period_fraction", fraction=fraction)

    @classmethod
    def date_range(cls, start_date: str, end_date: str) -> "StreamGraphSelection":
        if start_date > end_date:
            raise ValueError("start_date must be less than or equal to end_date")
        return cls(kind="date_range", start_date=start_date, end_date=end_date)

    def describe(self, *, source_total_edges: int | None = None) -> str:
        if self.kind == "full":
            return "full graph"
        if self.kind == "period_fraction":
            if source_total_edges is None:
                return f"chronological prefix fraction={self.fraction:.2%} of sorted period"
            selected_edges = int(source_total_edges * float(self.fraction))
            return (
                f"chronological prefix fraction={self.fraction:.2%} of sorted period "
                f"({selected_edges:,} / {source_total_edges:,} edges)"
            )
        if self.kind == "date_range":
            return f"date range {self.start_date}..{self.end_date}"
        raise ValueError(f"Unsupported selection kind: {self.kind}")


@dataclass(frozen=True, slots=True)
class StreamGraphDescriptor:
    """Human-readable description of a stream graph selection."""

    parquet_path: str
    label: str
    selection: StreamGraphSelection
    source_total_edges: int
    selected_edges_estimate: int | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class StreamGraphDataset:
    """Reference to a stream-graph parquet plus an optional data selection."""

    parquet_path: str
    label: str | None = None
    selection: StreamGraphSelection = field(default_factory=StreamGraphSelection.full)
    source_total_edges: int | None = None

    @classmethod
    def from_parquet(cls, parquet_path: str, *, label: str | None = None) -> "StreamGraphDataset":
        dataset = cls(parquet_path=str(Path(parquet_path)), label=label)
        dataset._validate_schema()
        dataset.source_total_edges = dataset._parquet_file().metadata.num_rows
        return dataset

    @property
    def resolved_label(self) -> str:
        return self.label or Path(self.parquet_path).name

    def describe(self) -> StreamGraphDescriptor:
        source_total_edges = self._source_total_edges()
        selected_edges_estimate = None
        if self.selection.kind == "full":
            selected_edges_estimate = source_total_edges
        elif self.selection.kind == "period_fraction":
            selected_edges_estimate = int(source_total_edges * float(self.selection.fraction))
        return StreamGraphDescriptor(
            parquet_path=self.parquet_path,
            label=self.resolved_label,
            selection=self.selection,
            source_total_edges=source_total_edges,
            selected_edges_estimate=selected_edges_estimate,
        )

    def slice_period_fraction(self, fraction: float, *, label: str | None = None) -> "StreamGraphDataset":
        return StreamGraphDataset(
            parquet_path=self.parquet_path,
            label=label or self.resolved_label,
            selection=StreamGraphSelection.period_fraction(fraction),
            source_total_edges=self._source_total_edges(),
        )

    def slice_first_fraction(self, fraction: float, *, label: str | None = None) -> "StreamGraphDataset":
        """Compatibility alias for the chronological prefix selection.

        # TODO(REFACTORING): remove this alias after downstream callers migrate
        # to `slice_period_fraction`.
        """

        return self.slice_period_fraction(fraction, label=label)

    def slice_date_range(
        self,
        start_date: str,
        end_date: str,
        *,
        label: str | None = None,
    ) -> "StreamGraphDataset":
        return StreamGraphDataset(
            parquet_path=self.parquet_path,
            label=label or self.resolved_label,
            selection=StreamGraphSelection.date_range(start_date, end_date),
            source_total_edges=self._source_total_edges(),
        )

    def read_table(self, *, columns: list[str] | None = None) -> pa.Table:
        return self._load_table(columns=columns)

    def write_parquet(self, output_path: str | Path) -> str:
        table = self._load_table(columns=list(STREAM_GRAPH_COLUMNS))
        destination = Path(output_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(table, destination)
        return str(destination)

    def _source_total_edges(self) -> int:
        if self.source_total_edges is not None:
            return self.source_total_edges
        self.source_total_edges = self._parquet_file().metadata.num_rows
        return self.source_total_edges

    def _parquet_file(self) -> pq.ParquetFile:
        return pq.ParquetFile(self.parquet_path)

    def _validate_schema(self) -> None:
        available_columns = set(self._parquet_file().schema_arrow.names)
        missing_columns = [column for column in STREAM_GRAPH_COLUMNS if column not in available_columns]
        if missing_columns:
            raise ValueError(
                f"Stream graph parquet at {self.parquet_path} is missing required columns: {missing_columns}"
            )

    def _load_table(self, *, columns: list[str] | None = None) -> pa.Table:
        if self.selection.kind == "full":
            return pq.read_table(self.parquet_path, columns=columns)
        if self.selection.kind == "period_fraction":
            return self._load_period_fraction_table(columns=columns)
        if self.selection.kind == "date_range":
            dataset = ds.dataset(self.parquet_path, format="parquet")
            return dataset.to_table(
                columns=columns,
                filter=(
                    (ds.field("timestamp") >= _date_to_unix_start(str(self.selection.start_date)))
                    & (ds.field("timestamp") <= _date_to_unix_end(str(self.selection.end_date)))
                ),
            )
        raise ValueError(f"Unsupported selection kind: {self.selection.kind}")

    def _load_period_fraction_table(self, *, columns: list[str] | None = None) -> pa.Table:
        total_rows = self._source_total_edges()
        selected_rows = int(total_rows * float(self.selection.fraction))
        if selected_rows <= 0:
            schema = self._parquet_file().schema_arrow
            if columns is not None:
                schema = pa.schema([schema.field(name) for name in columns])
            return pa.Table.from_batches([], schema=schema)

        parquet_file = self._parquet_file()
        batches = []
        remaining = selected_rows
        for batch in parquet_file.iter_batches(columns=columns):
            if remaining <= 0:
                break
            if batch.num_rows > remaining:
                batches.append(batch.slice(0, remaining))
                remaining = 0
                break
            batches.append(batch)
            remaining -= batch.num_rows

        if not batches:
            schema = self._parquet_file().schema_arrow
            if columns is not None:
                schema = pa.schema([schema.field(name) for name in columns])
            return pa.Table.from_batches([], schema=schema)
        return pa.Table.from_batches(batches)


def open_stream_graph(
    local_path: str | None = None,
    *,
    remote_path: str | None = None,
    download_backend: str = "yadisk",
    cache_dir: str | None = None,
    token_env: str = "YADISK_TOKEN",
    label: str | None = None,
) -> StreamGraphDataset:
    """Open a local or remotely resolved stream-graph parquet."""

    parquet_path = resolve_data_path(
        local_path,
        remote_path=remote_path,
        backend=download_backend,
        cache_dir=cache_dir,
        token_env=token_env,
        artifact_name="stream graph parquet",
    )
    if parquet_path is None:
        raise ValueError("open_stream_graph requires local_path or remote_path")
    return StreamGraphDataset.from_parquet(parquet_path, label=label)


__all__ = [
    "STREAM_GRAPH_COLUMNS",
    "StreamGraphDataset",
    "StreamGraphDescriptor",
    "StreamGraphSelection",
    "open_stream_graph",
]

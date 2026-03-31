from __future__ import annotations

from pathlib import Path

import pandas as pd

from payment_graph_forecasting.config.base import DataConfig
from payment_graph_forecasting.infra.datasets import resolve_stream_graph_dataset


def test_resolve_stream_graph_dataset_uses_local_parquet(tmp_path):
    parquet_path = tmp_path / "stream.parquet"
    pd.DataFrame(
        {
            "src_idx": [0],
            "dst_idx": [1],
            "timestamp": [1],
            "btc": [1.0],
            "usd": [2.0],
        }
    ).to_parquet(parquet_path, index=False)

    resolved = resolve_stream_graph_dataset(
        DataConfig(source="stream_graph", parquet_path=str(parquet_path))
    )

    assert resolved.source == "stream_graph"
    assert resolved.parquet_path == str(parquet_path)


def test_resolve_stream_graph_dataset_converts_jodie_csv(tmp_path):
    raw_csv = tmp_path / "wikipedia.csv"
    raw_csv.write_text(
        "\n".join(
            [
                "src,dst,t,y,f1,f2",
                "1,2,30,1,0.5,1.5",
                "0,1,10,0,2.0,3.0",
            ]
        ),
        encoding="utf-8",
    )

    resolved = resolve_stream_graph_dataset(
        DataConfig(
            source="jodie_csv",
            raw_path=str(raw_csv),
            cache_dir=str(tmp_path / "cache"),
        )
    )

    df = pd.read_parquet(resolved.parquet_path)
    assert resolved.source == "jodie_csv"
    assert resolved.metadata["resolved_from"] == "jodie_csv"
    assert Path(resolved.metadata["raw_csv_path"]) == raw_csv
    assert list(df.columns) == ["src_idx", "dst_idx", "timestamp", "btc", "usd"]

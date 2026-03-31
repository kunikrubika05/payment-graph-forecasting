from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from payment_graph_forecasting.config.base import DataConfig
from payment_graph_forecasting.infra.datasets import resolve_stream_graph_dataset
from scripts.build_dense_validation_bundle import build_dense_validation_bundle
from scripts.convert_edge_table_to_stream_graph import convert_edge_table_to_stream_graph


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


def test_resolve_stream_graph_dataset_converts_generic_edge_table_csv(tmp_path):
    raw_csv = tmp_path / "edges.csv"
    raw_csv.write_text(
        "\n".join(
            [
                "source,target,ts,amount",
                "10,20,30,1.5",
                "11,21,10,2.5",
            ]
        ),
        encoding="utf-8",
    )

    resolved = resolve_stream_graph_dataset(
        DataConfig(
            source="edge_table_csv",
            raw_path=str(raw_csv),
            cache_dir=str(tmp_path / "cache"),
            extra={
                "src_col": "source",
                "dst_col": "target",
                "timestamp_col": "ts",
                "btc_col": "amount",
                "default_usd": 7.0,
                "remap_nodes": True,
            },
        )
    )

    df = pd.read_parquet(resolved.parquet_path)
    assert resolved.metadata["resolved_from"] == "edge_table_csv"
    assert list(df.columns) == ["src_idx", "dst_idx", "timestamp", "btc", "usd"]
    assert df["timestamp"].tolist() == [10, 30]
    assert df["src_idx"].tolist() == [1, 0]
    assert df["usd"].tolist() == [7.0, 7.0]


def test_convert_edge_table_to_stream_graph_supports_parquet_input(tmp_path):
    input_path = tmp_path / "edges.parquet"
    pd.DataFrame(
        {
            "u": [5, 6],
            "v": [7, 8],
            "t": [2, 1],
        }
    ).to_parquet(input_path, index=False)
    output_path = tmp_path / "stream.parquet"

    convert_edge_table_to_stream_graph(
        input_path,
        output_path,
        input_format="parquet",
        src_col="u",
        dst_col="v",
        timestamp_col="t",
        default_btc=3.0,
        default_usd=4.0,
    )

    df = pd.read_parquet(output_path)
    assert df["timestamp"].tolist() == [1, 2]
    assert df["btc"].tolist() == [3.0, 3.0]
    assert df["usd"].tolist() == [4.0, 4.0]


def test_build_dense_validation_bundle_remaps_nodes_and_filters_features(tmp_path):
    sg_path = tmp_path / "stream.parquet"
    pd.DataFrame(
        {
            "src_idx": [100, 100, 200],
            "dst_idx": [200, 300, 300],
            "timestamp": [1, 2, 3],
            "btc": [1.0, 2.0, 3.0],
            "usd": [4.0, 5.0, 6.0],
        }
    ).to_parquet(sg_path, index=False)
    features_path = tmp_path / "features.parquet"
    pd.DataFrame(
        {
            "node_idx": [100, 200, 300, 999],
            "f0": [0.1, 0.2, 0.3, 9.9],
        }
    ).to_parquet(features_path, index=False)

    bundle = build_dense_validation_bundle(
        sg_path,
        tmp_path / "bundle",
        max_edges=2,
        features_path=features_path,
    )

    sg_dense = pd.read_parquet(bundle["stream_graph_path"])
    feat_dense = pd.read_parquet(bundle["features_path"])
    node_mapping = np.load(bundle["node_mapping_path"])

    assert bundle["num_edges"] == 2
    assert bundle["num_nodes"] == 3
    assert sg_dense["src_idx"].tolist() == [0, 0]
    assert sg_dense["dst_idx"].tolist() == [1, 2]
    assert feat_dense["node_idx"].tolist() == [0, 1, 2]
    assert node_mapping.tolist() == [0, 1, 2]

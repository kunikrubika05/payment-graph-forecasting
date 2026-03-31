from __future__ import annotations

import pandas as pd

from scripts.convert_jodie_csv_to_stream_graph import convert_jodie_csv_to_stream_graph


def test_convert_jodie_csv_to_stream_graph_shifts_dst_and_keeps_contract(tmp_path):
    input_csv = tmp_path / "wikipedia.csv"
    input_csv.write_text(
        "\n".join(
            [
                "src,dst,t,y,f1,f2",
                "1,2,30,1,0.5,1.5",
                "0,1,10,0,2.0,3.0",
            ]
        ),
        encoding="utf-8",
    )

    output_parquet = tmp_path / "stream_graph.parquet"
    convert_jodie_csv_to_stream_graph(input_csv, output_parquet)

    df = pd.read_parquet(output_parquet)

    assert list(df.columns) == ["src_idx", "dst_idx", "timestamp", "btc", "usd"]
    assert df["timestamp"].tolist() == [10, 30]
    assert df["src_idx"].tolist() == [0, 1]
    assert df["dst_idx"].tolist() == [3, 4]
    assert df["btc"].tolist() == [2.0, 0.5]
    assert df["usd"].tolist() == [3.0, 1.5]

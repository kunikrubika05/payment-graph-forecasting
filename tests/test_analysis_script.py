from __future__ import annotations

import json

import pandas as pd

from scripts import analyze_stream_graph, slice_stream_graph, visualize_stream_graph
from src.visualization.stream_graph import visualize_stream_graph as legacy_visualize_stream_graph


def test_analyze_stream_graph_script_defaults_to_lightweight_fraction(tmp_path, capsys):
    parquet_path = tmp_path / "stream.parquet"
    pd.DataFrame(
        {
            "src_idx": [0, 1, 2, 3, 4],
            "dst_idx": [1, 2, 3, 4, 0],
            "timestamp": [1, 2, 3, 4, 5],
            "btc": [1.0, 2.0, 3.0, 4.0, 5.0],
            "usd": [10.0, 20.0, 30.0, 40.0, 50.0],
        }
    ).to_parquet(parquet_path, index=False)
    report_path = tmp_path / "report.json"

    exit_code = analyze_stream_graph.main(
        ["--input", str(parquet_path), "--output-json", str(report_path)]
    )

    output = capsys.readouterr().out
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert exit_code == 0
    assert "Selection: chronological prefix fraction=10.00% of sorted period (0 / 5 edges)" in output
    assert report["source_total_edges"] == 5
    assert report["num_edges"] == 0


def test_analyze_stream_graph_script_supports_full_opt_in(tmp_path, capsys):
    parquet_path = tmp_path / "stream.parquet"
    pd.DataFrame(
        {
            "src_idx": [0, 1],
            "dst_idx": [1, 0],
            "timestamp": [10, 20],
            "btc": [1.0, 2.0],
            "usd": [10.0, 20.0],
        }
    ).to_parquet(parquet_path, index=False)

    exit_code = analyze_stream_graph.main(["--input", str(parquet_path), "--full"])

    output = capsys.readouterr().out
    assert exit_code == 0
    assert "Selection: full graph" in output
    assert "Selected edges: 2" in output


def test_slice_stream_graph_script_is_thin_wrapper_over_package_api(tmp_path, capsys):
    parquet_path = tmp_path / "stream.parquet"
    pd.DataFrame(
        {
            "src_idx": [0, 0, 1, 2],
            "dst_idx": [1, 2, 2, 0],
            "timestamp": [1, 2, 3, 4],
            "btc": [1.0, 2.0, 3.0, 4.0],
            "usd": [10.0, 20.0, 30.0, 40.0],
        }
    ).to_parquet(parquet_path, index=False)
    output_path = tmp_path / "slice.parquet"

    exit_code = slice_stream_graph.main(
        ["--input", str(parquet_path), "--fraction", "0.5", "--output", str(output_path)]
    )

    output = capsys.readouterr().out
    sliced_df = pd.read_parquet(output_path)
    stats = json.loads(output_path.with_suffix(".json").read_text(encoding="utf-8"))
    assert exit_code == 0
    assert sliced_df["timestamp"].tolist() == [1, 2]
    assert stats["selection_description"] == "chronological prefix fraction=50.00% of sorted period (2 / 4 edges)"
    assert "Selection: chronological prefix fraction=50.00% of sorted period (2 / 4 edges)" in output


def test_visualize_stream_graph_script_is_thin_wrapper_over_legacy_visualization_api(
    monkeypatch,
    tmp_path,
    capsys,
):
    parquet_path = tmp_path / "stream.parquet"
    pd.DataFrame(
        {
            "src_idx": [0, 0, 1, 2, 3],
            "dst_idx": [1, 2, 2, 3, 0],
            "timestamp": [1, 2, 3, 4, 5],
            "btc": [1.0, 2.0, 3.0, 4.0, 5.0],
            "usd": [10.0, 20.0, 30.0, 40.0, 50.0],
        }
    ).to_parquet(parquet_path, index=False)
    output_path = tmp_path / "view.svg"
    metadata_path = tmp_path / "view.json"
    view_json_path = tmp_path / "renderable.json"
    calls: list[tuple[str, int | None, int, int, str]] = []

    def _wrapped_visualize(*args, **kwargs):
        calls.append(
            (
                str(args[1]),
                kwargs.get("sample_edges"),
                kwargs.get("max_nodes"),
                kwargs.get("max_edges"),
                kwargs.get("layout"),
            )
        )
        return legacy_visualize_stream_graph(*args, **kwargs)

    monkeypatch.setattr(visualize_stream_graph, "visualize_stream_graph", _wrapped_visualize)

    exit_code = visualize_stream_graph.main(
        [
            "--input",
            str(parquet_path),
            "--fraction",
            "0.8",
            "--sample-edges",
            "3",
            "--max-nodes",
            "3",
            "--max-edges",
            "2",
            "--output",
            str(output_path),
            "--metadata-json",
            str(metadata_path),
            "--view-json",
            str(view_json_path),
        ]
    )

    output = capsys.readouterr().out
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    renderable = json.loads(view_json_path.read_text(encoding="utf-8"))
    assert exit_code == 0
    assert output_path.exists()
    assert calls == [(str(output_path), 3, 3, 2, "flow")]
    assert metadata["output_format"] == "svg"
    assert metadata["view"]["selection_description"] == "chronological prefix fraction=80.00% of sorted period (4 / 5 edges)"
    assert renderable["sampled_edges"] == 3
    assert "Selection: chronological prefix fraction=80.00% of sorted period (4 / 5 edges)" in output

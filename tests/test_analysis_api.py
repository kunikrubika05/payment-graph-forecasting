from __future__ import annotations

import pandas as pd

import payment_graph_forecasting as pgf
from payment_graph_forecasting.analysis import analyze_stream_graph
from payment_graph_forecasting.data import StreamGraphDataset, open_stream_graph


def _write_stream_graph(tmp_path):
    parquet_path = tmp_path / "stream.parquet"
    pd.DataFrame(
        {
            "src_idx": [0, 0, 1, 2, 2],
            "dst_idx": [1, 1, 2, 2, 0],
            "timestamp": [10, 20, 30, 40, 50],
            "btc": [1.0, 2.0, 3.0, 4.0, 5.0],
            "usd": [10.0, 20.0, 30.0, 40.0, 50.0],
        }
    ).to_parquet(parquet_path, index=False)
    return parquet_path


def test_top_level_package_exports_analysis_surface():
    assert pgf.open_stream_graph is open_stream_graph
    assert pgf.StreamGraphDataset is StreamGraphDataset
    assert pgf.analyze_stream_graph is analyze_stream_graph


def test_stream_graph_analysis_report_matches_small_graph(tmp_path):
    parquet_path = _write_stream_graph(tmp_path)

    report = analyze_stream_graph(open_stream_graph(str(parquet_path), label="toy"))

    assert report.label == "toy"
    assert report.selection_description == "full graph"
    assert report.source_total_edges == 5
    assert report.num_edges == 5
    assert report.num_nodes == 3
    assert report.unique_sources == 3
    assert report.unique_destinations == 3
    assert report.timestamp_min == 10
    assert report.timestamp_max == 50
    assert report.total_btc == 15.0
    assert report.total_usd == 150.0
    assert report.mean_btc == 3.0
    assert report.median_btc == 3.0
    assert report.max_btc == 5.0
    assert report.unique_directed_edges == 4
    assert report.repeated_pair_events == 1
    assert report.self_loops == 1
    assert report.mean_out_degree == 4 / 3
    assert report.median_out_degree == 1.0
    assert report.max_out_degree == 2
    assert report.mean_in_degree == 4 / 3
    assert report.median_in_degree == 1.0
    assert report.max_in_degree == 2
    assert report.density_directed == 4 / 6


def test_fraction_slice_keeps_honest_selection_metadata(tmp_path):
    parquet_path = _write_stream_graph(tmp_path)

    report = analyze_stream_graph(open_stream_graph(str(parquet_path)).slice_period_fraction(0.4))

    assert report.num_edges == 2
    assert report.source_total_edges == 5
    assert report.selection_description == "chronological prefix fraction=40.00% of sorted period (2 / 5 edges)"
    assert "Selection: chronological prefix fraction=40.00% of sorted period (2 / 5 edges)" in report.to_text()
    assert "Selected edges: 2" in report.to_text()
    assert "Source edges: 5" in report.to_text()


def test_date_range_slice_and_write_parquet(tmp_path):
    parquet_path = _write_stream_graph(tmp_path)

    dataset = open_stream_graph(str(parquet_path)).slice_date_range("1970-01-01", "1970-01-01")
    output_path = tmp_path / "slice.parquet"
    written = dataset.write_parquet(output_path)
    sliced_df = pd.read_parquet(written)

    assert sliced_df["timestamp"].tolist() == [10, 20, 30, 40, 50]


def test_descriptor_exposes_selection_context(tmp_path):
    parquet_path = _write_stream_graph(tmp_path)

    descriptor = open_stream_graph(str(parquet_path)).slice_period_fraction(0.6).describe()

    assert descriptor.source_total_edges == 5
    assert descriptor.selected_edges_estimate == 3
    assert descriptor.selection.describe(source_total_edges=descriptor.source_total_edges) == "chronological prefix fraction=60.00% of sorted period (3 / 5 edges)"


def test_period_fraction_slice_matches_training_prefix_semantics(tmp_path):
    parquet_path = _write_stream_graph(tmp_path)

    dataset = open_stream_graph(str(parquet_path)).slice_period_fraction(0.4)
    timestamps = dataset.read_table(columns=["timestamp"]).to_pandas()["timestamp"].tolist()

    assert timestamps == [10, 20]

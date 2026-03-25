"""Tests for the stream graph pipeline (src/build_stream_graph.py)."""

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).parent.parent
SAMPLES_DIR = ROOT / "data" / "samples"


@pytest.fixture
def tmp_output(tmp_path):
    """Create a temporary output directory with a small node mapping."""
    mapping_df = pd.DataFrame({
        "entity_id": np.array([1, 2, 3, 100, 200, 300], dtype=np.int64),
        "node_index": np.arange(6, dtype=np.int64),
    })
    mapping_df.to_parquet(tmp_path / "node_mapping.parquet", index=False)
    return tmp_path


@pytest.fixture
def sample_stream_parquet(tmp_path):
    """Create a sample stream graph parquet file."""
    df = pd.DataFrame({
        "SRC_ID": [1, 2, 3, 0, 1, 2],
        "DST_ID": [2, 3, 1, 1, 1, 100],
        "TIMESTAMP": [1000, 1001, 1002, 1003, 1004, 1005],
        "VALUE_SATOSHI": [100000000, 200000000, 50000000, 10000000, 300000000, 150000000],
        "VALUE_USD": [1000.0, 2000.0, 500.0, 100.0, 3000.0, 1500.0],
    })
    out = tmp_path / "stream_graph_test.parquet"
    df.to_parquet(out, index=False)
    return tmp_path


class TestDateRangeToTimestamps:
    """Tests for _date_range_to_timestamps."""

    def test_basic(self):
        from src.build_stream_graph import _date_range_to_timestamps
        start_ts, end_ts = _date_range_to_timestamps("2020-06-01", "2020-06-01")
        assert start_ts < end_ts
        assert end_ts - start_ts < 86400

    def test_multi_day(self):
        from src.build_stream_graph import _date_range_to_timestamps
        start_ts, end_ts = _date_range_to_timestamps("2020-06-01", "2020-08-31")
        days = (end_ts - start_ts) / 86400
        assert 91 <= days <= 92


class TestLoadNodeMapping:
    """Tests for _load_node_mapping."""

    def test_load_local(self, tmp_output):
        from src.build_stream_graph import _load_node_mapping
        mapping = _load_node_mapping(tmp_output)
        assert len(mapping) == 6
        assert mapping[1] == 0
        assert mapping[300] == 5

    def test_missing_no_token(self, tmp_path):
        from src.build_stream_graph import _load_node_mapping
        with pytest.raises(SystemExit):
            _load_node_mapping(tmp_path, token=None)


class TestStepProcess:
    """Tests for the process step."""

    def test_filters_entity_zero(self, sample_stream_parquet, tmp_output):
        from src.build_stream_graph import step_process
        step_process(
            input_dir=sample_stream_parquet,
            output_dir=tmp_output,
            start_date="1970-01-01",
            end_date="2030-12-31",
            fmt="parquet",
        )
        out_files = list((tmp_output / "stream_graph").glob("*.parquet"))
        assert len(out_files) == 1
        df = pd.read_parquet(out_files[0])
        assert len(df) > 0
        assert "src_idx" in df.columns
        assert "timestamp" in df.columns

    def test_removes_self_loops(self, sample_stream_parquet, tmp_output):
        from src.build_stream_graph import step_process
        step_process(
            input_dir=sample_stream_parquet,
            output_dir=tmp_output,
            start_date="1970-01-01",
            end_date="2030-12-31",
            fmt="parquet",
        )
        out_files = list((tmp_output / "stream_graph").glob("*.parquet"))
        df = pd.read_parquet(out_files[0])
        assert (df["src_idx"] != df["dst_idx"]).all()

    def test_sorted_by_timestamp(self, sample_stream_parquet, tmp_output):
        from src.build_stream_graph import step_process
        step_process(
            input_dir=sample_stream_parquet,
            output_dir=tmp_output,
            start_date="1970-01-01",
            end_date="2030-12-31",
            fmt="parquet",
        )
        out_files = list((tmp_output / "stream_graph").glob("*.parquet"))
        df = pd.read_parquet(out_files[0])
        assert (df["timestamp"].diff().dropna() >= 0).all()

    def test_creates_stats_json(self, sample_stream_parquet, tmp_output):
        from src.build_stream_graph import step_process
        step_process(
            input_dir=sample_stream_parquet,
            output_dir=tmp_output,
            start_date="1970-01-01",
            end_date="2030-12-31",
            fmt="parquet",
        )
        json_files = list((tmp_output / "stream_graph").glob("*.json"))
        assert len(json_files) == 1
        with open(json_files[0]) as f:
            stats = json.load(f)
        assert "num_edges" in stats
        assert "num_nodes" in stats
        assert stats["num_edges"] > 0

    def test_date_filter_empty(self, sample_stream_parquet, tmp_output):
        from src.build_stream_graph import step_process
        with pytest.raises(SystemExit):
            step_process(
                input_dir=sample_stream_parquet,
                output_dir=tmp_output,
                start_date="2020-01-01",
                end_date="2020-12-31",
                fmt="parquet",
            )

    def test_btc_conversion(self, sample_stream_parquet, tmp_output):
        from src.build_stream_graph import step_process
        step_process(
            input_dir=sample_stream_parquet,
            output_dir=tmp_output,
            start_date="1970-01-01",
            end_date="2030-12-31",
            fmt="parquet",
        )
        out_files = list((tmp_output / "stream_graph").glob("*.parquet"))
        df = pd.read_parquet(out_files[0])
        assert df["btc"].dtype == np.float32
        assert df["usd"].dtype == np.float32

    def test_output_columns(self, sample_stream_parquet, tmp_output):
        from src.build_stream_graph import step_process
        step_process(
            input_dir=sample_stream_parquet,
            output_dir=tmp_output,
            start_date="1970-01-01",
            end_date="2030-12-31",
            fmt="parquet",
        )
        out_files = list((tmp_output / "stream_graph").glob("*.parquet"))
        df = pd.read_parquet(out_files[0])
        assert list(df.columns) == ["src_idx", "dst_idx", "timestamp", "btc", "usd"]

    def test_skip_existing(self, sample_stream_parquet, tmp_output):
        from src.build_stream_graph import step_process
        step_process(
            input_dir=sample_stream_parquet,
            output_dir=tmp_output,
            start_date="1970-01-01",
            end_date="2030-12-31",
            fmt="parquet",
        )
        out_files = list((tmp_output / "stream_graph").glob("*.parquet"))
        mtime = out_files[0].stat().st_mtime
        step_process(
            input_dir=sample_stream_parquet,
            output_dir=tmp_output,
            start_date="1970-01-01",
            end_date="2030-12-31",
            fmt="parquet",
        )
        assert out_files[0].stat().st_mtime == mtime


class TestCSVFormat:
    """Tests with CSV sample files."""

    @pytest.mark.skipif(
        not (SAMPLES_DIR / "orbitaal-stream_graph-2016_07_08.csv").exists(),
        reason="CSV samples not available",
    )
    def test_csv_samples(self, tmp_output):
        from src.build_stream_graph import step_process

        csv_df = pd.read_csv(SAMPLES_DIR / "orbitaal-stream_graph-2016_07_08.csv")
        all_ids = np.unique(np.concatenate([
            csv_df["SRC_ID"].values, csv_df["DST_ID"].values
        ]))
        all_ids = all_ids[all_ids != 0]
        mapping_df = pd.DataFrame({
            "entity_id": all_ids,
            "node_index": np.arange(len(all_ids), dtype=np.int64),
        })
        mapping_df.to_parquet(tmp_output / "node_mapping.parquet", index=False)

        step_process(
            input_dir=SAMPLES_DIR,
            output_dir=tmp_output,
            start_date="2016-07-08",
            end_date="2016-07-09",
            fmt="csv",
        )
        out_files = list((tmp_output / "stream_graph").glob("*.parquet"))
        assert len(out_files) == 1
        df = pd.read_parquet(out_files[0])
        assert len(df) > 0
        assert list(df.columns) == ["src_idx", "dst_idx", "timestamp", "btc", "usd"]
        assert (df["timestamp"].diff().dropna() >= 0).all()


class TestStepDownload:
    """Tests for download step (mocked)."""

    def test_skip_existing(self, tmp_path):
        from src.build_stream_graph import step_download, STREAM_GRAPH_FILENAME
        fake_archive = tmp_path / STREAM_GRAPH_FILENAME
        fake_archive.write_bytes(b"fake")
        step_download(tmp_path)


class TestStepExtract:
    """Tests for extract step."""

    def test_missing_archive(self, tmp_path):
        from src.build_stream_graph import step_extract
        with pytest.raises(SystemExit):
            step_extract(tmp_path, year=2020)

    def test_skip_existing(self, tmp_path):
        from src.build_stream_graph import step_extract, STREAM_GRAPH_FILENAME
        edges_dir = tmp_path / "STREAM_GRAPH" / "EDGES"
        edges_dir.mkdir(parents=True)
        (edges_dir / "something_2020_something.parquet").write_bytes(b"fake")
        fake_archive = tmp_path / STREAM_GRAPH_FILENAME
        fake_archive.write_bytes(b"fake")
        step_extract(tmp_path, year=2020)

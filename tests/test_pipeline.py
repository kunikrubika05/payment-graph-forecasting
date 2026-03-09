"""
Tests for build_pipeline.py using CSV samples.

Run: pytest tests/test_pipeline.py -v
"""

import shutil
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Add src to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from build_pipeline import (
    _collect_entity_ids_from_csv,
    _extract_date,
    _process_single_snapshot,
    step_mapping,
    step_snapshots,
)

SAMPLES_DIR = Path(__file__).parent.parent / "data" / "samples"
SNAPSHOT_08 = SAMPLES_DIR / "orbitaal-snapshot-2016_07_08.csv"
SNAPSHOT_09 = SAMPLES_DIR / "orbitaal-snapshot-2016_07_09.csv"

# Skip all tests if CSV samples are not present
pytestmark = pytest.mark.skipif(
    not SNAPSHOT_08.exists(), reason="CSV samples not found in data/samples/"
)


class TestExtractDate:
    def test_csv_pattern(self):
        assert _extract_date("orbitaal-snapshot-2016_07_08") == "2016-07-08"

    def test_parquet_pattern(self):
        assert _extract_date("orbitaal-snapshot-date-2016-07-08-file-id-123.snappy") == "2016-07-08"

    def test_unknown_pattern(self):
        assert _extract_date("random_file") is None


class TestCollectEntityIds:
    def test_returns_set_of_ints(self):
        ids = _collect_entity_ids_from_csv(SNAPSHOT_08)
        assert isinstance(ids, set)
        assert len(ids) > 0
        assert all(isinstance(i, (int, np.integer)) for i in list(ids)[:10])

    def test_includes_src_and_dst(self):
        df = pd.read_csv(SNAPSHOT_08, nrows=100)
        ids = set(df["SRC_ID"]) | set(df["DST_ID"])
        collected = _collect_entity_ids_from_csv(SNAPSHOT_08)
        assert ids.issubset(collected)


class TestStepMapping:
    def test_builds_mapping_from_csv(self, tmp_path):
        mapping_df = step_mapping(
            input_dir=SAMPLES_DIR,
            output_dir=tmp_path,
            fmt="csv",
            n_workers=1,
        )

        assert isinstance(mapping_df, pd.DataFrame)
        assert "entity_id" in mapping_df.columns
        assert "node_index" in mapping_df.columns
        assert len(mapping_df) > 0

        # Indices are dense 0..N-1
        assert mapping_df["node_index"].min() == 0
        assert mapping_df["node_index"].max() == len(mapping_df) - 1

        # No duplicates
        assert mapping_df["entity_id"].is_unique
        assert mapping_df["node_index"].is_unique

        # Entity 0 excluded
        assert 0 not in mapping_df["entity_id"].values

        # File saved
        assert (tmp_path / "node_mapping.parquet").exists()

    def test_skip_if_exists(self, tmp_path):
        # First run
        step_mapping(SAMPLES_DIR, tmp_path, fmt="csv", n_workers=1)
        mtime1 = (tmp_path / "node_mapping.parquet").stat().st_mtime

        # Second run should skip
        step_mapping(SAMPLES_DIR, tmp_path, fmt="csv", n_workers=1)
        mtime2 = (tmp_path / "node_mapping.parquet").stat().st_mtime
        assert mtime1 == mtime2


class TestStepSnapshots:
    def test_builds_snapshots_from_csv(self, tmp_path):
        # Build mapping first
        step_mapping(SAMPLES_DIR, tmp_path, fmt="csv", n_workers=1)

        # Build snapshots
        step_snapshots(SAMPLES_DIR, tmp_path, fmt="csv", n_workers=1)

        # Check output files
        snapshot_dir = tmp_path / "daily_snapshots"
        assert snapshot_dir.exists()

        parquet_files = list(snapshot_dir.glob("*.parquet"))
        assert len(parquet_files) == 2  # 2016-07-08 and 2016-07-09

        # Check file names
        names = {f.stem for f in parquet_files}
        assert "2016-07-08" in names
        assert "2016-07-09" in names

        # Check stats file
        stats_path = tmp_path / "daily_stats.csv"
        assert stats_path.exists()
        stats_df = pd.read_csv(stats_path)
        assert len(stats_df) == 2

    def test_snapshot_content(self, tmp_path):
        step_mapping(SAMPLES_DIR, tmp_path, fmt="csv", n_workers=1)
        step_snapshots(SAMPLES_DIR, tmp_path, fmt="csv", n_workers=1)

        # Read a processed snapshot
        df = pd.read_parquet(tmp_path / "daily_snapshots" / "2016-07-08.parquet")

        assert "src_idx" in df.columns
        assert "dst_idx" in df.columns
        assert "btc" in df.columns

        # All indices are non-negative integers
        assert (df["src_idx"] >= 0).all()
        assert (df["dst_idx"] >= 0).all()

        # No self-loops
        assert (df["src_idx"] != df["dst_idx"]).all()

        # BTC values are positive
        assert (df["btc"] > 0).all()

    def test_global_mapping_consistency(self, tmp_path):
        """Same entity_id in both days maps to same node_index."""
        step_mapping(SAMPLES_DIR, tmp_path, fmt="csv", n_workers=1)
        step_snapshots(SAMPLES_DIR, tmp_path, fmt="csv", n_workers=1)

        mapping_df = pd.read_parquet(tmp_path / "node_mapping.parquet")
        entity_to_idx = dict(zip(mapping_df["entity_id"], mapping_df["node_index"]))

        # Load original CSVs and verify mapping
        for csv_file in [SNAPSHOT_08, SNAPSHOT_09]:
            raw = pd.read_csv(csv_file)
            raw = raw[(raw["SRC_ID"] != 0) & (raw["DST_ID"] != 0)]
            raw = raw[raw["SRC_ID"] != raw["DST_ID"]]

            date = _extract_date(csv_file.stem)
            processed = pd.read_parquet(tmp_path / "daily_snapshots" / f"{date}.parquet")

            # Check a sample of mappings
            sample = raw.head(100)
            for _, row in sample.iterrows():
                expected_src = entity_to_idx[row["SRC_ID"]]
                expected_dst = entity_to_idx[row["DST_ID"]]
                match = processed[
                    (processed["src_idx"] == expected_src)
                    & (processed["dst_idx"] == expected_dst)
                ]
                assert len(match) > 0, (
                    f"Mapping mismatch: SRC_ID={row['SRC_ID']} -> {expected_src}, "
                    f"DST_ID={row['DST_ID']} -> {expected_dst} not found in processed"
                )

    def test_idempotent(self, tmp_path):
        """Running snapshots twice doesn't duplicate data."""
        step_mapping(SAMPLES_DIR, tmp_path, fmt="csv", n_workers=1)
        step_snapshots(SAMPLES_DIR, tmp_path, fmt="csv", n_workers=1)
        step_snapshots(SAMPLES_DIR, tmp_path, fmt="csv", n_workers=1)

        parquet_files = list((tmp_path / "daily_snapshots").glob("*.parquet"))
        assert len(parquet_files) == 2

        stats_df = pd.read_csv(tmp_path / "daily_stats.csv")
        assert len(stats_df) == 2

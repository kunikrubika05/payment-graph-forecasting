"""Pipeline for building stream graph from ORBITAAL dataset.

Downloads the stream graph archive from Zenodo, extracts the relevant year,
filters to a target date range, applies global node mapping, and produces
a single chronologically sorted parquet file suitable for temporal GNN models
(TGN, DyGFormer, TGAT, GraphMixer, etc.).

Output format (one parquet file, sorted by timestamp):
    src_idx (int64)   - source node (global mapping)
    dst_idx (int64)   - destination node (global mapping)
    timestamp (int64) - UNIX timestamp of the transaction
    btc (float32)     - VALUE_SATOSHI / 1e8
    usd (float32)     - VALUE_USD

Usage:
    # Full pipeline on dev machine
    python src/build_stream_graph.py --steps download extract process upload \\
        --start-date 2020-06-01 --end-date 2020-08-31

    # Process only (archive already extracted)
    python src/build_stream_graph.py --steps process upload \\
        --start-date 2020-06-01 --end-date 2020-08-31

    # Test on CSV samples
    python src/build_stream_graph.py --steps process \\
        --input-dir data/samples --format csv \\
        --start-date 2016-07-08 --end-date 2016-07-09
"""

import argparse
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from src.yadisk_utils import (
    create_remote_folder_recursive,
    download_file,
    upload_file,
)

ROOT = Path(__file__).parent.parent
DEFAULT_RAW_DIR = ROOT / "data" / "raw"
DEFAULT_OUTPUT_DIR = ROOT / "data" / "processed"

ZENODO_RECORD = "12581515"
ZENODO_BASE = f"https://zenodo.org/records/{ZENODO_RECORD}/files"
STREAM_GRAPH_URL = f"{ZENODO_BASE}/orbitaal-stream_graph.tar.gz?download=1"
STREAM_GRAPH_FILENAME = "orbitaal-stream_graph.tar.gz"
STREAM_GRAPH_SIZE_GB = 23.9

REMOTE_BASE = "orbitaal_processed"


def step_download(raw_dir: Path):
    """Download stream graph archive from Zenodo using wget with resume.

    Args:
        raw_dir: Local directory to save the downloaded archive.
    """
    raw_dir.mkdir(parents=True, exist_ok=True)
    dest = raw_dir / STREAM_GRAPH_FILENAME

    if dest.exists():
        size_gb = dest.stat().st_size / 1e9
        print(f"[skip] {dest.name} already exists ({size_gb:.1f} GB)")
        return

    print(f"[download] {STREAM_GRAPH_FILENAME} ({STREAM_GRAPH_SIZE_GB} GB)...")
    cmd = ["wget", "-c", "-O", str(dest), STREAM_GRAPH_URL]
    subprocess.run(cmd, check=True)
    print(f"[done] {dest.name}")


def step_extract(raw_dir: Path, year: int = 2020):
    """Extract only the target year file from the stream graph archive.

    Uses tar --wildcards to extract only files matching the target year,
    saving significant disk space (~50-80 GB full vs ~3-5 GB for one year).

    Args:
        raw_dir: Directory containing the tar.gz archive.
        year: Year to extract (default 2020).
    """
    archive = raw_dir / STREAM_GRAPH_FILENAME
    if not archive.exists():
        print(f"[error] Archive not found: {archive}")
        sys.exit(1)

    extract_dir = raw_dir / "STREAM_GRAPH" / "EDGES"
    existing = list(extract_dir.glob(f"*{year}*")) if extract_dir.exists() else []
    if existing:
        print(f"[skip] Found {len(existing)} files for year {year} in {extract_dir}")
        return

    print(f"[extract] Extracting files for year {year} from {archive.name}...")
    print(f"  This extracts only *{year}* files, not the entire archive.")
    cmd = [
        "tar", "-xzf", str(archive),
        "-C", str(raw_dir),
        "--wildcards", f"*{year}*",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        if "Not found in archive" in result.stderr or result.returncode == 2:
            print(f"  [warn] tar --wildcards may not be supported, trying full extract + filter...")
            cmd_full = ["tar", "-xzf", str(archive), "-C", str(raw_dir)]
            subprocess.run(cmd_full, check=True)
        else:
            print(f"  [error] tar failed: {result.stderr}")
            sys.exit(1)

    extracted = list(extract_dir.glob(f"*{year}*")) if extract_dir.exists() else []
    print(f"[done] Extracted {len(extracted)} files for year {year}")


def _date_range_to_timestamps(start_date: str, end_date: str):
    """Convert date strings to UNIX timestamp range (inclusive, full days).

    Args:
        start_date: Start date in YYYY-MM-DD format.
        end_date: End date in YYYY-MM-DD format.

    Returns:
        Tuple of (start_ts, end_ts) as integers.
    """
    start_dt = datetime.strptime(start_date, "%Y-%m-%d")
    end_dt = datetime.strptime(end_date, "%Y-%m-%d").replace(
        hour=23, minute=59, second=59
    )
    return int(start_dt.timestamp()), int(end_dt.timestamp())


def _load_node_mapping(output_dir: Path, token: str = None) -> pd.Series:
    """Load node_mapping.parquet, downloading from Yandex.Disk if needed.

    Args:
        output_dir: Local output directory.
        token: Yandex.Disk OAuth token (for downloading).

    Returns:
        pandas Series: entity_id -> node_index.
    """
    mapping_path = output_dir / "node_mapping.parquet"

    if not mapping_path.exists() and token:
        print("[mapping] Downloading node_mapping.parquet from Yandex.Disk...")
        mapping_path.parent.mkdir(parents=True, exist_ok=True)
        success = download_file(
            f"{REMOTE_BASE}/node_mapping.parquet",
            str(mapping_path),
            token,
        )
        if not success:
            print("[error] Failed to download node_mapping.parquet")
            sys.exit(1)
        print("[mapping] Downloaded successfully")

    if not mapping_path.exists():
        print("[error] node_mapping.parquet not found and no YADISK_TOKEN to download it")
        sys.exit(1)

    print("[mapping] Loading node_mapping.parquet...")
    mapping_df = pd.read_parquet(mapping_path)
    entity_to_idx = pd.Series(
        mapping_df["node_index"].values,
        index=mapping_df["entity_id"].values,
    )
    print(f"[mapping] Loaded {len(entity_to_idx):,} entities "
          f"(~{entity_to_idx.nbytes / 1e9:.1f} GB)")
    return entity_to_idx


def step_process(
    input_dir: Path,
    output_dir: Path,
    start_date: str,
    end_date: str,
    fmt: str = "parquet",
    token: str = None,
):
    """Process stream graph files: filter by date, apply mapping, save.

    Reads raw stream graph parquet/csv files, filters to the target date range,
    removes entity 0 and self-loops, applies global node mapping, and saves
    a single chronologically sorted parquet file.

    Args:
        input_dir: Directory with raw stream graph files.
        output_dir: Directory for processed output.
        start_date: Start date (YYYY-MM-DD, inclusive).
        end_date: End date (YYYY-MM-DD, inclusive).
        fmt: Input format ("parquet" or "csv").
        token: Yandex.Disk OAuth token for downloading node_mapping.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    period_tag = f"{start_date}__{end_date}"
    out_path = output_dir / "stream_graph" / f"{period_tag}.parquet"

    if out_path.exists():
        size_mb = out_path.stat().st_size / 1e6
        print(f"[skip] {out_path} already exists ({size_mb:.1f} MB)")
        return

    entity_to_idx = _load_node_mapping(output_dir, token)

    if fmt == "parquet":
        files = sorted(input_dir.rglob("*stream_graph*.parquet"))
    else:
        files = sorted(input_dir.glob("*stream_graph*.csv"))

    if not files:
        print(f"[error] No stream graph files found in {input_dir}")
        sys.exit(1)

    print(f"[process] Found {len(files)} stream graph files")
    print(f"[process] Filtering to {start_date} .. {end_date}")

    start_ts, end_ts = _date_range_to_timestamps(start_date, end_date)
    print(f"[process] Timestamp range: {start_ts} .. {end_ts}")

    chunks = []
    total_raw = 0
    total_filtered = 0

    for i, filepath in enumerate(files, 1):
        t0 = time.time()

        if fmt == "parquet":
            df = pq.read_table(filepath).to_pandas()
        else:
            df = pd.read_csv(filepath)

        raw_count = len(df)
        total_raw += raw_count

        if "TIMESTAMP" in df.columns:
            df = df[(df["TIMESTAMP"] >= start_ts) & (df["TIMESTAMP"] <= end_ts)]
        else:
            print(f"  [warn] {filepath.name}: no TIMESTAMP column, skipping date filter")

        df = df[(df["SRC_ID"] != 0) & (df["DST_ID"] != 0)]
        df = df[df["SRC_ID"] != df["DST_ID"]]

        if len(df) == 0:
            elapsed = time.time() - t0
            print(f"  [{i}/{len(files)}] {filepath.name}: {raw_count:,} raw -> 0 after filter "
                  f"({elapsed:.1f}s)")
            continue

        df["src_idx"] = df["SRC_ID"].map(entity_to_idx)
        df["dst_idx"] = df["DST_ID"].map(entity_to_idx)

        before = len(df)
        df = df.dropna(subset=["src_idx", "dst_idx"])
        if len(df) < before:
            print(f"  [warn] Dropped {before - len(df)} unmapped edges")

        df["src_idx"] = df["src_idx"].astype(np.int64)
        df["dst_idx"] = df["dst_idx"].astype(np.int64)
        df["timestamp"] = df["TIMESTAMP"].astype(np.int64)
        df["btc"] = (df["VALUE_SATOSHI"] / 1e8).astype(np.float32)
        df["usd"] = df["VALUE_USD"].astype(np.float32)

        result = df[["src_idx", "dst_idx", "timestamp", "btc", "usd"]]
        chunks.append(result)
        total_filtered += len(result)

        elapsed = time.time() - t0
        print(f"  [{i}/{len(files)}] {filepath.name}: {raw_count:,} raw -> {len(result):,} "
              f"({elapsed:.1f}s)")

    if not chunks:
        print("[error] No edges found in the specified date range")
        sys.exit(1)

    print(f"\n[process] Concatenating {len(chunks)} chunks ({total_filtered:,} edges total)...")
    result_df = pd.concat(chunks, ignore_index=True)

    print("[process] Sorting by timestamp...")
    result_df.sort_values("timestamp", kind="mergesort", inplace=True)
    result_df.reset_index(drop=True, inplace=True)

    unique_src = result_df["src_idx"].nunique()
    unique_dst = result_df["dst_idx"].nunique()
    unique_nodes = len(
        np.union1d(result_df["src_idx"].values, result_df["dst_idx"].values)
    )
    ts_min = result_df["timestamp"].min()
    ts_max = result_df["timestamp"].max()

    print(f"\n[process] Stream graph summary:")
    print(f"  Period: {start_date} to {end_date}")
    print(f"  Edges: {len(result_df):,}")
    print(f"  Unique nodes: {unique_nodes:,}")
    print(f"  Unique sources: {unique_src:,}")
    print(f"  Unique destinations: {unique_dst:,}")
    print(f"  Timestamp range: {ts_min} .. {ts_max}")
    print(f"  Total BTC: {result_df['btc'].sum():,.2f}")
    print(f"  Total USD: {result_df['usd'].sum():,.2f}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    result_df.to_parquet(out_path, index=False)
    size_mb = out_path.stat().st_size / 1e6
    print(f"\n[process] Saved to {out_path} ({size_mb:.1f} MB)")

    stats = {
        "period": period_tag,
        "start_date": start_date,
        "end_date": end_date,
        "num_edges": len(result_df),
        "num_nodes": unique_nodes,
        "timestamp_min": int(ts_min),
        "timestamp_max": int(ts_max),
        "total_btc": float(result_df["btc"].sum()),
        "total_usd": float(result_df["usd"].sum()),
    }

    import json
    stats_path = out_path.with_suffix(".json")
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"[process] Stats saved to {stats_path}")


def step_upload(output_dir: Path, start_date: str, end_date: str):
    """Upload processed stream graph to Yandex.Disk.

    Args:
        output_dir: Local directory with processed files.
        start_date: Start date (for filename).
        end_date: End date (for filename).
    """
    token = os.environ.get("YADISK_TOKEN")
    if not token:
        print("[error] YADISK_TOKEN environment variable not set")
        sys.exit(1)

    period_tag = f"{start_date}__{end_date}"
    remote_dir = f"{REMOTE_BASE}/stream_graph"

    create_remote_folder_recursive(remote_dir, token)

    for suffix in [".parquet", ".json"]:
        local_path = output_dir / "stream_graph" / f"{period_tag}{suffix}"
        if local_path.exists():
            remote_path = f"{remote_dir}/{period_tag}{suffix}"
            print(f"[upload] {local_path.name} -> {remote_path}")
            success = upload_file(str(local_path), remote_path, token)
            if success:
                print(f"[upload] Done: {local_path.name}")
            else:
                print(f"[upload] FAILED: {local_path.name}")


def main():
    """CLI entry point for the stream graph pipeline."""
    parser = argparse.ArgumentParser(
        description="ORBITAAL stream graph pipeline"
    )
    parser.add_argument(
        "--steps", nargs="+", required=True,
        choices=["download", "extract", "process", "upload"],
        help="Pipeline steps to run",
    )
    parser.add_argument(
        "--input-dir", type=Path, default=DEFAULT_RAW_DIR,
        help="Directory with raw data",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR,
        help="Directory for processed output",
    )
    parser.add_argument(
        "--start-date", type=str, default="2020-06-01",
        help="Start date (YYYY-MM-DD, inclusive)",
    )
    parser.add_argument(
        "--end-date", type=str, default="2020-08-31",
        help="End date (YYYY-MM-DD, inclusive)",
    )
    parser.add_argument(
        "--format", choices=["parquet", "csv"], default="parquet",
        help="Input file format",
    )
    parser.add_argument(
        "--year", type=int, default=2020,
        help="Year to extract from archive (for extract step)",
    )

    args = parser.parse_args()
    token = os.environ.get("YADISK_TOKEN")

    print(f"Input:  {args.input_dir}")
    print(f"Output: {args.output_dir}")
    print(f"Format: {args.format}")
    print(f"Period: {args.start_date} to {args.end_date}")
    print()

    t0 = time.time()

    for step in args.steps:
        step_t0 = time.time()
        print(f"{'='*60}")
        print(f"STEP: {step}")
        print(f"{'='*60}")

        if step == "download":
            step_download(args.input_dir)
        elif step == "extract":
            step_extract(args.input_dir, args.year)
        elif step == "process":
            step_process(
                args.input_dir, args.output_dir,
                args.start_date, args.end_date,
                args.format, token,
            )
        elif step == "upload":
            step_upload(args.output_dir, args.start_date, args.end_date)

        elapsed = time.time() - step_t0
        print(f"[{step}] completed in {elapsed:.1f}s\n")

    total = time.time() - t0
    print(f"Pipeline finished in {total:.1f}s")


if __name__ == "__main__":
    main()

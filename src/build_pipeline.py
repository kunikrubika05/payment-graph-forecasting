"""
Pipeline for building payment graphs from ORBITAAL dataset.

Usage:
    # Full pipeline on dev machine (download from Zenodo, build, upload to Yandex.Disk):
    python build_pipeline.py --steps download extract mapping snapshots upload

    # Build from local CSV samples (for testing):
    python build_pipeline.py --steps mapping snapshots --input-dir ../data/samples --format csv

    # Individual steps:
    python build_pipeline.py --steps download --zenodo-files snapshot-day nodetable
    python build_pipeline.py --steps extract
    python build_pipeline.py --steps mapping
    python build_pipeline.py --steps snapshots
    python build_pipeline.py --steps upload
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).parent.parent
DEFAULT_RAW_DIR = ROOT / "data" / "raw"
DEFAULT_OUTPUT_DIR = ROOT / "data" / "processed"

ZENODO_RECORD = "12581515"
ZENODO_BASE = f"https://zenodo.org/records/{ZENODO_RECORD}/files"

ZENODO_FILES = {
    "snapshot-day": {
        "url": f"{ZENODO_BASE}/orbitaal-snapshot-day.tar.gz?download=1",
        "filename": "orbitaal-snapshot-day.tar.gz",
        "size_gb": 24.8,
    },
    "nodetable": {
        "url": f"{ZENODO_BASE}/orbitaal-nodetable.tar.gz?download=1",
        "filename": "orbitaal-nodetable.tar.gz",
        "size_gb": 24.9,
    },
    "stream-graph": {
        "url": f"{ZENODO_BASE}/orbitaal-stream_graph.tar.gz?download=1",
        "filename": "orbitaal-stream_graph.tar.gz",
        "size_gb": 23.9,
    },
}

# CSV samples (small files for testing)
CSV_SAMPLES = {
    "stream-08": f"{ZENODO_BASE}/orbitaal-stream_graph-2016_07_08.csv?download=1",
    "stream-09": f"{ZENODO_BASE}/orbitaal-stream_graph-2016_07_09.csv?download=1",
    "snapshot-08": f"{ZENODO_BASE}/orbitaal-snapshot-2016_07_08.csv?download=1",
    "snapshot-09": f"{ZENODO_BASE}/orbitaal-snapshot-2016_07_09.csv?download=1",
}


# ---------------------------------------------------------------------------
# Step 1: Download
# ---------------------------------------------------------------------------

def step_download(raw_dir: Path, zenodo_files: list[str]):
    """Download files from Zenodo using wget with resume support."""
    raw_dir.mkdir(parents=True, exist_ok=True)

    for key in zenodo_files:
        if key not in ZENODO_FILES:
            print(f"Unknown file key: {key}. Available: {list(ZENODO_FILES.keys())}")
            continue

        info = ZENODO_FILES[key]
        dest = raw_dir / info["filename"]

        if dest.exists():
            print(f"[skip] {dest.name} already exists ({dest.stat().st_size / 1e9:.1f} GB)")
            continue

        print(f"[download] {info['filename']} ({info['size_gb']} GB)...")
        cmd = ["wget", "-c", "-O", str(dest), info["url"]]
        subprocess.run(cmd, check=True)
        print(f"[done] {dest.name}")


# ---------------------------------------------------------------------------
# Step 2: Extract
# ---------------------------------------------------------------------------

def step_extract(raw_dir: Path):
    """Extract tar.gz archives."""
    for archive in sorted(raw_dir.glob("*.tar.gz")):
        print(f"[extract] {archive.name}...")
        subprocess.run(
            ["tar", "-xzf", str(archive), "-C", str(raw_dir)],
            check=True,
        )
        print(f"[done] {archive.name}")


# ---------------------------------------------------------------------------
# Step 3: Build global node mapping
# ---------------------------------------------------------------------------

def _collect_entity_ids_from_parquet(filepath: Path) -> set[int]:
    """Read a single parquet file and return set of unique entity IDs."""
    df = pq.read_table(filepath, columns=["SRC_ID", "DST_ID"]).to_pandas()
    src = set(df["SRC_ID"].unique())
    dst = set(df["DST_ID"].unique())
    return src | dst


def _collect_entity_ids_from_csv(filepath: Path) -> set[int]:
    """Read a single CSV file and return set of unique entity IDs."""
    df = pd.read_csv(filepath, usecols=["SRC_ID", "DST_ID"])
    src = set(df["SRC_ID"].unique())
    dst = set(df["DST_ID"].unique())
    return src | dst


def step_mapping(input_dir: Path, output_dir: Path, fmt: str = "parquet",
                 n_workers: int = 4):
    """Build global entity_id -> node_index mapping.

    Scans all snapshot files, collects unique entity IDs,
    assigns dense indices 0..N-1, saves mapping as parquet.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    mapping_path = output_dir / "node_mapping.parquet"

    if mapping_path.exists():
        print(f"[skip] {mapping_path} already exists")
        return pd.read_parquet(mapping_path)

    # Find snapshot files
    if fmt == "parquet":
        # Look in SNAPSHOT/EDGES/day/ directory structure from ORBITAAL tar.gz
        files = sorted(input_dir.rglob("*snapshot*.parquet"))
    else:
        files = sorted(input_dir.glob("*snapshot*.csv"))

    if not files:
        print(f"[error] No snapshot files found in {input_dir}")
        sys.exit(1)

    print(f"[mapping] Scanning {len(files)} files for unique entity IDs...")
    all_ids = set()

    collect_fn = _collect_entity_ids_from_parquet if fmt == "parquet" else _collect_entity_ids_from_csv

    if n_workers > 1 and len(files) > 1:
        with ProcessPoolExecutor(max_workers=n_workers) as pool:
            futures = {pool.submit(collect_fn, f): f for f in files}
            for i, future in enumerate(as_completed(futures), 1):
                ids = future.result()
                all_ids |= ids
                if i % 100 == 0 or i == len(files):
                    print(f"  [{i}/{len(files)}] unique IDs so far: {len(all_ids):,}")
    else:
        for i, f in enumerate(files, 1):
            ids = collect_fn(f)
            all_ids |= ids
            if i % 10 == 0 or i == len(files):
                print(f"  [{i}/{len(files)}] unique IDs so far: {len(all_ids):,}")

    # Remove entity 0 (special)
    all_ids.discard(0)

    # Sort for deterministic mapping
    sorted_ids = sorted(all_ids)
    mapping_df = pd.DataFrame({
        "entity_id": sorted_ids,
        "node_index": range(len(sorted_ids)),
    })

    mapping_df.to_parquet(mapping_path, index=False)
    print(f"[mapping] Saved {len(mapping_df):,} entities -> {mapping_path}")

    return mapping_df


# ---------------------------------------------------------------------------
# Step 4: Build daily snapshots
# ---------------------------------------------------------------------------

def _process_single_snapshot(args):
    """Process one snapshot file: apply mapping, compute stats, save."""
    filepath, mapping_path, output_dir, fmt = args

    # Load mapping
    mapping_df = pd.read_parquet(mapping_path)
    entity_to_idx = dict(zip(mapping_df["entity_id"], mapping_df["node_index"]))

    # Read snapshot
    if fmt == "parquet":
        df = pq.read_table(filepath).to_pandas()
    else:
        df = pd.read_csv(filepath)

    # Extract date from filename
    name = filepath.stem
    # Try to parse date from filename patterns:
    #   parquet: orbitaal-snapshot-date-YYYY-MM-DD-file-id-N.snappy
    #   csv: orbitaal-snapshot-2016_07_08
    date_str = _extract_date(name)
    if not date_str:
        date_str = name  # fallback

    # Clean: remove self-loops and entity 0
    df = df[(df["SRC_ID"] != 0) & (df["DST_ID"] != 0)]
    df = df[df["SRC_ID"] != df["DST_ID"]]

    # Apply mapping
    df["src_idx"] = df["SRC_ID"].map(entity_to_idx)
    df["dst_idx"] = df["DST_ID"].map(entity_to_idx)

    # Drop rows where mapping failed (shouldn't happen, but safety)
    before = len(df)
    df = df.dropna(subset=["src_idx", "dst_idx"])
    if len(df) < before:
        print(f"  [warn] {date_str}: dropped {before - len(df)} unmapped edges")

    df["src_idx"] = df["src_idx"].astype(np.int64)
    df["dst_idx"] = df["dst_idx"].astype(np.int64)
    df["btc"] = (df["VALUE_SATOSHI"] / 1e8).astype(np.float32)
    if "VALUE_USD" in df.columns:
        df["usd"] = df["VALUE_USD"].astype(np.float32)
        out_df = df[["src_idx", "dst_idx", "btc", "usd"]]
    else:
        out_df = df[["src_idx", "dst_idx", "btc"]]

    # Save
    out_path = output_dir / "daily_snapshots" / f"{date_str}.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_parquet(out_path, index=False)

    # Stats
    unique_nodes = set(out_df["src_idx"]) | set(out_df["dst_idx"])
    stats = {
        "date": date_str,
        "num_nodes": len(unique_nodes),
        "num_edges": len(out_df),
        "total_btc": float(out_df["btc"].sum()),
    }
    if "usd" in out_df.columns:
        stats["total_usd"] = float(out_df["usd"].sum())

    return stats


def _extract_date(filename: str) -> str | None:
    """Extract date string from ORBITAAL filename patterns."""
    import re

    # CSV pattern: orbitaal-snapshot-2016_07_08
    m = re.search(r"(\d{4})_(\d{2})_(\d{2})", filename)
    if m:
        return f"{m.group(1)}-{m.group(2)}-{m.group(3)}"

    # Parquet pattern: orbitaal-snapshot-date-2016-07-08-file-id-123
    m = re.search(r"date-(\d{4})-(\d{2})-(\d{2})", filename)
    if m:
        return f"{m.group(1)}-{m.group(2)}-{m.group(3)}"

    return None


def step_snapshots(input_dir: Path, output_dir: Path, fmt: str = "parquet",
                   n_workers: int = 4):
    """Build daily snapshot parquet files with global node indices."""
    mapping_path = output_dir / "node_mapping.parquet"
    if not mapping_path.exists():
        print("[error] node_mapping.parquet not found. Run 'mapping' step first.")
        sys.exit(1)

    # Find snapshot files
    if fmt == "parquet":
        files = sorted(input_dir.rglob("*snapshot*.parquet"))
    else:
        files = sorted(input_dir.glob("*snapshot*.csv"))

    if not files:
        print(f"[error] No snapshot files found in {input_dir}")
        sys.exit(1)

    snapshot_dir = output_dir / "daily_snapshots"
    snapshot_dir.mkdir(parents=True, exist_ok=True)

    # Check which dates are already processed
    existing = {f.stem for f in snapshot_dir.glob("*.parquet")}
    tasks = []
    for f in files:
        date_str = _extract_date(f.stem)
        if date_str and date_str in existing:
            continue
        tasks.append((f, mapping_path, output_dir, fmt))

    if not tasks:
        print(f"[skip] All {len(files)} snapshots already processed")
        return

    print(f"[snapshots] Processing {len(tasks)} files ({len(existing)} already done)...")

    all_stats = []
    if n_workers > 1 and len(tasks) > 1:
        with ProcessPoolExecutor(max_workers=n_workers) as pool:
            futures = {pool.submit(_process_single_snapshot, t): t for t in tasks}
            for i, future in enumerate(as_completed(futures), 1):
                stats = future.result()
                all_stats.append(stats)
                if i % 100 == 0 or i == len(tasks):
                    print(f"  [{i}/{len(tasks)}] last: {stats['date']} "
                          f"({stats['num_nodes']:,} nodes, {stats['num_edges']:,} edges)")
    else:
        for i, t in enumerate(tasks, 1):
            stats = _process_single_snapshot(t)
            all_stats.append(stats)
            print(f"  [{i}/{len(tasks)}] {stats['date']} "
                  f"({stats['num_nodes']:,} nodes, {stats['num_edges']:,} edges)")

    # Save/update stats
    stats_path = output_dir / "daily_stats.csv"
    new_stats_df = pd.DataFrame(all_stats)
    if stats_path.exists():
        old_stats_df = pd.read_csv(stats_path)
        stats_df = pd.concat([old_stats_df, new_stats_df]).drop_duplicates(subset="date")
    else:
        stats_df = new_stats_df
    stats_df.sort_values("date").to_csv(stats_path, index=False)
    print(f"[snapshots] Stats saved to {stats_path}")


# ---------------------------------------------------------------------------
# Step 5: Upload to Yandex.Disk
# ---------------------------------------------------------------------------

def step_upload(output_dir: Path, remote_path: str = "orbitaal_processed"):
    """Upload processed data to Yandex.Disk via REST API."""
    token = os.environ.get("YADISK_TOKEN")
    if not token:
        print("[error] YADISK_TOKEN environment variable not set")
        sys.exit(1)

    import urllib.request
    import urllib.parse

    api_base = "https://cloud-api.yandex.net/v1/disk/resources"
    headers = {"Authorization": f"OAuth {token}"}

    def yadisk_request(url, method="GET", data=None):
        req = urllib.request.Request(url, headers=headers, method=method, data=data)
        with urllib.request.urlopen(req) as resp:
            if resp.status in (200, 201):
                return json.loads(resp.read())
        return None

    def ensure_folder(path):
        url = f"{api_base}?path={urllib.parse.quote(path)}"
        try:
            yadisk_request(url, method="PUT")
        except urllib.error.HTTPError:
            pass  # folder may already exist

    def upload_file(local_path: Path, remote_file_path: str):
        url = (f"{api_base}/upload"
               f"?path={urllib.parse.quote(remote_file_path)}"
               f"&overwrite=true")
        resp = yadisk_request(url)
        if not resp or "href" not in resp:
            print(f"  [error] Failed to get upload URL for {remote_file_path}")
            return False

        upload_url = resp["href"]
        with open(local_path, "rb") as f:
            req = urllib.request.Request(upload_url, data=f.read(), method="PUT")
            urllib.request.urlopen(req)
        return True

    # Create remote folders
    ensure_folder(remote_path)
    ensure_folder(f"{remote_path}/daily_snapshots")

    # Upload node_mapping
    mapping_file = output_dir / "node_mapping.parquet"
    if mapping_file.exists():
        print(f"[upload] {mapping_file.name}...")
        upload_file(mapping_file, f"{remote_path}/node_mapping.parquet")

    # Upload stats
    stats_file = output_dir / "daily_stats.csv"
    if stats_file.exists():
        print(f"[upload] {stats_file.name}...")
        upload_file(stats_file, f"{remote_path}/daily_stats.csv")

    # Upload daily snapshots
    snapshot_dir = output_dir / "daily_snapshots"
    if snapshot_dir.exists():
        files = sorted(snapshot_dir.glob("*.parquet"))
        print(f"[upload] {len(files)} daily snapshots...")
        for i, f in enumerate(files, 1):
            upload_file(f, f"{remote_path}/daily_snapshots/{f.name}")
            if i % 100 == 0 or i == len(files):
                print(f"  [{i}/{len(files)}]")

    print("[upload] Done")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="ORBITAAL graph building pipeline")
    parser.add_argument(
        "--steps", nargs="+", required=True,
        choices=["download", "extract", "mapping", "snapshots", "upload"],
        help="Pipeline steps to run",
    )
    parser.add_argument(
        "--input-dir", type=Path, default=DEFAULT_RAW_DIR,
        help="Directory with raw data (parquet/csv files)",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR,
        help="Directory for processed output",
    )
    parser.add_argument(
        "--format", choices=["parquet", "csv"], default="parquet",
        help="Input file format",
    )
    parser.add_argument(
        "--zenodo-files", nargs="+", default=["snapshot-day"],
        choices=list(ZENODO_FILES.keys()),
        help="Which Zenodo files to download",
    )
    parser.add_argument(
        "--workers", type=int, default=4,
        help="Number of parallel workers",
    )

    args = parser.parse_args()

    print(f"Input:  {args.input_dir}")
    print(f"Output: {args.output_dir}")
    print(f"Format: {args.format}")
    print()

    t0 = time.time()

    for step in args.steps:
        step_t0 = time.time()
        print(f"{'='*60}")
        print(f"STEP: {step}")
        print(f"{'='*60}")

        if step == "download":
            step_download(args.input_dir, args.zenodo_files)
        elif step == "extract":
            step_extract(args.input_dir)
        elif step == "mapping":
            step_mapping(args.input_dir, args.output_dir, args.format, args.workers)
        elif step == "snapshots":
            step_snapshots(args.input_dir, args.output_dir, args.format, args.workers)
        elif step == "upload":
            step_upload(args.output_dir)

        elapsed = time.time() - step_t0
        print(f"[{step}] completed in {elapsed:.1f}s\n")

    total = time.time() - t0
    print(f"Pipeline finished in {total:.1f}s")


if __name__ == "__main__":
    main()

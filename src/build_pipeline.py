"""Pipeline for building payment graphs from ORBITAAL dataset.

Downloads raw data from Zenodo, builds a global node mapping,
produces daily snapshot parquet files with dense node indices,
and uploads results to Yandex.Disk.

Usage:
    python build_pipeline.py --steps download extract mapping snapshots upload
    python build_pipeline.py --steps mapping snapshots --input-dir ../data/samples --format csv
    python build_pipeline.py --steps download --zenodo-files snapshot-day nodetable
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

CSV_SAMPLES = {
    "stream-08": f"{ZENODO_BASE}/orbitaal-stream_graph-2016_07_08.csv?download=1",
    "stream-09": f"{ZENODO_BASE}/orbitaal-stream_graph-2016_07_09.csv?download=1",
    "snapshot-08": f"{ZENODO_BASE}/orbitaal-snapshot-2016_07_08.csv?download=1",
    "snapshot-09": f"{ZENODO_BASE}/orbitaal-snapshot-2016_07_09.csv?download=1",
}


def step_download(raw_dir: Path, zenodo_files: list[str]):
    """Download files from Zenodo using wget with resume support.

    Args:
        raw_dir: Local directory to save downloaded files.
        zenodo_files: List of file keys from ZENODO_FILES to download.
    """
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


def step_extract(raw_dir: Path):
    """Extract all tar.gz archives in the raw data directory.

    Args:
        raw_dir: Directory containing tar.gz files.
    """
    for archive in sorted(raw_dir.glob("*.tar.gz")):
        print(f"[extract] {archive.name}...")
        subprocess.run(
            ["tar", "-xzf", str(archive), "-C", str(raw_dir)],
            check=True,
        )
        print(f"[done] {archive.name}")


def _collect_entity_ids_from_parquet(filepath: Path) -> np.ndarray:
    """Read a single parquet snapshot and return unique entity IDs."""
    df = pq.read_table(filepath, columns=["SRC_ID", "DST_ID"]).to_pandas()
    return np.unique(np.concatenate([df["SRC_ID"].values, df["DST_ID"].values]))


def _collect_entity_ids_from_csv(filepath: Path) -> np.ndarray:
    """Read a single CSV snapshot and return unique entity IDs."""
    df = pd.read_csv(filepath, usecols=["SRC_ID", "DST_ID"])
    return np.unique(np.concatenate([df["SRC_ID"].values, df["DST_ID"].values]))


def step_mapping(input_dir: Path, output_dir: Path, fmt: str = "parquet",
                 n_workers: int = 4):
    """Build global entity_id -> node_index mapping.

    Scans all snapshot files, collects unique entity IDs, assigns dense
    indices 0..N-1, and saves the mapping as a parquet file. Entity 0
    (coinbase/special) is excluded.

    Uses numpy arrays with batch processing for memory efficiency:
    320M int64 in numpy ≈ 2.5 GB vs ≈ 25 GB in a Python set.

    Args:
        input_dir: Directory containing raw snapshot files.
        output_dir: Directory to save node_mapping.parquet.
        fmt: Input format, "parquet" or "csv".
        n_workers: Unused (kept for CLI compatibility).

    Returns:
        DataFrame with columns (entity_id, node_index).
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    mapping_path = output_dir / "node_mapping.parquet"

    if mapping_path.exists():
        print(f"[skip] {mapping_path} already exists")
        return pd.read_parquet(mapping_path)

    if fmt == "parquet":
        files = sorted(input_dir.rglob("*snapshot*.parquet"))
    else:
        files = sorted(input_dir.glob("*snapshot*.csv"))

    if not files:
        print(f"[error] No snapshot files found in {input_dir}")
        sys.exit(1)

    print(f"[mapping] Scanning {len(files)} files for unique entity IDs...")

    collect_fn = _collect_entity_ids_from_parquet if fmt == "parquet" else _collect_entity_ids_from_csv

    BATCH_LIMIT = 2 * 1024**3
    batch_ids = []
    batch_bytes = 0
    unique_ids = np.array([], dtype=np.int64)

    for i, f in enumerate(files, 1):
        file_ids = collect_fn(f)
        batch_ids.append(file_ids)
        batch_bytes += file_ids.nbytes

        if batch_bytes >= BATCH_LIMIT or i == len(files):
            merged = np.concatenate(batch_ids)
            new_unique = np.unique(merged)
            if len(unique_ids) > 0:
                unique_ids = np.unique(np.concatenate([unique_ids, new_unique]))
            else:
                unique_ids = new_unique
            batch_ids = []
            batch_bytes = 0
            print(f"  [{i}/{len(files)}] unique IDs so far: {len(unique_ids):,} "
                  f"(~{unique_ids.nbytes / 1e9:.1f} GB)")
        elif i % 200 == 0:
            print(f"  [{i}/{len(files)}] processing...")

    unique_ids = unique_ids[unique_ids != 0]

    mapping_df = pd.DataFrame({
        "entity_id": unique_ids,
        "node_index": np.arange(len(unique_ids), dtype=np.int64),
    })

    mapping_df.to_parquet(mapping_path, index=False)
    print(f"[mapping] Saved {len(mapping_df):,} entities -> {mapping_path}")

    return mapping_df


def _extract_date(filename: str) -> str | None:
    """Extract YYYY-MM-DD date string from an ORBITAAL filename.

    Supports two patterns:
        - CSV: orbitaal-snapshot-2016_07_08 -> 2016-07-08
        - Parquet: orbitaal-snapshot-date-2016-07-08-file-id-123 -> 2016-07-08

    Args:
        filename: Filename stem (without extension).

    Returns:
        Date string in YYYY-MM-DD format, or None if no pattern matches.
    """
    import re

    m = re.search(r"(\d{4})_(\d{2})_(\d{2})", filename)
    if m:
        return f"{m.group(1)}-{m.group(2)}-{m.group(3)}"

    m = re.search(r"date-(\d{4})-(\d{2})-(\d{2})", filename)
    if m:
        return f"{m.group(1)}-{m.group(2)}-{m.group(3)}"

    return None


def _process_single_snapshot(filepath, entity_to_idx, output_dir, fmt):
    """Process one snapshot file: apply global mapping, compute stats, save.

    Args:
        filepath: Path to the raw snapshot file.
        entity_to_idx: pandas Series mapping entity_id -> node_index.
        output_dir: Base output directory (daily_snapshots/ subdirectory is used).
        fmt: Input format, "parquet" or "csv".

    Returns:
        Dict with daily statistics (date, num_nodes, num_edges, total_btc, total_usd).
    """
    if fmt == "parquet":
        df = pq.read_table(filepath).to_pandas()
    else:
        df = pd.read_csv(filepath)

    date_str = _extract_date(filepath.stem)
    if not date_str:
        date_str = filepath.stem

    df = df[(df["SRC_ID"] != 0) & (df["DST_ID"] != 0)]
    df = df[df["SRC_ID"] != df["DST_ID"]]

    df["src_idx"] = df["SRC_ID"].map(entity_to_idx)
    df["dst_idx"] = df["DST_ID"].map(entity_to_idx)

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

    out_path = output_dir / "daily_snapshots" / f"{date_str}.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_parquet(out_path, index=False)

    unique_nodes = len(np.union1d(out_df["src_idx"].values, out_df["dst_idx"].values))
    stats = {
        "date": date_str,
        "num_nodes": unique_nodes,
        "num_edges": len(out_df),
        "total_btc": float(out_df["btc"].sum()),
    }
    if "usd" in out_df.columns:
        stats["total_usd"] = float(out_df["usd"].sum())

    return stats


def step_snapshots(input_dir: Path, output_dir: Path, fmt: str = "parquet",
                   n_workers: int = 4):
    """Build daily snapshot parquet files with global node indices.

    Loads the global mapping once as a pandas Series (~5 GB for 320M entities)
    and processes files sequentially to avoid duplicating the mapping across
    worker processes.

    Args:
        input_dir: Directory containing raw snapshot files.
        output_dir: Directory with node_mapping.parquet and for daily_snapshots/ output.
        fmt: Input format, "parquet" or "csv".
        n_workers: Unused (sequential processing for memory safety).
    """
    mapping_path = output_dir / "node_mapping.parquet"
    if not mapping_path.exists():
        print("[error] node_mapping.parquet not found. Run 'mapping' step first.")
        sys.exit(1)

    if fmt == "parquet":
        files = sorted(input_dir.rglob("*snapshot*.parquet"))
    else:
        files = sorted(input_dir.glob("*snapshot*.csv"))

    if not files:
        print(f"[error] No snapshot files found in {input_dir}")
        sys.exit(1)

    snapshot_dir = output_dir / "daily_snapshots"
    snapshot_dir.mkdir(parents=True, exist_ok=True)

    existing = {f.stem for f in snapshot_dir.glob("*.parquet")}
    pending_files = []
    for f in files:
        date_str = _extract_date(f.stem)
        if date_str and date_str in existing:
            continue
        pending_files.append(f)

    if not pending_files:
        print(f"[skip] All {len(files)} snapshots already processed")
        return

    print(f"[snapshots] Loading mapping...")
    mapping_df = pd.read_parquet(mapping_path)
    entity_to_idx = pd.Series(
        mapping_df["node_index"].values,
        index=mapping_df["entity_id"].values,
    )
    print(f"[snapshots] Mapping loaded: {len(entity_to_idx):,} entities "
          f"(~{entity_to_idx.nbytes / 1e9:.1f} GB)")

    print(f"[snapshots] Processing {len(pending_files)} files ({len(existing)} already done)...")

    all_stats = []
    for i, f in enumerate(pending_files, 1):
        stats = _process_single_snapshot(f, entity_to_idx, output_dir, fmt)
        all_stats.append(stats)
        if i % 100 == 0 or i == len(pending_files):
            print(f"  [{i}/{len(pending_files)}] {stats['date']} "
                  f"({stats['num_nodes']:,} nodes, {stats['num_edges']:,} edges)")

    stats_path = output_dir / "daily_stats.csv"
    new_stats_df = pd.DataFrame(all_stats)
    if stats_path.exists():
        old_stats_df = pd.read_csv(stats_path)
        stats_df = pd.concat([old_stats_df, new_stats_df]).drop_duplicates(subset="date")
    else:
        stats_df = new_stats_df
    stats_df.sort_values("date").to_csv(stats_path, index=False)
    print(f"[snapshots] Stats saved to {stats_path}")


def step_upload(output_dir: Path, remote_path: str = "orbitaal_processed"):
    """Upload processed data to Yandex.Disk via REST API.

    Requires YADISK_TOKEN environment variable with a valid OAuth token.

    Args:
        output_dir: Local directory with processed files.
        remote_path: Remote folder name on Yandex.Disk.
    """
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
            pass

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

    ensure_folder(remote_path)
    ensure_folder(f"{remote_path}/daily_snapshots")

    mapping_file = output_dir / "node_mapping.parquet"
    if mapping_file.exists():
        print(f"[upload] {mapping_file.name}...")
        upload_file(mapping_file, f"{remote_path}/node_mapping.parquet")

    stats_file = output_dir / "daily_stats.csv"
    if stats_file.exists():
        print(f"[upload] {stats_file.name}...")
        upload_file(stats_file, f"{remote_path}/daily_stats.csv")

    snapshot_dir = output_dir / "daily_snapshots"
    if snapshot_dir.exists():
        files = sorted(snapshot_dir.glob("*.parquet"))
        print(f"[upload] {len(files)} daily snapshots...")
        for i, f in enumerate(files, 1):
            upload_file(f, f"{remote_path}/daily_snapshots/{f.name}")
            if i % 100 == 0 or i == len(files):
                print(f"  [{i}/{len(files)}]")

    print("[upload] Done")


def main():
    """CLI entry point for the ORBITAAL graph building pipeline."""
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

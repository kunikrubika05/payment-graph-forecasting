"""Slice a stream graph parquet by date range.

Usage:
    PYTHONPATH=. python scripts/slice_stream_graph.py \
        --input stream_graph/2020-06-01__2020-08-31.parquet \
        --start 2020-07-01 --end 2020-07-07 \
        --output stream_graph/2020-07-01__2020-07-07.parquet

Can also download from Yandex.Disk first:
    YADISK_TOKEN="..." PYTHONPATH=. python scripts/slice_stream_graph.py \
        --yadisk-path orbitaal_processed/stream_graph/2020-06-01__2020-08-31.parquet \
        --start 2020-07-01 --end 2020-07-07 \
        --output /tmp/stream_graph_1week.parquet
"""

import argparse
import os
import sys
from datetime import datetime

import numpy as np
import pandas as pd


def unix_ts(date_str: str) -> int:
    """Convert 'YYYY-MM-DD' to UNIX timestamp (seconds)."""
    return int(datetime.strptime(date_str, "%Y-%m-%d").timestamp())


def main():
    parser = argparse.ArgumentParser(description="Slice stream graph by date range")
    parser.add_argument("--input", type=str, default=None,
                        help="Local path to input parquet")
    parser.add_argument("--yadisk-path", type=str, default=None,
                        help="Yandex.Disk path to download from")
    parser.add_argument("--start", type=str, required=True,
                        help="Start date inclusive (YYYY-MM-DD)")
    parser.add_argument("--end", type=str, required=True,
                        help="End date inclusive (YYYY-MM-DD)")
    parser.add_argument("--output", type=str, required=True,
                        help="Output parquet path")
    args = parser.parse_args()

    if args.input is None and args.yadisk_path is None:
        print("ERROR: specify --input or --yadisk-path")
        sys.exit(1)

    input_path = args.input

    if args.yadisk_path and not input_path:
        token = os.environ.get("YADISK_TOKEN", "")
        if not token:
            print("ERROR: YADISK_TOKEN required for --yadisk-path")
            sys.exit(1)
        from src.yadisk_utils import download_file
        input_path = "/tmp/stream_graph_full.parquet"
        print(f"Downloading {args.yadisk_path} -> {input_path}...")
        download_file(args.yadisk_path, input_path, token)
        print("Done.")

    print(f"Reading {input_path}...")
    df = pd.read_parquet(input_path)
    print(f"  Total: {len(df):,} edges, columns: {list(df.columns)}")

    ts_start = unix_ts(args.start)
    ts_end = unix_ts(args.end) + 86400 - 1

    mask = (df["timestamp"] >= ts_start) & (df["timestamp"] <= ts_end)
    df_slice = df[mask].copy()
    print(f"  After filter [{args.start}, {args.end}]: {len(df_slice):,} edges")

    unique_nodes = set(df_slice["src_idx"].unique()) | set(df_slice["dst_idx"].unique())
    print(f"  Unique nodes: {len(unique_nodes):,}")

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    df_slice.to_parquet(args.output, index=False)
    size_mb = os.path.getsize(args.output) / 1e6
    print(f"  Saved: {args.output} ({size_mb:.1f} MB)")

    stats = {
        "start_date": args.start,
        "end_date": args.end,
        "num_edges": len(df_slice),
        "num_nodes": len(unique_nodes),
        "ts_min": int(df_slice["timestamp"].min()),
        "ts_max": int(df_slice["timestamp"].max()),
        "source": args.input or args.yadisk_path,
    }
    import json
    stats_path = args.output.replace(".parquet", ".json")
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"  Stats: {stats_path}")


if __name__ == "__main__":
    main()

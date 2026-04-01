"""Thin CLI wrapper around the package-facing stream-graph slicing API."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from payment_graph_forecasting.analysis import analyze_stream_graph
from payment_graph_forecasting.data import open_stream_graph


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Slice a stream graph parquet")
    parser.add_argument("--input", type=str, default=None, help="Local path to input parquet")
    parser.add_argument("--remote-path", type=str, default=None, help="Remote path to download first")
    parser.add_argument("--start", type=str, default=None, help="Start date inclusive (YYYY-MM-DD)")
    parser.add_argument("--end", type=str, default=None, help="End date inclusive (YYYY-MM-DD)")
    parser.add_argument(
        "--fraction",
        type=float,
        default=None,
        help="Take the first N fraction of the chronologically sorted period. Matches training fraction semantics.",
    )
    parser.add_argument("--output", type=str, required=True, help="Output parquet path")
    parser.add_argument("--cache-dir", type=str, default=None, help="Optional download cache dir")
    parser.add_argument("--token-env", type=str, default="YADISK_TOKEN", help="Token env for remote downloads")
    return parser


def _select_dataset(args: argparse.Namespace):
    dataset = open_stream_graph(
        args.input,
        remote_path=args.remote_path,
        cache_dir=args.cache_dir,
        token_env=args.token_env,
    )
    if args.fraction is not None:
        return dataset.slice_period_fraction(args.fraction)
    if args.start and args.end:
        return dataset.slice_date_range(args.start, args.end)
    raise ValueError("specify --fraction or both --start and --end")


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    selected = _select_dataset(args)
    output_path = selected.write_parquet(args.output)
    report = analyze_stream_graph(selected)
    stats_path = str(Path(output_path).with_suffix(".json"))
    with open(stats_path, "w", encoding="utf-8") as fh:
        json.dump(report.to_dict(), fh, indent=2)
    print(f"Saved slice: {output_path}")
    print(f"Saved stats: {stats_path}")
    print(f"Selection: {report.selection_description}")
    print(f"Selected edges: {report.num_edges:,}")
    print(f"Selected nodes: {report.num_nodes:,}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

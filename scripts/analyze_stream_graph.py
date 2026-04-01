"""Thin CLI wrapper around the package-facing stream-graph analysis API."""

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
    parser = argparse.ArgumentParser(description="Analyze a stream graph parquet")
    parser.add_argument("--input", type=str, default=None, help="Local path to input parquet")
    parser.add_argument("--remote-path", type=str, default=None, help="Remote path to download first")
    parser.add_argument(
        "--fraction",
        type=float,
        default=0.1,
        help="Analyze the first N fraction of the chronologically sorted period. Matches training fraction semantics.",
    )
    parser.add_argument("--start", type=str, default=None, help="Start date inclusive (YYYY-MM-DD)")
    parser.add_argument("--end", type=str, default=None, help="End date inclusive (YYYY-MM-DD)")
    parser.add_argument("--full", action="store_true", help="Analyze the full input parquet instead of a slice")
    parser.add_argument("--output-json", type=str, default=None, help="Optional path to save the report as JSON")
    parser.add_argument("--output-slice", type=str, default=None, help="Optional path to materialize the selected slice")
    parser.add_argument("--label", type=str, default=None, help="Optional display label for the report")
    parser.add_argument("--cache-dir", type=str, default=None, help="Optional download cache dir")
    parser.add_argument("--token-env", type=str, default="YADISK_TOKEN", help="Token env for remote downloads")
    return parser


def _select_dataset(args: argparse.Namespace):
    dataset = open_stream_graph(
        args.input,
        remote_path=args.remote_path,
        cache_dir=args.cache_dir,
        token_env=args.token_env,
        label=args.label,
    )
    if args.full:
        return dataset
    if args.start or args.end:
        if not (args.start and args.end):
            raise ValueError("both --start and --end are required for date-range slicing")
        return dataset.slice_date_range(args.start, args.end)
    return dataset.slice_period_fraction(args.fraction)


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    selected = _select_dataset(args)
    if args.output_slice:
        selected.write_parquet(args.output_slice)
    report = analyze_stream_graph(selected)
    print(report.to_text())
    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as fh:
            json.dump(report.to_dict(), fh, indent=2)
        print(f"\nSaved report JSON: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

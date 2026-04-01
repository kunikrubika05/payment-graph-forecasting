"""Thin CLI wrapper around the legacy/internal stream-graph visualization API."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from payment_graph_forecasting.data import open_stream_graph
from src.visualization import (
    DEFAULT_LAYOUT,
    DEFAULT_MAX_RENDER_EDGES,
    DEFAULT_MAX_RENDER_NODES,
    DEFAULT_SAMPLE_EDGES,
    SUPPORTED_LAYOUTS,
    save_stream_graph_visualization_artifact,
    save_stream_graph_visualization_view,
    visualize_stream_graph,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build a lightweight visualization of a stream graph")
    parser.add_argument("--input", type=str, default=None, help="Local path to input parquet")
    parser.add_argument("--remote-path", type=str, default=None, help="Remote path to download first")
    parser.add_argument(
        "--fraction",
        type=float,
        default=0.1,
        help="Visualize a selected chronological prefix. Matches training fraction semantics.",
    )
    parser.add_argument("--start", type=str, default=None, help="Start date inclusive (YYYY-MM-DD)")
    parser.add_argument("--end", type=str, default=None, help="End date inclusive (YYYY-MM-DD)")
    parser.add_argument("--full", action="store_true", help="Visualize the full input parquet instead of a slice")
    parser.add_argument("--output", type=str, required=True, help="Output SVG path")
    parser.add_argument("--label", type=str, default=None, help="Optional display label")
    parser.add_argument("--layout", choices=sorted(SUPPORTED_LAYOUTS), default=DEFAULT_LAYOUT)
    parser.add_argument("--sample-edges", type=int, default=DEFAULT_SAMPLE_EDGES)
    parser.add_argument(
        "--scan-full-selection",
        action="store_true",
        help="Opt in to scanning the full selected graph before building the renderable view",
    )
    parser.add_argument("--max-nodes", type=int, default=DEFAULT_MAX_RENDER_NODES)
    parser.add_argument("--max-edges", type=int, default=DEFAULT_MAX_RENDER_EDGES)
    parser.add_argument("--metadata-json", type=str, default=None, help="Optional path to save artifact metadata")
    parser.add_argument("--view-json", type=str, default=None, help="Optional path to save the renderable view as JSON")
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
    artifact = visualize_stream_graph(
        selected,
        args.output,
        sample_edges=None if args.scan_full_selection else args.sample_edges,
        max_nodes=args.max_nodes,
        max_edges=args.max_edges,
        layout=args.layout,
    )
    metadata_path = args.metadata_json or str(Path(args.output).with_suffix(".json"))
    save_stream_graph_visualization_artifact(artifact, metadata_path)
    if args.view_json:
        save_stream_graph_visualization_view(artifact.view, args.view_json)
    print(artifact.to_text())
    print(f"\nSaved visualization: {artifact.output_path}")
    print(f"Saved metadata: {metadata_path}")
    if args.view_json:
        print(f"Saved renderable view: {args.view_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

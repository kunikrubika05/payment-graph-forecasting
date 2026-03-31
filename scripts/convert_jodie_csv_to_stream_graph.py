"""Convert a JODIE-style CSV dataset into the repo stream-graph parquet format.

The output format matches the parquet contract used by the stream-graph
library runners:

    src_idx   int64
    dst_idx   int64
    timestamp int64
    btc       float32
    usd       float32

JODIE datasets are bipartite. To preserve that property in the unified node
index space, destination ids are shifted by ``max(src) + 1`` exactly like
PyG's ``JODIEDataset`` implementation.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def convert_jodie_csv_to_stream_graph(
    input_csv: str | Path,
    output_parquet: str | Path,
    *,
    skip_header: bool = True,
    feature_mode: str = "first2",
) -> Path:
    """Convert a raw JODIE CSV file to the local stream-graph parquet contract."""

    read_kwargs = {"header": None}
    if skip_header:
        read_kwargs["skiprows"] = 1
    df = pd.read_csv(input_csv, **read_kwargs)

    if df.shape[1] < 4:
        raise ValueError("Expected at least 4 columns in JODIE CSV: src, dst, timestamp, label")

    src = df.iloc[:, 0].astype("int64")
    dst = df.iloc[:, 1].astype("int64")
    timestamp = df.iloc[:, 2].astype("int64")
    label = df.iloc[:, 3].astype("float32")
    msg = df.iloc[:, 4:]

    dst = dst + int(src.max()) + 1

    if feature_mode == "first2":
        if msg.shape[1] >= 2:
            btc = msg.iloc[:, 0].astype("float32")
            usd = msg.iloc[:, 1].astype("float32")
        elif msg.shape[1] == 1:
            btc = msg.iloc[:, 0].astype("float32")
            usd = label.astype("float32")
        else:
            btc = label.astype("float32")
            usd = 0.0 * label.astype("float32")
    elif feature_mode == "label+zeros":
        btc = label.astype("float32")
        usd = 0.0 * label.astype("float32")
    else:
        raise ValueError(f"Unsupported feature_mode: {feature_mode}")

    out_df = pd.DataFrame(
        {
            "src_idx": src,
            "dst_idx": dst,
            "timestamp": timestamp,
            "btc": btc.astype("float32"),
            "usd": usd.astype("float32"),
        }
    ).sort_values("timestamp", kind="stable")

    output_path = Path(output_parquet)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_parquet(output_path, index=False)
    return output_path


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convert JODIE-style CSV (e.g. Wikipedia) to stream-graph parquet format"
    )
    parser.add_argument("--input-csv", type=str, required=True)
    parser.add_argument("--output-parquet", type=str, required=True)
    parser.add_argument("--feature-mode", type=str, default="first2", choices=["first2", "label+zeros"])
    parser.add_argument(
        "--no-skip-header",
        action="store_true",
        help="Do not skip the first CSV row. JODIE raw files normally need skiprows=1.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    convert_jodie_csv_to_stream_graph(
        args.input_csv,
        args.output_parquet,
        skip_header=not args.no_skip_header,
        feature_mode=args.feature_mode,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

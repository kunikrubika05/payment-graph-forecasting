"""Convert a generic edge table to the unified stream-graph parquet contract."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def convert_edge_table_to_stream_graph(
    input_path: str | Path,
    output_parquet: str | Path,
    *,
    input_format: str = "csv",
    src_col: str = "src",
    dst_col: str = "dst",
    timestamp_col: str = "timestamp",
    btc_col: str | None = None,
    usd_col: str | None = None,
    delimiter: str = ",",
    has_header: bool = True,
    default_btc: float = 1.0,
    default_usd: float = 0.0,
    remap_nodes: bool = False,
    bipartite: bool = False,
) -> Path:
    """Convert a generic edge table into the local stream-graph parquet format."""

    if input_format == "csv":
        read_kwargs: dict[str, object] = {"sep": delimiter}
        if not has_header:
            read_kwargs["header"] = None
        df = pd.read_csv(input_path, **read_kwargs)
        if not has_header:
            df = df.rename(
                columns={
                    0: src_col,
                    1: dst_col,
                    2: timestamp_col,
                    3: btc_col or "btc",
                    4: usd_col or "usd",
                }
            )
    elif input_format == "parquet":
        df = pd.read_parquet(input_path)
    else:
        raise ValueError(f"Unsupported input_format: {input_format}")

    required = {src_col, dst_col, timestamp_col}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in edge table: {sorted(missing)}")

    out_df = pd.DataFrame(
        {
            "src_idx": df[src_col].astype("int64"),
            "dst_idx": df[dst_col].astype("int64"),
            "timestamp": df[timestamp_col].astype("int64"),
        }
    )

    if bipartite:
        out_df["dst_idx"] = out_df["dst_idx"] + int(out_df["src_idx"].max()) + 1

    if remap_nodes:
        unique_nodes = pd.Index(sorted(set(out_df["src_idx"]) | set(out_df["dst_idx"])))
        node_to_new = {int(node): idx for idx, node in enumerate(unique_nodes)}
        out_df["src_idx"] = out_df["src_idx"].map(node_to_new).astype("int64")
        out_df["dst_idx"] = out_df["dst_idx"].map(node_to_new).astype("int64")

    if btc_col is not None and btc_col in df.columns:
        out_df["btc"] = df[btc_col].astype("float32")
    else:
        out_df["btc"] = float(default_btc)

    if usd_col is not None and usd_col in df.columns:
        out_df["usd"] = df[usd_col].astype("float32")
    else:
        out_df["usd"] = float(default_usd)

    out_df = out_df.sort_values("timestamp", kind="stable")
    output_path = Path(output_parquet)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_parquet(output_path, index=False)
    return output_path


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Convert a generic edge table to stream-graph parquet")
    parser.add_argument("--input-path", required=True)
    parser.add_argument("--output-parquet", required=True)
    parser.add_argument("--input-format", choices=["csv", "parquet"], default="csv")
    parser.add_argument("--src-col", default="src")
    parser.add_argument("--dst-col", default="dst")
    parser.add_argument("--timestamp-col", default="timestamp")
    parser.add_argument("--btc-col", default=None)
    parser.add_argument("--usd-col", default=None)
    parser.add_argument("--delimiter", default=",")
    parser.add_argument("--no-header", action="store_true")
    parser.add_argument("--default-btc", type=float, default=1.0)
    parser.add_argument("--default-usd", type=float, default=0.0)
    parser.add_argument("--remap-nodes", action="store_true")
    parser.add_argument("--bipartite", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    convert_edge_table_to_stream_graph(
        input_path=args.input_path,
        output_parquet=args.output_parquet,
        input_format=args.input_format,
        src_col=args.src_col,
        dst_col=args.dst_col,
        timestamp_col=args.timestamp_col,
        btc_col=args.btc_col,
        usd_col=args.usd_col,
        delimiter=args.delimiter,
        has_header=not args.no_header,
        default_btc=args.default_btc,
        default_usd=args.default_usd,
        remap_nodes=args.remap_nodes,
        bipartite=args.bipartite,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

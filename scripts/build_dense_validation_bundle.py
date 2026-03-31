"""Build a tiny dense validation bundle from a stream-graph parquet."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def build_dense_validation_bundle(
    stream_graph_path: str | Path,
    output_dir: str | Path,
    *,
    max_edges: int = 10_000,
    features_path: str | Path | None = None,
) -> dict[str, str | int]:
    """Create a dense tiny bundle with remapped ids and optional features."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    sg = pd.read_parquet(stream_graph_path).iloc[:max_edges].copy()
    unique_nodes = pd.Index(sorted(set(sg["src_idx"].tolist()) | set(sg["dst_idx"].tolist())))
    node_to_new = {int(node): idx for idx, node in enumerate(unique_nodes)}

    sg["src_idx"] = sg["src_idx"].map(node_to_new).astype(np.int64)
    sg["dst_idx"] = sg["dst_idx"].map(node_to_new).astype(np.int64)
    sg_out = output_dir / "stream_graph_dense.parquet"
    sg.to_parquet(sg_out, index=False)

    mapping = np.arange(len(unique_nodes), dtype=np.int64)
    mapping_out = output_dir / "node_mapping_dense.npy"
    np.save(mapping_out, mapping)

    result: dict[str, str | int] = {
        "stream_graph_path": str(sg_out),
        "node_mapping_path": str(mapping_out),
        "num_edges": int(len(sg)),
        "num_nodes": int(len(unique_nodes)),
    }

    if features_path is not None:
        feat = pd.read_parquet(features_path)
        feat = feat[feat["node_idx"].isin(unique_nodes)].copy()
        feat["node_idx"] = feat["node_idx"].map(node_to_new).astype(np.int64)
        feat = feat.sort_values("node_idx").reset_index(drop=True)
        feat_out = output_dir / "features_dense.parquet"
        feat.to_parquet(feat_out, index=False)
        result["features_path"] = str(feat_out)

    return result


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build a dense tiny validation bundle")
    parser.add_argument("--stream-graph-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-edges", type=int, default=10_000)
    parser.add_argument("--features-path", default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    bundle = build_dense_validation_bundle(
        stream_graph_path=args.stream_graph_path,
        output_dir=args.output_dir,
        max_edges=args.max_edges,
        features_path=args.features_path,
    )
    print(bundle)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Package-facing build helpers for optional C++ and CUDA extensions."""

from __future__ import annotations

import argparse
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class ExtensionBuildSpec:
    """Declarative description of an extension build target."""

    name: str
    source_file: str
    build_dir: str
    kind: str

    @property
    def is_cuda(self) -> bool:
        return self.kind == "cuda"


SRC_MODELS_DIR = Path(__file__).resolve().parents[2] / "src" / "models"

TEMPORAL_SAMPLING_CPP = ExtensionBuildSpec(
    name="temporal_sampling_cpp",
    source_file="csrc/temporal_sampling.cpp",
    build_dir="csrc/build",
    kind="cpp",
)
TEMPORAL_SAMPLING_CUDA = ExtensionBuildSpec(
    name="temporal_sampling_cuda",
    source_file="csrc/temporal_sampling.cu",
    build_dir="csrc/build_cuda",
    kind="cuda",
)
GRAPH_METRICS_CPP = ExtensionBuildSpec(
    name="graph_metrics_cpp",
    source_file="csrc/graph_metrics.cpp",
    build_dir="csrc/build_gm",
    kind="cpp",
)
GRAPH_METRICS_CUDA = ExtensionBuildSpec(
    name="graph_metrics_cuda",
    source_file="csrc/graph_metrics.cu",
    build_dir="csrc/build_gm_cuda",
    kind="cuda",
)


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the CLI parser for extension compilation."""

    parser = argparse.ArgumentParser(
        description="Build optional C++/CUDA extensions for payment_graph_forecasting"
    )
    parser.add_argument(
        "--cuda",
        action="store_true",
        help="Build temporal sampling CUDA extension",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Build both temporal sampling C++ and CUDA extensions",
    )
    parser.add_argument(
        "--graph-metrics",
        action="store_true",
        help="Build graph_metrics C++ extension",
    )
    parser.add_argument(
        "--graph-metrics-cuda",
        action="store_true",
        help="Build graph_metrics CUDA extension",
    )
    return parser


def _compile_extension(spec: ExtensionBuildSpec, *, models_dir: Path = SRC_MODELS_DIR) -> Any:
    from torch.utils.cpp_extension import load

    source_path = models_dir / spec.source_file
    build_path = models_dir / spec.build_dir
    build_path.mkdir(parents=True, exist_ok=True)

    print(f"Compiling {spec.name} from {source_path}...")
    kwargs: dict[str, Any] = {
        "name": spec.name,
        "sources": [str(source_path)],
        "build_directory": str(build_path),
        "verbose": True,
    }
    if spec.is_cuda:
        kwargs["extra_cuda_cflags"] = ["-O3"]
    else:
        kwargs["extra_cflags"] = ["-O3"]
    module = load(**kwargs)
    print(f"{spec.name} built successfully!")
    print(f"Build directory: {build_path}")
    return module


def _warn_if_cuda_unavailable() -> None:
    import torch

    if not torch.cuda.is_available():
        print("WARNING: CUDA not available on this machine.")
        print("Building anyway (for cross-compilation or later use)...")


def ensure_build_prerequisites(spec: ExtensionBuildSpec) -> None:
    """Validate local prerequisites before attempting a build."""

    if shutil.which("ninja") is None:
        raise RuntimeError(
            f"Building {spec.name} requires the `ninja` executable. "
            "Install it in the active environment before running the extension build CLI."
        )


def build_extension(spec: ExtensionBuildSpec, *, models_dir: Path = SRC_MODELS_DIR) -> Any:
    """Build a specific extension target."""

    ensure_build_prerequisites(spec)
    if spec.is_cuda:
        _warn_if_cuda_unavailable()
    return _compile_extension(spec, models_dir=models_dir)


def build_temporal_sampling_cpp(*, models_dir: Path = SRC_MODELS_DIR) -> Any:
    return build_extension(TEMPORAL_SAMPLING_CPP, models_dir=models_dir)


def build_temporal_sampling_cuda(*, models_dir: Path = SRC_MODELS_DIR) -> Any:
    return build_extension(TEMPORAL_SAMPLING_CUDA, models_dir=models_dir)


def build_graph_metrics_cpp(*, models_dir: Path = SRC_MODELS_DIR) -> Any:
    return build_extension(GRAPH_METRICS_CPP, models_dir=models_dir)


def build_graph_metrics_cuda(*, models_dir: Path = SRC_MODELS_DIR) -> Any:
    return build_extension(GRAPH_METRICS_CUDA, models_dir=models_dir)


def selected_build_specs(args: argparse.Namespace) -> list[ExtensionBuildSpec]:
    """Resolve which build targets should run for the provided CLI args."""

    selected: list[ExtensionBuildSpec] = []
    if args.all:
        selected.extend([TEMPORAL_SAMPLING_CPP, TEMPORAL_SAMPLING_CUDA])
    elif args.cuda:
        selected.append(TEMPORAL_SAMPLING_CUDA)
    elif not args.graph_metrics and not args.graph_metrics_cuda:
        selected.append(TEMPORAL_SAMPLING_CPP)

    if args.graph_metrics:
        selected.append(GRAPH_METRICS_CPP)
    if args.graph_metrics_cuda:
        selected.append(GRAPH_METRICS_CUDA)
    return selected


def run_selected_builds(
    args: argparse.Namespace,
    *,
    models_dir: Path = SRC_MODELS_DIR,
) -> list[Any]:
    """Run all requested extension builds."""

    results: list[Any] = []
    for spec in selected_build_specs(args):
        results.append(build_extension(spec, models_dir=models_dir))
    return results


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint for optional extension builds."""

    parser = build_arg_parser()
    args = parser.parse_args(argv)
    try:
        run_selected_builds(args)
    except RuntimeError as exc:
        print(f"ERROR: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Build C++ and CUDA extensions for temporal sampling and graph metrics.

Usage:
    cd payment-graph-forecasting
    source venv/bin/activate

    # Temporal sampling — C++ only (CPU)
    python src/models/build_ext.py

    # Temporal sampling — CUDA (requires NVCC)
    python src/models/build_ext.py --cuda

    # Temporal sampling — both
    python src/models/build_ext.py --all

    # Common neighbors — C++ only
    python src/models/build_ext.py --graph-metrics

    # Common neighbors — CUDA
    python src/models/build_ext.py --graph-metrics-cuda

    # Everything at once
    python src/models/build_ext.py --all --graph-metrics --graph-metrics-cuda

After building, extensions are cached and loaded automatically
by temporal_graph_sampler.py, data_utils.py, and graph_metrics.py.
"""

import argparse
import os
import sys


def build_cpp():
    """Compile the temporal_sampling_cpp (CPU) extension."""
    from torch.utils.cpp_extension import load

    src_dir = os.path.dirname(os.path.abspath(__file__))
    cpp_file = os.path.join(src_dir, "csrc", "temporal_sampling.cpp")
    build_dir = os.path.join(src_dir, "csrc", "build")
    os.makedirs(build_dir, exist_ok=True)

    print(f"Compiling C++ extension from {cpp_file}...")
    module = load(
        name="temporal_sampling_cpp",
        sources=[cpp_file],
        build_directory=build_dir,
        extra_cflags=["-O3"],
        verbose=True,
    )
    print("C++ extension built successfully!")
    print(f"Module: {module}")
    print(f"Build directory: {build_dir}")
    return module


def build_cuda():
    """Compile the temporal_sampling_cuda (GPU) extension."""
    import torch
    if not torch.cuda.is_available():
        print("WARNING: CUDA not available on this machine.")
        print("Building anyway (for cross-compilation or later use)...")

    from torch.utils.cpp_extension import load

    src_dir = os.path.dirname(os.path.abspath(__file__))
    cu_file = os.path.join(src_dir, "csrc", "temporal_sampling.cu")
    build_dir = os.path.join(src_dir, "csrc", "build_cuda")
    os.makedirs(build_dir, exist_ok=True)

    print(f"Compiling CUDA extension from {cu_file}...")
    module = load(
        name="temporal_sampling_cuda",
        sources=[cu_file],
        build_directory=build_dir,
        extra_cuda_cflags=["-O3"],
        verbose=True,
    )
    print("CUDA extension built successfully!")
    print(f"Module: {module}")
    print(f"Build directory: {build_dir}")
    return module


def build_graph_metrics_cpp():
    """Compile the graph_metrics_cpp (CPU) extension for common neighbors."""
    from torch.utils.cpp_extension import load

    src_dir = os.path.dirname(os.path.abspath(__file__))
    cpp_file = os.path.join(src_dir, "csrc", "graph_metrics.cpp")
    build_dir = os.path.join(src_dir, "csrc", "build_gm")
    os.makedirs(build_dir, exist_ok=True)

    print(f"Compiling graph_metrics C++ extension from {cpp_file}...")
    module = load(
        name="graph_metrics_cpp",
        sources=[cpp_file],
        build_directory=build_dir,
        extra_cflags=["-O3"],
        verbose=True,
    )
    print("graph_metrics C++ extension built successfully!")
    return module


def build_graph_metrics_cuda():
    """Compile the graph_metrics_cuda (GPU) extension for common neighbors."""
    import torch
    if not torch.cuda.is_available():
        print("WARNING: CUDA not available on this machine.")
        print("Building anyway (for cross-compilation or later use)...")

    from torch.utils.cpp_extension import load

    src_dir = os.path.dirname(os.path.abspath(__file__))
    cu_file = os.path.join(src_dir, "csrc", "graph_metrics.cu")
    build_dir = os.path.join(src_dir, "csrc", "build_gm_cuda")
    os.makedirs(build_dir, exist_ok=True)

    print(f"Compiling graph_metrics CUDA extension from {cu_file}...")
    module = load(
        name="graph_metrics_cuda",
        sources=[cu_file],
        build_directory=build_dir,
        extra_cuda_cflags=["-O3"],
        verbose=True,
    )
    print("graph_metrics CUDA extension built successfully!")
    return module


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build C++/CUDA extensions for temporal sampling and graph metrics"
    )
    parser.add_argument("--cuda", action="store_true",
                        help="Build temporal sampling CUDA extension")
    parser.add_argument("--all", action="store_true",
                        help="Build both temporal sampling C++ and CUDA extensions")
    parser.add_argument("--graph-metrics", action="store_true",
                        help="Build graph_metrics C++ extension (common neighbors)")
    parser.add_argument("--graph-metrics-cuda", action="store_true",
                        help="Build graph_metrics CUDA extension (common neighbors)")
    args = parser.parse_args()

    if args.all:
        build_cpp()
        print()
        build_cuda()
    elif args.cuda:
        build_cuda()
    elif not args.graph_metrics and not args.graph_metrics_cuda:
        build_cpp()

    if args.graph_metrics:
        print()
        build_graph_metrics_cpp()

    if args.graph_metrics_cuda:
        print()
        build_graph_metrics_cuda()

"""Build C++ and CUDA extensions for accelerated temporal neighbor sampling.

Usage:
    cd payment-graph-forecasting
    source venv/bin/activate

    # Build C++ only (CPU)
    python src/models/build_ext.py

    # Build CUDA extension (requires NVCC)
    python src/models/build_ext.py --cuda

    # Build both
    python src/models/build_ext.py --all

After building, extensions are cached and loaded automatically
by temporal_graph_sampler.py and data_utils.py on subsequent imports.
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build C++/CUDA extensions for temporal sampling"
    )
    parser.add_argument("--cuda", action="store_true",
                        help="Build CUDA extension")
    parser.add_argument("--all", action="store_true",
                        help="Build both C++ and CUDA extensions")
    args = parser.parse_args()

    if args.all:
        build_cpp()
        print()
        build_cuda()
    elif args.cuda:
        build_cuda()
    else:
        build_cpp()

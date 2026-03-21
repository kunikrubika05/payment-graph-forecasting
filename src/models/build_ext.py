"""Build C++ extension for accelerated temporal neighbor sampling.

Usage:
    cd payment-graph-forecasting
    source venv/bin/activate
    python src/models/build_ext.py

After building, the extension is cached and loaded automatically
by data_utils.py on subsequent imports.
"""

import os
import sys


def build():
    """Compile the temporal_sampling_cpp extension."""
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


if __name__ == "__main__":
    build()

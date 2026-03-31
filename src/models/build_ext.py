"""Legacy compatibility shim for optional extension builds.

Preferred command:
    python -m payment_graph_forecasting.infra.extensions

Legacy command remains supported:
    python src/models/build_ext.py
"""

from __future__ import annotations

import os
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from payment_graph_forecasting.infra.extensions import (
    build_graph_metrics_cpp,
    build_graph_metrics_cuda,
    build_temporal_sampling_cpp,
    build_temporal_sampling_cuda,
    main,
)

# TODO(REFACTORING): remove this shim after callers migrate to
# `python -m payment_graph_forecasting.infra.extensions`.
build_cpp = build_temporal_sampling_cpp
build_cuda = build_temporal_sampling_cuda


if __name__ == "__main__":
    raise SystemExit(main())

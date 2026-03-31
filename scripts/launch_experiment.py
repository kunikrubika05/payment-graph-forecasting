#!/usr/bin/env python3
"""Thin wrapper around the new YAML-based experiment launcher."""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from payment_graph_forecasting.experiments.launcher import main


if __name__ == "__main__":
    raise SystemExit(main())

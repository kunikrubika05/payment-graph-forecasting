"""Unified experiment launcher driven by YAML specs."""

from __future__ import annotations

import argparse
from pathlib import Path

from payment_graph_forecasting.config.base import ExperimentSpec
from payment_graph_forecasting.config.yaml_io import load_experiment_spec
from payment_graph_forecasting.models.registry import get_model_adapter


def launch_experiment(spec: ExperimentSpec):
    """Resolve a model adapter and launch an experiment."""

    adapter = get_model_adapter(spec.model_name)
    return adapter.run(spec)


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint for launching YAML-defined experiments."""

    parser = argparse.ArgumentParser(description="Launch payment-graph-forecasting experiment from YAML spec")
    parser.add_argument("spec", nargs="?", type=str, help="Path to experiment YAML file")
    parser.add_argument("--config", dest="config", type=str, default=None, help="Path to experiment YAML file")
    parser.add_argument("--dry-run", action="store_true", help="Resolve the experiment without running training")
    args = parser.parse_args(argv)

    spec_path = args.config or args.spec
    if spec_path is None:
        parser.error("either positional spec or --config must be provided")

    spec = load_experiment_spec(Path(spec_path))
    if args.dry_run:
        spec.runtime.dry_run = True
    launch_experiment(spec)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

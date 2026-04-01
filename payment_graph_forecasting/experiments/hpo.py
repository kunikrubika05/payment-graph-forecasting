"""Package-facing HPO launcher wrapping legacy Optuna entrypoints."""

from __future__ import annotations

import argparse
import sys
from contextlib import contextmanager
from typing import Callable


def _graphmixer_hpo_main() -> None:
    from src.models.GraphMixer.graphmixer_hpo import main

    main()


def _eagle_hpo_main() -> None:
    from src.models.EAGLE.eagle_hpo import main

    main()


def _glformer_hpo_main() -> None:
    from src.models.GLFormer.glformer_hpo import main

    main()


def _hyperevent_hpo_main() -> None:
    from src.models.HyperEvent.hyperevent_hpo import main

    main()


HPO_REGISTRY: dict[str, Callable[[], None]] = {
    "graphmixer": _graphmixer_hpo_main,
    "eagle": _eagle_hpo_main,
    "glformer": _glformer_hpo_main,
    "hyperevent": _hyperevent_hpo_main,
}


def get_hpo_entrypoint(model_name: str) -> Callable[[], None]:
    """Return the HPO entrypoint for a supported model."""

    try:
        return HPO_REGISTRY[model_name]
    except KeyError as exc:
        raise KeyError(
            f"Unknown HPO model '{model_name}'. Known models: {sorted(HPO_REGISTRY)}"
        ) from exc


@contextmanager
def _patched_argv(argv: list[str]):
    previous = sys.argv[:]
    sys.argv = argv
    try:
        yield
    finally:
        sys.argv = previous


def run_hpo(model_name: str, argv: list[str] | None = None) -> int:
    """Run a model-specific HPO entrypoint through the package surface."""

    entrypoint = get_hpo_entrypoint(model_name)
    cli_argv = [f"payment_graph_forecasting.experiments.hpo:{model_name}"]
    if argv:
        cli_argv.extend(argv)
    with _patched_argv(cli_argv):
        entrypoint()
    return 0


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint for package-facing HPO dispatch."""

    parser = argparse.ArgumentParser(
        description="Run model-specific HPO through the payment_graph_forecasting package"
    )
    parser.add_argument("model", choices=sorted(HPO_REGISTRY), help="Model name to optimize")
    args, remainder = parser.parse_known_args(argv)
    return run_hpo(args.model, remainder)


if __name__ == "__main__":
    raise SystemExit(main())

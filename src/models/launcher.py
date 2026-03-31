"""Legacy GraphMixer CLI wrapper.

This file preserves the historical entrypoint while delegating to the new
package runner.
"""

# TODO(REFACTORING): remove legacy src.models.launcher wrapper after callers migrate to payment_graph_forecasting.experiments.runners.graphmixer.

from payment_graph_forecasting.experiments.runners.graphmixer import (
    build_graphmixer_arg_parser,
    main,
    run_graphmixer_experiment as run_experiment,
)

__all__ = ["build_graphmixer_arg_parser", "main", "run_experiment"]


if __name__ == "__main__":
    raise SystemExit(main())

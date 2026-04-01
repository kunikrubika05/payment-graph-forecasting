"""Legacy sg-GraphMixer CLI wrapper.

This file preserves the historical entrypoint while delegating to the new
package runner.
"""

from payment_graph_forecasting.experiments.runners.sg_graphmixer import (
    build_sg_graphmixer_arg_parser,
    main,
    run_sg_graphmixer_experiment as run_experiment,
)

__all__ = ["build_sg_graphmixer_arg_parser", "main", "run_experiment"]


if __name__ == "__main__":
    raise SystemExit(main())

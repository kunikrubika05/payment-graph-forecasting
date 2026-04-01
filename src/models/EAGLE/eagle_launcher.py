"""Legacy EAGLE CLI wrapper.

This file preserves the historical entrypoint while delegating to the new
package runner.
"""

from payment_graph_forecasting.experiments.runners.eagle import (
    build_eagle_arg_parser,
    main,
    run_eagle_experiment as run_experiment,
)

__all__ = ["build_eagle_arg_parser", "main", "run_experiment"]


if __name__ == "__main__":
    raise SystemExit(main())

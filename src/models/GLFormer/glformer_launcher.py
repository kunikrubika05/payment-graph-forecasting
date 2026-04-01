"""Legacy GLFormer CLI wrapper.

This file preserves the historical entrypoint while delegating to the new
package runner.
"""

from payment_graph_forecasting.experiments.runners.glformer import (
    build_glformer_arg_parser,
    main,
    run_glformer_experiment as run_experiment,
)

__all__ = ["build_glformer_arg_parser", "main", "run_experiment"]


if __name__ == "__main__":
    raise SystemExit(main())

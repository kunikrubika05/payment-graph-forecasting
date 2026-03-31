"""Legacy GLFormer CUDA CLI wrapper.

This file preserves the historical entrypoint while delegating to the unified
package GLFormer runner, which now models sampler backend selection directly.
"""

# TODO(REFACTORING): remove legacy src.models.GLFormer_cuda.glformer_launcher
# wrapper after callers migrate to
# payment_graph_forecasting.experiments.runners.glformer with
# `--sampling-backend`.

from payment_graph_forecasting.experiments.runners.glformer import (
    build_glformer_arg_parser,
    main,
    run_glformer_experiment as run_experiment,
)

__all__ = ["build_glformer_arg_parser", "main", "run_experiment"]


if __name__ == "__main__":
    raise SystemExit(main())

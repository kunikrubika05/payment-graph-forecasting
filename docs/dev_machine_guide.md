# GPU Dev Machine Guide

Short operational checklist for running the library on a remote GPU machine.

One provider used in the project: [immers.cloud](https://immers.cloud/).

## Baseline Requirements

- Ubuntu 24.04 CUDA image
- Python 3.10+
- working GPU visible through `nvidia-smi`
- project environment installed with `.[dl,hpo,dev]`
- `tmux` for long-running remote processes

## Minimal Environment Check

After the repository is installed on the machine, verify the package-facing
entrypoints:

```bash
./venv/bin/python -m payment_graph_forecasting.experiments.launcher --help
./venv/bin/python -m payment_graph_forecasting.experiments.hpo --help
./venv/bin/python -m payment_graph_forecasting.infra.extensions --help
```

## Minimal Launch Check

Before a long run, verify dry-runs on representative example specs:

```bash
./venv/bin/python -m payment_graph_forecasting.experiments.launcher --config exps/examples/graphmixer_library.yaml --dry-run
./venv/bin/python -m payment_graph_forecasting.experiments.launcher --config exps/examples/dygformer_library.yaml --dry-run
./venv/bin/python -m payment_graph_forecasting.experiments.launcher --config exps/examples/pairwise_mlp_library.yaml --dry-run
```

## Optional Extensions

Build optional C++/CUDA extensions only when the target workflow actually needs
them. The maintained entrypoint is:

```bash
./venv/bin/python -m payment_graph_forecasting.infra.extensions --help
```

Actual compilation requires `ninja` in the active environment.

## Practical Notes

- User-facing execution should go through `payment_graph_forecasting.*`, not `src/*`.
- Keeping `YADISK_TOKEN` in the environment is convenient for runs that use remote storage.
- Visualization remains legacy/internal tooling and is not a core part of the GPU workflow.

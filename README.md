# payment-graph-forecasting

Library-facing package for temporal payment-graph experiments and runtime
primitives.

The canonical user surface is `payment_graph_forecasting.*`.
The `src/*` tree remains in the repository as legacy/internal backend code and
historical tooling. It is not the primary user-facing API.

Full documentation lives in [docs/README.md](docs/README.md).

## Installation

Use the project virtual environment:

```bash
./venv/bin/pip install -e ".[dl,hpo,dev]"
```

For GPU machines, install the matching PyTorch build first, then install the
project extras.

## Canonical Package Surface

Top-level entrypoints:

- `payment_graph_forecasting.load_experiment_spec`
- `payment_graph_forecasting.build_execution_plan`
- `payment_graph_forecasting.launch_experiment`
- `payment_graph_forecasting.describe_cuda_capabilities`
- `payment_graph_forecasting.TemporalGraphSampler`
- `payment_graph_forecasting.CommonNeighbors`

Main package areas:

- `payment_graph_forecasting.config`
- `payment_graph_forecasting.models`
- `payment_graph_forecasting.training`
- `payment_graph_forecasting.evaluation`
- `payment_graph_forecasting.analysis`
- `payment_graph_forecasting.experiments`
- `payment_graph_forecasting.infra`

Canonical model names:

- `graphmixer`
- `sg_graphmixer`
- `eagle`
- `glformer`
- `hyperevent`
- `pairwise_mlp`
- `dygformer`

## YAML Launches

Reference YAML specs live in `exps/examples/`.

Examples:

```bash
./venv/bin/python -m payment_graph_forecasting.experiments.launcher --help
./venv/bin/python -m payment_graph_forecasting.experiments.launcher --config exps/examples/graphmixer_library.yaml --dry-run
./venv/bin/python -m payment_graph_forecasting.experiments.launcher --config exps/examples/dygformer_library.yaml --dry-run
./venv/bin/python -m payment_graph_forecasting.experiments.launcher --config exps/examples/sg_graphmixer_library.yaml --dry-run
./venv/bin/python -m payment_graph_forecasting.experiments.launcher --config exps/examples/eagle_library.yaml --dry-run
./venv/bin/python -m payment_graph_forecasting.experiments.launcher --config exps/examples/glformer_library.yaml --dry-run
./venv/bin/python -m payment_graph_forecasting.experiments.launcher --config exps/examples/hyperevent_library.yaml --dry-run
./venv/bin/python -m payment_graph_forecasting.experiments.launcher --config exps/examples/pairwise_mlp_library.yaml --dry-run
./venv/bin/python -m payment_graph_forecasting.experiments.hpo --help
```

Minimal Python usage:

```python
from payment_graph_forecasting import load_experiment_spec, launch_experiment

spec = load_experiment_spec("exps/examples/graphmixer_library.yaml")
result = launch_experiment(spec)
```

## Runtime And Extensions

CUDA/runtime helpers:

```python
import payment_graph_forecasting as pgf

caps = pgf.describe_cuda_capabilities()
print(caps)
```

Optional extension build entrypoint:

```bash
./venv/bin/python -m payment_graph_forecasting.infra.extensions --help
```

## Analysis And Scripts

The lightweight analysis API is package-facing under
`payment_graph_forecasting.analysis`.

Supported user scripts:

```bash
./venv/bin/python scripts/analyze_stream_graph.py --help
./venv/bin/python scripts/slice_stream_graph.py --help
./venv/bin/python scripts/visualize_stream_graph.py --help
```

Visualization remains in the repository as legacy/internal tooling under
`src/visualization`.

## Validation

```bash
./venv/bin/python -c "import payment_graph_forecasting; import payment_graph_forecasting.models; import payment_graph_forecasting.training; import payment_graph_forecasting.evaluation"
./venv/bin/pytest tests -q
```

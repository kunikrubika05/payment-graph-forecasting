# Documentation

This directory contains the maintained user-facing documentation for the
repository.

## Canonical Surface

The canonical user surface is `payment_graph_forecasting.*`.

Primary package areas:

| Package | Purpose |
| --- | --- |
| `payment_graph_forecasting.config` | Typed experiment specs and YAML loading |
| `payment_graph_forecasting.models` | Canonical model exports and adapters |
| `payment_graph_forecasting.training` | Training wrappers |
| `payment_graph_forecasting.evaluation` | Evaluation wrappers |
| `payment_graph_forecasting.analysis` | Lightweight analysis API |
| `payment_graph_forecasting.data` | Stream-graph dataset access and slicing helpers |
| `payment_graph_forecasting.sampling` | Sampling strategies and runtime samplers |
| `payment_graph_forecasting.cuda` | CUDA capability helpers |
| `payment_graph_forecasting.graph_metrics` | Runtime graph-metric wrappers |
| `payment_graph_forecasting.experiments` | Unified launcher and HPO entrypoints |
| `payment_graph_forecasting.infra` | Runtime, extension-build, and support helpers |

Canonical model names:

- `graphmixer`
- `sg_graphmixer`
- `eagle`
- `glformer`
- `hyperevent`
- `pairwise_mlp`
- `dygformer`

Reference YAML specs live in `exps/examples/`.

There is also an AI assistant prompt for supported library workflows:
[library_assistant_prompt.md](library_assistant_prompt.md).

## Maintained Command Surface

Launcher and support CLIs:

```bash
./venv/bin/python -m payment_graph_forecasting.experiments.launcher --help
./venv/bin/python -m payment_graph_forecasting.experiments.hpo --help
./venv/bin/python -m payment_graph_forecasting.infra.extensions --help
```

Example dry-runs:

```bash
./venv/bin/python -m payment_graph_forecasting.experiments.launcher --config exps/examples/graphmixer_library.yaml --dry-run
./venv/bin/python -m payment_graph_forecasting.experiments.launcher --config exps/examples/dygformer_library.yaml --dry-run
./venv/bin/python -m payment_graph_forecasting.experiments.launcher --config exps/examples/sg_graphmixer_library.yaml --dry-run
./venv/bin/python -m payment_graph_forecasting.experiments.launcher --config exps/examples/eagle_library.yaml --dry-run
./venv/bin/python -m payment_graph_forecasting.experiments.launcher --config exps/examples/glformer_library.yaml --dry-run
./venv/bin/python -m payment_graph_forecasting.experiments.launcher --config exps/examples/hyperevent_library.yaml --dry-run
./venv/bin/python -m payment_graph_forecasting.experiments.launcher --config exps/examples/pairwise_mlp_library.yaml --dry-run
```

Analysis and tooling scripts:

```bash
./venv/bin/python scripts/analyze_stream_graph.py --help
./venv/bin/python scripts/slice_stream_graph.py --help
./venv/bin/python scripts/visualize_stream_graph.py --help
```

Validation:

```bash
./venv/bin/python -c "import payment_graph_forecasting; import payment_graph_forecasting.models; import payment_graph_forecasting.training; import payment_graph_forecasting.evaluation"
./venv/bin/pytest tests -q
```

## Package Coverage

### `payment_graph_forecasting.config`

Typed experiment-spec objects and YAML loading.

Notable modules:

- `config.base`
- `config.yaml_io`

### `payment_graph_forecasting.models`

Model classes, registry access, and adapter-based execution plans.

Notable modules:

- `models.base`
- `models.registry`
- `models.graphmixer`
- `models.sg_graphmixer`
- `models.eagle`
- `models.glformer`
- `models.hyperevent`
- `models.pairwise_mlp`
- `models.dygformer`

### `payment_graph_forecasting.training`

Package-facing training wrappers for supported model paths.

Notable modules:

- `training.api`
- `training.amp`
- `training.epoch`
- `training.temporal_context`
- `training.trainer`

### `payment_graph_forecasting.evaluation`

Package-facing evaluation wrappers and ranking support.

Notable modules:

- `evaluation.api`
- `evaluation.ranking_loop`
- `evaluation.temporal_ranking`

### `payment_graph_forecasting.analysis`

Stream-graph analysis reports and report formatting.

Notable modules:

- `analysis.stream_graph`

### `payment_graph_forecasting.data`

Dataset descriptors, chronological slicing, date-range slicing, and parquet resolution.

Notable modules:

- `data.stream_graph`

### `payment_graph_forecasting.sampling`

Negative sampling strategies, temporal sampler backends, neighbor batches, and feature batches.

Notable modules:

- `sampling.strategy`
- `sampling.temporal`

### `payment_graph_forecasting.cuda`

Capability probes for optional C++/CUDA acceleration.

### `payment_graph_forecasting.graph_metrics`

Graph metric wrappers including `CommonNeighbors`.

### `payment_graph_forecasting.experiments`

Unified experiment launching, HPO dispatch, result helpers, and runner utilities.

Notable modules:

- `experiments.launcher`
- `experiments.hpo`
- `experiments.hpo_artifacts`
- `experiments.results`
- `experiments.runner_utils`
- `experiments.runners.*`

### `payment_graph_forecasting.infra`

Dataset resolution, local/remote data access, runtime environment helpers, and optional extension build entrypoints.

Notable modules:

- `infra.data_access`
- `infra.datasets`
- `infra.runtime`
- `infra.extensions`
- `infra.upload.*`

## Additional Docs

- [cuda_module_api.md](cuda_module_api.md) — CUDA/runtime API and extension entrypoints
- [cuda_module_overview.md](cuda_module_overview.md) — short overview of the CUDA-backed primitives
- [cuda_temporal_sampling.md](cuda_temporal_sampling.md) — temporal sampling notes
- [cuda_common_neighbors.md](cuda_common_neighbors.md) — common-neighbors notes
- [dev_machine_guide.md](dev_machine_guide.md) — GPU dev-machine checklist
- [evaluation_protocols_temporal_lp.md](evaluation_protocols_temporal_lp.md) — ranking evaluation note

Visualization remains available only as legacy/internal tooling under
`src/visualization`.

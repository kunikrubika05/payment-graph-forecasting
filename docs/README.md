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

## Additional Docs

- [cuda_module_api.md](cuda_module_api.md) — CUDA/runtime API and extension entrypoints
- [cuda_module_overview.md](cuda_module_overview.md) — short overview of the CUDA-backed primitives
- [cuda_temporal_sampling.md](cuda_temporal_sampling.md) — temporal sampling notes
- [cuda_common_neighbors.md](cuda_common_neighbors.md) — common-neighbors notes
- [dev_machine_guide.md](dev_machine_guide.md) — trimmed GPU dev-machine checklist
- [evaluation_protocols_temporal_lp_ru.md](evaluation_protocols_temporal_lp_ru.md) — background note on ranking evaluation protocols

Visualization remains available only as legacy/internal tooling under
`src/visualization`.

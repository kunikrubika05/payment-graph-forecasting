# GraphMixerTime Backend

`src/models/GraphMixer/` is no longer the main documentation surface. It now
serves as the backend layer for library-facing adapters and runners.

## Canonical package-facing paths

Use the package-facing launcher for the snapshot-style / sliding-window
`graphmixer` variant:

```bash
./venv/bin/python -m payment_graph_forecasting.experiments.launcher \
    --config exps/examples/graphmixer_library.yaml --dry-run
```

Important distinction:

- `graphmixer` in the package is the snapshot/sliding-window GraphMixer path
- `sg_graphmixer` is the separate sg-baselines-aligned stream-graph variant
- `graphmixer_hpo.py` still keeps a legacy recommendation for the
  `GraphMixerTime + CUDA sampler` path, because that path does not yet have a
  dedicated package model name separate from the two variants above

For optional compiled extensions:

```bash
./venv/bin/python -m payment_graph_forecasting.infra.extensions
./venv/bin/python -m payment_graph_forecasting.infra.extensions --all
```

## Implementation files

- [graphmixer.py](graphmixer.py)
- [graphmixer_train.py](graphmixer_train.py)
- [graphmixer_evaluate.py](graphmixer_evaluate.py)
- [data_utils.py](data_utils.py)
- [graphmixer_hpo.py](graphmixer_hpo.py)

## Compatibility note

The legacy scripts in this directory are still supported as bridges, but they
should not be treated as the canonical product surface. Follow the package docs
in [README.md](../../../README.md), [docs/README.md](../../../docs/README.md),
and [docs/design/sg_graphmixer_vs_graphmixer.md](../../../docs/design/sg_graphmixer_vs_graphmixer.md).

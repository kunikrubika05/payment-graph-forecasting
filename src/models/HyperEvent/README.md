# HyperEvent

`src/models/HyperEvent/` now serves mainly as the implementation/backend layer
for the library-facing `payment_graph_forecasting` package.

## Canonical entrypoints

For new runs, use the package-facing surfaces:

```bash
./venv/bin/python -m payment_graph_forecasting.experiments.launcher \
    --config exps/examples/hyperevent_library.yaml --dry-run

./venv/bin/python -m payment_graph_forecasting.experiments.hpo hyperevent \
    --parquet-path /tmp/sg_baselines_data/stream_graph.parquet \
    --n-trials 30 --hpo-epochs 10 \
    --output /tmp/hyperevent_hpo
```

The generated HPO follow-up artifacts are now package-facing:

- `best_experiment.yaml`
- `best_train_command.sh` calling
  `python -m payment_graph_forecasting.experiments.launcher --config ...`

## Model notes

- Model family: `hyperevent`
- Data regime: stream graph
- Architectural note: unlike EAGLE/GLFormer, HyperEvent uses adjacency-table
  style local history instead of full TemporalCSR.
- Main implementation files:
  - [hyperevent.py](hyperevent.py)
  - [hyperevent_train.py](hyperevent_train.py)
  - [hyperevent_evaluate.py](hyperevent_evaluate.py)
  - [data_utils.py](data_utils.py)

## Compatibility note

Legacy scripts such as `hyperevent_launcher.py` and `hyperevent_hpo.py` still
exist for compatibility, but they are no longer the recommended documentation
surface. If package and legacy commands disagree, follow the package-facing
docs in [README.md](../../../README.md) and [docs/README.md](../../../docs/README.md).

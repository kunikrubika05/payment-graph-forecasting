# EAGLE-Time

`src/models/EAGLE/` now serves mainly as the implementation/backend layer for
the library-facing `payment_graph_forecasting` package.

## Canonical entrypoints

For new runs, use the package-facing surfaces:

```bash
./venv/bin/python -m payment_graph_forecasting.experiments.launcher \
    --config exps/examples/eagle_library.yaml --dry-run

./venv/bin/python -m payment_graph_forecasting.experiments.hpo eagle \
    --parquet-path /tmp/sg_baselines_data/stream_graph.parquet \
    --features-path /tmp/sg_baselines_data/features_10.parquet \
    --node-mapping-path /tmp/sg_baselines_data/node_mapping_10.npy \
    --fraction 0.10 --node-feat-dim 15 \
    --n-trials 30 --hpo-epochs 15 \
    --output /tmp/eagle_hpo
```

The generated HPO follow-up artifacts are now package-facing:

- `best_experiment.yaml`
- `best_train_command.sh` calling
  `python -m payment_graph_forecasting.experiments.launcher --config ...`

## Model notes

- Model family: `eagle`
- Data regime: stream graph
- Eval regime: sg-baselines-compatible ranking protocol
- Main implementation files:
  - [eagle.py](eagle.py)
  - [eagle_train.py](eagle_train.py)
  - [eagle_evaluate.py](eagle_evaluate.py)
  - [data_utils.py](data_utils.py)

## Compatibility note

Legacy scripts such as `eagle_launcher.py` and `eagle_hpo.py` still exist for
compatibility, but they are no longer the recommended documentation surface.
If package and legacy commands disagree, follow the package-facing docs in
[README.md](../../../README.md) and [docs/README.md](../../../docs/README.md).

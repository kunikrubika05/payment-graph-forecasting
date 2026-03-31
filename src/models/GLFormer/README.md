# GLFormer

`src/models/GLFormer/` now serves mainly as the implementation/backend layer for
the library-facing `payment_graph_forecasting` package.

## Canonical entrypoints

For new runs, use the package-facing surfaces:

```bash
./venv/bin/python -m payment_graph_forecasting.experiments.launcher \
    --config exps/examples/glformer_library.yaml --dry-run

./venv/bin/python -m payment_graph_forecasting.experiments.hpo glformer \
    --parquet-path /tmp/sg_baselines_data/stream_graph.parquet \
    --edge-feat-dim 2 \
    --n-trials 6 --hpo-epochs 10 \
    --output /tmp/glformer_hpo
```

For optional compiled backends:

```bash
./venv/bin/python -m payment_graph_forecasting.infra.extensions
./venv/bin/python -m payment_graph_forecasting.infra.extensions --all --graph-metrics
```

The generated HPO follow-up artifacts are now package-facing:

- `best_experiment.yaml`
- `best_train_command.sh` calling
  `python -m payment_graph_forecasting.experiments.launcher --config ...`

## Model notes

- Model family: `glformer`
- Data regime: stream graph
- Runtime note: CUDA sampler-backed execution is now expressed through
  `sampling.backend` / `sampling_backend` under the same library model, not as a
  separate canonical model name.
- Main implementation files:
  - [glformer.py](glformer.py)
  - [glformer_train.py](glformer_train.py)
  - [glformer_evaluate.py](glformer_evaluate.py)
  - [data_utils.py](data_utils.py)

## Compatibility note

Legacy scripts such as `glformer_launcher.py` and `glformer_hpo.py` still exist
for compatibility, but they are no longer the recommended documentation
surface. If package and legacy commands disagree, follow the package-facing docs
in [README.md](../../../README.md) and [docs/README.md](../../../docs/README.md).

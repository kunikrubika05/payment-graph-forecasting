# Stream Graph Baselines

Baselines for temporal link prediction on the ORBITAAL stream graph.

## Protocol

### Data

- stream graph: `2020-06-01__2020-08-31.parquet`
- two supported periods: `10%` and `25%` chronological prefixes of the full graph
- split: chronological `train 70% / val 15% / test 15%`

### Features

Each candidate pair can use:

- 15 node features for the source
- 15 node features for the destination
- 4 pair features:
  - `CN_undirected`
  - `AA_undirected`
  - `CN_directed`
  - `AA_directed`

All features are computed from the train partition only.

### Negatives

- train: `negative_ratio=5` per positive
- evaluation: `n_negatives=100` per query
- mixed historical and random negatives

### Hyperparameter Search

- grid search over model-specific hyperparameters
- selection metric: validation `MRR`
- validation is used for selection only
- test is not used during hyperparameter search

### Metrics

- `MRR` as the primary metric
- `Hits@1`
- `Hits@3`
- `Hits@10`
- per-source ranking over `{dst_true} ∪ negatives`

## Baseline Families

### Heuristics

| Method | Description |
| --- | --- |
| CN | Common Neighbors |
| Jaccard | Jaccard coefficient |
| AA | Adamic-Adar |
| PA | Preferential Attachment |

### ML Models

| Model | Hyperparameter grid |
| --- | --- |
| LogReg | `C × penalty` |
| CatBoost | `iterations × depth × lr` |
| RF | `n_estimators × max_depth × min_samples_leaf` |

## Run

```bash
YADISK_TOKEN="..." ./venv/bin/python -m sg_baselines.run --period all --output /tmp/sg_baselines_results --upload 2>&1 | tee /tmp/sg_baselines.log
YADISK_TOKEN="..." ./venv/bin/python -m sg_baselines.run --period period_10 --output /tmp/sg_baselines_results --upload 2>&1 | tee /tmp/sg_baselines.log
YADISK_TOKEN="..." ./venv/bin/python -m sg_baselines.run --period period_10 --skip-ml --output /tmp/sg_baselines_results 2>&1 | tee /tmp/sg_baselines.log
YADISK_TOKEN="..." ./venv/bin/python -m sg_baselines.run --period period_10 --models catboost --skip-heuristics --output /tmp/sg_baselines_results 2>&1 | tee /tmp/sg_baselines.log
```

## Correctness Guarantees

1. Node features are computed from train only.
2. Adjacency artifacts are built from train only.
3. Pair features are computed from train adjacency only.
4. Hyperparameter search uses validation, not test.
5. Negative sampling excludes positives from the evaluated split.
6. Train, validation, and test remain chronological and non-overlapping.
7. Final reporting is done on test only.

## Result Layout

```text
exp_sg_{10,25}/
  config.json
  hp_search_{model}.json
  summary.json
  error.txt
```

## Machine Requirements

- RAM: 16 GB or more
- CPU: 4 cores or more
- disk: at least 20 GB free
- GPU: not required

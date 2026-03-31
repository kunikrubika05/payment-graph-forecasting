# Experiments

This directory contains all experiments conducted in the project.
Each experiment lives in its own subdirectory with a standardized structure.

## Examples

Library-facing example YAML specs are stored in `exps/examples/`.

Current examples:

- `graphmixer_library.yaml`
- `sg_graphmixer_library.yaml`
- `eagle_library.yaml`
- `glformer_library.yaml`
- `hyperevent_library.yaml`
- `pairwise_mlp_library.yaml`

These files are package-launcher smoke/reference configs rather than historical
experiment records.

```bash
./venv/bin/python -m payment_graph_forecasting.experiments.launcher \
    --config exps/examples/graphmixer_library.yaml --dry-run
```

## Naming convention

```
exp_NNN_<brief_description>/
```

- `NNN` — sequential number (001, 002, ...) for chronological ordering
- `<brief_description>` — snake_case summary of what the experiment is about

Examples:
```
exp_001_graph_level_sarimax/
exp_002_link_pred_node2vec/
exp_003_graph_level_prophet_vs_tft/
```

## Experiment directory structure

Each experiment directory must contain at minimum a `README.md`:

```
exp_NNN_<name>/
  README.md             # REQUIRED: description, setup, results
  train.py              # Training script (if applicable)
  evaluate.py           # Evaluation script (if applicable)
  configs/              # Hyperparameter configs (if applicable)
  notebooks/            # Jupyter notebooks (if applicable)
  results/              # Local results (metrics, small plots)
  ...
```

## Experiment README.md template

Each experiment's README.md should follow this structure:

```markdown
# EXP-NNN: <Title>

**Date:** YYYY-MM-DD
**Author:** <name>
**Status:** planned | in progress | complete | abandoned

## Goal

What are we trying to learn or achieve?

## Setup

- **Task:** link prediction / graph forecasting / edge weight / ...
- **Data subset:** e.g. 2010-07-17 to 2020-12-31 (excluding pre-market and truncated last day)
- **Model:** ...
- **Hyperparameters:** ...

## Results

| Metric | Value |
|--------|-------|
| ...    | ...   |

## Artifacts on Yandex.Disk

If model weights, large plots, or other heavy artifacts are stored on Yandex.Disk:

- Weights: `orbitaal_experiments/exp_NNN_<name>/checkpoints/`
- Plots: `orbitaal_experiments/exp_NNN_<name>/plots/`

## Conclusions

What did we learn? What to try next?
```

## Index

_List of experiments will be maintained here as they are added._

<!-- Add entries like:
| # | Name | Task | Status | Key result |
|---|------|------|--------|------------|
| 001 | graph_level_sarimax | Graph forecasting | complete | RMSE = ... |
-->

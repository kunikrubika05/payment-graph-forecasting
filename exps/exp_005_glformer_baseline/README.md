# EXP-005: GLFormer Baseline (C++/Python sampling)

**Date:** 2026-03-26
**Author:** team
**Status:** planned

## Goal

Establish GLFormer training performance baseline using the standard C++/Python
neighbor sampling backend. This experiment serves as the control for comparing
against the CUDA-accelerated version (exp_006).

## Setup

- **Task:** Temporal link prediction
- **Model:** GLFormerTime (Adaptive Token Mixer, concatenation predictor)
- **Data:** 1-week slice of stream graph (2020-07-01 to 2020-07-07)
  - Sliced from `2020-06-01__2020-08-31.parquet` via `scripts/slice_stream_graph.py`
  - Chronological split: 70% train / 15% val / 15% test
  - Undirected edges (both directions)
- **Sampling backend:** C++ (via `sample_neighbors_batch` from `data_utils.py`)
- **Epochs:** 10 (timing comparison, not convergence)
- **Hyperparameters:**
  - hidden_dim: 100, num_neighbors: 20, num_glformer_layers: 2
  - batch_size: 200, lr: 0.0001, weight_decay: 1e-5, dropout: 0.1
  - edge_feat_dim: 2 (btc + usd), seed: 42, AMP: enabled
  - max_val_edges: 5000

## How to run

```bash
# 1. Slice the graph (on dev machine)
YADISK_TOKEN="..." PYTHONPATH=. python scripts/slice_stream_graph.py \
    --yadisk-path orbitaal_processed/stream_graph/2020-06-01__2020-08-31.parquet \
    --start 2020-07-01 --end 2020-07-07 \
    --output /tmp/stream_graph_1week.parquet

# 2. Run baseline through the package-facing GLFormer path
./venv/bin/python -m payment_graph_forecasting.experiments.launcher \
    --config exps/examples/glformer_library.yaml --dry-run
```

Note:
- this experiment note is historical;
- if the experiment is re-run, prefer a dedicated YAML spec under `exps/` and
  the package launcher instead of the old `src/models/GLFormer/glformer_launcher.py`.

## Results

_To be filled after running._

| Metric | Value |
|--------|-------|
| Mean epoch time (sec) | |
| Total training time (min) | |
| Best val MRR | |
| Final train loss | |

## Artifacts on Yandex.Disk

- `orbitaal_processed/experiments/exp_006_glformer/<run_name>/`

## Conclusions

_To be filled after comparison with exp_006._

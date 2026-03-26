# EXP-006: GLFormer with CUDA-Accelerated Sampling

**Date:** 2026-03-26
**Author:** team
**Status:** planned

## Goal

Measure the training speedup from using TemporalGraphSampler with CUDA backend
for neighbor sampling and feature gathering in GLFormer. Compare epoch times and
verify that metrics match the baseline (exp_005).

## Setup

- **Task:** Temporal link prediction
- **Model:** GLFormerTime (identical architecture to exp_005)
- **Data:** Same 1-week slice (2020-07-01 to 2020-07-07)
- **Sampling backend:** CUDA (via `TemporalGraphSampler`, auto-detected)
- **Epochs:** 10 (matching exp_005)
- **Hyperparameters:** Identical to exp_005

## What is different

| Component | exp_005 (baseline) | exp_006 (CUDA) |
|-----------|-------------------|----------------|
| Neighbor sampling | `sample_neighbors_batch()` (C++) | `TemporalGraphSampler.sample_neighbors()` (CUDA) |
| Feature gathering | `featurize_neighbors()` (C++) | `TemporalGraphSampler.featurize()` (CUDA) |
| Model architecture | GLFormerTime | GLFormerTime (identical) |
| Training code | GLFormer/glformer_train.py | GLFormer_cuda/glformer_train.py |

## How to run

```bash
# (after slicing the graph — see exp_005 README)

PYTHONPATH=. python src/models/GLFormer_cuda/glformer_launcher.py \
    --parquet-path /tmp/stream_graph_1week.parquet \
    --epochs 10 --batch-size 200 --num-neighbors 20 \
    --sampling-backend auto \
    --output /tmp/exp_006_results \
    2>&1 | tee /tmp/exp_006.log
```

## Expected outcome

- Epoch time: 2-10x faster (sampling bottleneck eliminated)
- Metrics: same val MRR and train loss (same model, same seed)

## Results

_To be filled after running._

| Metric | exp_005 | exp_006 | Speedup |
|--------|---------|---------|---------|
| Mean epoch time (sec) | | | |
| Total training time (min) | | | |
| Best val MRR | | | |
| Final train loss | | | |

## Artifacts on Yandex.Disk

- `orbitaal_processed/experiments/exp_006_glformer_cuda/<run_name>/`

## Conclusions

_To be filled after comparison._

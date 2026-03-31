# `cuda_exp_graphmixer_a10`

## Summary

`cuda_exp_graphmixer_a10` was an environment-specific experiment branch for
running GraphMixer with accelerated temporal sampling on NVIDIA A10 hardware.
It is not treated as a separate library model. The branch existed to compare
sampling/runtime backends and to capture hardware-specific setup needed for the
experiment machine.

## Purpose

The experiment was used to answer questions like:

- how much faster GraphMixer training becomes with accelerated sampling;
- whether CUDA-backed sampling materially changes end-to-end throughput versus
  Python/C++ backends;
- what environment/setup was required on the target A10 machine.

## Main Artifacts

- launcher: `src/models/cuda_exp_graphmixer_a10/launcher.py`
- training path: `src/models/cuda_exp_graphmixer_a10/train.py`
- environment setup: `src/models/cuda_exp_graphmixer_a10/setup_a10.sh`
- scripted run wrapper: `src/models/cuda_exp_graphmixer_a10/run_experiment.sh`

## What Was Varied

The branch focused on backend/runtime concerns rather than a different model
family. Relevant dimensions included:

- temporal sampling backend;
- CUDA availability and compiled extensions;
- hardware-specific setup for A10 / compute capability 8.6;
- resulting throughput and training wall-clock time.

## Relationship To The Library Refactor

Current refactoring direction treats this branch as a legacy experiment record,
not as a canonical standalone model. Where possible, backend/runtime choices
should be expressed through the unified library runtime/sampling layers instead
of separate launchers.

## Metrics Placeholder

Fill this section with the measured results from the original experiment.

| Metric | Value | Notes |
| --- | --- | --- |
| Train backend | TBD | |
| Eval backend | TBD | |
| GPU | NVIDIA A10 | |
| Train time | TBD | |
| Eval time | TBD | |
| Best val MRR | TBD | |
| Test MRR | TBD | |

## Open Refactoring Note

- `cuda_exp_graphmixer_a10` should remain documented for reproducibility.
- Its reusable backend/runtime pieces should continue to be folded into the
  main library pathways rather than maintained as a separate product surface.

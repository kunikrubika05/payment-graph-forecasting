# Refactoring Completion Plan

## Goal

This note defines the remaining work needed to treat the current refactor as
functionally complete before the dedicated full-library validation pass.

The project is already past the architecture-invention stage. The remaining
work is mostly about removing ambiguity:

- which surfaces are canonical;
- which legacy paths are still intentional;
- which bridges are temporary and must be removed later;
- which components still need one more code-level migration before the runtime
  validation phase on a dev machine.

## Current State

Already completed:

- package-facing model registry and adapters;
- typed YAML experiment specs;
- unified package launcher;
- runtime / device / AMP infrastructure;
- library-facing runners for `graphmixer`, `sg_graphmixer`, `eagle`,
  `glformer`, `hyperevent`, `pairwise_mlp`;
- package-facing HPO dispatcher;
- package-facing extension build CLI;
- package-facing docs and example YAML specs for primary model variants.

Still transitional:

- `BaseRunnerAdapter -> argparse.Namespace` bridge in
  `payment_graph_forecasting.models.base`;
- legacy HPO mains still own Optuna trial logic;
- `graphmixer_hpo.py` still emits a legacy follow-up recommendation for the
  `GraphMixerTime + CUDA sampler` path;
- `pairwise_mlp` still uses a legacy core pipeline instead of the unified
  trainer/evaluator stack, although its package-facing remote storage contract
  is now explicit rather than hardcoded;
- several legacy launchers remain as compatibility wrappers;
- A10-specific setup/run scripts remain historical experiment support.

## Canonical Surfaces

These should now be treated as canonical:

- `python -m payment_graph_forecasting.experiments.launcher`
- `python -m payment_graph_forecasting.experiments.hpo`
- `python -m payment_graph_forecasting.infra.extensions`
- YAML specs in `exps/examples/` and future experiment YAMLs under `exps/`
- library packages under `payment_graph_forecasting.*`

These should be treated as compatibility or archival surfaces:

- `src/models/*_launcher.py`
- `src/models/*_hpo.py`
- `src/models/build_ext.py`
- `src/models/cuda_exp_graphmixer_a10/*`

## Remaining Refactoring Work

### 1. Decide long-lived compatibility policy

Before deleting anything, explicitly classify each legacy surface:

- keep as long-lived compatibility shim;
- keep temporarily with a removal condition;
- archive as experiment support only;
- remove after full validation.

The main targets are:

- `src/models/launcher.py`
- `src/models/EAGLE/eagle_launcher.py`
- `src/models/GLFormer/glformer_launcher.py`
- `src/models/HyperEvent/hyperevent_launcher.py`
- `src/models/GLFormer_cuda/glformer_launcher.py`
- `src/models/sg_graphmixer/launcher.py`
- `src/models/build_ext.py`

### 2. Close remaining code-level bridges

These are the most likely remaining code tasks before the validation phase:

- reduce or replace the `argparse` bridge in `BaseRunnerAdapter`;
- decide whether `payment_graph_forecasting.experiments.hpo` should stay a thin
  dispatcher for the validation phase or whether one more migration step is
  needed first;
- decide whether `pairwise_mlp` is acceptable as-is for the validation phase as
  a legacy-core model with an explicit package-facing storage contract, or
  whether one more trainer/evaluator migration step is worth doing now;
- document or formalize the special GraphMixer split:
  `graphmixer`, `sg_graphmixer`, and the separate `GraphMixerTime + CUDA sampler`
  HPO/training branch.

### 3. Prepare validation package

Before using SSH/dev-machine time, the repo should have:

- explicit validation entrypoints;
- a concrete smoke matrix for each model family;
- documented expectations for optional dependencies, compiled extensions, and
  hardware-specific paths.

## Exit Criteria For “Refactor Complete”

The refactor should be considered complete enough to move into the dedicated
validation phase only when:

- canonical package-facing surfaces are unambiguous;
- remaining legacy entrypoints are either documented shims or documented
  archival paths;
- all primary model variants have runnable package-facing smoke configs;
- no unresolved code-level migration question is blocking full-library
  validation;
- the repository has a validation checklist that can be executed end to end on
  a dev machine.

## Immediate Next Steps

Recommended order:

1. Write a deprecation/compatibility matrix for all remaining legacy launchers
   and shims.
2. Resolve the highest-risk code-level bridge still affecting canonical usage.
3. Freeze the refactor scope.
4. Move to the dedicated full functional validation pass.

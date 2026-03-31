# Legacy Surface Matrix

## Purpose

This matrix classifies the remaining legacy code surfaces so the next
refactoring step can be chosen from real code rather than intuition.

The key distinction is:

- thin compatibility wrappers over package-facing entrypoints;
- legacy modules that still contain independent orchestration logic;
- archival experiment support that should not be treated as canonical product
  surface.

## Matrix

| Surface | Current role | Status | Recommended action |
| --- | --- | --- | --- |
| `src/models/launcher.py` | GraphMixer wrapper over package runner | thin shim | Keep for compatibility during validation phase; removable later |
| `src/models/EAGLE/eagle_launcher.py` | EAGLE wrapper over package runner | thin shim | Keep for compatibility during validation phase; removable later |
| `src/models/GLFormer/glformer_launcher.py` | GLFormer wrapper over package runner | thin shim | Keep for compatibility during validation phase; removable later |
| `src/models/GLFormer_cuda/glformer_launcher.py` | GLFormer CUDA wrapper over unified package runner | thin shim | Keep for compatibility during validation phase; removable later |
| `src/models/HyperEvent/hyperevent_launcher.py` | HyperEvent wrapper over package runner | thin shim | Keep for compatibility during validation phase; removable later |
| `src/models/sg_graphmixer/launcher.py` | sg-GraphMixer wrapper over package runner | thin shim | Keep for compatibility during validation phase; removable later |
| `src/models/build_ext.py` | Wrapper over package extension CLI | thin shim | Keep as compatibility shim during validation phase |
| `src/models/pairwise_mlp/run.py` | Legacy core training/evaluation pipeline behind package runner | active legacy core | Acceptable for validation phase now that remote storage is explicit; migrate later if the unified trainer/evaluator stack should absorb it |
| `src/models/GraphMixer/graphmixer_hpo.py` | Owns GraphMixerTime+CUDA-sampler HPO logic | active legacy core | Keep for now; needs explicit model-contract decision before deeper migration |
| `src/models/EAGLE/eagle_hpo.py` | Owns Optuna trial logic; package-facing follow-up artifacts | transitional HPO core | Acceptable for validation phase; migrate later if desired |
| `src/models/GLFormer/glformer_hpo.py` | Owns Optuna trial logic; package-facing follow-up artifacts | transitional HPO core | Acceptable for validation phase; migrate later if desired |
| `src/models/HyperEvent/hyperevent_hpo.py` | Owns Optuna trial logic; package-facing follow-up artifacts | transitional HPO core | Acceptable for validation phase; migrate later if desired |
| `src/models/cuda_exp_graphmixer_a10/launcher.py` | Historical experiment branch | archival experiment support | Keep as archival support; do not treat as canonical model surface |
| `src/models/cuda_exp_graphmixer_a10/setup_a10.sh` | A10-specific environment setup | archival machine support | Keep as historical/hardware note until validation phase is done |
| `src/models/cuda_exp_graphmixer_a10/run_experiment.sh` | A10-specific scripted run | archival machine support | Keep as historical/hardware note until validation phase is done |

## Interpretation

The previous launcher-duplication hotspot is now closed: the remaining legacy
launchers above have been reduced to thin wrappers over package runners.

The main remaining refactoring hotspot before the full-library validation phase
is now the split between thin wrapper surfaces and legacy cores that still own
substantial orchestration logic:

1. `src/models/GraphMixer/graphmixer_hpo.py`
2. `src/models/pairwise_mlp/run.py`
3. transitional HPO mains for `EAGLE`, `GLFormer`, and `HyperEvent`
4. the library-level `argparse` bridge in `BaseRunnerAdapter`

## Recommended next code step

Either:

- freeze these remaining bridges for the validation phase and move on to the
  full functional test protocol;

or, if one more refactoring step is desired before validation:

- reduce the `BaseRunnerAdapter -> argparse.Namespace` bridge or formalize it as
  an accepted transitional layer for the validation phase.

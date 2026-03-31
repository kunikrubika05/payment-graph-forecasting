# Tooling Migration Status

## Goal

This note tracks the status of experiment tooling around the new
`payment_graph_forecasting.*` package surface.

The core library-facing launcher and model adapters are already in place.
What remains is the supporting tooling layer: HPO scripts, extension-build
helpers, machine-setup scripts, and historical experiment wrappers.

## Current Status

### Already package-facing

- `payment_graph_forecasting.experiments.launcher`
- `payment_graph_forecasting.experiments.hpo` (thin dispatcher over legacy HPO mains)
- `payment_graph_forecasting.infra.extensions` (package-facing extension build CLI)
- generated HPO follow-up artifacts for `eagle`, `glformer`, and `hyperevent`
  now emit package-facing `best_experiment.yaml` plus a launcher command
- typed YAML experiment specs in `payment_graph_forecasting.config`
- library-facing runners for:
  - `graphmixer`
  - `sg_graphmixer`
  - `eagle`
  - `glformer`
  - `hyperevent`
  - `pairwise_mlp`
- unified runtime/device/AMP layer
- unified training/evaluation wrapper surface

### Still legacy-only

These still keep their trial logic in legacy modules even though HPO dispatch
now has a package-facing wrapper:

- `src/models/GraphMixer/graphmixer_hpo.py`
- `src/models/EAGLE/eagle_hpo.py`
- `src/models/GLFormer/glformer_hpo.py`
- `src/models/HyperEvent/hyperevent_hpo.py`
- `src/models/cuda_exp_graphmixer_a10/setup_a10.sh`
- `src/models/cuda_exp_graphmixer_a10/run_experiment.sh`

### Mixed / transitional

- `GLFormer_cuda` execution logic has been partially folded into the package
  `glformer` path via explicit `sampling_backend`, but the old launcher and
  HPO/docs paths still exist.
- `src/models/GraphMixer/graphmixer_hpo.py` still emits a legacy follow-up
  train command because its `GraphMixerTime + CUDA sampler` path does not yet
  have a first-class package model distinct from snapshot `graphmixer` and
  `sg_graphmixer`.
- `src/models/build_ext.py` is now a compatibility shim over the package-facing
  extension build module and should no longer be treated as the canonical entrypoint.

## Recommended Migration Order

### 1. HPO wrappers

Lowest-risk next migration target:

- keep the new package-facing HPO dispatcher and progressively migrate trial
  logic out of legacy modules;
- replace generated "best train command" outputs so they point to the package
  launcher / YAML configs where practical.

Why first:

- mostly orchestration;
- easy to smoke-test with imports / `--help` / small dry-run-style checks;
- lower regression risk than deleting legacy launchers.

### 2. Extension build tooling

Second migration target:

- keep `src/models/build_ext.py` only as a thin compatibility shim over the
  package-facing extension CLI;
- document which models actually benefit from compiled extensions;
- keep build/test expectations explicit.

### 3. Machine/setup scripts

Third migration target:

- update setup/dev-machine guidance to call package-facing launch paths first;
- keep hardware-specific scripts as legacy experiment support where needed;
- avoid treating A10-specific scripts as canonical product surface.

### 4. Legacy launcher cleanup

Only after the steps above:

- evaluate which `src/models/*_launcher.py` files are still needed;
- keep compatibility shims where external callers may still depend on them;
- remove only when package-facing parity is demonstrated.

## Testing Guidance

For tooling migration work, prefer these checks:

- import smoke checks;
- `--help` smoke checks for CLIs;
- YAML loading and package launcher dry-run checks;
- targeted unit tests around wrapper dispatch;
- full `./venv/bin/pytest tests -q` before closing a step.

## Practical Interpretation

The project is already past the stage where the package launcher is
experimental. The remaining work is mostly tooling consolidation and legacy
surface reduction, not core architecture invention.

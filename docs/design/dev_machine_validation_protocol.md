# Dev-Machine Validation Protocol

## Goal

This document defines the first full validation pass after the library-facing
refactor. The intent is to validate canonical package surfaces end to end
without turning the first pass into a multi-hour benchmark campaign.

The recommended order is:

1. CPU-only validation on a dev machine.
2. GPU validation after the CPU path is confirmed.

## Recommended CPU Machine

For the first validation pass, use a machine with:

- Ubuntu 24.04
- 8-16 vCPU
- 32 GB RAM minimum, 64 GB preferred
- 100+ GB SSD
- Python 3.10+
- internet access for dataset download / Yandex.Disk access

Use 16 vCPU / 64 GB RAM if the plan includes:

- the ORBITAAL summer 2020 stream graph;
- sg-baselines period artifacts;
- a second external dataset smoke pass.

## Install

```bash
python3 -m venv venv
source venv/bin/activate
pip install -e ".[dl,hpo,dev]"
```

Optional external-dataset path:

```bash
pip install torch-geometric
```

Optional extension builds:

```bash
./venv/bin/python -m payment_graph_forecasting.infra.extensions --help
```

If actual compilation is needed, `ninja` must be available in the active
environment.

## Dataset Matrix

### ORBITAAL / team dataset

Primary validation target:

- summer 2020 stream graph
- 25% chronological prefix for the first CPU pass

Use cases:

- package launcher on stream-graph runners;
- sg-baselines-aligned path;
- upload/download/listing against team storage.

### Wikipedia

Secondary validation target:

- JODIE / temporal Wikipedia interaction dataset
- converted into the repository stream-graph parquet contract

Use cases:

- validate that the library can operate on a non-team temporal graph;
- validate a lightweight external-data adapter path;
- check that the package launcher is not implicitly tied to ORBITAAL-only data.

## Preparation

### ORBITAAL assets

Have these available locally or via Yandex.Disk:

- summer 2020 stream graph parquet
- for sg-GraphMixer / EAGLE-style node-feature paths:
  - `features_25.parquet`
  - `node_mapping_25.npy`
- for sg-baselines-aligned path:
  - adjacency / node-feature artifacts for `period_25`

### Wikipedia conversion

Preferred path:

1. obtain the raw JODIE/Wikipedia CSV;
2. convert it with:

```bash
./venv/bin/python scripts/convert_jodie_csv_to_stream_graph.py \
    --input-csv /path/to/wikipedia.csv \
    --output-parquet /tmp/wikipedia_stream_graph.parquet
```

This produces the local stream-graph parquet contract:

- `src_idx`
- `dst_idx`
- `timestamp`
- `btc`
- `usd`

The feature names are compatibility placeholders for the library runners. For
Wikipedia they simply carry the first two message dimensions or a fallback
label-based surrogate.

## 60-Minute CPU Validation Plan

### Phase 1: setup and smoke (10 minutes)

- verify branch / repo state;
- create venv and install package extras;
- run:
  - `./venv/bin/python -m payment_graph_forecasting.experiments.launcher --help`
  - `./venv/bin/python -m payment_graph_forecasting.experiments.hpo --help`
  - `./venv/bin/python -m payment_graph_forecasting.infra.extensions --help`
- run full `./venv/bin/pytest tests -q`

### Phase 2: ORBITAAL package dry-runs (10 minutes)

- run package dry-runs for:
  - `graphmixer`
  - `sg_graphmixer`
  - `eagle`
  - `glformer`
  - `hyperevent`
  - `pairwise_mlp`
- confirm output directories and resolved configs.

### Phase 3: ORBITAAL CPU real runs (20 minutes)

Run short real executions on the 25% summer 2020 data.

Recommended:

- `eagle`: 1 epoch, small batch, `device=cpu`
- `glformer`: 1 epoch, small batch, `device=cpu`
- `hyperevent`: 1 epoch, small batch, `device=cpu`

Optional, if artifacts are already in place:

- `sg_graphmixer`: 1 epoch or a constrained query budget on `period_25`
- `pairwise_mlp`: precompute availability check plus one constrained run

For this first CPU pass, success means:

- data loads cleanly;
- training loop starts;
- at least one checkpoint / result artifact is produced;
- final summary is written if the path completes within budget.

### Phase 4: Wikipedia external-data pass (10 minutes)

- obtain or download the raw Wikipedia/JODIE CSV;
- convert it with `scripts/convert_jodie_csv_to_stream_graph.py`;
- run at least:
  - package dry-run on the converted parquet
  - one short real CPU run, preferably `hyperevent` or `glformer`

Success means:

- no ORBITAAL-specific assumptions leak into the package path;
- the converted parquet is accepted by at least one real training/eval path.

### Phase 5: storage validation (10 minutes)

- create a dedicated validation remote path under the team experiments root;
- run one package experiment with:
  - `upload.enabled=true`
  - explicit `upload.remote_dir`
  - explicit `upload.token_env`
- verify:
  - upload succeeds;
  - the remote directory exists;
  - expected result files are visible there;
  - download or listing works for the same location.

## Recommended YAML Profiles

### ORBITAAL stream-graph CPU profiles

Use package configs derived from:

- `exps/examples/eagle_library.yaml`
- `exps/examples/glformer_library.yaml`
- `exps/examples/hyperevent_library.yaml`

Adjust:

- `runtime.device: cpu`
- `training.epochs: 1`
- smaller `training.batch_size`
- constrained eval budgets where supported
- `upload.enabled: false` except during the storage validation phase

### Wikipedia profile

Derive from `glformer_library.yaml` or `hyperevent_library.yaml` and replace:

- `data.parquet_path`
- optional feature-related fields if not available
- experiment name / output dir

## Out Of Scope For The First CPU Pass

Not required in the 60-minute CPU session:

- CUDA extension compilation;
- benchmark-quality metrics;
- long HPO runs;
- parity benchmarking against historical experiment numbers;
- A10-specific scripts.

These belong to the subsequent GPU validation phase.

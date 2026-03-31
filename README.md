# payment-graph-forecasting

Forecasting the dynamics of the payment graph based on open payment data.

## Dataset

**ORBITAAL** — Bitcoin transaction network between entities (clustered addresses), covering 2009-01-03 to 2021-01-25. Source: [Zenodo](https://zenodo.org/records/12581515).

**Processed data:** [Yandex.Disk](https://disk.yandex.ru/d/uJavr5EtMWj4Jg) — daily snapshots, node mapping, statistics.

## Processed graph format

The pipeline (`src/build_pipeline.py`) produces the following structure:

```
data/processed/
├── node_mapping.parquet    # Global entity_id → node_index mapping
├── daily_stats.csv         # Per-day graph statistics
└── daily_snapshots/
    ├── 2009-01-03.parquet
    ├── 2009-01-04.parquet
    ├── ...
    └── 2021-01-25.parquet
```

### node_mapping.parquet

Global mapping from original ORBITAAL `entity_id` to dense `node_index` (0..N-1). Built once across all daily snapshots, so the same entity always has the same index regardless of the day.

| Column | Type | Description |
|--------|------|-------------|
| `entity_id` | int64 | Original ORBITAAL entity ID |
| `node_index` | int64 | Dense index (0..N-1) |

Entity 0 (special/coinbase) is excluded.

### Daily snapshots (e.g. `2016-07-08.parquet`)

Each file is one day's aggregated transaction graph. Self-loops and entity 0 are removed.

| Column | Type | Description |
|--------|------|-------------|
| `src_idx` | int64 | Sender node index (from global mapping) |
| `dst_idx` | int64 | Receiver node index (from global mapping) |
| `btc` | float32 | Transaction value in BTC |
| `usd` | float32 | Transaction value in USD (daily rate) |

**Loading into PyTorch Geometric:**

```python
import pandas as pd
import torch
from torch_geometric.data import Data

df = pd.read_parquet("data/processed/daily_snapshots/2016-07-08.parquet")
data = Data(
    edge_index=torch.tensor([df["src_idx"].values, df["dst_idx"].values], dtype=torch.long),
    edge_attr=torch.tensor(df[["btc", "usd"]].values, dtype=torch.float),
)
```

### daily_stats.csv

Pre-computed statistics per day for quick analysis and graph-level forecasting.

| Column | Type | Description |
|--------|------|-------------|
| `date` | str | Date (YYYY-MM-DD) |
| `num_nodes` | int | Active entities that day |
| `num_edges` | int | Edges after cleaning |
| `total_btc` | float | Sum of all transaction values (BTC) |
| `total_usd` | float | Sum of all transaction values (USD) |

## Stream graph format

The stream graph pipeline (`src/build_stream_graph.py`) produces a single parquet file with all transactions sorted chronologically — the standard format for temporal GNN models (TGN, DyGFormer, TGAT).

| Column | Type | Description |
|--------|------|-------------|
| `src_idx` | int64 | Sender node index (from global mapping) |
| `dst_idx` | int64 | Receiver node index (from global mapping) |
| `timestamp` | int64 | UNIX timestamp of the transaction (seconds) |
| `btc` | float32 | Transaction value in BTC |
| `usd` | float32 | Transaction value in USD (daily rate) |

**Loading for temporal link prediction:**

```python
import pandas as pd

df = pd.read_parquet("data/processed/stream_graph/2020-06-01__2020-08-31.parquet")
# Filter to a specific week (data is sorted by timestamp)
week_start = pd.Timestamp("2020-07-01").timestamp()
week_end = pd.Timestamp("2020-07-07 23:59:59").timestamp()
week_df = df[(df["timestamp"] >= week_start) & (df["timestamp"] <= week_end)]
```

## Installation

The project uses `pyproject.toml` with optional dependency groups. Install only what you need:

```bash
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip setuptools wheel

# Core only (data pipeline, feature computation)
pip install -e .

# Core + ML baselines (scikit-learn, catboost, statsmodels, prophet)
pip install -e ".[baselines]"

# Core + deep learning (torch, ninja)
pip install -e ".[dl]"

# Core + hyperparameter optimization (optuna)
pip install -e ".[hpo]"

# Core + visualization (matplotlib, networkx)
pip install -e ".[viz]"

# Core + testing (pytest)
pip install -e ".[dev]"

# Everything
pip install -e ".[all]"

# Combine groups as needed
pip install -e ".[dl,hpo,dev]"
```

**PyTorch with CUDA:** on GPU machines, install torch first with the correct CUDA version:
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install -e ".[dl,hpo,dev]"
```

## Pipeline usage

```bash
# Test on CSV samples (locally)
python src/build_pipeline.py --steps mapping snapshots \
    --input-dir data/samples --format csv

# Full pipeline on dev machine (Zenodo → process → Yandex.Disk)
python src/build_pipeline.py --steps download extract mapping snapshots upload \
    --zenodo-files snapshot-day

# Stream graph pipeline (for temporal GNN models)
python src/build_stream_graph.py --steps download extract process upload \
    --start-date 2020-06-01 --end-date 2020-08-31

# Run tests
pytest tests/ -v
```

## Library API And YAML Experiments

The repository now exposes a library-facing package:

```python
from payment_graph_forecasting import load_experiment_spec, launch_experiment

spec = load_experiment_spec("exps/examples/graphmixer_library.yaml")
result = launch_experiment(spec)
```

You can also launch experiments from YAML directly:

```bash
./venv/bin/python -m payment_graph_forecasting.experiments.launcher \
    exps/examples/graphmixer_library.yaml --dry-run
```

The new package path is `payment_graph_forecasting.*`. Legacy `src.*` imports
still work through compatibility adapters while the refactor is in progress.

Current library-facing model names:

- `graphmixer`
- `sg_graphmixer`
- `eagle`
- `glformer`
- `hyperevent`
- `pairwise_mlp`

`graphmixer` and `sg_graphmixer` intentionally remain separate variants. They
belong to the same GraphMixer family, but correspond to different data and
evaluation regimes.

Example specs currently live in `exps/examples/`:

- `graphmixer_library.yaml`
- `sg_graphmixer_library.yaml`
- `eagle_library.yaml`
- `glformer_library.yaml`
- `hyperevent_library.yaml`
- `pairwise_mlp_library.yaml`

Examples:

```bash
./venv/bin/python -m payment_graph_forecasting.experiments.launcher \
    --config exps/examples/graphmixer_library.yaml --dry-run

./venv/bin/python -m payment_graph_forecasting.experiments.launcher \
    --config exps/examples/sg_graphmixer_library.yaml --dry-run

./venv/bin/python -m payment_graph_forecasting.experiments.launcher \
    --config exps/examples/eagle_library.yaml --dry-run

./venv/bin/python -m payment_graph_forecasting.experiments.launcher \
    --config exps/examples/pairwise_mlp_library.yaml --dry-run
```

Optional compiled extensions now also have a package-facing entrypoint:

```bash
# Requires `ninja` in the active environment for actual compilation.
./venv/bin/python -m payment_graph_forecasting.infra.extensions
./venv/bin/python -m payment_graph_forecasting.infra.extensions --all --graph-metrics
```

Legacy `python src/models/build_ext.py ...` remains available as a compatibility
shim while the refactor is being finalized.

## Project structure

```
payment_graph_forecasting/
├── config/              # Typed experiment specs and YAML loading
├── evaluation/          # Stable evaluation wrappers
├── experiments/         # Unified launcher and library-facing runners
├── infra/               # Runtime, extension-build, and upload infrastructure
├── models/              # Library model exports and adapters
├── sampling/            # Unified sampling config/representation
└── training/            # Stable training wrappers
src/
├── analyze.py           # Exploratory data analysis on CSV samples
├── build_graphs.py      # PaymentGraph class (PyG-compatible, pickle format)
├── build_pipeline.py    # Main pipeline: download → mapping → snapshots → upload
├── build_stream_graph.py # Stream graph pipeline: download → extract → process → upload
├── compute_features.py  # Graph-level and node-level feature computation
├── baselines/           # Legacy baseline pipelines
└── models/              # Legacy model implementations and migration shims
docs/
├── design/              # Architecture / refactoring notes
└── experiments/         # Notes about legacy experiment branches
tests/
├── test_library_api.py       # Library surface / launcher tests
├── test_model_adapter_base.py # Adapter contract tests
├── test_*_runner.py          # Runner-level smoke tests
└── ...
```

New library-facing work should go through `payment_graph_forecasting.*`.

# payment-graph-forecasting

Forecasting the dynamics of the payment graph based on open payment data.

## Dataset

**ORBITAAL** — Bitcoin transaction network between entities (clustered addresses), covering 2009-01-03 to 2021-01-25. Source: [Zenodo](https://zenodo.org/records/12581515).

**Processed data:** [Yandex.Disk](https://disk.yandex.ru/client/disk/orbitaal_processed) — daily snapshots, node mapping, statistics.

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

## Pipeline usage

```bash
# Install dependencies
pip install -r requirements.txt

# Test on CSV samples (locally)
python src/build_pipeline.py --steps mapping snapshots \
    --input-dir data/samples --format csv

# Full pipeline on dev machine (Zenodo → process → Yandex.Disk)
python src/build_pipeline.py --steps download extract mapping snapshots upload \
    --zenodo-files snapshot-day

# Run tests
pytest tests/ -v
```

## Project structure

```
src/
├── analyze.py          # Exploratory data analysis on CSV samples
├── build_graphs.py     # PaymentGraph class (PyG-compatible, pickle format)
├── build_pipeline.py   # Main pipeline: download → mapping → snapshots → upload
└── visualize.py        # Visualization scripts
tests/
└── test_pipeline.py    # Pipeline tests (11 tests)
```

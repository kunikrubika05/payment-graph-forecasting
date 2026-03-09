# CLAUDE.md

This file is automatically loaded by Claude Code at the start of every conversation.
It contains project context, rules, and technical instructions for the AI assistant.

> **Update policy:** After completing a meaningful chunk of work (experiment, refactor, new feature),
> propose updates to this file. Do NOT modify it without explicit user approval.
> You MAY suggest specific edits — the user will validate and approve them.

---

## Project overview

**Topic:** Forecasting the dynamics of the payment graph based on open payment data.
**Type:** Course project (3 people). The end goal is to build an open-source Python library
for analysis, training, and validation of forecasting models on transaction graphs.

**Dataset:** ORBITAAL — Bitcoin entity-level transaction network (2009-01-03 to 2021-01-25).
Entities are clusters of Bitcoin addresses grouped by common-input heuristic.
Source: https://zenodo.org/records/12581515. Total raw size ~157 GB.

**Key insight from literature:** For macro-level forecasting of Bitcoin transaction activity,
simple structural metrics (num_nodes, num_edges) can be as informative as complex graph features.
External economic factors (price, news, market activity) matter more than internal graph topology.

### Task formulations (from project description)

1. **Link Prediction** — will edge (A->B) appear at time T+1? (GNN: GraphSAGE, GCN, GAT; Node2Vec + classifier)
2. **Edge Weight Prediction** — transaction volume between A and B (Temporal GNN, RNN)
3. **Graph-level Forecasting** — predict num_nodes, num_edges, total_volume as time series (SARIMAX, Prophet, TFT, hybrid models)
4. **Node Activity Prediction** — will entity be active tomorrow? (classification)

**Not yet decided** which task to prioritize. This is the next discussion point.

### Key findings from literature review

| Paper | Task | Key result |
|-------|------|------------|
| Bianconi & Agrawal (2017) | Graph-level forecasting | Simple linear regression on (nodes, edges) matches models with complex graph features |
| Ma & Mahmoudinia (2024) | Fee forecasting (time series) | SARIMAX beats Time2Vec and TFT on small data |
| Wei, Zhang & Liu (2020) | Link prediction (DLForecast) | Time-decayed graphs + node embedding + MMU ensemble, >60% accuracy |
| Zhao, Wang & Wei (2025) | Link prediction (HIN+GNN) | 83.82% accuracy on Ethereum, 75% on Bitcoin after UTXO adaptation |
| Zhou et al (2025) | Link prediction (criminal activity) | GAT best for known addresses (92.6%), RF best for new addresses (94-96.7% recall) |

---

## What has been done

### Data pipeline (complete)

Full pipeline in `src/build_pipeline.py`: download -> extract -> mapping -> snapshots -> upload.
All 5 steps have been executed on the dev machine. Results uploaded to Yandex.Disk.

**Processed data structure on Yandex.Disk** (`orbitaal_processed/`):
```
orbitaal_processed/
  node_mapping.parquet        # 320M+ entities -> dense 0..N-1 index
  daily_stats.csv             # per-day statistics (4401 rows)
  daily_snapshots/
    2009-01-03.parquet        # one file per day
    ...
    2021-01-25.parquet        # 4401 files total
```

**Raw data on dev machine** (`data/raw/`):
```
data/raw/
  orbitaal-snapshot-day.tar.gz     # 24.8 GB (extracted -> SNAPSHOT/EDGES/day/)
  orbitaal-nodetable.tar.gz        # 24.9 GB (downloaded, NOT extracted yet)
  orbitaal-stream_graph.tar.gz     # 23.9 GB (downloaded, NOT extracted yet)
```

### Data quality notes

- **5 missing dates**: 2009-01-04 through 2009-01-08 (no blocks mined between genesis and block 1)
- **257 days with 0 edges** (245 in 2009, 12 in 2010) — all transactions involved entity 0 (coinbase), which is excluded
- **USD = 0 before 2010-07-17** — no market price existed before Mt. Gox launch. This is correct, not a bug
- **Last day (2021-01-25) is truncated**: 62K nodes vs ~500K typical. Exclude from training
- **float32 precision artifacts** in btc values (e.g. 134.08999633789062) — acceptable for ML
- **~14.6% node overlap** between consecutive days — most entities appear and disappear within one day

### Tests

11 tests in `tests/test_pipeline.py` — all passing. Tests use CSV samples from `data/samples/`.

---

## Project structure

```
src/
  build_pipeline.py     # Main pipeline (download/extract/mapping/snapshots/upload)
  build_graphs.py       # PaymentGraph class (legacy prototype, pickle format)
  analyze.py            # EDA on CSV samples
  visualize.py          # Visualization scripts
tests/
  test_pipeline.py      # 11 tests for the pipeline
data/
  samples/              # CSV samples for 2016-07-08 and 2016-07-09 (halving day)
  raw/                  # Raw ORBITAAL data (on dev machine only, gitignored)
  processed/            # Processed parquet snapshots (on dev machine only, gitignored)
exps/
  README.md             # Experiment structure and naming conventions
  exp_NNN_<name>/       # Individual experiment directories
```

### Experiments

All experiments live in the `exps/` directory. See `exps/README.md` for the structure,
naming conventions, and how to document results. Each experiment gets its own subdirectory
with a README.md describing setup, hyperparameters, results, and links to artifacts on Yandex.Disk.

---

## Rules

### What the agent CAN do

- Edit source code and configuration files
- Run tests (`pytest tests/ -v`)
- Propose changes to CLAUDE.md (with user approval)
- Analyze data, write scripts, debug issues

### What the agent CANNOT do

- **Do NOT run git commands** (commit, push, branch, merge). The user handles all git operations.
- **Do NOT access the dev machine** via SSH or run remote commands. Only the user works on the dev machine.
- **Do NOT commit or push** unless the user explicitly asks to prepare git commands for them to copy-paste.
- **Do NOT modify CLAUDE.md** without explicit user approval.

### Git workflow (for reference — executed by the user)

1. Create a feature/fix branch: `git checkout -b feature/<name>`
2. Stage and commit changes
3. Push: `git push --set-upstream origin feature/<name>` (this prints a PR link)
4. Open the PR link in the browser, review, merge via GitHub UI
5. After merge:
```bash
git checkout main
git pull
git branch -d feature/<name>
```

Branch naming: `feature/<name>`, `fix/<name>`, `experiment/<name>`

### Code quality

- **Write tests** for any new functionality. Place them in `tests/`.
- Run `pytest tests/ -v` before proposing changes as ready.
- Use the project's virtual environment (venv). Do NOT use system pip.
- **Write docstrings** for all public functions, classes, and modules (Google style).
- **Do NOT write inline comments.** Code should be self-explanatory. Docstrings only.
- Maintain documentation in `docs/` as the project grows (see `docs/README.md`).
- Communicate with the user in Russian.

### Security

- Do NOT commit secrets (tokens, IPs, passwords). Use environment variables.
- Sensitive values (Yandex.Disk token, dev machine IP) are stored as env vars or provided by the user at runtime.

---

## Technical reference

### Dev machine

- **OS:** Ubuntu 22.04, **CPU:** 8 cores, **RAM:** 64 GB, **SSD:** 200 GB
- **Access:** SSH with key-based auth (`ssh -i <path_to_key> -l cursach <DEV_MACHINE_IP>`)
- **Python venv** is set up on the machine with all dependencies
- **tmux** is used for long-running processes (pipeline steps take hours)
- **No VPN** on the dev machine. Zenodo is accessible directly (~9 MB/s)

The agent does not connect to the dev machine. If something needs to be run there,
provide the user with commands to copy-paste.

### Yandex.Disk API

Used for sharing processed data and experiment artifacts between team members.

**Setup:**
1. Create an OAuth token at https://oauth.yandex.ru (scope: cloud_api:disk.*)
2. Set as environment variable: `export YADISK_TOKEN="your_token_here"`
3. The upload step in `build_pipeline.py` reads it from `YADISK_TOKEN` env var

**API basics (for reference when writing upload/download code):**
- Base URL: `https://cloud-api.yandex.net/v1/disk/resources`
- Auth header: `Authorization: OAuth <token>`
- Create folder: `PUT /v1/disk/resources?path=<remote_path>`
- Get upload URL: `GET /v1/disk/resources/upload?path=<remote_path>&overwrite=true` — returns JSON with `href`, then PUT file contents to that URL
- Get download URL: `GET /v1/disk/resources/download?path=<remote_path>` — returns JSON with `href`

**Current structure on Yandex.Disk:**
```
orbitaal_processed/
  node_mapping.parquet
  daily_stats.csv
  daily_snapshots/
    2009-01-03.parquet ... 2021-01-25.parquet
```

### Loading processed data (code snippets)

```python
# Load a daily snapshot into PyTorch Geometric
import pandas as pd
import torch
from torch_geometric.data import Data

df = pd.read_parquet("data/processed/daily_snapshots/2016-07-08.parquet")
data = Data(
    edge_index=torch.tensor([df["src_idx"].values, df["dst_idx"].values], dtype=torch.long),
    edge_attr=torch.tensor(df[["btc", "usd"]].values, dtype=torch.float),
)

# Load daily stats for time-series analysis
stats = pd.read_csv("data/processed/daily_stats.csv", parse_dates=["date"])

# Load node mapping
mapping = pd.read_parquet("data/processed/node_mapping.parquet")
# mapping: entity_id -> node_index (dense 0..N-1)
```

### Memory considerations (lessons learned)

Processing 320M+ entities requires careful memory management on 64 GB RAM:

| Structure | Memory for 320M entries |
|-----------|------------------------|
| Python `set` of ints | ~25 GB |
| Python `dict` (int->int) | ~32 GB |
| `numpy` int64 array | ~2.5 GB |
| `pandas` Series (int64) | ~5 GB |

- **Mapping step** uses numpy arrays with batch processing (2 GB batches, periodic `np.unique` merges)
- **Snapshots step** processes files sequentially with one shared pandas Series for the mapping
- **Do NOT use multiprocessing** for steps that require the full mapping — each worker duplicates it

### ORBITAAL dataset reference

| File | Size | Content | Status |
|------|------|---------|--------|
| `orbitaal-snapshot-day.tar.gz` | 24.8 GB | Daily aggregated graphs (4401 parquet files) | Downloaded, extracted, processed |
| `orbitaal-nodetable.tar.gz` | 24.9 GB | Entity metadata (names, timestamps, balances) | Downloaded, NOT extracted |
| `orbitaal-stream_graph.tar.gz` | 23.9 GB | Timestamped transactions by year (13 files) | Downloaded, NOT extracted |

**Snapshot columns:** SRC_ID, DST_ID, VALUE_SATOSHI, VALUE_USD
**Stream graph columns:** SRC_ID, DST_ID, TIMESTAMP, VALUE_SATOSHI, VALUE_USD
**Node table columns:** ID, NAME, FIRST_TIMESTAMP, LAST_TIMESTAMP, BALANCE_SATOSHI

---

### Current decisions

- **Stream graph is deferred.** We focus on snapshot-day (daily aggregated graphs) for now.
  Stream graph (`orbitaal-stream_graph.tar.gz`) is kept on the dev machine for future use
  (link prediction with temporal resolution, sub-daily analysis). Do not delete it.
- **Raw snapshot-day tar.gz and extracted parquet** can be deleted from the dev machine
  after confirming processed data is intact on Yandex.Disk.

---

## Open questions

1. Which task formulation to prioritize? (Graph-level forecasting is easiest to start; link prediction is most impactful)
2. Which baselines to implement first? (SARIMAX on daily_stats for graph-level; heuristic methods for link prediction)
3. How to handle the pre-2010 period with sparse/empty graphs? (Likely just filter to 2010+ or 2010-07-17+)

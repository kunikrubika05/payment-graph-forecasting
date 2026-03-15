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

**Raw data:** Первая дев-машинка (8 cores / 64 GB / 200 GB SSD) была уничтожена 2026-03-15.
Raw-файлы (`orbitaal-nodetable.tar.gz`, `orbitaal-stream_graph.tar.gz`) утеряны.
Все обработанные данные и 9 завершённых экспериментов сохранены на Яндекс.Диске.
При необходимости raw-данные можно скачать заново с Zenodo.

### Data quality notes

- **5 missing dates**: 2009-01-04 through 2009-01-08 (no blocks mined between genesis and block 1)
- **257 days with 0 edges** (245 in 2009, 12 in 2010) — all transactions involved entity 0 (coinbase), which is excluded
- **USD = 0 before 2010-07-17** — no market price existed before Mt. Gox launch. This is correct, not a bug
- **Last day (2021-01-25) is truncated**: 62K nodes vs ~500K typical. Exclude from training
- **float32 precision artifacts** in btc values (e.g. 134.08999633789062) — acceptable for ML
- **~14.6% node overlap** between consecutive days — most entities appear and disappear within one day

### Feature computation pipeline

`src/compute_features.py`: computes graph-level and node-level features from daily snapshots.
Uses scipy.sparse for PageRank, clustering, connected components; custom Batagelj-Zaversnik
for k-core. No networkx dependency at runtime.

**Graph-level features** (~40 columns): degree stats, Gini, BTC/USD aggregates, WCC/SCC,
clustering, triangles, PageRank stats, k-core, assortativity, reciprocity.

**Node-level features** (26 columns per node): degree, weighted degree, balance, transaction
stats (avg/median/max/min/std BTC), unique counterparties, PageRank, clustering, k-core, triangles.

**OOM-защита:** перед вычислением `A @ A` (для кластеризации и треугольников) оценивается
размер результата как `sum(deg²) * 12 bytes`. Если >40 ГБ — эти фичи пропускаются (= 0).
Это затрагивает ~50 дней периода Bitcoin-пузыря (конец 2017 — начало 2018), где графы
особенно плотные. Все остальные фичи для этих дней считаются нормально.

**Output:**
```
data/processed/
  graph_features.csv            # one row per day, ~40 columns
  node_features/
    2009-01-03.parquet          # one file per day, only active nodes
    ...
```

Supports `--upload` mode: uploads node features to Yandex.Disk in batches and deletes
local copies to conserve disk space. Supports resume (skips already processed days).

**Статус:** Все 4401 день посчитаны. Данные на Яндекс.Диске.

**Что есть:**
- **graph_features.csv** — 4401 строка данных (все дни с 2009-01-03 по 2021-01-25).
  На дев-машине и на Яндекс.Диске.
- **Node features** — 4144 файла на Яндекс.Диске. Из 4401 дня 257 имеют 0 рёбер
  (245 в 2009, 12 в 2010) — все транзакции шли через entity 0 (coinbase), который
  исключён. Для этих 257 дней node features НЕ существуют — это ожидаемое поведение.
- **~50 дней (2017-12-02 — 2018-01-20)**: clustering_coeff и triangle_count = 0
  из-за OOM-защиты (`A @ A` на графах ~500K узлов требует >60 ГБ RAM).
  Все остальные ~35 фичей для этих дней посчитаны полностью.

### Baseline experiments pipeline

`src/baselines/` — полный пайплайн для обучения и оценки бейзлайнов. Запускается одной
командой, создаёт tmux-сессии, работает автономно. Результаты загружаются на Яндекс.Диск.

**Задачи:**
1. **Link Prediction** — LogReg, CatBoost, RandomForest на агрегированных node features.
   Sliding window (W дней → предсказание рёбер дня W+1). Два режима: A (единая модель)
   и B (live-update). Per-source negative sampling: 50/50 historical + random.
2. **Graph-level Forecasting** — ARIMA, SARIMAX, Holt-Winters, Prophet, naive baselines
   на временных рядах из `graph_features.csv` (num_nodes, num_edges, total_btc, total_usd).
3. **Heuristic Link Prediction** — Common Neighbors, Jaccard, Adamic-Adar, Preferential Attachment.

**Протокол оценки (TGB-style):**
- Per-source ranking: для каждого positive edge (s, d_true) фиксируется s, строится candidate
  set {d_true} ∪ {neg_1, ..., neg_q}, модель ранжирует кандидатов, ранг d_true → метрики.
- n_negatives=100 (per TGB standard для больших графов, как tgbl-coin).
- Negative sampling: 50% historical (соседи из окна, отсутствующие в target day) + 50% random.
- Метрики: filtered MRR (primary), Hits@1, Hits@3, Hits@10 — per-query, усреднённые.
- Обучение: бинарная классификация (pos+neg пары) с per-source historical+random negatives.
- Референсные значения (TGB tgbl-coin): PA=0.481, PA_rec=0.584, TGN=0.586, PopTrack=0.725 (MRR).
- Подробнее: `docs/evaluation_protocols_temporal_lp_ru.md`.

**Периоды:** 10 блоков по ~90 дней из разных эпох Bitcoin (2012–2020).
Окна агрегации: W ∈ {3, 7, 14, 30}. Feature modes: base (50 фичей) и extended (100 фичей).
HP search встроен в пайплайн (grid search по val-метрике PR-AUC на подвыборке 500K сэмплов,
финальная модель обучается на полном датасете до 2M сэмплов).
Mode B: retrain каждые 5 дней (retrain_interval=5), только для mid_2015q3 (тяжёлые периоды
исключены из-за бюджета 24ч).
HP grids: LogReg 8, CatBoost 12, RF 12 комбинаций.
Pair features используют float32 для экономии памяти (50% vs float64).

**Модули:**
- `src/yadisk_utils.py` — download/upload с Яндекс.Диска (retry, рекурсивные папки)
- `src/baselines/config.py` — конфигурация экспериментов (периоды, HP-сетки, ExperimentConfig)
- `src/baselines/data_loader.py` — загрузка node_features и daily_snapshots с Я.Диска
- `src/baselines/feature_engineering.py` — mean/time-weighted агрегация, pair features
- `src/baselines/evaluation.py` — ranking metrics (MRR, Hits@K), time series metrics (MAE, RMSE, MAPE, sMAPE)
- `src/baselines/experiment_logger.py` — config.json, metrics.jsonl, summary.json, модели, предсказания
- `src/baselines/link_prediction.py` — LP pipeline (Mode A + Mode B)
- `src/baselines/graph_forecasting.py` — time series forecasting pipeline
- `src/baselines/heuristic_baselines.py` — heuristic scores pipeline
- `src/baselines/runner.py` — обработка очереди конфигов с resume и error handling
- `src/baselines/launcher.py` — генерация 35 экспериментов (22 LP + 3 graph forecast + 10 heuristic), распределение по tmux-сессиям

**Запуск на дев-машине:**
```bash
cd ~/payment-graph-forecasting && git pull
source venv/bin/activate && pip install -r requirements.txt
PYTHONPATH=. python -m pytest tests/test_baselines.py -v
export YADISK_TOKEN="..."
export RF_N_JOBS=4          # RF параллелизм (cores / sessions)
export CATBOOST_THREADS=4   # CatBoost параллелизм
PYTHONPATH=. python src/baselines/launcher.py --sessions 4
```

**Мониторинг:** `tmux ls`, `tail -f /tmp/baseline_logs/baseline_N.log`,
`bash scripts/full_check.sh` (полный статус + проверка Яндекс.Диска).
**Resume:** при перезапуске runner пропускает эксперименты с готовым `summary.json`
(проверяет **локальный** `/tmp/baseline_results/`). При миграции на новую машинку
нужно предварительно скачать `summary.json` с Яндекс.Диска (скрипт `scripts/sync_completed.py`).
**Ошибки:** записываются в `error.txt` в папке эксперимента, runner переходит к следующему.

**Структура результатов на Яндекс.Диске:**
```
orbitaal_processed/experiments/
  exp_001_link_pred_baselines/
    period_mid_2015q3_w7_mean_modeA/
      config.json, metrics.jsonl, summary.json
      hp_search_results.json, feature_importance.json
      feature_correlations.csv, high_correlations.json
      model/best_logreg.pkl, best_catboost.cbm, best_rf.pkl
      predictions/2015-09-15_logreg.parquet
  exp_002_graph_level_baselines/
  exp_003_heuristic_baselines/
```

**Статус baseline экспериментов (2026-03-15):**

Всего 35 экспериментов (22 LP + 3 graph forecast + 10 heuristic).

Завершено 9/35, все на Яндекс.Диске:
- `period_mature_2020q2_w3_mean_modeA` (LP)
- `period_mature_2020q2_w7_mean_modeA` (LP)
- `period_mature_2020q2_w14_mean_modeA` (LP)
- `period_mature_2020q2_w30_mean_modeA` (LP)
- `period_mature_2020q2_w7_time_weighted_modeA` (LP)
- `period_late_2020q4_w7_mean_modeA` (LP)
- `period_peak_2018q2_w7_mean_modeA` (LP)
- `period_peak_2018q2_w7_time_weighted_modeA` (LP)
- `period_post_peak_2019q1_w7_mean_modeA` (LP)

Оставшиеся 26: 13 LP + 10 heuristic + 3 graph forecasting.
Самые тяжёлые периоды (2018-2020) уже посчитаны. Оставшиеся LP — 2012-2017 (легче).

**Предварительные результаты (MRR, TGB-style ranking):**
- LogReg: 0.25-0.35 (слабая линейная модель)
- CatBoost: 0.48-0.62 (сопоставимо с TGB PA=0.481, TGN=0.586)
- RF: 0.49-0.55 (близко к CatBoost)
- Heuristic CN: ~0.72, AA: ~0.71, Jaccard: ~0.65, PA: ~0.53

**Оптимизация heuristic (2026-03-15):**
Heuristic baselines были оптимизированы: Python-циклы в compute_CN/Jaccard/AA заменены
на vectorized sparse matrix operations (`adj[src].multiply(adj[dst]).sum(axis=1)`).
Все 4 эвристики теперь считаются за один проход (вместо 4× генерации кандидатов).
Ожидаемое ускорение: ~100-500× (5 ч/день → ~1-3 мин/день).

### Tests

82 tests total — all passing:
- `tests/test_pipeline.py` — 11 tests for the data pipeline
- `tests/test_compute_features.py` — 36 tests for feature computation (correctness,
  disk cleanup, resume, edge cases)
- `tests/test_baselines.py` — 35 tests for baseline pipeline (feature engineering,
  ranking metrics, experiment logger, config, link prediction helpers, heuristic helpers)

---

## Project structure

```
src/
  build_pipeline.py     # Main pipeline (download/extract/mapping/snapshots/upload)
  compute_features.py   # Graph-level and node-level feature computation
  yadisk_utils.py       # Yandex.Disk API utilities (download/upload)
  build_graphs.py       # PaymentGraph class (legacy prototype, pickle format)
  analyze.py            # EDA on CSV samples
  visualize.py          # Visualization scripts
  baselines/
    config.py           # Experiment configuration and HP grids
    data_loader.py      # Load data from Yandex.Disk
    feature_engineering.py  # Feature aggregation and pair features
    evaluation.py       # Classification and time series metrics
    experiment_logger.py    # Logging: config, metrics, models, predictions
    link_prediction.py  # Link prediction pipeline (Mode A + B)
    graph_forecasting.py    # ARIMA, SARIMAX, Holt-Winters, Prophet
    heuristic_baselines.py  # CN, Jaccard, Adamic-Adar, PA
    runner.py           # Queue-based experiment runner
    launcher.py         # Multi-session tmux orchestrator
tests/
  test_pipeline.py      # 11 tests for the pipeline
  test_compute_features.py  # 36 tests for feature computation
  test_baselines.py     # 32 tests for baseline pipeline
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
- **Always activate venv** before running any Python commands: `source venv/bin/activate && PYTHONPATH=. <command>`. Do NOT use system Python or pip.
- **Write docstrings** for all public functions, classes, and modules (Google style).
- **Do NOT write inline comments.** Code should be self-explanatory. Docstrings only.
- Maintain documentation in `docs/` as the project grows (see `docs/README.md`).
- **Update documentation** when implementing new features or changing protocols. This includes:
  updating CLAUDE.md (with user approval), relevant docs in `docs/`, and test descriptions.
- Communicate with the user in Russian.

### Logging and progress

- **Always use `tee`** when running long processes: `command 2>&1 | tee /tmp/logfile.log`
- **Progress bars** must show: current item, total, percentage, speed (sec/item), ETA, elapsed time
- **Log file per process** — if a process crashes, the log must survive for diagnosis
- **Never run long processes without logging** — silent crashes are unacceptable

### Resource utilization

- **Do NOT let dev machine CPU/GPU idle** while waiting. If a long process uses 1 core, consider running independent work on remaining cores.
- **Parallelization safety:** for heavy graphs (2017+, ~500K nodes), max 1-2 workers on 64 GB RAM. Sparse matrix multiply (A²·A for triangles/clustering) is the bottleneck — each worker can use 10-20 GB peak.
- **OOM risk:** if a tmux session silently disappears, it was likely killed by OOM-killer. Check `dmesg | grep -i oom` or the log file.

### Security

- Do NOT commit secrets (tokens, IPs, passwords). Use environment variables.
- Sensitive values (Yandex.Disk token, dev machine IP) are stored as env vars or provided by the user at runtime.

---

## Technical reference

### Dev machine

Первая машинка (8 cores / 64 GB / 200 GB SSD) уничтожена 2026-03-15.
Рекомендуемый конфиг новой машинки: **16 cores / 64 GB RAM / 100 GB SSD**.
- 16 cores: RF_N_JOBS=4 × 4 сессии = 16 cores (полная утилизация)
- 64 GB RAM: запас на случай тяжёлых графов (sparse matrix ops на ~500K узлов)
- 100 GB SSD: достаточно для кэша данных (~20 GB результатов)

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
    2009-01-03.parquet ... 2021-01-25.parquet  # 4401 files
  node_features/
    2009-01-03.parquet ... 2021-01-25.parquet  # 4144 файлов (257 пустых дней не имеют node features)
  graph_features.csv                           # 4401 строка
  experiments/                                 # Результаты экспериментов (создаётся launcher.py)
    exp_001_link_pred_baselines/
    exp_002_graph_level_baselines/
    exp_003_heuristic_baselines/
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
  Stream graph нужно будет скачать заново с Zenodo при необходимости (старая машинка уничтожена).
- **Processed data** полностью на Яндекс.Диске. Новая машинка качает их через API по требованию.

---

## Open questions

1. ~~Which baselines to implement first?~~ **Done** — LP (LogReg/CatBoost/RF), graph-level (ARIMA/SARIMAX/Prophet), heuristic (CN/Jaccard/AA/PA).
2. How to handle the pre-2010 period with sparse/empty graphs? (Likely just filter to 2010+ or 2010-07-17+)
3. Next steps after baselines: GNN models (GraphSAGE, GAT), Node2Vec embeddings, temporal GNNs?
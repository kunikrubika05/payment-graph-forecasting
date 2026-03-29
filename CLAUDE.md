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

1. **Link Prediction** — will edge (A->B) appear at time T, if we know all previous information?
2. **Edge Weight Prediction** — transaction volume between A and B.

**Decision (2026-03-19):** Focus on **Link Prediction**.

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
```

Supports `--upload` mode: uploads node features to Yandex.Disk in batches and deletes
local copies to conserve disk space. Supports resume (skips already processed days).

### Baseline experiments pipeline

`src/baselines/` — полный пайплайн для обучения и оценки бейзлайнов. Запускается одной
командой, создаёт tmux-сессии, работает автономно. Результаты загружаются на Яндекс.Диск.

**Задачи:**
1. **Link Prediction** — LogReg, CatBoost, RandomForest на агрегированных node features.
   Sliding window (W дней → предсказание рёбер дня W+1). Два режима: A (единая модель)
   и B (live-update). Per-source negative sampling: 50/50 historical + random.
2. **Heuristic Link Prediction** — Common Neighbors, Jaccard, Adamic-Adar, Preferential Attachment.

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
  exp_xxx_<short_desc>
```

### DL models for temporal link prediction

`src/models/` — пайплайн глубокого обучения для temporal link prediction.
Первая модель: **GraphMixer** (Cong et al., ICLR 2023).

**Архитектура GraphMixer:**
- LinkEncoder: MLP-Mixer over K=20 последних рёбер (temporal edge sequences)
- NodeEncoder: node features + mean-pool 1-hop neighbor features
- LinkClassifier: additive MLP (fc_src(h_src) + fc_dst(h_dst) → score)
- 138K параметров, 540 KB. Не использует attention или GNN.

**C++ расширение** (`src/models/csrc/temporal_sampling.cpp`):
- TemporalCSR: CSR с binary search по timestamp
- sample_neighbors_batch: пакетный сэмплинг K соседей
- featurize_neighbors: заполнение feature-массивов
- Компиляция: `python src/models/build_ext.py` (pybind11 через torch.utils.cpp_extension)
- Fallback на Python/NumPy если не скомпилировано

**Протокол оценки:** идентичен бейзлайнам (TGB-style: 50 hist + 50 random negatives,
per-source ranking, MRR/Hits@K). Использует ту же `compute_ranking_metrics` из
`src/baselines/evaluation.py`.

**Запуск:**
```bash
python src/models/build_ext.py  # компиляция C++ (один раз)
YADISK_TOKEN="..." PYTHONPATH=. python src/models/launcher.py \
    --period mature_2020q2 --output /tmp/graphmixer_results \
    2>&1 | tee /tmp/graphmixer.log
```

**Документация:** `docs/graphmixer_pipeline.md` — полное описание архитектуры, пайплайна,
параметров CLI и структуры результатов.

**Инфраструктура:** `docs/dev_machine_guide.md` — инструкция по работе с GPU машиной (immers.cloud).

### CUDA temporal sampling module

`src/models/temporal_graph_sampler.py` — универсальный модуль для GPU-ускоренного
temporal neighbor sampling. Три бэкенда: Python (NumPy), C++ (pybind11), CUDA.

**Что делает:**
1. **sample_neighbors** — для batch запросов (node, time) находит K последних соседей
   с timestamp < time через binary search в CSR. Соседи = исходящие рёбра (src→dst).
2. **featurize** — собирает node/edge features и relative timestamps для найденных соседей.
3. **sample_negatives** — генерация негативных кандидатов (random, historical, mixed).

**Файлы:**
- `src/models/temporal_graph_sampler.py` — Python wrapper, `TemporalGraphSampler` класс
- `src/models/csrc/temporal_sampling.cu` — CUDA-ядра (`sample_neighbors_kernel`,
  `featurize_neighbors_kernel`, `TemporalCSR_CUDA`)
- `src/models/csrc/temporal_sampling.cpp` — C++ бэкенд (существовал ранее)
- `src/models/build_ext.py` — компиляция (`--cuda` / `--all`)
- `tests/test_temporal_sampler.py` — 30 тестов (корректность всех бэкендов, cross-validation)
- `scripts/bench_sampling.py` — бенчмарк (5 сценариев, warmup + measurement)

**Бенчмарк (V100-32GB):**

| Сценарий | Python | C++ | CUDA | Speedup vs Python |
|----------|--------|-----|------|-------------------|
| часть от GraphMixer (B=512, K=20) | 23ms | 2.2ms | 0.2ms | 110x |
| часть от DyGFormer (B=2048, K=512) | 1740ms | 104ms | 1.0ms | 1689x |

**Setup на GPU машине:** `bash scripts/setup_v100.sh` — автоматическая настройка
V100 (PyTorch 2.5.1+cu121, компиляция расширений, тесты).

**Статус (2026-03-26):** Модуль завершён. 30/30 тестов passing. Бенчмарк подтверждён на V100.

### GLFormer_cuda — CUDA-ускоренный pipeline обучения

`src/models/GLFormer_cuda/` — версия GLFormer pipeline, использующая
`TemporalGraphSampler` (CUDA) вместо стандартного C++/Python сэмплинга.

**Отличие от GLFormer/:** только бэкенд сэмплинга. Модель (`GLFormerTime`),
loss, optimizer, протокол оценки — идентичны. Код train/evaluate/launcher
переписан для вызова `sampler.sample_neighbors()` + `sampler.featurize()`.

**Файлы:**
- `src/models/GLFormer_cuda/data_utils.py` — `build_cuda_sampler()`, re-exports
- `src/models/GLFormer_cuda/glformer_train.py` — `train_glformer_cuda()`, `prepare_glformer_batch_cuda()`
- `src/models/GLFormer_cuda/glformer_evaluate.py` — `evaluate_tgb_style()` с CUDA сэмплингом
- `src/models/GLFormer_cuda/glformer_launcher.py` — CLI launcher (`--sampling-backend`)

**Сравнительный эксперимент exp_005 vs exp_006 (2026-03-27, V100, 3 эпохи):**
- Данные: 1 неделя (4.7M рёбер, 9.4M undirected), batch_size=4000, K=20
- **Результат: CUDA speedup = 0%.** Epoch_time C++ ≈ CUDA ≈ 3.5 мин
- Причина: при K=20, batch=4000 сэмплинг занимает <2% времени эпохи.
  Bottleneck — GPU forward/backward pass. CUDA-сэмплинг помогает только когда
  сэмплинг сам является узким местом (>20% epoch_time).
- Яндекс.Диск: `exp_005_glformer/` оставить, `exp_006_glformer_cuda/` можно удалить.

**Починки, сделанные в ходе экспериментов (2026-03-27):**
- `log1p` нормализация BTC/USD в `src/models/EAGLE/data_utils.py` — фикс NaN loss от AMP float16 overflow
- `torch.amp.autocast("cuda")` и `torch.amp.GradScaler("cuda", ...)` — фикс FutureWarning во всех 4 файлах
- Validation/test sampler в GLFormer_cuda переведён на C++ backend (CUDA per-edge overhead = 5x slower)
- `--max-test-edges` флаг добавлен в оба launcher'а и `run_cuda_comparison.sh`
- Исправлен `YADISK_EXPERIMENTS_BASE` в `GLFormer/glformer_launcher.py` (был exp_006, стало exp_005)

### Stream graph pipeline

`src/build_stream_graph.py` — пайплайн для получения stream graph из ORBITAAL dataset.
Скачивает архив с Zenodo, извлекает нужный год, фильтрует по датам, применяет глобальный
node mapping и сохраняет один parquet-файл, отсортированный по timestamp.

**Формат выходного файла** (один parquet, стандарт для TGN/DyGFormer/TGAT):
```
src_idx (int64)   — source node (global mapping)
dst_idx (int64)   — destination node (global mapping)
timestamp (int64) — UNIX timestamp транзакции (секунды)
btc (float32)     — VALUE_SATOSHI / 1e8
usd (float32)     — VALUE_USD
```

Файл отсортирован по timestamp. Для обучения на подпериоде достаточно простой фильтрации
по timestamp — данные уже в хронологическом порядке.

**Запуск на дев-машине (CPU, без GPU):**
```bash
PYTHONPATH=. python src/build_stream_graph.py \
    --steps download extract process upload \
    --start-date 2020-06-01 --end-date 2020-08-31 \
    2>&1 | tee /tmp/stream_graph.log

PYTHONPATH=. python src/build_stream_graph.py \
    --steps extract process upload \
    --start-date 2020-06-01 --end-date 2020-08-31 \
    2>&1 | tee /tmp/stream_graph.log
```

**Рекомендуемый конфиг:** `cpu.4.8.120` (4 vCPU, 8 GB RAM, 120 GB SSD) — 2837 ₽/мес.
8 GB RAM хватит (mapping ~5 GB + данные ~3 GB). 120 GB SSD — запас для tar-распаковки.
GPU не нужен. Время: ~1 час (скачивание 45 мин + обработка 15 мин).

**Результат на Яндекс.Диске:**
```
orbitaal_processed/stream_graph/
  2020-06-01__2020-08-31.parquet   # stream graph, sorted by timestamp
  2020-06-01__2020-08-31.json      # статистика (num_edges, num_nodes, etc.)
```

### Static node features for stream graph

`scripts/compute_stream_node_features.py` — вычисление 15 статических node features
из train-части stream graph для двух конфигураций (10% и 25% рёбер).

**15 фичей (float32):** log_in_degree, log_out_degree, in_out_ratio, log_unique_in/out_cp,
log_total/avg_btc_in/out, recency, activity_span, log_event_rate, burstiness,
out/in_counterparty_entropy.

**Протокол:** фичи считаются ТОЛЬКО на train-части (первые 70% рёбер периода).
Val (15%) и test (15%) НЕ участвуют — предотвращение data leakage.
- features_10: train = первые 7% от полного stream graph (4.3M рёбер, 1.9M активных нод)
- features_25: train = первые 17.5% от полного stream graph (10.8M рёбер, 4.3M активных нод)

**Memory-efficient:** глобальное пространство индексов = 330M нод (весь ORBITAAL),
но фичи хранятся только для активных нод (sparse формат с колонкой `node_idx`).
Маппинг через `np.searchsorted` — 0 доп. памяти vs 37 GB для dense массива.

**Хелпер для загрузки в модель:**
```python
from scripts.compute_stream_node_features import load_node_features
# dense = load_node_features("features_10.parquet", num_nodes_in_period)
# dense[node_idx] → float32[15], неактивные = 0
```

**Запуск:**
```bash
YADISK_TOKEN="..." PYTHONPATH=. python scripts/compute_stream_node_features.py \
    --input /tmp/stream_graph_full.parquet \
    --output-dir /tmp/stream_features/ --upload \
    2>&1 | tee /tmp/stream_features.log
```

**Статус (2026-03-29):** Завершено. Результаты на Яндекс.Диске.
features_10.parquet (60.5 MB), features_25.parquet (141.1 MB).

### Sparse adjacency matrices and pair features (CN, AA)

`scripts/compute_stream_adjacency.py` — построение бинарных sparse adjacency матриц
(directed + undirected) из train-части stream graph для двух периодов (10% и 25%).

**Зачем:** pair-level фичи (Common Neighbors, Adamic-Adar) нельзя предпосчитать для всех
пар (N² ≈ 10¹²). Хранится adjacency матрица, из которой CN/AA вычисляются на лету
за миллисекунды на batch. CN — самая сильная эвристика (MRR 0.73 на бейзлайнах),
и именно pair-level структуры не хватает node-only моделям.

**Протокол:** матрицы строятся ТОЛЬКО из train-рёбер (первые 70% периода).
Ноды из val/test, которых нет в train → CN=0, AA=0 (корректно, нет data leakage).

**Формат:** scipy CSR в `.npz`, float32, бинарные (0/1). Индексы — **локальные**
(0..n_active-1). Маппинг local→global в `node_mapping_{label}.npy`.

**ВАЖНО — единый маппинг:** `node_mapping` из adjacency **идентичен** `node_idx`
из `features_{label}.parquet` (оба = `np.unique(concat([src_train, dst_train]))`).
Это гарантировано конструкцией и проверяется assert'ами в скрипте.

**Файлы на Яндекс.Диске** (для каждого периода 10/25):
```
stream_graph/
  adj_{label}_directed.npz    # CSR (n_active × n_active), A[i,j]=1 if edge i→j
  adj_{label}_undirected.npz  # CSR symmetric, A[i,j]=A[j,i]=1 if edge i→j or j→i
  node_mapping_{label}.npy    # local_idx → global_idx (int64)
  adj_{label}.json            # metadata (nnz, n_nodes, etc.)
```

**Использование pair features в модели (CN + AA):**
```python
from scipy import sparse
from scripts.compute_stream_adjacency import compute_cn, compute_aa

# Загрузка (один раз при старте):
adj = sparse.load_npz("adj_10_undirected.npz")
mapping = np.load("node_mapping_10.npy")

# В каждом batch:
# 1. Global → local маппинг
local_src = np.searchsorted(mapping, batch_src_global)
local_dst = np.searchsorted(mapping, batch_dst_global)
# 2. Проверка что ноды есть в маппинге (val/test ноды без train-истории)
valid_src = np.isin(batch_src_global, mapping)
valid_dst = np.isin(batch_dst_global, mapping)
valid = valid_src & valid_dst
# 3. Вычисление pair features (только для valid пар, остальные = 0)
cn = np.zeros(len(batch_src_global), dtype=np.float32)
aa = np.zeros(len(batch_src_global), dtype=np.float32)
if valid.any():
    cn[valid] = compute_cn(adj, local_src[valid], local_dst[valid])
    aa[valid] = compute_aa(adj, local_src[valid], local_dst[valid])
```

**Полный набор фичей для модели (на одну пару src→dst):**
- 15 node features src + 15 node features dst = 30 (из features parquet)
- CN_undirected, AA_undirected, CN_directed, AA_directed = 4 pair features
- **Итого: 34 фичи на пару**

**Запуск:**
```bash
YADISK_TOKEN="..." PYTHONPATH=. python scripts/compute_stream_adjacency.py \
    --input /tmp/stream_graph_full.parquet \
    --output-dir /tmp/stream_adjacency/ --upload \
    2>&1 | tee /tmp/stream_adjacency.log
```

**Статус (2026-03-29):** Код готов, 29 тестов passing. Ждёт запуска на дев-машине.

### Tests

235 tests total — all passing:
- `tests/test_pipeline.py` — 11 tests for the data pipeline
- `tests/test_compute_features.py` — 36 tests for feature computation (correctness,
  disk cleanup, resume, edge cases)
- `tests/test_baselines.py` — 35 tests for baseline pipeline (feature engineering,
  ranking metrics, experiment logger, config, link prediction helpers, heuristic helpers)
- `tests/test_models.py` — 42 tests for DL models (GraphMixer architecture, TemporalCSR,
  neighbor sampling, featurization, C++ extension correctness, chronological split)
- `tests/test_stream_graph.py` — 16 tests for stream graph pipeline (date filtering,
  entity 0 removal, self-loops, timestamp sorting, node mapping, CSV/parquet formats)
- `tests/test_temporal_sampler.py` — 30 tests for CUDA sampling module (Python/C++/CUDA
  backend equivalence, negative sampling, edge cases, cross-validation)
- `tests/test_stream_node_features.py` — 36 tests for stream node features (shape, dtype,
  finite, inactive nodes, hand-crafted per-feature checks, sparse format, load_node_features)
- `tests/test_stream_adjacency.py` — 29 tests for adjacency matrices (CSR format, binary,
  directed/undirected, CN/AA correctness, mapping consistency with node features, edge cases)

---

## Project structure

```
src/
  build_pipeline.py     # Main pipeline (download/extract/mapping/snapshots/upload)
  build_stream_graph.py # Stream graph pipeline (download/extract/process/upload)
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
  models/
    __init__.py         # DL models package
    graphmixer.py       # GraphMixer architecture (MLP-Mixer, no attention/GNN)
    data_utils.py       # Event stream, TemporalCSR, neighbor sampling, featurization
    temporal_graph_sampler.py  # Unified sampler: Python/C++/CUDA backends
    train.py            # Training loop, validation, early stopping
    evaluate.py         # TGB-style evaluation (50 hist + 50 random negatives)
    launcher.py         # CLI for running experiments
    build_ext.py        # C++/CUDA extension build script (--cuda / --all)
    csrc/
      temporal_sampling.cpp  # C++ pybind11: TemporalCSR, batch sampling, featurization
      temporal_sampling.cu   # CUDA: GPU-parallel sampling + featurization
    GLFormer/           # GLFormer model (standard C++/Python sampling)
    GLFormer_cuda/      # GLFormer with CUDA-accelerated sampling
tests/
  test_pipeline.py      # 11 tests for the pipeline
  test_compute_features.py  # 36 tests for feature computation
  test_baselines.py     # 35 tests for baseline pipeline
  test_models.py        # 42 tests for DL models and C++ extension
  test_stream_graph.py  # 16 tests for stream graph pipeline
  test_stream_node_features.py  # 36 tests for stream node features
  test_stream_adjacency.py      # 29 tests for adjacency matrices and pair features
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

### Code quality

- **Write tests** for any new functionality. Place them in `tests/`.
- Run `pytest tests/ -v` before proposing changes as ready.
- **Always activate venv** before running any Python commands: `source venv/bin/activate && <command>`. Do NOT use system Python or pip. After `pip install -e .` the `src` package is importable without `PYTHONPATH=.`.
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

### Security

- Do NOT commit secrets (tokens, IPs, passwords). Use environment variables.
- Sensitive values (Yandex.Disk token, dev machine IP) are stored as env vars or provided by the user at runtime.

---

## Technical reference

### Dev machines

**GPU машина** (для DL обучения): immers.cloud
- **Конфиг:** V100-PCIE-32GB, 32GB RAM, 128GB SSD, Ubuntu 24.04 + CUDA 12.8
- **ВАЖНО:** V100 = Volta (compute capability 7.0). Нужен **проприетарный** драйвер
  (`nvidia-driver-570-server`), НЕ open (`nvidia-driver-570-open` несовместим с Volta).
- **ВАЖНО:** 16GB RAM НЕ хватает для 4.7M рёбер с undirected=True (OOM при загрузке).
  Минимум 32GB RAM.
- **Setup:** `bash scripts/setup_v100.sh` — полная автоматическая настройка (драйвер,
  venv, PyTorch 2.5.1+cu121, PyG, C++/CUDA extensions, тесты). Если драйвер требует
  reboot — перезапустить скрипт после перезагрузки.
- **PyTorch:** 2.5.1+cu121 (TORCH_CUDA_ARCH_LIST="7.0")
- **Расширения:** temporal_sampling_cpp + temporal_sampling_cuda (30/30 тестов)

**A10 (Ampere CC 8.6) — уроки из настройки (2026-03-28):**
- Setup-скрипт для A10: `bash src/models/cuda_exp_graphmixer_a10/setup_a10.sh`
- Перед установкой драйвера скрипт проверяет `nvidia-smi` — если работает, пропускает установку.
  Immers.cloud A10 поставляется с nvidia-driver-570 уже установленным.
- **ВАЖНО:** нужен `python3.12-dev` для компиляции C++/CUDA расширений (без него `Python.h` не найден).
  Скрипт устанавливает его автоматически.
- **OOM:** 3 месяца stream graph (~56M directed edges, ~112M undirected) не помещаются в RAM.
  Для бенчмарка использовать 1 неделю (`stream_graph/week.parquet`, 4.7M directed, 9.4M undirected).
  Нарезать через `scripts/slice_stream_graph.py`.

**Скачивание файла с Яндекс.Диска на машину (нет CLI в yadisk_utils.py):**
```bash
# Сохранить длинный путь в переменную (избежать проблем с переносом строк при вставке):
P="orbitaal_processed/stream_graph/2020-06-01__2020-08-31.parquet"
# Нарезать нужный период и скачать:
PYTHONPATH=. python scripts/slice_stream_graph.py --yadisk-path "$P" --start 2020-07-01 --end 2020-07-07 --output stream_graph/week.parquet
```
**Причина проблем:** `yadisk_utils.py` не имеет CLI (`if __name__ == "__main__"` отсутствует).
Многострочные команды с `\` разбиваются терминалом при вставке — всегда использовать переменные
для длинных путей и однострочные команды.

Общее:
- **Access:** SSH with key-based auth
- **Python venv** is set up on the machine with all dependencies
- **tmux** is used for long-running processes
- The agent does not connect to the dev machine. Provide commands to copy-paste.

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
  stream_graph/
    2020-06-01__2020-08-31.parquet             # stream graph (sorted by UNIX timestamp)
    2020-06-01__2020-08-31.json                # statistics
    features_10.parquet                        # node features from train of first 10% edges (1.9M nodes, 60 MB)
    features_10.json                           # metadata
    features_25.parquet                        # node features from train of first 25% edges (4.3M nodes, 141 MB)
    features_25.json                           # metadata
    adj_10_directed.npz                        # CSR binary adjacency, directed, train of 10%
    adj_10_undirected.npz                      # CSR binary adjacency, undirected (symmetric)
    node_mapping_10.npy                        # local→global index mapping for 10%
    adj_10.json                                # metadata
    adj_25_directed.npz                        # same for 25%
    adj_25_undirected.npz
    node_mapping_25.npy
    adj_25.json
  experiments/                                 # Результаты экспериментов (создаётся launcher.py)
    exp_***_<short_desc>
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

## Next steps (2026-03-29)

1. ~~Baselines~~ **Done** (33/35). ML бейзлайны — needs review.
2. ~~GraphMixer~~ **Done**. Test MRR=0.430 (vs CN=0.732). Первый DL baseline.
3. ~~Stream-graph pipeline~~ **Done.** Результат на Яндекс.Диске.
4. ~~CUDA temporal sampling module~~ **Done.** 30/30 тестов, бенчмарк 50-1700x speedup.
5. ~~GLFormer_cuda pipeline~~ **Done.** Код готов, эксперимент проведён.
6. ~~GPU машина setup~~ **Done.** `scripts/setup_v100.sh` работает, все extensions собраны.
7. ~~exp_005 / exp_006 GLFormer comparison~~ **Done (2026-03-27).** CUDA speedup = 0% при K=20 (ожидаемо).
8. **GLFormer нормальное обучение** (задача коллеги). Нужно: batch_size=4000, K=20, 100 эпох,
   patience=20, lr=0.0001, на 1-недельных данных (2020-07-01_2020-07-07). Ожидаемое время: ~6 часов.
9. **DyGFormer** — следующая основная DL-модель. Требует реализации с нуля под наш формат данных.
   CUDA-сэмплинг при K=512 даёт speedup в sampling, но transformer-forward доминирует → реальный
   прирост скромный. Тем не менее модель сильнее GLFormer (interaction-aware, K=512).
10. ~~Static node features for stream graph~~ **Done (2026-03-29).** 15 фичей, sparse формат,
    features_10 (60 MB) + features_25 (141 MB) на Яндекс.Диске.
11. **Adjacency matrices for pair features (CN, AA).** Код готов, 29 тестов. Ждёт запуска.
12. Graph-level forecasting **deprioritized**.
12. Pre-2010 period: filter to 2010+ or 2010-07-17+ (sparse/empty graphs before that).
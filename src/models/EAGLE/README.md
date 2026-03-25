# EAGLE-Time: Temporal Link Prediction на Stream Graph

Реализация модели **EAGLE-Time** из статьи *"EAGLE: Expressive dynamics-Aware Graph LEarning" (Yue et al., 2024)* для задачи предсказания ссылок (temporal link prediction) на потоковом графе транзакций.

---

## Архитектура

EAGLE-Time использует только **временные паттерны взаимодействий** (delta times) — без признаков рёбер и узлов. Это делает модель быстрее GraphMixer и позволяет сосредоточиться на динамике графа.

```
Для каждой пары (src, dst) при времени t:
  1. Найти K последних соседей src до t  → delta times (t - t_neighbor_i)
  2. Найти K последних соседей dst до t  → delta times
  3. Закодировать каждый набор через EAGLETimeEncoder:
       cos(delta_t * omega) → Linear(100) → MLP-Mixer × L слоёв → mean-pool → Linear
  4. Предсказать оценку: relu(fc_src(h_src) + fc_dst(h_dst)) → Linear(1)
```

### Модули

| Файл | Описание |
|------|----------|
| [eagle.py](eagle.py) | Архитектура: `EAGLETimeEncoding`, `EAGLEMixerBlock`, `EAGLETimeEncoder`, `EAGLETime` |
| [eagle_train.py](eagle_train.py) | Цикл обучения, валидация, ранняя остановка |
| [eagle_evaluate.py](eagle_evaluate.py) | TGB-style оценка (50 hist + 50 random negatives) |
| [eagle_launcher.py](eagle_launcher.py) | CLI-точка входа для запуска экспериментов |
| [eagle_hpo.py](eagle_hpo.py) | Подбор гиперпараметров через Optuna |
| [tppr.py](tppr.py) | Temporal Personalized PageRank (структурный скоринг) |
| [data_utils.py](data_utils.py) | Загрузка stream graph, конвертация в `TemporalEdgeData` |

### Режимы признаков

| `edge_feat_dim` | `node_feat_dim` | Описание |
|:---:|:---:|----------|
| 0 | 0 | Только временные паттерны (оригинальный EAGLE-Time) |
| 2 | 0 | + признаки рёбер-соседей (btc/usd каждой транзакции из истории) |
| 0 | D | + собственные признаки узла (D-мерный вектор из `node_feats`) |
| 2 | D | Все три сигнала |

**Признаки рёбер** (`edge_feat_dim > 0`): для каждого из K соседних взаимодействий берётся вектор признаков этого ребра (btc, usd) и конкатенируется к временному кодированию `cos(Δt·ω)`. Это обогащает MLP-Mixer информацией об объёме транзакций.

**Признаки узлов** (`node_feat_dim > 0`): собственный вектор каждого запрашиваемого узла (из `data.node_feats`) проецируется линейным слоем и прибавляется к pooled embedding после MLP-Mixer. Схема аналогична NodeEncoder в GraphMixer.

### Параметры модели (по умолчанию)

| Параметр | Значение | Описание |
|----------|----------|----------|
| `hidden_dim` | 100 | Размер скрытых представлений |
| `num_neighbors` | 20 | K последних соседей |
| `num_mixer_layers` | 1 | Количество MLP-Mixer блоков |
| `time_dim` | 100 | Размерность временного кодирования |
| `token_expansion` | 0.5 | Расширение в token-mixing MLP |
| `channel_expansion` | 4.0 | Расширение в channel-mixing MLP |
| `dropout` | 0.1 | Dropout |
| `edge_feat_dim` | 0 | Размерность признаков рёбер (0 = отключено) |
| `node_feat_dim` | 0 | Размерность признаков узлов (0 = отключено) |

~120K параметров в режиме time-only; при `edge_feat_dim=2, node_feat_dim=26` ~135K.

---

## Формат данных

Модель обучается на **stream graph** — файле parquet с транзакциями, каждая из которых имеет точный Unix-timestamp.

```python
import pandas as pd
import torch
from torch_geometric.data import TemporalData

df = pd.read_parquet("stream_graph/2020-06-01_2020-08-31.parquet")
data = TemporalData(
    src=torch.tensor(df["src_idx"].values, dtype=torch.long),
    dst=torch.tensor(df["dst_idx"].values, dtype=torch.long),
    t=torch.tensor(df["timestamp"].values, dtype=torch.long),
    msg=torch.tensor(df[["btc", "usd"]].values, dtype=torch.float),
)
```

Колонки parquet-файла:

| Колонка | Тип | Описание |
|---------|-----|----------|
| `src_idx` | int | Индекс отправителя (0..N-1 из node_mapping) |
| `dst_idx` | int | Индекс получателя |
| `timestamp` | int64 | Unix-время транзакции |
| `btc` | float | Объём в BTC |
| `usd` | float | Объём в USD |

Данные автоматически сортируются по `timestamp` и (при `undirected=True`) дополняются обратными рёбрами. Разбивка: **70% train / 15% val / 15% test** по хронологии.

---

## Установка зависимостей

```bash
# Базовые зависимости
pip install -r requirements.txt

# torch_geometric (необходима для TemporalData)
pip install torch_geometric

# Для HPO
pip install optuna

# C++ ускорение (рекомендуется, ускоряет в 3-5x)
python src/models/build_ext.py
```

---

## Быстрый старт

### Обучение (time-only, без признаков)

```bash
PYTHONPATH=. python src/models/EAGLE/eagle_launcher.py \
    --parquet-path stream_graph/2020-06-01_2020-08-31.parquet \
    --epochs 100 \
    --output /tmp/eagle_results \
    2>&1 | tee /tmp/eagle_train.log
```

### Обучение с признаками рёбер (btc + usd)

```bash
PYTHONPATH=. python src/models/EAGLE/eagle_launcher.py \
    --parquet-path stream_graph/2020-06-01_2020-08-31.parquet \
    --edge-feat-dim 2 \
    --epochs 100 \
    --output /tmp/eagle_results_ef \
    2>&1 | tee /tmp/eagle_train_ef.log
```

### Обучение с признаками рёбер + узлов

```bash
PYTHONPATH=. python src/models/EAGLE/eagle_launcher.py \
    --parquet-path stream_graph/2020-06-01_2020-08-31.parquet \
    --edge-feat-dim 2 \
    --node-feat-dim 26 \
    --epochs 100 \
    --output /tmp/eagle_results_full \
    2>&1 | tee /tmp/eagle_train_full.log
```

> **Примечание:** Признаки узлов (`--node-feat-dim`) требуют, чтобы в parquet-файле были загружены реальные векторы узлов. По умолчанию `data.node_feats` содержит нули — загрузите данные через `data_utils.temporal_data_to_edge_data` с передачей `node_feats` вручную или используйте `build_event_stream` из `src.models.data_utils` с включёнными node features.

### Загрузка результатов на Яндекс.Диск

```bash
YADISK_TOKEN="ваш_токен" PYTHONPATH=. python src/models/EAGLE/eagle_launcher.py \
    --parquet-path stream_graph/2020-06-01_2020-08-31.parquet \
    --epochs 100 \
    --output /tmp/eagle_results
```

### Подбор гиперпараметров (Optuna)

```bash
PYTHONPATH=. python src/models/EAGLE/eagle_hpo.py \
    --parquet-path stream_graph/2020-06-01_2020-08-31.parquet \
    --n-trials 30 \
    --hpo-epochs 15 \
    --output /tmp/eagle_hpo \
    2>&1 | tee /tmp/eagle_hpo.log
```

После завершения HPO в `/tmp/eagle_hpo/best_train_command.sh` будет готовая команда для финального обучения с лучшими параметрами.

### TPPR (структурный скоринг)

```bash
PYTHONPATH=. python src/models/EAGLE/tppr.py \
    --parquet-path stream_graph/2020-06-01_2020-08-31.parquet \
    --topk 100 --alpha 0.9 --beta 0.8 \
    --output /tmp/eagle_tppr \
    2>&1 | tee /tmp/eagle_tppr.log
```

---

## CLI-параметры (eagle_launcher.py)

### Данные

| Флаг | По умолчанию | Описание |
|------|-------------|----------|
| `--parquet-path` | (обязательный) | Путь к stream graph parquet-файлу |
| `--train-ratio` | 0.7 | Доля рёбер для обучения |
| `--val-ratio` | 0.15 | Доля рёбер для валидации |
| `--output` | `/tmp/eagle_results` | Папка для результатов |

### Гиперпараметры модели

| Флаг | По умолчанию | Описание |
|------|-------------|----------|
| `--hidden-dim` | 100 | Размер скрытого представления |
| `--num-neighbors` | 20 | K соседей для сэмплинга |
| `--num-mixer-layers` | 1 | Количество MLP-Mixer блоков |
| `--token-expansion` | 0.5 | Token-mixing expansion |
| `--channel-expansion` | 4.0 | Channel-mixing expansion |
| `--dropout` | 0.1 | Dropout rate |
| `--edge-feat-dim` | 0 | Размерность признаков рёбер (0 = отключено, 2 = btc+usd) |
| `--node-feat-dim` | 0 | Размерность признаков узлов (0 = отключено) |

### Обучение

| Флаг | По умолчанию | Описание |
|------|-------------|----------|
| `--epochs` | 100 | Максимальное число эпох |
| `--batch-size` | 200 | Рёбер в батче |
| `--lr` | 0.001 | Learning rate (Adam) |
| `--weight-decay` | 5e-5 | Weight decay |
| `--patience` | 10 | Early stopping patience |
| `--seed` | 42 | Random seed |
| `--no-amp` | (флаг) | Отключить mixed precision |
| `--max-val-edges` | 5000 | Макс. рёбер для оценки на val |

---

## Структура результатов

После запуска в `--output/eagle_<parquet_name>/`:

```
eagle_2020-06-01_2020-08-31/
  config.json           # гиперпараметры эксперимента
  data_summary.json     # статистика датасета (nodes, edges, splits)
  metrics.jsonl         # метрики по эпохам (JSONL)
  training_curves.csv   # кривые обучения (CSV для построения графиков)
  best_model.pt         # веса лучшей модели (по val MRR)
  final_results.json    # тестовые метрики + timing
  experiment.log        # полный лог эксперимента
```

### Метрики оценки

Используется **TGB-style ranking protocol** (идентичен бейзлайнам):
- Для каждого положительного ребра `(src, dst_true, t)` генерируются 50 исторических + 50 случайных негативных кандидатов.
- Ранжируются 101 кандидат, определяется ранг `dst_true`.
- **Метрики:** MRR (primary), Hits@1, Hits@3, Hits@10.

---

## Программный интерфейс

```python
from src.models.EAGLE.data_utils import load_stream_graph_data, TemporalCSR
from src.models.EAGLE.eagle_train import train_eagle
from src.models.EAGLE.eagle_evaluate import evaluate_tgb_style
import numpy as np
import torch

# Загрузка данных
data, train_mask, val_mask, test_mask = load_stream_graph_data(
    "stream_graph/2020-06-01_2020-08-31.parquet",
    train_ratio=0.7,
    val_ratio=0.15,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Вариант 1: только временные паттерны ---
model, history = train_eagle(
    data=data, train_mask=train_mask, val_mask=val_mask,
    output_dir="/tmp/eagle_time_only", device=device, num_epochs=50,
)

# --- Вариант 2: временные паттерны + признаки рёбер (btc, usd) ---
model, history = train_eagle(
    data=data, train_mask=train_mask, val_mask=val_mask,
    output_dir="/tmp/eagle_edge_feats", device=device, num_epochs=50,
    edge_feat_dim=2,  # размерность data.edge_feats (btc + usd)
)

# --- Вариант 3: все признаки ---
# (требует загрузить реальные node_feats в data.node_feats)
model, history = train_eagle(
    data=data, train_mask=train_mask, val_mask=val_mask,
    output_dir="/tmp/eagle_full", device=device, num_epochs=50,
    edge_feat_dim=2,
    node_feat_dim=data.node_feats.shape[1],  # 1 (заглушка) или 26 (реальные)
)

# Оценка на тестовом наборе
all_before_test = train_mask | val_mask
test_csr = TemporalCSR(
    data.num_nodes,
    data.src[all_before_test],
    data.dst[all_before_test],
    data.timestamps[all_before_test],
    np.where(all_before_test)[0].astype(np.int64),
)
metrics = evaluate_tgb_style(model, data, test_csr, test_mask, device)
print(f"MRR={metrics['mrr']:.4f}  Hits@10={metrics['hits@10']:.3f}")
```

---

## Зависимости

Ключевые зависимости модуля:

- `torch >= 2.0.0` — модель и обучение
- `torch_geometric` — `TemporalData` для stream graph
- `numpy >= 1.24.0` — работа с массивами
- `pandas >= 2.0.0`, `pyarrow >= 12.0.0` — чтение parquet
- `tqdm >= 4.65.0` — прогресс-бары
- `optuna` — HPO (только для `eagle_hpo.py`)
- `src.models.data_utils` — `TemporalEdgeData`, `TemporalCSR`, C++ extension
- `src.baselines.evaluation` — `compute_ranking_metrics`
- `src.yadisk_utils` — загрузка результатов на Яндекс.Диск

C++ расширение (`src/models/csrc/temporal_sampling.cpp`) даёт 3-5x ускорение сэмплинга соседей. Если не скомпилировано — автоматически используется Python-fallback.

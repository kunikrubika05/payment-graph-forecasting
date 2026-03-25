# GraphMixer — временное предсказание связей на stream graph

Адаптация модели **GraphMixer** для работы с stream graph parquet-форматом.

Исходная статья:
> Cong et al. "Do We Really Need Complicated Model Architectures For Temporal Networks?" ICLR 2023.

---

## Архитектура

### Идея

GraphMixer заменяет сложные GNN-механизмы на простой **MLP-Mixer** поверх K последних рёбер узла. Главный тезис статьи: внимание к *временным рёбрам* (какие взаимодействия были?) важнее структурной агрегации (с кем?).

Ключевые компоненты:

- **LinkEncoder**: MLP-Mixer over K most-recent edges per node.
  - `FeatEncoder`: конкатенирует cosine time encoding и edge features → project to hidden_dim.
  - `MixerBlock × L`: token-mixing (across K positions) + channel-mixing (across hidden_dim), оба через 2-слойный MLP с GeLU.
  - Masked mean-pool → [B, hidden_dim] эмбеддинг узла.
- **NodeEncoder** (опциональный, `--node-feat-dim > 0`):
  - Проецирует `own_node_feats + mean(neighbor_node_feats)` → hidden_dim.
  - В stream graph формате node features = нули → не рекомендуется.
- **LinkClassifier** (аддитивный предиктор):
  - `score = out(ReLU(fc_src(h_src) + fc_dst(h_dst)))` — скаляр (логит).
  - Сравните с конкатенационным предиктором GLFormer: additive predictor не моделирует src-dst взаимодействие явно.

### Блоки

```
Вход: K последних соседей узла u в момент t
        ↓
FixedTimeEncoding: cos(delta_t * omega_i) → [B, K, time_dim]
        ↓
FeatEncoder: Linear([time_enc, edge_feats]) → [B, K, hidden_dim]   ← токены
        ↓
MixerBlock × L:
    ┌── LayerNorm + token-mixing MLP (across K dim, транспонируем)
    ├── residual
    ├── LayerNorm + channel-mixing MLP (across hidden_dim)
    └── residual
        ↓
LayerNorm + masked mean pool → [B, hidden_dim]
        ↓
(опц., node_feat_dim > 0) + NodeEncoder(own_feats + mean_neighbor_feats)
        ↓
Эмбеддинг узла Z_u ∈ R^{hidden_dim} (или R^{2·hidden_dim})
```

**Предиктор ребра (аддитивный):**
```
score(u,v) = out(ReLU(fc_src(Z_u) + fc_dst(Z_v)))  →  скаляр (логит)
```

---

## Отличия от EAGLE и GLFormer

| Аспект | EAGLE-Time | GLFormer | GraphMixer |
|--------|-----------|----------|-----------:|
| Token mixer | MLP-Mixer (параллельный) | Adaptive Token Mixer (каузальный, time-aware) | MLP-Mixer (параллельный) |
| Временное взвешивание | Нет | softmax(-Δt) в ATM | Нет (delta_t как вход в FeatEncoder) |
| Слои | Независимые | Иерархические (dilated) | Независимые |
| Предиктор | Аддитивный | Конкатенационный | Аддитивный |
| NodeEncoder | Нет | Нет | Опциональный |
| Learning rate | 0.001 | 0.0001 | 0.0001 |
| Число параметров | ~150K | ~120K | ~100K |

---

## Формат данных

Stream graph parquet-файл (стандарт проекта):

| Колонка    | Тип     | Описание                                    |
|-----------|---------|---------------------------------------------|
| `src_idx` | int64   | Источник (dense индекс 0..N-1)              |
| `dst_idx` | int64   | Назначение (dense индекс 0..N-1)            |
| `timestamp` | int64 | UNIX-timestamp транзакции                   |
| `btc`     | float32 | Сумма транзакции в BTC                      |
| `usd`     | float32 | Сумма транзакции в USD                      |

---

## Разбивка данных

Хронологическая (по позиции ребра в сортированном по времени массиве):

- **Train**: первые 70% рёбер
- **Val**: следующие 15%
- **Test**: последние 15%

Параметры: `--train-ratio 0.7 --val-ratio 0.15`.

CSR-структуры:
- При обучении: CSR только из train-рёбер
- При val/test: CSR из train+val рёбер (тестовые рёбра никогда не попадают в CSR)

---

## Режимы признаков

| Режим | `--edge-feat-dim` | `--node-feat-dim` | Описание |
|-------|------------------|------------------|----------|
| Временно́й | 0 | 0 | только delta times |
| Рёберные признаки | 2 | 0 | btc+usd каждого соседа **(рекомендуется)** |
| С node features | 2 | >0 | + NodeEncoder (не рекомендуется для stream graph) |

**Рекомендуемая конфигурация** для нашего датасета: `--edge-feat-dim 2 --node-feat-dim 0`.

---

## Установка зависимостей

```bash
pip install -r requirements.txt
python src/models/build_ext.py  # компиляция C++ TemporalCSR (один раз)
```

---

## Запуск

### HPO (подбор гиперпараметров)

```bash
PYTHONPATH=. python src/models/GraphMixer/graphmixer_hpo.py \
    --parquet-path stream_graph/2020-06-01_2020-08-31.parquet \
    --edge-feat-dim 2 \
    --n-trials 30 --hpo-epochs 15 \
    --output /tmp/graphmixer_hpo \
    2>&1 | tee /tmp/graphmixer_hpo.log
```

HPO перебирает: `hidden_dim`, `num_neighbors`, `num_mixer_layers`, `lr`, `weight_decay`, `dropout`, `batch_size`. Результат — файл `hpo_results.json` и скрипт `best_train_command.sh`.

### Обучение

```bash
YADISK_TOKEN="..." PYTHONPATH=. python src/models/GraphMixer/graphmixer_launcher.py \
    --parquet-path stream_graph/2020-06-01_2020-08-31.parquet \
    --edge-feat-dim 2 \
    --epochs 100 \
    --num-mixer-layers 2 \
    --hidden-dim 100 \
    --num-neighbors 20 \
    --lr 0.0001 \
    --output /tmp/graphmixer_results \
    2>&1 | tee /tmp/graphmixer_train.log
```

---

## Параметры CLI

### `graphmixer_launcher.py`

| Параметр | По умолчанию | Описание |
|----------|-------------|----------|
| `--parquet-path` | обязательный | Путь к stream graph parquet |
| `--train-ratio` | 0.7 | Доля рёбер для обучения |
| `--val-ratio` | 0.15 | Доля рёбер для валидации |
| `--output` | /tmp/graphmixer_results | Директория для результатов |
| `--epochs` | 100 | Максимальное число эпох |
| `--batch-size` | 200 | Размер батча |
| `--lr` | 0.0001 | Learning rate Adam |
| `--weight-decay` | 1e-5 | L2-регуляризация |
| `--num-neighbors` | 20 | K соседей на узел |
| `--hidden-dim` | 100 | Размерность скрытого слоя |
| `--num-mixer-layers` | 2 | Число MLP-Mixer блоков (1-3) |
| `--dropout` | 0.1 | Dropout |
| `--patience` | 20 | Early stopping (эпох без улучшения val MRR) |
| `--seed` | 42 | Random seed |
| `--no-amp` | False | Отключить AMP (mixed precision) |
| `--max-val-edges` | 5000 | Макс. рёбер при валидации |
| `--edge-feat-dim` | 2 | Размерность признаков рёбер соседей |
| `--node-feat-dim` | 0 | Размерность признаков узла-запроса |

### `graphmixer_hpo.py`

Дополнительные параметры: `--n-trials` (30), `--hpo-epochs` (15).

---

## Структура выходных данных

```
/tmp/graphmixer_results/graphmixer_<parquet_stem>/
    config.json           # гиперпараметры и конфигурация модели
    best_model.pt         # чекпоинт лучшей модели по val MRR
    metrics.jsonl         # метрики по эпохам (JSON Lines)
    training_curves.csv   # кривые обучения для построения графиков
    data_summary.json     # статистика датасета
    final_results.json    # итоговые метрики (test MRR, Hits@K, время)
    summary.json          # best_epoch, best_val_mrr
    experiment.log        # лог эксперимента
```

На Яндекс.Диске: `orbitaal_processed/experiments/exp_007_graphmixer/<exp_name>/`

---

## API

```python
from src.models.GraphMixer import GraphMixerTime
from src.models.GraphMixer.data_utils import load_stream_graph_data, build_temporal_csr
from src.models.GraphMixer.graphmixer_train import train_graphmixer
from src.models.GraphMixer.graphmixer_evaluate import evaluate_tgb_style

# Загрузка данных
data, train_mask, val_mask, test_mask = load_stream_graph_data(
    "stream_graph/2020-06-01_2020-08-31.parquet"
)

# Обучение
model, history = train_graphmixer(
    data=data,
    train_mask=train_mask,
    val_mask=val_mask,
    output_dir="/tmp/graphmixer_exp",
    edge_feat_dim=2,
    num_mixer_layers=2,
    hidden_dim=100,
)

# Оценка
import torch
test_csr = build_temporal_csr(data, train_mask | val_mask)
metrics = evaluate_tgb_style(
    model=model, data=data, csr=test_csr, eval_mask=test_mask,
    device=torch.device("cuda"),
)
print(f"Test MRR: {metrics['mrr']:.4f}")
```

---

## Зависимости

- `torch >= 2.0`
- `torch_geometric` (для TemporalData в data_utils)
- `numpy`, `pandas`, `tqdm`
- `optuna` (только для graphmixer_hpo.py)
- Скомпилированное C++ расширение (`python src/models/build_ext.py`)

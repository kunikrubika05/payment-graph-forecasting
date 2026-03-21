# GraphMixer: пайплайн temporal link prediction

## Обзор

GraphMixer (Cong et al., ICLR 2023, arXiv:2302.11636) — модель для temporal link prediction,
использующая только MLP (без attention и GNN). Реализована в `src/models/`.

**Задача:** для заданного ребра (A → B, время T) предсказать, появится ли оно.

**Целевой бейзлайн:** Common Neighbors (MRR 0.46–0.73 на наших данных).

## Архитектура модели

GraphMixer состоит из 3 компонентов:

### 1. LinkEncoder (MLP-Mixer)

Для каждого узла берёт K=20 последних рёбер (до момента запроса) и кодирует их:

- **Вход:** K рёбер с признаками `[btc, usd]` и относительными временными метками `Δt = t_query - t_edge`
- **Time Encoding:** `cos(Δt × ω)` с экспоненциально распределёнными частотами, 100 компонент, нетренируемые
- **FeatEncoder:** конкатенация `[time_enc, edge_feats]` → Linear → 100-dim вектор для каждого ребра
- **MLP-Mixer:** 2 блока, каждый содержит:
  - Token-mixing: MLP через ось K (определяет важность рёбер)
  - Channel-mixing: MLP через ось hidden_dim (комбинирует признаки)
  - Residual connections + LayerNorm
- **Mean-pool** → 100-dim вектор

### 2. NodeEncoder

- **Вход:** собственные node features (25 признаков) + средние node features K ближайших соседей
- **Обработка:** `node_feat + mean(neighbor_feats)` → Linear → 100-dim вектор

### 3. LinkClassifier

- **Представление узла:** `h = [link_enc; node_enc]` — конкатенация, 200-dim
- **Скор пары:** `score = MLP(fc_src(h_src) + fc_dst(h_dst))` → скаляр
- Аддитивная схема (как в TransE/DistMult)

### Размер модели

| Компонент | Параметры | Размер |
|-----------|-----------|--------|
| LinkEncoder | 95,220 | 372 KB |
| NodeEncoder | 2,600 | 10 KB |
| LinkClassifier | 40,301 | 157 KB |
| **Итого** | **138,121** | **540 KB** |

## Данные

### Формат event stream

Дневные снимки ORBITAAL конвертируются в единый поток событий:

- Каждое ребро из `daily_snapshots/*.parquet` → событие `(src, dst, timestamp=day_index, btc, usd)`
- `undirected=True` → для каждого ребра добавляется обратное (×2 рёбер, +11% AP по статье)
- Node features (25 признаков) усредняются по всем дням, где узел активен
- Узлы ремаппятся в dense `0..N-1`

### Edge features (2 признака)

- `btc` — объём транзакции в BTC
- `usd` — объём транзакции в USD

### Node features (25 признаков)

`in_degree`, `out_degree`, `total_degree`, `weighted_in_btc`, `weighted_out_btc`,
`weighted_in_usd`, `weighted_out_usd`, `balance_btc`, `balance_usd`,
`avg_in_btc`, `avg_out_btc`, `median_in_btc`, `median_out_btc`,
`max_in_btc`, `max_out_btc`, `min_in_btc`, `min_out_btc`,
`std_in_btc`, `std_out_btc`, `unique_in_counterparties`, `unique_out_counterparties`,
`pagerank`, `clustering_coeff`, `k_core`, `triangle_count`.

## Пайплайн обучения

### Шаг 1: Подготовка данных (CPU)

```bash
YADISK_TOKEN="..." PYTHONPATH=. python src/models/launcher.py --period mature_2020q2
```

1. Скачивание `daily_snapshots/` и `node_features/` с Яндекс.Диска для периода
2. `build_event_stream()` → `TemporalEdgeData`
3. `chronological_split(data, 0.6, 0.2)` → train/val/test маски

### Шаг 2: Построение TemporalCSR (CPU)

CSR-подобная структура для запросов "K последних соседей узла до момента T":

- Binary search по отсортированным timestamp для каждого узла
- **C++ расширение** (`temporal_sampling_cpp`) для ~3-5x ускорения vs Python
- Гарантия отсутствия temporal leakage

### Шаг 3: Обучение (GPU)

Для каждой эпохи:

1. Shuffle тренировочных рёбер
2. Мини-батчи (600 рёбер):
   - Positive: ребро `(src, dst, t)` из данных
   - Negative: `(src, random_node, t)` — 1 random negative на positive
   - Для каждого узла: sample K=20 ближайших соседей из CSR
   - Forward через модель → pos_score, neg_score
   - **Loss:** `BCEWithLogitsLoss`, gradient clipping `max_norm=1.0`
3. Валидация: subsample до 5000 val-рёбер, 100 random negatives, MRR
4. **Early stopping:** patience=20 по val_MRR

### Шаг 4: TGB-style evaluation (GPU)

Для каждого тестового ребра `(src, dst_true, t)`:

1. Генерация 100 негативов: 50 исторических + 50 случайных
2. Candidate set: `{dst_true} ∪ {neg_1, ..., neg_100}` = 101 кандидат
3. Скоринг всех 101 пар через модель
4. Ранг `dst_true` (filtered ranking)
5. Метрики: **MRR** (primary), Hits@1, Hits@3, Hits@10

Протокол идентичен бейзлайнам — результаты напрямую сравнимы.

## C++ расширение

### Что ускоряет

- `TemporalCSR` — построение CSR, binary search по timestamp
- `sample_neighbors_batch` — пакетный сэмплинг K соседей для batch узлов
- `featurize_neighbors` — заполнение массивов node/edge features для соседей

### Сборка

```bash
source venv/bin/activate
python src/models/build_ext.py
```

Компилирует `src/models/csrc/temporal_sampling.cpp` через `torch.utils.cpp_extension`.
Результат кэшируется в `src/models/csrc/build/`. При последующих импортах загружается мгновенно.

### Fallback

Если C++ расширение не скомпилировано — автоматический fallback на Python/NumPy.
Функционально идентично, но медленнее в 3-5x.

## Запуск

### Локально (тесты)

```bash
source venv/bin/activate
python src/models/build_ext.py                    # компиляция C++ (один раз)
PYTHONPATH=. python -m pytest tests/test_models.py -v  # 42 теста
```

### На дев-машине (обучение)

```bash
source venv/bin/activate && pip install -r requirements.txt
python src/models/build_ext.py
export YADISK_TOKEN="..."
PYTHONPATH=. python src/models/launcher.py \
    --period mature_2020q2 \
    --output /tmp/graphmixer_results \
    2>&1 | tee /tmp/graphmixer.log
```

### Параметры CLI

| Параметр | Default | Описание |
|----------|---------|----------|
| `--period` | `mature_2020q2` | Период из `PERIODS` (10 вариантов) |
| `--window` | `7` | Контекстное окно (дни) |
| `--epochs` | `100` | Макс. эпох |
| `--batch-size` | `600` | Размер батча |
| `--lr` | `0.0001` | Learning rate (Adam) |
| `--num-neighbors` | `20` | K соседей на узел |
| `--hidden-dim` | `100` | Скрытая размерность |
| `--num-mixer-layers` | `2` | Число MLP-Mixer блоков |
| `--dropout` | `0.1` | Dropout |
| `--patience` | `20` | Early stopping patience |
| `--train-ratio` | `0.6` | Доля train |
| `--val-ratio` | `0.2` | Доля val |

### Результаты

Сохраняются в `--output` директорию:

```
graphmixer_mature_2020q2_w7/
  config.json          # гиперпараметры
  metrics.jsonl        # метрики по эпохам
  best_model.pt        # лучшая модель (по val MRR)
  summary.json         # итоги обучения
  final_results.json   # финальные метрики на test set
  experiment.log       # полный лог
```

## Структура модулей

```
src/models/
  __init__.py           # пакет
  graphmixer.py         # архитектура модели (FixedTimeEncoding, MLP-Mixer, LinkClassifier)
  data_utils.py         # event stream, TemporalCSR, neighbor sampling, featurization
  train.py              # training loop, validation, early stopping
  evaluate.py           # TGB-style evaluation (50 hist + 50 random negatives)
  launcher.py           # CLI для запуска экспериментов
  build_ext.py          # сборка C++ расширения
  csrc/
    temporal_sampling.cpp  # C++ pybind11: TemporalCSR, batch sampling, featurization
    build/                 # скомпилированные .so файлы (gitignored)
```

## Ссылки

- Статья: Cong et al., "Do We Really Need Complicated Model Architectures For Temporal Networks?", ICLR 2023
- arXiv: https://arxiv.org/abs/2302.11636
- Reference implementation: `GRAPH_MIXER/GraphMixer-main/`

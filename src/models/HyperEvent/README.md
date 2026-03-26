# HyperEvent

Реализация модели **HyperEvent** для предсказания временны́х связей в stream graph.

**Статья:** Gao, Wu, Ding — "HyperEvent: A Strong Baseline for Dynamic Link Prediction via Relative Structural Encoding" (arXiv:2507.11836, 2025).
**Репозиторий авторов:** https://github.com/jianjianGJ/HyperEvent

---

## Идея

HyperEvent — это простой, но конкурентоспособный бейзлайн, который **не использует обучаемые эмбеддинги узлов** и **не требует памяти о прошлых состояниях**. Вместо этого модель:

1. Для каждого запроса `(u*, v*, t*)` извлекает контекст `H = S_{u*} ∪ S_{v*}` — последние `n_latest` соседей каждого из запросных узлов из **таблицы смежности**.
2. Для каждого события `e = (u, v)` в контексте вычисляет **12-мерный вектор относительного структурного кодирования** (relational vector), описывающий структурные отношения между `{u, v}` и `{u*, v*}`.
3. Подаёт последовательность этих векторов в **стандартный Transformer**, который выдаёт логит вероятности существования связи.

---

## Архитектура

```
Stream graph events
        │
        ▼
  AdjacencyTable               ← n_neighbor последних соседей на узел
  (per-node circular buffer)
        │
        ▼
  Context extraction           ← n_latest событий от u* и v*
  H = S_{u*} ∪ S_{v*}         ← до 2·n_latest событий
        │
        ▼
  Relational vector (12-dim)   ← для каждого e=(u,v) в H:
  ┌────────────────────────────────────────────────────────────┐
  │ d0(u*,u)  d0(u*,v)  d0(v*,u)  d0(v*,v)  — прямое соседство│
  │ d1(u*,u)  d1(u*,v)  d1(v*,u)  d1(v*,v)  — 1-hop overlap   │
  │ d2(u*,u)  d2(u*,v)  d2(v*,u)  d2(v*,v)  — 2-hop overlap   │
  └────────────────────────────────────────────────────────────┘
        │
        ▼
  Linear(12 → d_model)         ← проекция
  + sinusoidal positional enc
        │
        ▼
  TransformerEncoder            ← n_layers слоёв, n_heads голов, GELU
        │
        ▼
  masked mean-pool → Linear(d_model → 1)   ← логит
```

### Корреляционные меры

| Мера | Формула |
|------|---------|
| `d0(a, b)` | `count(a ∈ adj[b]) / max(\|adj[b]\|, 1)` |
| `d1(a, b)` | `\|adj[a] ∩ adj[b]\| / max(\|adj[a]\|·\|adj[b]\|, 1)` |
| `d2(a, b)` | `\|2hop(a) ∩ 2hop(b)\| / max(\|2hop(a)\|·\|2hop(b)\|, 1)` |

`2hop(x) = ∪_{y ∈ adj[x]} adj[y][-k2:]`, где `k2 = floor(√n_neighbor)`.

---

## Сравнение с EAGLE и GLFormer

| Аспект | EAGLE | GLFormer | HyperEvent |
|--------|-------|----------|------------|
| Энкодер узла | MLP-Mixer над delta-time + edge feats | AdaptiveTokenMixer (причинный) | — (нет эмбеддингов узлов) |
| Входные фичи | delta-time, btc, usd | delta-time, btc, usd | 12-мерный relational vector |
| Предиктор | аддитивный MLP | конкатенация MLP | Transformer (глобальный) |
| Память о состоянии | TemporalCSR (полная история) | TemporalCSR | AdjacencyTable (n_neighbor записей) |
| Параметры (типично) | ~140K | ~200K | ~25K (d_model=64) |
| Порядок обучения | перемешанный | перемешанный | **хронологический** |
| Edge features (btc/usd) | опционально | опционально | не используются |

**Ключевое отличие обучения:** HyperEvent обрабатывает рёбра в хронологическом порядке (без перемешивания), потому что таблица смежности накапливает историю и должна отражать корректное состояние графа на момент каждого события.

---

## Параметры

| Параметр | По умолчанию | Описание |
|----------|-------------|----------|
| `--n-neighbor` | 20 | Ёмкость таблицы смежности (n_neighbor в статье) |
| `--n-latest` | 10 | Количество контекстных событий на узел |
| `--d-model` | 64 | Размерность скрытого слоя Transformer |
| `--n-heads` | 4 | Количество голов внимания |
| `--n-layers` | 3 | Количество слоёв Transformer encoder |
| `--dropout` | 0.1 | Dropout |
| `--lr` | 0.0001 | Learning rate Adam (как в статье) |
| `--weight-decay` | 1e-5 | Weight decay |
| `--epochs` | 50 | Максимальное число эпох |
| `--batch-size` | 200 | Рёбер на батч |
| `--patience` | 20 | Ранняя остановка |
| `--max-val-edges` | 5000 | Рёбер в val-метрике за эпоху |

### Выбор `n_neighbor`

Из ablation-исследования в статье:
- **Быстро меняющиеся графы** (частые взаимодействия): `n_neighbor = 10–20`
- **Стабильные реляционные графы**: `n_neighbor = 30–50`
- Для Bitcoin-транзакций рекомендуется начинать с `n_neighbor = 20`

---

## Запуск

### Обучение

```bash
YADISK_TOKEN="..." PYTHONPATH=. python src/models/HyperEvent/hyperevent_launcher.py \
    --parquet-path stream_graph/2020-06-01_2020-08-31.parquet \
    --n-neighbor 20 --n-latest 10 \
    --epochs 50 --batch-size 200 \
    --output /tmp/hyperevent_results \
    2>&1 | tee /tmp/hyperevent.log
```

### Подбор гиперпараметров

```bash
PYTHONPATH=. python src/models/HyperEvent/hyperevent_hpo.py \
    --parquet-path stream_graph/2020-06-01_2020-08-31.parquet \
    --n-trials 30 --hpo-epochs 10 \
    --output /tmp/hyperevent_hpo \
    2>&1 | tee /tmp/hyperevent_hpo.log
```

HPO ищет по:
- `n_neighbor` ∈ {10, 15, 20, 30, 50}
- `n_latest` ∈ {5, 8, 10}
- `d_model` ∈ {32, 64, 128}
- `n_heads` ∈ {2, 4, 8} (невалидные комбинации с `d_model` автоматически откидываются)
- `n_layers` ∈ [1, 3]
- `lr` ∈ [1e-4, 1e-2] (log-uniform)
- `weight_decay` ∈ [1e-6, 1e-3] (log-uniform)
- `dropout` ∈ [0.0, 0.3] (шаг 0.05)
- `batch_size` ∈ {100, 200, 400}

---

## Структура результатов

```
/tmp/hyperevent_results/
  hyperevent_<parquet_name>/
    config.json           # гиперпараметры модели
    data_summary.json     # статистика датасета
    training_curves.csv   # per-epoch метрики
    metrics.jsonl         # per-epoch метрики (JSONL)
    summary.json          # best epoch, best val MRR
    best_model.pt         # веса лучшей модели
    final_results.json    # итоговые метрики на тест-сете
    experiment.log        # полный лог эксперимента
```

Результаты загружаются на Яндекс.Диск в:
```
orbitaal_processed/experiments/exp_008_hyperevent/hyperevent_<parquet_name>/
```

---

## Протокол оценки

Идентичен EAGLE и GLFormer (TGB-style):
- **50 исторических негативов**: узлы из `adj[u*]`, исключая `true_dst`
- **50 случайных негативов**: равномерная выборка из всех узлов
- **Per-source ranking**: ранг `true_dst` среди 101 кандидата
- **Метрики**: MRR (primary), Hits@1, Hits@3, Hits@10
- **Streaming**: таблица смежности обновляется после каждого тест-ребра

---

## Особенности реализации

### AdjacencyTable vs TemporalCSR

HyperEvent использует простую таблицу смежности (кольцевой буфер) вместо полного TemporalCSR:

| | TemporalCSR (EAGLE/GLFormer) | AdjacencyTable (HyperEvent) |
|-|-------------------------------|------------------------------|
| Хранение | полная история узла | последние n_neighbor соседей |
| Lookup | binary search по времени | O(1) |
| Память | O(E) всего | O(V · n_neighbor) = фиксировано |

### Хронологический порядок обучения

В отличие от EAGLE/GLFormer, рёбра НЕ перемешиваются. Таблица смежности накапливается последовательно, и каждый батч использует её состояние на начало батча. Это обеспечивает корректное отражение структурного контекста для каждого события.

### Производительность

Основной bottleneck — Python-вычисление relational vectors (set-операции для d0/d1/d2). Для нашего датасета (~11M рёбер, batch=200) ожидаемое время:
- ~1–3 сек/батч на CPU
- ~50–100 мин/эпоха при 50 эпохах

Для ускорения можно использовать `--n-neighbor 10` или `--n-latest 5`.

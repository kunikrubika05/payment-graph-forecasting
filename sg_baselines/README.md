# Stream Graph Baselines

Бейзлайны для temporal link prediction на stream graph (ORBITAAL Bitcoin dataset).

## Протокол

### Данные
- **Stream graph**: `2020-06-01__2020-08-31.parquet` (~61.5M рёбер)
- **Два периода**: 10% (6.1M рёбер) и 25% (15.4M рёбер) от полного графа
- **Split**: train 70% / val 15% / test 15% (хронологический, по порядку рёбер)

### Фичи (34 на пару)
- 15 node features для src (из `features_{10,25}.parquet`)
- 15 node features для dst
- 4 pair features: CN_undirected, AA_undirected, CN_directed, AA_directed

Все фичи вычислены **только из train** — нет data leakage.

### Негативы
- **Train**: `negative_ratio=5` на позитив (50% historical + 50% random)
- **Eval**: `n_negatives=100` на запрос (50% historical + 50% random)
- Historical = соседи src из train, отсутствующие в eval позитивах
- Random = из множества активных нод в train

### HP поиск
- Grid search по гиперпараметрам
- **Метрика отбора: MRR на val** (совпадает с финальной метрикой)
- Val **не используется для обучения** — только для оценки
- Test **не используется для HP поиска**

### Метрики (TGB-style)
- **MRR** (primary), Hits@1, Hits@3, Hits@10
- Per-source ranking: для каждого позитива (src, dst_true), кандидаты = {dst_true} ∪ {100 negatives}
- Ранг = позиция dst_true (1-based, ties broken conservatively)

## Бейзлайны

### Эвристики
| Метод | Описание |
|-------|----------|
| CN | Common Neighbors (undirected adj) |
| Jaccard | Jaccard coefficient |
| AA | Adamic-Adar |
| PA | Preferential Attachment |

### ML модели
| Модель | HP grid |
|--------|---------|
| LogReg | C × penalty = 8 комбинаций |
| CatBoost | iterations × depth × lr = 12 комбинаций |
| RF | n_estimators × max_depth × min_samples_leaf = 12 комбинаций |

## Запуск

```bash
# Все эксперименты
YADISK_TOKEN="..." PYTHONPATH=. python sg_baselines/run.py \
    --period all --output /tmp/sg_baselines_results --upload \
    2>&1 | tee /tmp/sg_baselines.log

# Только period_10
YADISK_TOKEN="..." PYTHONPATH=. python sg_baselines/run.py \
    --period period_10 --output /tmp/sg_baselines_results --upload \
    2>&1 | tee /tmp/sg_baselines.log

# Только эвристики
YADISK_TOKEN="..." PYTHONPATH=. python sg_baselines/run.py \
    --period period_10 --skip-ml --output /tmp/sg_baselines_results \
    2>&1 | tee /tmp/sg_baselines.log

# Только конкретные модели
YADISK_TOKEN="..." PYTHONPATH=. python sg_baselines/run.py \
    --period period_10 --models catboost --skip-heuristics \
    --output /tmp/sg_baselines_results \
    2>&1 | tee /tmp/sg_baselines.log
```

## Гарантии корректности (нет data leakage)

1. **Node features** вычислены только из train (70% периода) — `compute_stream_node_features.py`
2. **Adjacency** построена только из train — `compute_stream_adjacency.py`
3. **CN/AA** вычисляются из train adjacency — ноды из val/test без train-истории получают 0
4. **HP search** по MRR на val — test не участвует в подборе
5. **Negative sampling** исключает все позитивы текущего split'а
6. **Train/val/test** хронологически непересекающиеся (assert на timestamps)
7. **Финальная оценка** только на test

## Структура результатов

```
exp_sg_{10,25}/
  config.json              # Конфигурация эксперимента
  hp_search_{model}.json   # Результаты HP поиска для каждой ML модели
  summary.json             # Все метрики (heuristics + ML, val + test)
  error.txt                # Ошибка (если была)
```

## Требования к машине

- **RAM**: 16+ GB (stream graph ~2-4 GB, adjacency ~1-2 GB, features ~0.5 GB, training data ~2 GB)
- **CPU**: 4+ ядер (CatBoost и RF используют все ядра)
- **Disk**: 20 GB свободных
- **GPU**: не нужен
- **Время**: ~1-3 часа на period_10, ~3-8 часов на period_25 (зависит от CPU)

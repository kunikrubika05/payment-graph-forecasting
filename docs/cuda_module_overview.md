# CUDA-модуль: обзор

Модуль реализует GPU-ускоренные примитивы для обучения и инференса на
временных транзакционных графах. Два алгоритма, три бэкенда у каждого,
единый Python API, автовыбор бэкенда.

---

## Что реализовано

### 1. Temporal Neighbor Sampling — `TemporalGraphSampler`

Для узла u в момент t найти K последних соседей с timestamp < t.
Фундаментальная операция для GraphMixer, DyGFormer, TGN, TGAT.

**Методы:**

| Метод | Что делает |
|-------|-----------|
| `sample_neighbors(nodes, times, K)` | K последних соседей по времени |
| `featurize(neighbors)` | node/edge features + relative timestamps |
| `sample_negatives(nodes, times, n, strategy)` | negative sampling (random/historical/mixed) |

**Файлы:** `src/models/temporal_graph_sampler.py`,
`src/models/csrc/temporal_sampling.cpp`,
`src/models/csrc/temporal_sampling.cu`

**Тесты:** `tests/test_temporal_sampler.py` — 30 тестов

---

### 2. Common Neighbors — `CommonNeighbors`

Для пары (u, v) вычислить |N(u) ∩ N(v)|. Сильнейшая эвристика link
prediction (MRR = 0.73 в наших бейзлайнах). Может использоваться как
признак в нейросети вместо MinHash-аппроксимации (BUDDY).

**Методы:**

| Метод | Что делает |
|-------|-----------|
| `compute(src, dst)` | Точный CN для батча пар (не аппроксимация) |

**Файлы:** `src/models/graph_metrics.py`,
`src/models/csrc/graph_metrics.cpp`,
`src/models/csrc/graph_metrics.cu`

**Тесты:** `tests/test_graph_metrics.py` — 16 тестов

---

## Бэкенды

| Бэкенд | Реализация | Когда лучший |
|--------|-----------|-------------|
| `python` | NumPy / scipy | fallback, всегда доступен |
| `cpp` | C++ pybind11 | разреженные графы, нет GPU |
| `cuda` | кастомные CUDA-ядра | плотные графы / большой K |

`backend="auto"` → выбирает лучший из скомпилированных.

---

## Бенчмарки (V100-PCIE-32GB, CUDA 12.8)

### Temporal Sampling (`scripts/bench_sampling.py --cuda`)

```
Scenario           Backend  Sample(ms)  Feat(ms)  Total(ms)   Speedup
----------------------------------------------------------------------
graphmixer_small   python        2.74      4.82       7.55      1.0x
graphmixer_small   cpp           0.07      0.14       0.21     35.9x
graphmixer_small   cuda          0.08      0.07       0.15     49.9x

dygformer_small    python        3.04      8.46      11.49      1.0x
dygformer_small    cpp           0.40      3.63       4.04      2.8x
dygformer_small    cuda          0.09      0.09       0.18     64.2x

dygformer_medium   python       14.10     48.30      62.33      1.0x
dygformer_medium   cpp           2.12     32.07      34.22      1.8x
dygformer_medium   cuda          0.12      0.22       0.34    183.7x  ←

dygformer_large    python       14.20     49.16      63.53      1.0x
dygformer_large    cpp           2.14     33.08      35.26      1.8x
dygformer_large    cuda          0.13      0.23       0.36    175.9x

real_scale         python        2.79      5.89       8.64      1.0x
real_scale         cpp           0.38      2.21       2.61      3.3x
real_scale         cuda          0.09      0.09       0.18     48.5x
```

Пик: **184×** над Python на dygformer_medium (batch=2048, K=512).
Почему C++ только 1.8× на K=512: оба бэкенда упираются в memory bandwidth
при featurize — читают 2048×512 строк из node_feat, который не влезает
в кэш. CUDA читает коалесцированно на 900 GB/s → 54× над C++.

### Common Neighbors (`scripts/bench_cn.py --cuda`)

```
Scenario               Backend  Median(ms)   p95(ms)   Speedup
---------------------------------------------------------------
sparse_like_bitcoin    python       0.42       0.48      1.0x
sparse_like_bitcoin    cpp          0.07       0.08      5.9x  ← C++ лучший
sparse_like_bitcoin    cuda         1.18       1.24      0.4x  ← CUDA медленнее

breakeven              python       1.27       1.36      1.0x
breakeven              cpp          0.57       0.62      2.2x
breakeven              cuda         0.73       0.81      1.7x  ← C++ ещё лучше

dense_small            python       2.00       2.08      1.0x
dense_small            cpp          1.02       1.06      1.9x
dense_small            cuda         0.42       0.48      4.8x  ← CUDA лучший

dense_medium           python       8.51       8.88      1.0x
dense_medium           cpp          5.05       5.21      1.7x
dense_medium           cuda         0.71       0.75     12.0x

dense_large            python      35.79      37.14      1.0x
dense_large            cpp         20.85      22.40      1.7x
dense_large            cuda         1.66       1.71     21.5x  ←
```

Пик: **21.5×** над Python на dense_large (N=200K, avg_deg=1000, batch=2048).
Реальный breakeven (CUDA выгоднее C++) — avg_deg ≈ 100–150.
Почему C++ только 1.7× на dense: оба упираются в memory bandwidth
(CSR col_idx = 800 MB, случайный доступ ~20 GB/s).

---

## Применение в проекте

| Задача | Примитив | Бэкенд | Результат |
|--------|---------|--------|-----------|
| GraphMixer / GLFormer обучение | `sample_neighbors` + `featurize` | cpp / cuda | sampling < 2% эпохи |
| DyGFormer обучение (K=512) | `sample_neighbors` + `featurize` | **cuda** | 0.34ms вместо 62ms |
| DyGFormer package-facing train pipeline | `train_dygformer_cuda` + `TemporalGraphSampler` | **cuda** | эпоха на V100: `47.9 -> 27.1` мин при `1536/32` |
| DyGFormer speed-oriented config | `train_dygformer_cuda` + `TemporalGraphSampler` | **cuda** | оценка `~19.7` мин/эпоха при `3072/24` |
| PairwiseMLP precompute | `CommonNeighbors.compute` | cpp (sparse) | точный CN, 43M пар |
| BUDDY-like модель (dense) | `CommonNeighbors.compute` | **cuda** | точный CN вместо MinHash |
| Evaluation neg sampling | `sample_negatives` | cpp / cuda | < 1ms на батч |

### DyGFormer в package-facing API

Начиная с текущей интеграции, CUDA backend для DyGFormer включается не
через отдельный legacy launcher, а через конфиг фреймворка:

```yaml
experiment:
  model: dygformer

sampling:
  backend: cuda
  num_neighbors: 32
```

Дальше canonical entrypoint обычный:

```bash
python -m payment_graph_forecasting.experiments.launcher --config /path/to/spec.yaml
```

Это не единственный public surface. Для прямого library usage каноническими
считаются также:

- `payment_graph_forecasting.TemporalGraphSampler`
- `payment_graph_forecasting.CommonNeighbors`
- `payment_graph_forecasting.describe_cuda_capabilities()`

Внутри это работает так:

- [payment_graph_forecasting/models/dygformer.py](/Users/kunikrubika/Desktop/payment-graph-forecasting/payment_graph_forecasting/models/dygformer.py) прокидывает `sampling.backend` в runner
- [payment_graph_forecasting/training/api.py](/Users/kunikrubika/Desktop/payment-graph-forecasting/payment_graph_forecasting/training/api.py) выбирает `train_dygformer_cuda(...)`, если backend не `auto`
- [src/models/DyGFormer/dygformer_cuda_train.py](/Users/kunikrubika/Desktop/payment-graph-forecasting/src/models/DyGFormer/dygformer_cuda_train.py) выполняет CUDA-backed train loop

### Реальный эффект на ORBITAAL 10%

Изменение не ограничилось synthetic benchmark'ами. На реальном package-facing
DyGFormer pipeline для ORBITAAL stream-graph (summer 2020, 10%) были получены:

| Конфиг | До | После | Ускорение |
|--------|----|-------|-----------|
| `batch_size=1536`, `num_neighbors=32` | `~47.9 мин/epoch` | `~27.1 мин/epoch` | `~1.8x` |
| `batch_size=3072`, `num_neighbors=32` | н/д | `~26.6 мин/epoch` | лучший safe config |
| `batch_size=3072`, `num_neighbors=24` | н/д | `~19.7 мин/epoch` | speed-oriented config |

Ключевой вывод: после интеграции CUDA sampling bottleneck batch-prep почти
исчезает, и эпоха начинает упираться уже в сам forward/backward DyGFormer.

---

## Итог

```
2 алгоритма × 3 бэкенда = 6 реализаций
46 тестов (30 sampling + 16 CN), все passing
10 бенчмарк-сценариев (5 sampling + 5 CN)

Пиковый speedup — Temporal Sampling: 184× vs Python
Пиковый speedup — Common Neighbors:   22× vs Python
```

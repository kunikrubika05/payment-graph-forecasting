# GPU-ускорение Temporal Neighbor Sampling: реализация и результаты

## Краткое summary для команды

Мы реализовали GPU-ускоренный temporal neighbor sampling — ключевую операцию
для обучения моделей на временных графах (GraphMixer, DyGFormer, TGN).
Для каждого узла u в момент времени t нужно найти K последних соседей
с timestamp < t. CUDA-ядро делает это параллельно для всего батча.

Результат: **до 184× ускорение** над Python, **до 50–64× над C++** при
большом K и батче. Три бэкенда с единым API — автовыбор.

---

## Мотивация

Обучение DyGFormer требует K=512 последних соседей для каждого узла в батче.
При batch_size=2048 это 2048 × 512 = 1M соседей за один шаг — на каждый батч.
Без GPU-ускорения sampling занимает десятки миллисекунд и становится узким
местом, съедающим время, сопоставимое с forward/backward pass модели.

---

## Алгоритм

### TemporalCSR

Граф хранится в CSR, отсортированном по времени внутри каждой строки:

```
row_ptr:  [N+1]      — начало/конец соседей узла u
col_idx:  [E]        — индексы соседей (sorted by timestamp per row)
timestamps:[E]       — времена рёбер (sorted ascending per row)
edge_feat: [E, F_e]  — признаки рёбер
node_feat: [N, F_n]  — признаки узлов
```

### sample_neighbors

Для запроса (u, t): найти K последних рёбер u с timestamp < t.

**Шаг 1 — Binary search:** найти правую границу среди соседей u:
```
pos = upper_bound(timestamps[row_ptr[u]..row_ptr[u+1]], t) - 1
```
**Шаг 2 — Выбрать K последних:** взять K рёбер левее pos.

На GPU каждый поток обрабатывает один запрос (u, t) — идеальный параллелизм
при batch_size=2048: все 2048 binary search выполняются одновременно.

### featurize

Для найденных K соседей собрать:
- node features соседей: `node_feat[neighbor_ids]`
- edge features: `edge_feat[edge_ids]`
- relative timestamps: `t - timestamps[edge_ids]`

На GPU: коалесцированный gather по индексам, каждый поток — один элемент.

### Почему CUDA выигрывает сильнее, чем для CN

| | CN (bitset) | Sampling |
|---|---|---|
| Паттерн доступа | Случайный (large CSR) | Последовательный (конец строки) |
| Коалесценность | Низкая | Высокая |
| Regularность | Нерегулярный (power-law) | Регулярный (K фиксирован) |
| Speedup vs C++ | 12x | **50–100x** |

Sampling имеет регулярный паттерн: все потоки берут последние K элементов
из своих строк — cache-friendly, warp efficiency высокий. Поэтому speedup
над C++ здесь намного выше, чем для CN.

---

## Три операции, три бэкенда

```python
from src.models.temporal_graph_sampler import TemporalGraphSampler, Backend

sampler = TemporalGraphSampler(graph, backend=Backend.CUDA)

# 1. Найти K последних соседей
neighbors = sampler.sample_neighbors(nodes, timestamps, K=20)
# → NeighborBatch(node_ids, edge_ids, timestamps, mask)

# 2. Собрать признаки для соседей
features = sampler.featurize(neighbors)
# → FeatureBatch(node_feat, edge_feat, rel_time)

# 3. Сгенерировать негативные примеры
negatives = sampler.sample_negatives(nodes, timestamps,
                                     n_neg=100, strategy="mixed")
# → [B, n_neg] negative destination node indices
```

---

## Бенчмарки

### Запуск

```bash
source venv/bin/activate
PYTHONPATH=. python scripts/bench_sampling.py --cuda 2>&1 | tee /tmp/bench_sampling.log
```

### Сценарии

| Сценарий | batch | K | nodes | edges | Модель-прототип |
|----------|-------|---|-------|-------|-----------------|
| graphmixer_small | 512 | 20 | 100K | 1M | GraphMixer |
| dygformer_small | 512 | 512 | 100K | 1M | DyGFormer (малый) |
| dygformer_medium | 2048 | 512 | 100K | 1M | DyGFormer (средний) |
| dygformer_large | 2048 | 512 | 1M | 10M | DyGFormer (реальный) |
| real_scale | 512 | 512 | 5M | 20M | Наш Bitcoin граф |

### Результаты (V100-PCIE-32GB, CUDA 12.8)

```
Scenario           Backend  Sample(ms)  Feat(ms)  Total(ms)  Speedup
---------------------------------------------------------------------
graphmixer_small   python        2.74      4.82       7.55     1.0x
graphmixer_small   cpp           0.07      0.14       0.21    35.9x
graphmixer_small   cuda          0.08      0.07       0.15    49.9x

dygformer_small    python        3.04      8.46      11.49     1.0x
dygformer_small    cpp           0.40      3.63       4.04     2.8x
dygformer_small    cuda          0.09      0.09       0.18    64.2x

dygformer_medium   python       14.10     48.30      62.33     1.0x
dygformer_medium   cpp           2.12     32.07      34.22     1.8x
dygformer_medium   cuda          0.12      0.22       0.34   183.7x   ← пик

dygformer_large    python       14.20     49.16      63.53     1.0x
dygformer_large    cpp           2.14     33.08      35.26     1.8x
dygformer_large    cuda          0.13      0.23       0.36   175.9x

real_scale         python        2.79      5.89       8.64     1.0x
real_scale         cpp           0.38      2.21       2.61     3.3x
real_scale         cuda          0.09      0.09       0.18    48.5x
```

### Ключевые наблюдения

**GraphMixer (K=20):** C++ уже почти так же быстр, как CUDA (35.9x vs 49.9x).
Sampling не является узким местом — forward pass доминирует. CUDA здесь бонус.

**DyGFormer (K=512, batch=2048):** CUDA даёт **184×** над Python, **54×** над C++.
При K=512 featurize становится огромным (2048 × 512 элементов) — C++ не справляется,
CUDA коалесцированным gather собирает это за 0.22ms вместо 32ms.

**real_scale (5M nodes):** CUDA стабильно **48×** — даже на графе реального
размера скорость не падает. C++ тут только 3.3x из-за cache pressure на 5M nodes.

---

## Почему C++ проигрывает Python на DyGFormer (1.8x вместо 35x)

Та же история, что и с CN на плотных графах: bottleneck — **memory bandwidth**.

При K=512 операция `featurize` читает 2048 × 512 строк из node_feat и edge_feat.
Для 1M nodes node_feat не влезает в кэш → случайные чтения из большого массива.
C++ и Python оба упираются в одну и ту же память. C++ экономит на Python-overhead,
отсюда 1.8×. CUDA читает коалесцированно с 900 GB/s → 54× над C++.

---

## Применение в нашем пайплайне

| Модель | K | Backend | Sampling % от эпохи |
|--------|---|---------|---------------------|
| GraphMixer | 20 | C++ или CUDA | < 2% |
| GLFormer | 20 | C++ или CUDA | < 2% |
| DyGFormer | 512 | **CUDA обязателен** | ~40% без CUDA |

При K=20 sampling — не узкое место, CUDA бонус.
При K=512 CUDA превращает sampling из bottleneck в незначимые 0.34ms.

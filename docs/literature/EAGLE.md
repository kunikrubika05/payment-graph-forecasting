# EAGLE: Expressive dynamics-Aware Graph LEarning

> Historical literature/design note.
> Operational commands in this file are not the canonical project surface.
> For actual runs, use the package-facing docs in `README.md`, `docs/README.md`,
> `payment_graph_forecasting.experiments.launcher`,
> `payment_graph_forecasting.experiments.hpo`, and the YAML specs in
> `exps/examples/`.

## Источник

**Статья:** Yue et al., "Towards Expressive dynamics-Aware Graph LEarning", 2024
**Репозиторий:** https://github.com/TreeAI-Lab/EAGLE
**Наша реализация:** `src/models/eagle.py`, `src/models/eagle_train.py`, `src/models/tppr.py`

---

## 1. Мотивация и ключевая идея

Большинство моделей для temporal link prediction (TGN, TGAT, DyRep) используют сложные
механизмы: attention, memory modules, RNN. Они медленные и не всегда дают лучшее качество.

EAGLE задаёт вопрос: **что если использовать только временны́е паттерны взаимодействий,
без каких-либо признаков рёбер и узлов?**

Основные наблюдения авторов:

1. **Время — самый информативный сигнал.** На датасетах TGB (tgbl-wiki, tgbl-review,
   tgbl-coin) модели, использующие только timestamps, конкурируют с моделями, имеющими
   доступ к полному набору признаков.

2. **Структура графа несёт дополнительный сигнал.** Temporal Personalized PageRank (TPPR)
   может захватить структурную близость узлов с учётом временно́й динамики.

3. **Простая комбинация обоих сигналов побеждает сложные модели.** EAGLE-Hybrid
   (линейная смесь EAGLE-Time и TPPR скоров) достигает SOTA на нескольких бенчмарках.

### Сравнение с GraphMixer

| Аспект | GraphMixer | EAGLE-Time |
|--------|-----------|------------|
| Вход в энкодер | edge feats + time + node feats | **только delta times** |
| Time encoding | `cos(Δt × ω)`, равномерные частоты | `cos(Δt × ω)`, **лог-частоты** (9 порядков) |
| MLP-Mixer token expansion | 2.0 (расширение) | **0.5** (компрессия) |
| NodeEncoder | да (25 признаков + среднее соседей) | **нет** |
| Параметров | ~138K | **~120K** |
| Скорость обучения | базовая | **быстрее** (меньше данных на батч) |

---

## 2. Архитектура EAGLE

EAGLE состоит из трёх независимых компонентов, которые можно использовать по отдельности
или комбинировать.

### 2.1. EAGLE-Time (нейросетевой компонент)

Полностью основан на MLP-Mixer. Для каждого узла рассматриваются K последних соседей и
кодируются **только разницы во времени** `Δt = t_query - t_neighbor`.

```
Δt₁, Δt₂, ..., Δtₖ  →  TimeEncode  →  Linear  →  MLP-Mixer  →  MeanPool  →  h_node
```

#### TimeEncode (нетренируемое)

Вместо равномерных частот (как в GraphMixer) используются **логарифмически распределённые**:

```
ω_i = 1 / 10^(i × 9 / (dim - 1)),    i = 0, 1, ..., dim-1
```

Это покрывает 9 порядков величины: от ω₀ = 1 (секундные паттерны) до ω_{dim-1} = 10⁻⁹
(многолетние паттерны). Кодирование:

```
TimeEncode(Δt) = [cos(Δt × ω₀), cos(Δt × ω₁), ..., cos(Δt × ω_{dim-1})]
```

При Δt = 0 все компоненты равны 1 (cos(0) = 1). Для наших данных (дневные снимки,
Δt измеряется в номерах дней) это покрытие избыточно — основной полезный диапазон
находится в пределах 1–365 дней, но лог-частоты обеспечивают робастность.

#### MLP-Mixer блок

Каждый блок содержит два подблока с residual connections:

```
Token-mixing:   x' = x + FF_token(LayerNorm(x)ᵀ)ᵀ
Channel-mixing: x'' = x' + FF_channel(LayerNorm(x'))
```

**Token-mixing** (ось соседей):
- Вход: [batch, K, hidden_dim] → transpose → [batch, hidden_dim, K]
- FeedForward: `K → K/2 → K` (expansion = 0.5, **компрессия**)
- Определяет значимость каждого из K соседей

**Channel-mixing** (ось признаков):
- Вход: [batch, K, hidden_dim]
- FeedForward: `hidden_dim → hidden_dim × 4 → hidden_dim` (expansion = 4.0, стандарт)
- Комбинирует различные компоненты временного кодирования

Компрессивный token-mixing (0.5 вместо обычных 2–4) — ключевая дизайн-деталь EAGLE.
При малом числе токенов (K = 10–30) это предотвращает переобучение на позиционные
артефакты и стабилизирует обучение.

#### Pooling и MLP head

```
h_node = Linear(MeanPool(LayerNorm(mixer_output)))
```

Mean-pool по оси соседей → проекция в hidden_dim.

#### Edge predictor

Аддитивная схема (как TransE):

```
score(src, dst) = Linear(ReLU(W_src × h_src + W_dst × h_dst))
```

Поддерживает два режима:
- **Pairwise:** `h_src [B, d]`, `h_dst [B, d]` → `scores [B]`
- **Ranking:** `h_src [B, d]`, `h_dst [B, C, d]` → `scores [B, C]`

### 2.2. EAGLE-Structure (TPPR)

Temporal Personalized PageRank — алгоритмический (не нейросетевой) компонент.

#### Идея

Для каждого узла поддерживается словарь PPR (Personalized PageRank) — вектор близости
ко всем остальным узлам. При появлении нового ребра словари обновляются инкрементально
с учётом временно́го затухания.

#### Алгоритм обновления

При появлении ребра (src → dst) обновляется PPR обоих узлов.
Для узла s₁, получившего ребро к s₂:

**Первое ребро** (s₁ ещё не встречался):
```
PPR(s₁) = (1 - α) × PPR(s₂) + α × δ(s₂)
```

**Последующие рёбра:**
```
PPR(s₁) ← β × (norm_old / norm_new) × PPR(s₁) + β/norm_new × (1 - α) × PPR(s₂) + restart
```

Где:
- **α** (restart probability, default 0.9) — вес прямого соседства. Высокое α означает,
  что PPR концентрируется на непосредственных соседях.
- **β** (temporal decay, default 0.8) — затухание старых связей. При каждом новом ребре
  старый PPR умножается на β, что постепенно забывает устаревшие связи.
- **topk** (default 100) — максимальный размер PPR-словаря. При превышении отбрасываются
  записи с наименьшим весом.

#### Скоринг

Сходство двух узлов — скалярное произведение их PPR-векторов:

```
similarity(u, v) = Σ_w PPR(u)[w] × PPR(v)[w]
```

Эффективно вычисляется итерацией по меньшему словарю.

#### Характеристики

- **Сложность:** O(topk) на ребро (обновление + truncation)
- **Память:** O(num_nodes × topk) для PPR-словарей
- **Преимущество:** не требует GPU, работает на CPU
- **Ограничение:** O(num_edges) последовательных обновлений, не параллелизуется

### 2.3. EAGLE-Hybrid

Линейная комбинация скоров:

```
score_hybrid = η × score_TPPR + (1 - η) × score_Time
```

η подбирается grid search на валидационном множестве.
В нашей реализации гибридный режим пока не реализован (можно комбинировать вручную).

---

## 3. Наша реализация

### Модули

```
src/models/
  eagle.py              # Архитектура EAGLE-Time
  eagle_train.py        # Обучение и валидация (аналог train.py для GraphMixer)
  eagle_evaluate.py     # TGB-style оценка (аналог evaluate.py для GraphMixer)
  eagle_launcher.py     # CLI для полных экспериментов (аналог launcher.py)
  eagle_hpo.py          # Подбор гиперпараметров (Optuna)
  tppr.py               # TPPR + CLI для структурного бейзлайна
scripts/
  eagle_sanity_check.py # Быстрая проверка на синтетических данных
  run_eagle.py          # Полный пайплайн (тесты → HPO → обучение → TPPR)
tests/
  test_eagle.py         # 30 тестов для всех компонентов
```

### Отличия от авторской реализации

| Аспект | Оригинал (TreeAI-Lab) | Наша реализация |
|--------|----------------------|-----------------|
| TPPR | Numba @jitclass | Чистый Python (dict-based) |
| Neighbor sampling | Собственный NeighborFinder (Numba) | Общий TemporalCSR (C++ / Python fallback) |
| Негативы (обучение) | Random | Random (из общей инфраструктуры) |
| Негативы (оценка) | Random | **50 historical + 50 random** (TGB-протокол) |
| Mixed precision | Нет | **AMP (fp16)** на GPU |
| Данные | CSV + Pandas | TemporalEdgeData + NumPy (zero-copy) |
| HPO | Нет | **Optuna** (TPE + MedianPruner) |
| Early stopping | Свой EarlyStopMonitor | Простой patience counter |

### Переиспользование кода фреймворка

EAGLE-Time максимально переиспользует инфраструктуру, написанную для GraphMixer:

- `TemporalCSR` + `sample_neighbors_batch()` — neighbor sampling (C++ расширение)
- `generate_negatives_for_eval()` — генерация негативов для TGB-оценки
- `compute_ranking_metrics()` — MRR, Hits@K
- `prepare_sliding_window()` — загрузка данных + split
- `build_temporal_csr()` — построение CSR из масок
- `upload_directory()` — загрузка результатов на Яндекс.Диск
- `PERIODS` — конфигурация 10 периодов Bitcoin

Единственное отличие в подготовке батча: `prepare_eagle_batch()` извлекает **только**
delta_times и lengths (без edge_feats и node_feats), что делает его быстрее.

---

## 4. Размер модели

С параметрами по умолчанию (hidden_dim=100, K=20, 1 mixer layer):

| Компонент | Параметры | Описание |
|-----------|-----------|----------|
| EAGLETimeEncoding | 0 | Нетренируемые лог-частоты (buffer) |
| feat_encoder (Linear) | 10,100 | time_dim(100) → hidden_dim(100) |
| MixerBlock token_ff | 200 | K(20) → K×0.5(10) → K(20) |
| MixerBlock channel_ff | 80,400 | hidden(100) → hidden×4(400) → hidden(100) |
| LayerNorms (×3) | 600 | 3 × 2 × hidden(100) |
| mlp_head (Linear) | 10,100 | hidden(100) → hidden(100) |
| EdgePredictor | 20,301 | src_fc + dst_fc + out_fc |
| **Итого** | **~121K** | **~475 KB** |

---

## 5. Пайплайн обучения

### Подготовка данных

1. Скачивание `daily_snapshots/` и `node_features/` с Яндекс.Диска
2. `prepare_sliding_window(period, window, undirected=True)` → TemporalEdgeData + masks
3. `build_temporal_csr(data, train_mask)` → CSR для neighbor sampling

### Обучение (GPU, AMP)

Для каждой эпохи:

1. Shuffle тренировочных рёбер
2. Мини-батчи (200 рёбер по умолчанию):
   - Positive: `(src, dst, t)` из тренировочного множества
   - Negative: `(src, random_node, t)` — 1 random negative
   - Для каждого узла: sample K соседей → вычислить `Δt`
   - Forward → pos_logits, neg_logits
   - **Loss:** `BCEWithLogitsLoss`, gradient clipping `max_norm=1.0`
   - **AMP:** fp16 forward + backward на CUDA (×1.5–2 ускорение на A100)
3. Валидация: subsample до 5000 val-рёбер, 100 random negatives, MRR
4. **Early stopping:** patience=10 по val MRR, сохранение лучшей модели

### TGB-style evaluation

Для каждого тестового ребра `(src, dst_true, t)`:

1. Генерация негативов: 50 исторических (бывшие соседи src) + 50 случайных
2. Candidate set: `{dst_true} ∪ negatives` = 101 кандидат
3. Скоринг всех кандидатов через модель
4. Ранг `dst_true` (filtered ranking, tie-breaking = 0.5)
5. Метрики: **MRR** (primary), Hits@1, Hits@3, Hits@10

Протокол идентичен бейзлайнам и GraphMixer — результаты напрямую сравнимы.

---

## 6. Подбор гиперпараметров (Optuna)

### Пространство поиска

| Параметр | Тип | Диапазон | Default |
|----------|-----|----------|---------|
| hidden_dim | categorical | {50, 100, 200} | 100 |
| num_neighbors | categorical | {10, 15, 20, 30} | 20 |
| num_mixer_layers | int | [1, 3] | 1 |
| lr | log-float | [1e-4, 1e-2] | 0.001 |
| weight_decay | log-float | [1e-6, 1e-3] | 5e-5 |
| dropout | float | [0.0, 0.3] step 0.05 | 0.1 |
| batch_size | categorical | {200, 400, 600} | 200 |
| token_expansion | categorical | {0.5, 1.0, 2.0} | 0.5 |
| channel_expansion | categorical | {2.0, 4.0} | 4.0 |

### Стратегия

- **Sampler:** TPE (Tree-structured Parzen Estimator) — байесовская оптимизация
- **Pruner:** MedianPruner (n_startup=5, n_warmup=3) — отсекает слабые trials раньше
- **Бюджет:** 30 trials × 15 эпох ≈ 2–3 часа на A100
- **Метрика:** максимизация val MRR
- **Результат:** `hpo_results.json` + `best_train_command.sh`

---

## 7. Запуск

### Полный пайплайн (одной командой)

```bash
cd ~/payment-graph-forecasting
export YADISK_TOKEN="..."
PYTHONPATH=. python scripts/run_eagle.py
```

Выполняет: тесты → sanity check → HPO → обучение → TPPR → сравнительная таблица.

### По шагам

```bash
# 1. Тесты
PYTHONPATH=. python -m pytest tests/test_eagle.py -v

# 2. Sanity check (~2-5 мин)
PYTHONPATH=. python scripts/eagle_sanity_check.py

# 3. HPO (~2-3 ч на A100)
PYTHONPATH=. python src/models/eagle_hpo.py \
    --period mature_2020q2 --n-trials 30 --hpo-epochs 15 \
    --output /tmp/eagle_hpo 2>&1 | tee /tmp/eagle_hpo.log

# 4. Обучение (~2-4 ч на A100)
YADISK_TOKEN="..." PYTHONPATH=. python src/models/eagle_launcher.py \
    --period mature_2020q2 --epochs 100 \
    --output /tmp/eagle_results 2>&1 | tee /tmp/eagle_train.log

# 5. TPPR бейзлайн
YADISK_TOKEN="..." PYTHONPATH=. python src/models/tppr.py \
    --period mature_2020q2 --output /tmp/eagle_tppr \
    2>&1 | tee /tmp/eagle_tppr.log
```

### Параметры CLI (eagle_launcher.py)

| Параметр | Default | Описание |
|----------|---------|----------|
| `--period` | `mature_2020q2` | Период из 10 доступных |
| `--window` | `7` | Контекстное окно (дни) |
| `--epochs` | `100` | Макс. эпох |
| `--batch-size` | `200` | Размер батча |
| `--lr` | `0.001` | Learning rate |
| `--weight-decay` | `5e-5` | Weight decay |
| `--num-neighbors` | `20` | K соседей |
| `--hidden-dim` | `100` | Скрытая размерность |
| `--num-mixer-layers` | `1` | Число Mixer блоков |
| `--token-expansion` | `0.5` | Expansion factor для token-mixing |
| `--channel-expansion` | `4.0` | Expansion factor для channel-mixing |
| `--dropout` | `0.1` | Dropout |
| `--patience` | `10` | Early stopping patience |
| `--no-amp` | `false` | Отключить mixed precision |
| `--max-val-edges` | `5000` | Макс. val-рёбер на эпоху |

### Результаты

```
eagle_mature_2020q2_w7/
  config.json            # гиперпараметры
  data_summary.json      # статистика данных
  metrics.jsonl          # метрики по эпохам
  training_curves.csv    # для графиков
  best_model.pt          # лучшая модель (по val MRR)
  summary.json           # итоги обучения
  final_results.json     # финальные метрики на test set
  experiment.log         # полный лог
```

---

## 8. Теоретический анализ

### Почему работает только время?

В транзакционных графах (Bitcoin, финансы) временны́е паттерны несут сильный сигнал:

1. **Периодичность.** Многие сущности совершают транзакции с регулярностью (ежедневно,
   еженедельно). Log-частоты EAGLE покрывают все масштабы — от часов до лет.

2. **Бёрсты.** Всплески активности (множество транзакций за короткий период)
   характеризуют определённые типы узлов (биржи, миксеры). MLP-Mixer через
   token-mixing учится распознавать такие паттерны.

3. **Recency.** Недавние взаимодействия — сильный предиктор будущих. Маленькие Δt
   создают высокочастотные компоненты в time encoding, которые channel-mixing
   может использовать.

### Почему компрессивный token-mixing (0.5)?

При K = 20 соседях и expansion 0.5:
- Внутренняя размерность: 20 → 10 → 20
- Это bottleneck, который заставляет модель выучить компактное представление
  «профиля временно́й активности» вместо запоминания отдельных позиций
- При expansion ≥ 2.0 модель может переобучиться на порядок соседей,
  что не несёт полезного сигнала (соседи отсортированы по времени)

### Ограничения

1. **Нет признаков рёбер.** EAGLE-Time игнорирует объём транзакций (BTC, USD).
   Для финансовых данных это может быть существенная потеря — крупная транзакция
   несёт иной сигнал, чем пылевая.

2. **Нет признаков узлов.** PageRank, degree, k-core — полезные предикторы,
   которые EAGLE-Time не использует (CN с MRR 0.46–0.73 опирается именно на них).

3. **TPPR не параллелизуется.** Последовательная обработка O(num_edges) рёбер.
   Для крупных периодов (~500K рёбер) это десятки минут на CPU.

4. **Фиксированное K.** Все узлы представлены одинаковым числом соседей.
   Для узлов с 1–2 взаимодействиями представление сильно разрежено (padding).

---

## 9. Ожидаемые результаты

### Бенчмарки из статьи (TGB)

| Датасет | EAGLE-Time | EAGLE-Hybrid | GraphMixer | TGN |
|---------|-----------|-------------|-----------|-----|
| tgbl-wiki | 0.814 | 0.852 | 0.790 | 0.779 |
| tgbl-review | 0.872 | 0.888 | 0.857 | 0.662 |
| tgbl-coin | 0.774 | 0.805 | 0.726 | 0.586 |

### Ожидания на наших данных (Bitcoin ORBITAAL)

- **CN baseline:** MRR 0.46–0.73 (зависит от периода)
- **GraphMixer (ожидаемый):** MRR 0.55–0.75
- **EAGLE-Time (ожидаемый):** MRR 0.55–0.75 (сравнимо с GraphMixer, но быстрее)
- **TPPR:** MRR 0.3–0.5 (структурный сигнал без нейросети)

Финальные числа будут получены после обучения на дев-машине.

---

## 10. Ссылки

- Yue et al., "Towards Expressive dynamics-Aware Graph LEarning" (2024)
- Tolstikhin et al., "MLP-Mixer: An all-MLP Architecture for Vision", NeurIPS 2021
- Cong et al., "Do We Really Need Complicated Model Architectures For Temporal Networks?", ICLR 2023
- Page et al., "The PageRank Citation Ranking: Bringing Order to the Web", Stanford 1999
- Huang et al., "Temporal Graph Benchmark for Machine Learning on Temporal Graphs", NeurIPS 2024

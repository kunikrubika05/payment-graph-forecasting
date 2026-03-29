# Checklist корректности для stream graph экспериментов

## Данные на Яндекс.Диске

```
orbitaal_processed/stream_graph/
  2020-06-01__2020-08-31.parquet   # полный stream graph, 61.4M рёбер
  features_10.parquet              # 15 node features, train of 10% period (1.9M нод)
  features_25.parquet              # 15 node features, train of 25% period (4.3M нод)
  adj_10_directed.npz              # CSR binary, local indices
  adj_10_undirected.npz            # CSR symmetric
  node_mapping_10.npy              # local → global index
  adj_25_directed.npz / undirected / mapping
```

## Split протокол

Period = первые {fraction} рёбер stream graph (хронологически).
Внутри period: **train 70% / val 15% / test 15%** по порядку рёбер.

```
period_10: 6,143,581 рёбер → train=4,300,506 / val=921,537 / test=921,538
period_25: 15,358,954 рёбер → train=10,751,267 / val=~2.3M / test=~2.3M
```

Фичи и adjacency вычислены из **train** (первых 70% period).
Val/test НЕ участвуют в вычислении фичей.

## Протокол негативов

### Train (бинарная классификация)
- negative_ratio = 5 на позитив
- До 2 исторических (соседи src из train, минус позитивы src в текущем батче)
- 3-5 рандомных из **train нод** (active_nodes = node_mapping)
- Исключаются: все позитивы src, все исторические, сам src

### Eval (TGB-style ranking)
- n_negatives = 100 на запрос
- До 50 исторических (соседи src из train, минус **ВСЕ** позитивы src **из полного split'а**)
- 50-100 рандомных из **train нод**
- Candidate set = {dst_true} ∪ {100 negatives} = **101 кандидат**
- Ранг = 1-based, ties broken conservatively: `rank = count(score > true_score) + 1`
- Метрики: **MRR** (primary), Hits@1, Hits@3, Hits@10
- 50K queries субсэмплируются если в split больше уникальных рёбер

### Почему рандомные негативы из train нод
Мы не знаем о нодах из будущего (val/test). Рандомные негативы берутся из
`active_nodes = node_mapping` = все ноды, встреченные в train рёбрах.
Ноды, впервые появившиеся в val/test, не попадают в пул рандомных негативов.

## Гарантии нет leakage

| Компонент | Источник данных | Leakage? |
|-----------|----------------|----------|
| Node features (15 шт) | train edges only | ✓ Нет |
| Adjacency matrix | train edges only | ✓ Нет |
| CN/AA pair features | train adjacency | ✓ Нет |
| Train neighbors (neg sampling) | train edges only | ✓ Нет |
| Random neg pool (active_nodes) | train nodes only | ✓ Нет |
| HP search | val set only, MRR | ✓ Нет |
| Final metrics | test set only | ✓ Нет |

## Фичи (34 на пару src→dst)

| # | Фича | Источник |
|---|------|----------|
| 0-14 | src_{log_in_degree, log_out_degree, in_out_ratio, log_unique_in_cp, log_unique_out_cp, log_total_btc_in, log_total_btc_out, log_avg_btc_in, log_avg_btc_out, recency, activity_span, log_event_rate, burstiness, out_counterparty_entropy, in_counterparty_entropy} | features_{label}.parquet |
| 15-29 | dst_{те же 15} | features_{label}.parquet |
| 30 | cn_undirected | adj_{label}_undirected.npz |
| 31 | aa_undirected | adj_{label}_undirected.npz |
| 32 | cn_directed | adj_{label}_directed.npz |
| 33 | aa_directed | adj_{label}_directed.npz |

Ноды без train-истории → все 34 фичи = 0.

**ВАЖНО:** eval запросы фильтруются — оцениваются ТОЛЬКО рёбра где И src И dst_true
присутствуют в train node_mapping. Рёбра с новыми нодами исключаются, т.к. для них
все 101 кандидата получают score=0 → rank=1 → завышенный MRR.

## Для нейросетей: что нужно соблюсти для корректного сравнения

### Обязательные требования

1. **Тот же split**: train 70% / val 15% / test 15% от period (10% или 25%)
   ```python
   period_end = int(len(df) * fraction)
   period = df.iloc[:period_end]
   train_end = int(len(period) * 0.70)
   val_end = int(len(period) * 0.85)
   ```

2. **Тот же eval протокол**: 100 негативов (50 hist + 50 rand) из train нод,
   per-source ranking, MRR/Hits@K, 50K queries субсэмпл

3. **Те же данные**: stream graph `2020-06-01__2020-08-31.parquet`

4. **Temporal causality при обучении**: при обработке ребра с timestamp=t,
   модель видит ТОЛЬКО рёбра с timestamp < t. Никаких будущих рёбер.

5. **Neighbor sampling temporal**: если модель сэмплирует соседей для ноды в момент t,
   допустимы только рёбра с timestamp < t

6. **Node memory (TGN/DyGFormer)**: memory обновляется ПОСЛЕ forward pass на ребре,
   не до. При eval memory фиксируется на конец train.

7. **Негативы при обучении нейросети**: можно упростить до рандомных из train нод
   (DL модели обычно не используют исторические негативы при обучении, только при eval).
   Но eval протокол должен быть ИДЕНТИЧЕН бейзлайнам.

8. **Eval seed**: использовать те же seed'ы для воспроизводимости
   - val eval: seed = 42 + 300 (для final eval) или 42 + 200 (для HP/early stopping)
   - test eval: seed = 42 + 400

### Чеки перед запуском

- [ ] Split индексы совпадают с бейзлайнами (train_end, val_end, test_end)
- [ ] Eval функция использует `sample_negatives_for_eval` из `sg_baselines/sampling.py`
- [ ] `active_nodes` = `node_mapping` из adjacency (train ноды)
- [ ] `train_neighbors` = `build_train_neighbor_sets(train_edges)` — для исторических негативов
- [ ] `all_positives_per_src` строится из ВСЕГО split'а, не из субсэмпла
- [ ] MRR считается через `compute_ranking_metrics` из `src/baselines/evaluation.py`
- [ ] Нет будущих рёбер в neighbor sampling / message passing
- [ ] На val используется для early stopping / model selection (не test!)
- [ ] Final метрики ТОЛЬКО на test

### Бейзлайн результаты для сравнения (period_10)

Будут доступны в `orbitaal_processed/experiments/exp_sg_10/summary.json` после запуска.
Промежуточные (с текущего запуска, heuristics на 50K queries):
```
cn:      val MRR=0.8641, test MRR=0.8725
jaccard: val MRR=0.8517, test MRR=0.8621
aa:      val MRR=0.8621, test MRR=0.8709
pa:      val MRR=0.7405, test MRR=0.8036
```

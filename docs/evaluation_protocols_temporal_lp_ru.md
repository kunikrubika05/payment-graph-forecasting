# Протоколы оценки temporal link prediction на динамических транзакционных графах

## 1. Задача и постановка

Temporal link prediction (динамическое предсказание связей) — задача предсказания, появится ли ребро (s, d) в момент времени t' > t, если граф наблюдался до момента t. Два основных подхода:

- **Бинарная классификация** на выборке positive/negative рёбер, оценка через AUROC / AP.
- **Ранжирование** кандидатов-получателей для данного отправителя (и момента времени), оценка через MRR / Hits@K / Precision@K.

Современные бенчмарки (DGB, TGB, ROLAND) и анализ TGN/DyRep сходятся к **оценке в формате ранжирования с множественными негативами на один позитив**, отходя от бинарной AUROC с одним случайным негативом, которая теперь считается завышающей результаты для разреженных темпоральных графов.[^1][^2][^3]


## 2. Negative Sampling: текущий стандарт

### 2.1 Устаревший подход: один случайный негатив на позитив

Многие ранние статьи по temporal GNN, включая TGN и бенчмарки до TGB, оценивали dynamic link prediction как бинарную классификацию с **одним негативным ребром на позитив**, выбранным uniform-random из всех пар узлов. Эта стратегия генерирует слишком лёгкие негативы в разреженных графах и приводит к почти идеальным AUROC/AP, скрывая реальные различия моделей.[^2][^3][^1]

### 2.2 TGB (NeurIPS 2023): смесь исторических и случайных негативов

Для dynamic link property prediction TGB заменяет бинарную AUROC **протоколом ранжирования** с реалистичным negative sampler:[^1]

- Для каждого positive edge e_p = (s, d, t) **фиксируется source s и время t**, сэмплируется q негативных destinations.
- Candidate set: {d} ∪ {d_1, ..., d_q}, модель оценивается по рангу d в этом множестве.
- Негативы — **смесь 50/50**:
  - **Исторические негативы**: рёбра из обучающей выборки, отсутствующие в момент t.
  - **Случайные негативы**: узлы-получатели, выбранные uniform-random.
- Если исторических негативов недостаточно, остаток заполняется случайными.[^1]

Для малых графов TGB использует **все возможные destinations** (полное ранжирование); для больших — q выбирается как компромисс между полнотой и скоростью (например, tgbl-review: 100 негативов на позитив).[^1]

### 2.3 DGB: random vs historical vs inductive vs hard negatives

DGB формализует три семейства стратегий negative sampling для оценки:[^5][^2]

- **Random NS**: выборка из всех пар узлов, не являющихся рёбрами в момент t.
- **Historical NS**: выборка из рёбер, существовавших ранее, но отсутствующих в момент t; проверяет, может ли модель определить, *когда* повторяющееся ребро появится снова.
- **Inductive NS**: выборка из рёбер, появляющихся только в *будущей* части последовательности; проверяет обобщение на невиданные при обучении рёбра.

Показано, что random-only NS **массивно завышает** результаты и что historical/inductive негативы меняют ранжирование моделей.[^3][^2]

Дальнейшие работы уточняют **hard negative sampling** для *обучения*:

- **ENS** постепенно увеличивает сложность негативов.[^6][^7]
- **Nearest-neighbour hard negatives** для temporal GNN сэмплируют негативы с похожими эмбеддингами.[^8][^9]
- **Recently Popular NS (RP-NS)** для TGN/DyRep акцентирует негативы среди глобально популярных destinations.[^10]

Cornell et al. (2025) предупреждают, что embedding-based hard negatives делают оценку хрупкой, и рекомендуют фокусироваться на random/historical/inductive схемах.[^11][^12]

### 2.4 ROLAND: большие candidate sets со случайными негативами

ROLAND оценивает future link prediction на snapshot-based графах (включая Bitcoin OTC/Alpha) с **чисто случайным negative sampler**, но с большими candidate sets:[^13]

- Для каждого positive edge (u, v) в snapshot t+1 сэмплируется **10 001 негативный destination** для source u.
- Вычисляется ранг v среди 10 002 кандидатов, усредняется MRR.

### 2.5 Итого: что делать на практике

Для нового динамического транзакционного графа (Bitcoin-подобного), best practice по TGB и DGB:

- **Оценка**: задача ранжирования с множественными негативами на позитив.
- Для каждого (s, d, t) фиксируем s и t, сэмплируем q негативных destinations.
- **Смесь исторических и случайных негативов**; опционально inductive split.
- **Обучение**: random/historical негативы.

Чисто random one-negative оценка с AUROC/AP — устаревший подход.


## 3. Стратегии формирования candidate sets

### 3.1 Per-source candidate sets

Все современные протоколы TLP (TGB, DGB, ROLAND, DyRep) определяют кандидатов **per source node и time**:

- Для каждого positive event (s, d, t) определяем query q = (s, t).
- Candidate set C(q) = {d} ∪ N^-(q), где N^-(q) — негативные destinations для этого source и time.
- Оцениваем все d' ∈ C(q) и вычисляем ранг истинного destination.

Концептуально эквивалентно recommendation-style оценке: для "пользователя" s в момент t ранжируем список "товаров" (destinations).

### 3.2 Полное vs выборочное ранжирование

- **Полное ранжирование** — для малых графов. TGB tgbl-wiki оценивает каждое событие по *всем* destinations.
- **Выборочное ранжирование** — стандарт для больших графов:
  - TGB: 100 негативов на позитив (tgbl-review)
  - ROLAND: 10 001 негатив на позитив
  - DGB: десятки-сотни негативов в зависимости от протокола

Для транзакционных графов с миллионами адресов полное ранжирование невозможно.

### 3.3 Popularity-aware candidate sets для транзакционных графов

На транзакционных графах с сильной глобальной динамикой чисто случайные негативы недооценивают *популярные* destinations. Daniluk & Dąbrowski (NeurIPS 2023 workshop) показывают, что тривиальный baseline "recently popular nodes" (PopTrack), игнорирующий source и ранжирующий только по глобальной частоте destination, достигает **MRR ≈ 0.725 на tgbl-coin и 0.729 на tgbl-comment**, превосходя TGN, DyRep и EdgeBank.[^10]

Для Bitcoin-подобных графов, где доминируют глобальные тренды (биржи, миксеры), рекомендуется включать **popularity-focused вариант кандидатов** в дополнение к TGB-style.


## 4. Precision@K и Hits@K: глобальные vs per-source

### 4.1 Как метрики вычисляются в ключевых работах

- **DyRep**: для каждого события заменяют истинный destination на всех остальных, ранжируют, отчитываются по MAR и Hits@10. Метрики **per query**, затем усредняются.
- **TGB**: filtered MRR — reciprocal rank для каждого positive edge среди его негативов, затем усреднение. Per-query.
- **ROLAND**: MRR усреднением per-query reciprocal ranks по всем snapshot-ам.
- **DGB**: для ranking-метрик оперируют per query (per source–time pair) и агрегируют.

### 4.2 Критика глобальных метрик

Cornell et al. (2025) явно выделяют "combined nodes predictions" как ключевую проблему:[^12][^11]

- Вычисление метрик по *всем* scored edges вместе неявно предполагает равные base rates по source-ам и позволяет high-degree узлам доминировать.
- Рекомендация: вычислять ranking метрики **per source/query и затем усреднять**.

### 4.3 Рекомендуемый протокол для Precision@K / Hits@K

- Определяем evaluation на уровне **queries** q = (s, t).
- Для каждого query ранжируем candidate set C(q):
  - Hits@K: индикатор, что true destination в top-K.
  - MRR: 1/rank true destination.
- Усредняем метрику **по queries** (macro averaging).

Глобальное вычисление precision@K (pooling sources) **не рекомендуется** современной литературой.


## 5. Типичные результаты бейзлайнов

### 5.1 TGB транзакционные датасеты (tgbl-coin, tgbl-comment)

| Датасет | Модель / Baseline | Test MRR |
|--------|-------------------|----------|
| tgbl-coin (stablecoin tx) | DyRep | 0.452 ± 0.046 |
| | TGN | 0.586 ± 0.037 |
| | EdgeBank_tw (temporal memory heuristic) | 0.580 |
| | EdgeBank_∞ | 0.359 |
| | Preferential Attachment (PA) | 0.481 |
| | PA restricted to recent edges (PA_rec) | 0.584 |
| | PopTrack (recently popular, no learning) | 0.725 |

| Датасет | Модель / Baseline | Test MRR |
|--------|-------------------|----------|
| tgbl-comment (Reddit reply) | DyRep | 0.289 ± 0.033 |
| | TGN | 0.379 ± 0.021 |
| | EdgeBank_tw | 0.149 |
| | EdgeBank_∞ | 0.129 |
| | CN (Common Neighbours) | 0.131 |
| | CN_rec, AA_rec, RA_rec | ≈ 0.242–0.245 |
| | PopTrack | 0.729 |

Ключевые наблюдения для Bitcoin-подобных графов:

- Простые нейчённые эвристики (PA, CN/AA/RA на последних рёбрах) достигают **MRR ≈ 0.48–0.58** на tgbl-coin, сравнимо с TGN/DyRep.
- Чисто глобальная популярность (PopTrack) достигает **MRR ≈ 0.72–0.73**, показывая силу глобальной темпоральной динамики.

Реалистичные цели для LogReg/CatBoost: превзойти PA/PA_rec, EdgeBank и, в идеале, PopTrack.

### 5.2 ROLAND на Bitcoin OTC / Alpha trust графах

| Датасет | Модель | MRR |
|---------|--------|-----|
| Bitcoin-OTC | Best baseline (EvolveGCN) | ≈ 0.152 |
| | ROLAND-GRU | ≈ 0.220 |
| Bitcoin-Alpha | Best baseline | ≈ 0.201 |
| | ROLAND-GRU | ≈ 0.289 |

### 5.3 LogReg / CatBoost бейзлайны

**Нет общепринятого стандарта MRR/Hits@K для LogReg/CatBoost** на temporal link prediction:

- Бенчмарки TGB/DGB/ROLAND фокусируются на neural temporal GNN + непараметрические эвристики (EdgeBank, PA, CN, PopTrack).
- Gradient boosted trees и LogReg популярны для *node classification* на Bitcoin (Elliptic AML), но не для link prediction.

Защитимый подход: использовать **heuristic scores** (PA, CN/AA/RA, EdgeBank, PopTrack) как фичи для LogReg/CatBoost и ожидать результаты **сравнимые или немного лучше** raw heuristic scores.


## 6. Bitcoin-специфичные динамические транзакционные графы

- **ORBITAAL**: temporal entity–entity Bitcoin dataset (2009–2021) со stream и snapshot представлениями. Нет канонического LP-протокола.
- **TGB tgbl-coin**: stablecoin транзакции — ближайший стандартизированный датасет.
- **ROLAND Bitcoin OTC/Alpha**: trust графы, структурно похожие на entity-level транзакционные сети.

Рекомендуется: протокол TGB + popularity-aware candidate subsets.


## 7. Практические рекомендации для нашего проекта

1. **Train/test split**: хронологический, 70/15/15 по TGB. Использовать только историю до момента t при предсказании t' > t.
2. **Evaluation setting**: streaming (TGB) или live-update (ROLAND).
3. **Negative sampling (eval)**:
   - Per-source: для каждого (s, d, t) фиксируем s, t, сэмплируем q негативов (50/50 historical+random).
   - Дополнительно: popular-destinations evaluation (top-N популярных адресов).
4. **Candidate generation**: per-query candidate sets. Для больших графов — sampled (q=49 негативов + 1 true = 50 кандидатов).
5. **Метрики**:
   - Основная: **filtered MRR** (per-query, усреднённая).
   - Дополнительные: **Hits@1, Hits@3, Hits@10** (per-query).
   - Опционально: AUROC/AP для совместимости со старыми работами.
6. **Бейзлайны**:
   - Не-обучаемые: PA, PA_rec, CN/AA/RA и "recent" варианты, EdgeBank, PopTrack.
   - Классические ML: LogReg / CatBoost на фичах (degree, temporal counts, heuristic scores, recency).
   - Нейронные (следующий этап): TGN, DyRep, ROLAND-style GNN.

---

## Ссылки

[^1]: Huang et al. "Temporal Graph Benchmark for Machine Learning on Temporal Graphs" (NeurIPS 2023)
[^2]: Poursafaei et al. "Towards Better Evaluation for Dynamic Link Prediction" (NeurIPS 2022)
[^3]: DGB evaluation analysis
[^4]: Trivedi et al. "DyRep: Learning Representations over Dynamic Graphs" (ICLR 2019)
[^5]: DGB negative sampling strategies
[^6]: "Towards a better negative sampling strategy for dynamic graphs" (2024)
[^7]: ENS publication
[^8]: "Robust Training of Temporal GNNs using Nearest Neighbours based Hard Negatives" (2024)
[^9]: NN-based hard negatives
[^10]: Daniluk & Dąbrowski "Temporal graph models fail to capture global temporal dynamics" (NeurIPS 2023 workshop)
[^11]: Cornell et al. "Are We Really Measuring Progress?" (2025)
[^12]: Cornell et al. survey
[^13]: You et al. "ROLAND: Graph Learning Framework for Dynamic Graphs" (KDD 2022)

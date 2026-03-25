# GLFormer — временное предсказание связей на stream graph

Реализация модели **GLFormer** из статьи:
> Zou et al. "Global-Lens Transformers: Adaptive Token Mixing for Dynamic Link Prediction". AAAI 2026.

---

## Архитектура

### Идея

GLFormer заменяет self-attention в Transformer-based temporal graph moделях на **Adaptive Token Mixer** (ATM) — каузальную локальную агрегацию с учётом порядка взаимодействий и временного decay.

По сравнению с EAGLE-Time:
- **ATM вместо MLP-Mixer**: для каждой позиции i в последовательности соседей ATM агрегирует из позиций i+p (более старых соседей), взвешивая по обучаемому порядковому весу `w_p` и временному decay `softmax(-Δt)`.
- **Иерархия (dilated)**: каждый слой l использует смещения `[s_{l-1}, s_l]`, расширяя рецептивное поле с глубиной.
- **Конкатенационный предиктор** вместо аддитивного: `K2(ReLU(K1([h_src; h_dst])))` лучше моделирует взаимодействие между src и dst.
- **Co-occurrence**: опциональный сигнал — количество общих 1-hop соседей src и dst.

### Блоки

```
Вход: K последних соседей узла u в момент t
        ↓
GLFormerTimeEncoding: cosine(delta_t * omega_i) → [B, K, time_dim]
        ↓
Linear([time_enc, edge_feats]) → [B, K, hidden_dim]     ← токены
        ↓
GLFormerBlock × L (иерархические слои):
    ┌── LayerNorm + AdaptiveTokenMixer (смещения [s_{l-1}, s_l])
    ├── residual connection
    ├── LayerNorm + ChannelFFN (MLP, expansion=4.0)
    └── residual connection
        ↓
LayerNorm + masked mean pool → [B, hidden_dim]
        ↓
(опц.) + Linear(node_feats)
        ↓
Эмбеддинг узла Z_u ∈ R^{hidden_dim}
```

**Предиктор ребра:**
```
z(u,v) = K2(ReLU(K1([Z_u; Z_v; cooc_feat?])))  →  скаляр (логит)
```

### Adaptive Token Mixer (формулы из статьи)

Для каждой позиции i в последовательности (most-recent-first):

```
H_{i} = Σ_{p ∈ offsets} α_p^i · H_{i+p}

α_p^i = β · w_p  +  (1-β) · θ_p^i

θ_p^i = softmax(−(δ_{i+p} − δ_i))  по p
       (δ = query_time − neighbor_time, i+p старее → больший δ)
```

Где `w_p` — обучаемые веса по порядку (размер K), `β` — обучаемый скаляр (sigmoid).

### Иерархические слои

Слой l использует смещения `{s_{l-1}, ..., s_l}`, где `s_0=0`, `s_l = 2^l`:

| L | Слой 1 | Слой 2 | Слой 3 | Рецептивное поле |
|---|--------|--------|--------|-----------------|
| 1 | [0,2]  | —      | —      | 2               |
| 2 | [0,2]  | [2,4]  | —      | 6               |
| 3 | [0,2]  | [2,4]  | [4,8]  | 14              |

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

CSR-структуры для поиска соседей:
- При обучении: CSR только из train-рёбер
- При val/test: CSR из train+val рёбер (тестовые рёбра никогда не попадают в CSR)

---

## Режимы признаков

| Режим | `--edge-feat-dim` | `--node-feat-dim` | `--use-cooccurrence` | Описание |
|-------|------------------|------------------|---------------------|----------|
| Временно́й | 0 | 0 | нет | только delta times |
| Рёберные признаки | 2 | 0 | нет | btc+usd каждого соседа **(рекомендуется)** |
| Полный | 2 | >0 | нет | + собственные признаки узла |
| С co-occurrence | 2 | 0 | да | + количество общих соседей (src, dst) |

**Рекомендуемая конфигурация** для нашего датасета: `--edge-feat-dim 2`.

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
PYTHONPATH=. python src/models/GLFormer/glformer_hpo.py \
    --parquet-path stream_graph/2020-06-01_2020-08-31.parquet \
    --edge-feat-dim 2 \
    --n-trials 30 --hpo-epochs 15 \
    --output /tmp/glformer_hpo \
    2>&1 | tee /tmp/glformer_hpo.log
```

HPO перебирает: `hidden_dim`, `num_neighbors`, `num_glformer_layers`, `lr`, `weight_decay`, `dropout`, `batch_size`, `channel_expansion`. Результат — файл `hpo_results.json` и скрипт `best_train_command.sh`.

### Обучение

```bash
YADISK_TOKEN="..." PYTHONPATH=. python src/models/GLFormer/glformer_launcher.py \
    --parquet-path stream_graph/2020-06-01_2020-08-31.parquet \
    --edge-feat-dim 2 \
    --epochs 100 \
    --num-glformer-layers 2 \
    --hidden-dim 100 \
    --num-neighbors 20 \
    --lr 0.0001 \
    --output /tmp/glformer_results \
    2>&1 | tee /tmp/glformer_train.log
```

С co-occurrence:

```bash
PYTHONPATH=. python src/models/GLFormer/glformer_launcher.py \
    --parquet-path stream_graph/2020-06-01_2020-08-31.parquet \
    --edge-feat-dim 2 --use-cooccurrence --cooc-dim 16 \
    --epochs 100 --output /tmp/glformer_cooc
```

---

## Параметры CLI

### `glformer_launcher.py`

| Параметр | По умолчанию | Описание |
|----------|-------------|----------|
| `--parquet-path` | обязательный | Путь к stream graph parquet |
| `--train-ratio` | 0.7 | Доля рёбер для обучения |
| `--val-ratio` | 0.15 | Доля рёбер для валидации |
| `--output` | /tmp/glformer_results | Директория для результатов |
| `--epochs` | 100 | Максимальное число эпох |
| `--batch-size` | 200 | Размер батча |
| `--lr` | 0.0001 | Learning rate Adam |
| `--weight-decay` | 1e-5 | L2-регуляризация |
| `--num-neighbors` | 20 | K соседей на узел |
| `--hidden-dim` | 100 | Размерность скрытого слоя |
| `--num-glformer-layers` | 2 | Число GLFormer-блоков (1-3) |
| `--channel-expansion` | 4.0 | Коэффициент расширения FFN |
| `--dropout` | 0.1 | Dropout |
| `--patience` | 20 | Early stopping (эпох без улучшения val MRR) |
| `--seed` | 42 | Random seed |
| `--no-amp` | False | Отключить AMP (mixed precision) |
| `--max-val-edges` | 5000 | Макс. рёбер при валидации |
| `--edge-feat-dim` | 2 | Размерность признаков рёбер соседей |
| `--node-feat-dim` | 0 | Размерность признаков узла-запроса |
| `--use-cooccurrence` | False | Включить co-occurrence признаки |
| `--cooc-dim` | 16 | Размерность co-occurrence энкодинга |

### `glformer_hpo.py`

Дополнительные параметры: `--n-trials` (30), `--hpo-epochs` (15).

---

## Структура выходных данных

```
/tmp/glformer_results/glformer_<parquet_stem>/
    config.json           # гиперпараметры и конфигурация модели
    best_model.pt         # чекпоинт лучшей модели по val MRR
    metrics.jsonl         # метрики по эпохам (JSON Lines)
    training_curves.csv   # кривые обучения для построения графиков
    data_summary.json     # статистика датасета
    final_results.json    # итоговые метрики (test MRR, Hits@K, время)
    summary.json          # best_epoch, best_val_mrr
    experiment.log        # лог эксперимента
```

На Яндекс.Диске: `orbitaal_processed/experiments/exp_006_glformer/<exp_name>/`

---

## API

```python
from src.models.GLFormer import GLFormerTime
from src.models.GLFormer.data_utils import load_stream_graph_data
from src.models.GLFormer.glformer_train import train_glformer
from src.models.GLFormer.glformer_evaluate import evaluate_tgb_style

# Загрузка данных
data, train_mask, val_mask, test_mask = load_stream_graph_data(
    "stream_graph/2020-06-01_2020-08-31.parquet"
)

# Обучение
model, history = train_glformer(
    data=data,
    train_mask=train_mask,
    val_mask=val_mask,
    output_dir="/tmp/glformer_exp",
    edge_feat_dim=2,
    num_glformer_layers=2,
    hidden_dim=100,
)

# Оценка
from src.models.GLFormer.data_utils import build_temporal_csr
import torch

test_csr = build_temporal_csr(data, train_mask | val_mask)
metrics = evaluate_tgb_style(
    model=model, data=data, csr=test_csr, eval_mask=test_mask,
    device=torch.device("cuda"),
)
print(f"Test MRR: {metrics['mrr']:.4f}")
```

---

## Отличия от EAGLE

| Аспект | EAGLE-Time | GLFormer |
|--------|-----------|----------|
| Token mixer | MLP-Mixer (параллельный, не каузальный) | Adaptive Token Mixer (каузальный, time-aware) |
| Слои | Независимые, одинаковый рецептивный доступ | Иерархические, расширяющийся охват |
| Предиктор | Аддитивный: fc_src(h) + fc_dst(h) | Конкатенационный: MLP([h_src; h_dst]) |
| Edge features | Опциональные | Нативные (рекомендуется `--edge-feat-dim 2`) |
| Co-occurrence | Нет | Опциональные (флаг `--use-cooccurrence`) |
| Learning rate | 0.001 (по умолчанию) | 0.0001 (по умолчанию) |
| Patience | 10 | 20 |

---

## Зависимости

- `torch >= 2.0`
- `torch_geometric` (для TemporalData в data_utils)
- `numpy`, `pandas`, `tqdm`
- `optuna` (только для glformer_hpo.py)
- Скомпилированное C++ расширение (`python src/models/build_ext.py`)

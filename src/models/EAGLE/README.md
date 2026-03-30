# EAGLE-Time: Temporal Link Prediction на Stream Graph

Реализация модели **EAGLE-Time** из статьи *"EAGLE: Expressive dynamics-Aware Graph LEarning" (Yue et al., 2024)* для задачи предсказания ссылок (temporal link prediction) на потоковом графе транзакций.

**Протокол оценки полностью совместим с `sg_baselines/CORRECTNESS_CHECKLIST.md`.**

---

## Архитектура

EAGLE-Time использует **временные паттерны взаимодействий** (delta times) и может быть обогащён признаками рёбер и узлов.

```
Для каждой пары (src, dst) при времени t:
  1. Найти K последних соседей src до t  → delta times (t - t_neighbor_i)
  2. Найти K последних соседей dst до t  → delta times
  3. Закодировать каждый набор через EAGLETimeEncoder:
       cos(delta_t * omega) [+ edge_feats] → Linear → MLP-Mixer × L → mean-pool → Linear [+ node_fc(node_feats)]
  4. Предсказать оценку: relu(fc_src(h_src) + fc_dst(h_dst)) → Linear(1)
```

### Модули

| Файл | Описание |
|------|----------|
| [eagle.py](eagle.py) | Архитектура: `EAGLETimeEncoding`, `EAGLEMixerBlock`, `EAGLETimeEncoder`, `EAGLETime` |
| [eagle_train.py](eagle_train.py) | Цикл обучения, валидация, ранняя остановка |
| [eagle_evaluate.py](eagle_evaluate.py) | TGB-style оценка (sg_baselines протокол) |
| [eagle_launcher.py](eagle_launcher.py) | CLI-точка входа для запуска экспериментов |
| [eagle_hpo.py](eagle_hpo.py) | Подбор гиперпараметров через Optuna |
| [tppr.py](tppr.py) | Temporal Personalized PageRank (структурный скоринг) |
| [data_utils.py](data_utils.py) | Загрузка stream graph, node features, конвертация в `TemporalEdgeData` |

### Режимы признаков

| `edge_feat_dim` | `node_feat_dim` | Описание |
|:---:|:---:|----------|
| 0 | 0 | Только временные паттерны (оригинальный EAGLE-Time) |
| 2 | 0 | + признаки рёбер-соседей (btc/usd каждой транзакции из истории) |
| 0 | 15 | + 15 статических признаков узла из `features_10.parquet` |
| 2 | 15 | Все три сигнала |

### Параметры модели (по умолчанию)

| Параметр | Значение | Описание |
|----------|----------|----------|
| `hidden_dim` | 100 | Размер скрытых представлений |
| `num_neighbors` | 20 | K последних соседей |
| `num_mixer_layers` | 1 | Количество MLP-Mixer блоков |
| `time_dim` | 100 | Размерность временного кодирования |
| `token_expansion` | 0.5 | Расширение в token-mixing MLP |
| `channel_expansion` | 4.0 | Расширение в channel-mixing MLP |
| `dropout` | 0.1 | Dropout |

~120K параметров в режиме time-only; при `node_feat_dim=15` ~122K.

---

## Совместимость с sg_baselines

Eval протокол **идентичен** бейзлайнам (см. `sg_baselines/CORRECTNESS_CHECKLIST.md`):

| Компонент | Реализация |
|-----------|-----------|
| Split | train 70% / val 15% / test 15% от period (fraction от полного stream graph) |
| Негативы (eval) | `sample_negatives_for_eval` из `sg_baselines/sampling.py` |
| Random negatives | Из `active_nodes` (train ноды), НЕ из всех нод |
| Historical negatives | Исключают ВСЕ позитивы src из полного eval split'а |
| Query filtering | Только рёбра с src И dst_true в train node_mapping |
| 50K subsample | Да, `--max-test-edges 50000` |
| Rank | Conservative: `count(score > true_score) + 1` |
| Eval seeds | val: seed+300, test: seed+400 |
| Метрики | `compute_ranking_metrics` из `src/baselines/evaluation.py` |
| Node features | 15 фичей из `features_{label}.parquet` (train-only, no leakage) |

---

## Быстрый старт

### Запуск на period_10 с node features (рекомендуемый)

```bash
YADISK_TOKEN="..." PYTHONPATH=. python src/models/EAGLE/eagle_launcher.py \
    --parquet-path /tmp/sg_baselines_data/stream_graph.parquet \
    --features-path /tmp/sg_baselines_data/features_10.parquet \
    --node-mapping-path /tmp/sg_baselines_data/node_mapping_10.npy \
    --fraction 0.10 \
    --node-feat-dim 15 \
    --epochs 100 --batch-size 200 --patience 10 \
    --output /tmp/eagle_results \
    2>&1 | tee /tmp/eagle_train.log
```

### Обучение time-only (без признаков узлов)

```bash
PYTHONPATH=. python src/models/EAGLE/eagle_launcher.py \
    --parquet-path /tmp/sg_baselines_data/stream_graph.parquet \
    --fraction 0.10 \
    --node-mapping-path /tmp/sg_baselines_data/node_mapping_10.npy \
    --epochs 100 \
    --output /tmp/eagle_results_timeonly \
    2>&1 | tee /tmp/eagle_timeonly.log
```

### Подбор гиперпараметров (Optuna)

```bash
PYTHONPATH=. python src/models/EAGLE/eagle_hpo.py \
    --parquet-path /tmp/sg_baselines_data/stream_graph.parquet \
    --features-path /tmp/sg_baselines_data/features_10.parquet \
    --node-mapping-path /tmp/sg_baselines_data/node_mapping_10.npy \
    --fraction 0.10 --node-feat-dim 15 \
    --n-trials 30 --hpo-epochs 15 \
    --output /tmp/eagle_hpo \
    2>&1 | tee /tmp/eagle_hpo.log
```

---

## CLI-параметры (eagle_launcher.py)

### Данные

| Флаг | По умолчанию | Описание |
|------|-------------|----------|
| `--parquet-path` | (обязательный) | Путь к stream graph parquet-файлу |
| `--features-path` | None | Путь к features_{label}.parquet (15 node features) |
| `--node-mapping-path` | None | Путь к node_mapping_{label}.npy (active_nodes) |
| `--fraction` | None | Доля stream graph для period (0.10 или 0.25) |
| `--train-ratio` | 0.7 | Доля рёбер для обучения |
| `--val-ratio` | 0.15 | Доля рёбер для валидации |
| `--output` | `/tmp/eagle_results` | Папка для результатов |

### Гиперпараметры модели

| Флаг | По умолчанию | Описание |
|------|-------------|----------|
| `--hidden-dim` | 100 | Размер скрытого представления |
| `--num-neighbors` | 20 | K соседей для сэмплинга |
| `--num-mixer-layers` | 1 | Количество MLP-Mixer блоков |
| `--token-expansion` | 0.5 | Token-mixing expansion |
| `--channel-expansion` | 4.0 | Channel-mixing expansion |
| `--dropout` | 0.1 | Dropout rate |
| `--edge-feat-dim` | 0 | Размерность признаков рёбер (0 = отключено, 2 = btc+usd) |
| `--node-feat-dim` | 0 | Размерность признаков узлов (0 = авто из features, 15 = features_10) |

### Обучение

| Флаг | По умолчанию | Описание |
|------|-------------|----------|
| `--epochs` | 100 | Максимальное число эпох |
| `--batch-size` | 200 | Рёбер в батче |
| `--lr` | 0.001 | Learning rate (Adam) |
| `--weight-decay` | 5e-5 | Weight decay |
| `--patience` | 10 | Early stopping patience |
| `--seed` | 42 | Random seed |
| `--no-amp` | (флаг) | Отключить mixed precision |
| `--max-val-edges` | 5000 | Макс. рёбер для оценки на val (при обучении) |
| `--max-test-edges` | 50000 | Макс. queries для eval (50K по TGB протоколу) |

---

## Структура результатов

```
eagle_<name>_f10/
  config.json           # гиперпараметры эксперимента
  data_summary.json     # статистика датасета (nodes, edges, splits)
  metrics.jsonl         # метрики по эпохам (JSONL)
  training_curves.csv   # кривые обучения (CSV для построения графиков)
  best_model.pt         # веса лучшей модели (по val MRR)
  final_results.json    # val + test метрики + timing
  experiment.log        # полный лог эксперимента
```

### Бейзлайн результаты для сравнения (period_10)

```
cn:      val MRR=0.8641, test MRR=0.8725
jaccard: val MRR=0.8517, test MRR=0.8621
aa:      val MRR=0.8621, test MRR=0.8709
pa:      val MRR=0.7405, test MRR=0.8036
```

---

## Зависимости

- `torch >= 2.0.0` — модель и обучение
- `torch_geometric` — `TemporalData` для stream graph
- `numpy >= 1.24.0`, `pandas >= 2.0.0`, `pyarrow >= 12.0.0`
- `tqdm >= 4.65.0` — прогресс-бары
- `optuna` — HPO (только для `eagle_hpo.py`)
- `sg_baselines` — протокол негативов и оценки
- `src.baselines.evaluation` — `compute_ranking_metrics`
- `src.yadisk_utils` — загрузка результатов на Яндекс.Диск

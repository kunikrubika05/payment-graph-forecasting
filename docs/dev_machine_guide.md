# Инструкция по работе с GPU дев-машиной

Короткий operational checklist для запусков на GPU-машине.

Провайдер, который использовался в проекте: [immers.cloud](https://immers.cloud/).

## Базовые требования

- Ubuntu 24.04 CUDA image
- Python 3.10+
- доступ к GPU через `nvidia-smi`
- установленный проект с зависимостями `.[dl,hpo,dev]`
- при удалённой работе запускать долгие процессы в `tmux`

## Минимальная проверка окружения

После установки репозитория на машине проверь package-facing entrypoints:

```bash
./venv/bin/python -m payment_graph_forecasting.experiments.launcher --help
./venv/bin/python -m payment_graph_forecasting.experiments.hpo --help
./venv/bin/python -m payment_graph_forecasting.infra.extensions --help
```

## Минимальная проверка запуска

Перед долгим запуском проверь dry-run на примерах:

```bash
./venv/bin/python -m payment_graph_forecasting.experiments.launcher --config exps/examples/graphmixer_library.yaml --dry-run
./venv/bin/python -m payment_graph_forecasting.experiments.launcher --config exps/examples/dygformer_library.yaml --dry-run
./venv/bin/python -m payment_graph_forecasting.experiments.launcher --config exps/examples/pairwise_mlp_library.yaml --dry-run
```

## Optional Extensions

Собирать optional C++/CUDA extensions имеет смысл только когда они реально
нужны для запуска или бенчмарка. Package-facing entrypoint:

```bash
./venv/bin/python -m payment_graph_forecasting.infra.extensions --help
```

Для реальной сборки нужен `ninja` в активном окружении.

## Практические замечания

- Пользовательский запуск должен идти через `payment_graph_forecasting.*`, а не через `src/*`.
- Для удалённой машины удобно держать `YADISK_TOKEN` в окружении перед реальными запусками.
- Visualization остаётся legacy/internal tooling и не является основной частью GPU workflow.

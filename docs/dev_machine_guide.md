# Инструкция по работе с GPU дев-машиной (immers.cloud)

> Этот документ описывает полный пайплайн: от аренды GPU-машины на immers.cloud
> до запуска эксперимента и получения результатов. Написан для участников проекта
> payment-graph-forecasting.

## 1. Аренда машины

### Сервис
Используем **immers.cloud** (https://immers.cloud/). Российский провайдер GPU-серверов,
оплата в рублях, без VPN.

### Выбор конфигурации

При создании сервера выбираем:
- **Образ:** Ubuntu 24.04 CUDA (в разделе BIOS). Именно BIOS, не UEFI.
- **GPU:** Tesla T4 (15 GB VRAM) — достаточно для наших моделей (138K параметров).
  Для ускорения обучения можно взять A10 (24 GB) или A100 (40/80 GB).
- **CPU:** 16 vCPU (минимум 8)
- **RAM:** 64 GB (минимум 32 GB — нужно для загрузки данных ~24 GB)
- **SSD:** 100-160 GB (данные + venv с PyTorch занимают ~30 GB)

### Создание сервера

1. Зарегистрируйся на https://immers.cloud/, пополни баланс
2. Создай сервер с конфигурацией выше
3. При создании сервис сгенерирует **SSH-ключ** (пара .pem)
4. **Скачай .pem файл** — это единственный способ подключения, пароля нет
5. Запомни **IP-адрес** сервера (показывается после создания)

> **Важно:** .pem файл выдаётся один раз при создании. Если потеряешь — доступ к
> серверу будет утерян. Храни его надёжно, но **не коммить в git** (.pem в .gitignore).

## 2. Подключение по SSH

### Первое подключение

```bash
# Ограничить права на ключ (обязательно, иначе SSH откажет)
chmod 600 /path/to/your-key.pem

# Подключиться (пользователь всегда ubuntu)
ssh -i /path/to/your-key.pem ubuntu@<IP-АДРЕС>
```

При первом подключении SSH спросит про fingerprint — отвечай `yes`.

### Защита от разрыва SSH

SSH-соединение может разорваться при долгом простое. Чтобы избежать:

```bash
# Добавить в ~/.ssh/config на ЛОКАЛЬНОЙ машине
Host immers
    HostName <IP-АДРЕС>
    User ubuntu
    IdentityFile /path/to/your-key.pem
    ServerAliveInterval 60
    ServerAliveCountMax 3
```

После этого подключение: `ssh immers`

## 3. Настройка окружения (один раз после создания сервера)

```bash
# Обновить пакеты и установить python3-dev (нужен для C++ расширения)
sudo apt update && sudo apt install -y python3-venv python3-pip python3.12-dev

# Проверить GPU
nvidia-smi

# Склонировать репозиторий
cd ~ && git clone https://github.com/kunikrubika05/payment-graph-forecasting.git
cd payment-graph-forecasting

# Создать виртуальное окружение и установить зависимости
python3 -m venv venv
source venv/bin/activate
pip install -e ".[dl,hpo,dev]"

# Собрать optional C++/CUDA расширения, если они нужны
./venv/bin/python -m payment_graph_forecasting.infra.extensions

# Проверить что package launcher работает
./venv/bin/python -m payment_graph_forecasting.experiments.launcher \
    --config exps/examples/graphmixer_library.yaml --dry-run
```

## 4. Запуск эксперимента

### Всегда в tmux!

Эксперименты запускаются **только в tmux** — если SSH оборвётся, процесс продолжит работу.

```bash
# Создать tmux-сессию
tmux new -s pfg

# Внутри tmux: активировать venv и задать токен
cd ~/payment-graph-forecasting && source venv/bin/activate
export YADISK_TOKEN="<токен Яндекс.Диска>"

# Запустить эксперимент через package launcher
./venv/bin/python -m payment_graph_forecasting.experiments.launcher \
    --config exps/examples/graphmixer_library.yaml \
    2>&1 | tee /tmp/pfg_experiment.log

# Отсоединиться от tmux: Ctrl+B, затем D
```

### Рекомендуемый путь запуска

Для новых запусков используй YAML-конфиги через
`payment_graph_forecasting.experiments.launcher`.

Базовые примеры:

```bash
./venv/bin/python -m payment_graph_forecasting.experiments.launcher \
    --config exps/examples/graphmixer_library.yaml --dry-run

./venv/bin/python -m payment_graph_forecasting.experiments.launcher \
    --config exps/examples/sg_graphmixer_library.yaml --dry-run

./venv/bin/python -m payment_graph_forecasting.experiments.launcher \
    --config exps/examples/pairwise_mlp_library.yaml --dry-run
```

Типовые секции YAML:

| Поле | Описание |
|----------|----------|
| `experiment.model` | Имя library модели |
| `data.*` | Источник данных и data-specific параметры |
| `sampling.*` | Негативы, sampler backend и др. |
| `training.*` | Эпохи, batch size, lr, patience |
| `runtime.*` | `device`, `amp`, `dry_run`, `output_dir` |
| `model.*` | model-specific параметры |

### Legacy launchers

Старые `src/models/*` launchers всё ещё есть для совместимости и исторических
экспериментов, но это уже не основной рекомендуемый surface.

### Optional extension build

Для сборки optional C++/CUDA extensions используй package-facing entrypoint:

```bash
# Для реальной сборки нужен `ninja` в активном окружении.
./venv/bin/python -m payment_graph_forecasting.infra.extensions
./venv/bin/python -m payment_graph_forecasting.infra.extensions --all --graph-metrics
./venv/bin/python -m payment_graph_forecasting.infra.extensions --graph-metrics-cuda
```

`src/models/build_ext.py` сохранён как compatibility shim для старых инструкций
и исторических запусков.

Если расширение уже было собрано ранее, package-facing и legacy-backed runtime
path теперь используют prebuilt binary из локального build-кэша. `ninja`
нужен именно для пересборки, а не для каждого последующего запуска.

### Получение YADISK_TOKEN

1. Перейди на https://oauth.yandex.ru
2. Создай приложение с правами `cloud_api:disk.*`
3. Получи OAuth-токен
4. Используй как `export YADISK_TOKEN="..."`

Результаты автоматически загрузятся на Яндекс.Диск после завершения эксперимента.

## 5. Мониторинг

### Подключение к работающей сессии
```bash
# Из другого SSH-окна (или после переподключения)
tmux attach -t pfg
# Отсоединиться обратно: Ctrl+B, затем D
```

### Просмотр логов
```bash
# Последние строки лога
tail -20 /tmp/pfg_experiment.log

# Следить в реальном времени
tail -f /tmp/pfg_experiment.log
```

### Проверка GPU
```bash
# Однократно
nvidia-smi

# Следить каждые 2 секунды
watch -n 2 nvidia-smi
```

Что смотреть в nvidia-smi:
- **GPU-Util** — должен быть >80% во время тренировки (идеально 90%+)
- **Memory-Usage** — сколько VRAM занято
- **Power** — если Temp >80°C, GPU троттлит (для T4 норма до 70°C)

### Проверка что процесс жив
```bash
ps aux | grep payment_graph_forecasting.experiments.launcher
```

### Проверка RAM
```bash
free -h
```
Если Memory usage >90%, есть риск OOM-killer.

## 6. После завершения эксперимента

### Скачать результаты на локальную машину
```bash
# С ЛОКАЛЬНОЙ машины
scp -i /path/to/your-key.pem -r \
    ubuntu@<IP>:/tmp/graphmixer_results/ \
    ./GRAPH_MIXER/
```

Результаты также загружены на Яндекс.Диск (если YADISK_TOKEN был задан).

### Структура результатов
```
<run_name>/
    data_summary.json     # Статистика данных
    training_curves.csv   # Метрики по эпохам (если путь их пишет)
    metrics.jsonl         # Подробные метрики
    experiment.log        # Полный лог
    best_model.pt         # Лучшая модель
    final_results.json    # Итоговые метрики + тайминги
    summary.json          # Training summary (если путь его пишет)
```

### Удаление сервера
После получения результатов **удали сервер** в панели immers.cloud,
чтобы не тратить деньги. Все данные на сервере будут потеряны, но результаты
уже на Яндекс.Диске и/или скачаны локально.

## 7. Обновление кода между экспериментами

Если код изменился (новый коммит в main), на дев-машине:

```bash
cd ~/payment-graph-forecasting
git pull
source venv/bin/activate
pip install -e ".[dl,hpo,dev]"
./venv/bin/python -m payment_graph_forecasting.infra.extensions

./venv/bin/python -m payment_graph_forecasting.experiments.launcher \
    --config exps/examples/graphmixer_library.yaml --dry-run
```

Если были локальные изменения:
```bash
git checkout -- . && git pull
```

## 8. Типичные проблемы

| Проблема | Решение |
|----------|---------|
| `Permission denied (publickey)` | `chmod 600 your-key.pem` |
| `Python.h: No such file or directory` | `sudo apt install python3.12-dev` |
| SSH обрывается, процесс пропал | Процесс в tmux жив: `tmux attach -t pfg` |
| tmux-сессия исчезла | OOM-killer убил процесс: `dmesg \| grep -i oom` |
| GPU-Util = 0% во время тренировки | Возможен bottleneck на CPU / sampling backend, это не всегда ошибка |
| `CUDA out of memory` | Уменьшить `--batch-size` или `--num-neighbors` |
| pip install зависает/обрывается | SSH keepalive или запускать pip в tmux |

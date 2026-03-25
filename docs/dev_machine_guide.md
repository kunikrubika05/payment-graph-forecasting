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
pip install -r requirements.txt

# Собрать C++ расширение для neighbor sampling
PYTHONPATH=. python src/models/build_ext.py

# Проверить что всё работает
PYTHONPATH=. python -m pytest tests/test_models.py -v
```

## 4. Запуск эксперимента

### Всегда в tmux!

Эксперименты запускаются **только в tmux** — если SSH оборвётся, процесс продолжит работу.

```bash
# Создать tmux-сессию
tmux new -s gm

# Внутри tmux: активировать venv и задать токен
cd ~/payment-graph-forecasting && source venv/bin/activate
export YADISK_TOKEN="<токен Яндекс.Диска>"

# Запустить эксперимент
PYTHONPATH=. python src/models/launcher.py \
    --period mature_2020q2 \
    --window 7 \
    2>&1 | tee /tmp/graphmixer.log

# Отсоединиться от tmux: Ctrl+B, затем D
```

### Параметры launcher.py

| Параметр | Дефолт | Описание |
|----------|--------|----------|
| `--period` | (обязательный) | Период: early_2012q1, mature_2020q2, late_2020q4 и др. |
| `--window` | 7 | Размер окна (дни): 3, 7, 14, 30 |
| `--epochs` | 100 | Макс. число эпох |
| `--batch-size` | 600 | Размер батча |
| `--lr` | 0.0001 | Learning rate |
| `--hidden-dim` | 100 | Размер скрытого слоя |
| `--num-neighbors` | 20 | Число соседей для sampling |
| `--patience` | 20 | Early stopping patience |
| `--output` | /tmp/graphmixer_results | Куда сохранять результаты |

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
tmux attach -t gm
# Отсоединиться обратно: Ctrl+B, затем D
```

### Просмотр логов
```bash
# Последние строки лога
tail -20 /tmp/graphmixer.log

# Следить в реальном времени
tail -f /tmp/graphmixer.log
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
ps aux | grep launcher
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
graphmixer_<period>_w<window>/
    config.json           # Гиперпараметры
    data_summary.json     # Статистика данных (узлы, рёбра, split)
    training_curves.csv   # Метрики по эпохам
    metrics.jsonl         # Подробные метрики (JSON Lines)
    experiment.log        # Полный лог
    best_model.pt         # Лучшая модель (по val MRR)
    final_results.json    # Итоговые тестовые метрики + тайминги
    summary.json          # Краткая сводка (best epoch, best val MRR)
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
pip install -r requirements.txt
PYTHONPATH=. python src/models/build_ext.py
```

Если были локальные изменения (например, вручную правили requirements.txt):
```bash
git checkout -- . && git pull
```

## 8. Типичные проблемы

| Проблема | Решение |
|----------|---------|
| `Permission denied (publickey)` | `chmod 600 your-key.pem` |
| `Python.h: No such file or directory` | `sudo apt install python3.12-dev` |
| SSH обрывается, процесс пропал | Процесс в tmux жив: `tmux attach -t gm` |
| tmux-сессия исчезла | OOM-killer убил процесс: `dmesg \| grep -i oom` |
| GPU-Util = 0% во время тренировки | Bottleneck на CPU (neighbor sampling), это нормально между батчами |
| `CUDA out of memory` | Уменьшить `--batch-size` или `--num-neighbors` |
| pip install зависает/обрывается | SSH keepalive или запускать pip в tmux |

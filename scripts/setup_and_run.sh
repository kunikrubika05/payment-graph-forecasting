#!/bin/bash
set -e

echo "=== НАСТРОЙКА НОВОЙ ДЕВ-МАШИНКИ ==="
echo ""

if [ -z "$YADISK_TOKEN" ]; then
  echo "ERROR: YADISK_TOKEN не задан!"
  echo "Выполни: export YADISK_TOKEN=\"...\""
  exit 1
fi

echo "=== 1. Клонирование репо ==="
cd ~
if [ -d "payment-graph-forecasting" ]; then
  echo "  Репо уже есть, делаю git pull..."
  cd payment-graph-forecasting && git pull
else
  echo "  Клонирую..."
  git clone https://github.com/kunikrubika05/payment-graph-forecasting.git
  cd payment-graph-forecasting
fi

echo ""
echo "=== 2. Настройка Python venv ==="
if [ ! -d "venv" ]; then
  python3 -m venv venv
fi
source venv/bin/activate
pip install -q -r requirements.txt

echo ""
echo "=== 3. Тесты ==="
PYTHONPATH=. python -m pytest tests/ -v --tb=short
echo ""

echo "=== 4. Синхронизация завершённых экспериментов с Яндекс.Диска ==="
PYTHONPATH=. python scripts/sync_completed.py
echo ""

echo "=== 5. Проверка того, что синхронизировалось ==="
completed=$(find /tmp/baseline_results -name summary.json 2>/dev/null | wc -l)
echo "  Синхронизировано: $completed экспериментов (runner их пропустит)"
echo ""

NCORES=$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 8)
NSESSIONS=4
JOBS_PER_SESSION=$((NCORES / NSESSIONS))
if [ "$JOBS_PER_SESSION" -lt 1 ]; then
  JOBS_PER_SESSION=1
fi

echo "=== 6. Запуск экспериментов ==="
echo "  Cores: $NCORES, Sessions: $NSESSIONS, Jobs/session: $JOBS_PER_SESSION"
echo ""

export RF_N_JOBS=$JOBS_PER_SESSION
export CATBOOST_THREADS=$JOBS_PER_SESSION

PYTHONPATH=. python src/baselines/launcher.py --sessions $NSESSIONS

echo ""
echo "=== ГОТОВО ==="
echo "Мониторинг:"
echo "  tmux ls"
echo "  tail -f /tmp/baseline_logs/baseline_0.log"
echo "  bash scripts/full_check.sh"
echo "  bash scripts/monitor_loop.sh   # авто-обновление каждые 5 мин"

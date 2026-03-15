#!/bin/bash
INTERVAL=${1:-300}

echo "Мониторинг каждые $((INTERVAL / 60)) мин. Ctrl+C для выхода."
echo ""

while true; do
  clear
  bash "$(dirname "$0")/full_check.sh"
  echo ""
  echo "--- Следующая проверка через $((INTERVAL / 60)) мин ($(date -d "+${INTERVAL} seconds" '+%H:%M:%S' 2>/dev/null || date -v+${INTERVAL}S '+%H:%M:%S' 2>/dev/null || echo '?')) ---"

  alive=$(tmux ls 2>/dev/null | grep -c "baseline_")
  if [ "$alive" -eq 0 ]; then
    echo ""
    echo "!!! ВСЕ СЕССИИ ЗАВЕРШИЛИСЬ !!!"
    echo "Проверь результаты: find /tmp/baseline_results -name summary.json | wc -l"
    break
  fi

  sleep $INTERVAL
done

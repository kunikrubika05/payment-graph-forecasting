#!/bin/bash
echo "=== ПОЛНЫЙ МОНИТОРИНГ БЕЙЗЛАЙНОВ ($(date '+%Y-%m-%d %H:%M:%S')) ==="

echo ""
echo "=== 1. TMUX СЕССИИ ==="
tmux ls 2>/dev/null || echo "  НЕТ ЖИВЫХ СЕССИЙ"

echo ""
echo "=== 2. ЗАВЕРШЁННЫЕ ЭКСПЕРИМЕНТЫ ==="
completed=$(find /tmp/baseline_results -name summary.json 2>/dev/null | sort)
count=$(echo "$completed" | grep -c "summary" 2>/dev/null || echo 0)
echo "  Завершено: $count / 35"
echo "$completed" | while read f; do
  [ -z "$f" ] && continue
  dir=$(dirname "$f")
  name=$(echo "$dir" | sed 's|/tmp/baseline_results/||')
  echo "  [DONE] $name"
done

echo ""
echo "=== 3. В РАБОТЕ (config.json есть, summary.json нет) ==="
in_progress=0
for d in $(find /tmp/baseline_results -name config.json -exec dirname {} \; 2>/dev/null | sort); do
  if [ ! -f "$d/summary.json" ]; then
    name=$(echo "$d" | sed 's|/tmp/baseline_results/||')
    echo "  [WORK] $name"
    in_progress=$((in_progress + 1))
  fi
done
[ "$in_progress" -eq 0 ] && echo "  Нет"

echo ""
echo "=== 4. ПОСЛЕДНИЕ 3 СТРОКИ ЛОГОВ ==="
for i in 0 1 2 3; do
  log="/tmp/baseline_logs/baseline_${i}.log"
  if [ -f "$log" ]; then
    echo "--- baseline_$i ---"
    tail -3 "$log" 2>/dev/null
    echo ""
  fi
done

echo ""
echo "=== 5. ОШИБКИ ==="
errors=$(find /tmp/baseline_results -name error.txt 2>/dev/null)
if [ -n "$errors" ]; then
  echo "$errors" | while read f; do
    echo "  ОШИБКА: $f"
    head -3 "$f" | sed 's/^/    /'
  done
else
  echo "  Ошибок нет"
fi

echo ""
echo "=== 6. ИСПОЛЬЗОВАНИЕ РЕСУРСОВ ==="
echo "  CPU: $(uptime | awk -F'load average:' '{print $2}' | xargs) (load avg)"
echo "  RAM: $(free -h 2>/dev/null | awk '/Mem:/{print $3 "/" $2}' || vm_stat 2>/dev/null | head -3)"
echo "  Disk: $(df -h /tmp 2>/dev/null | awk 'NR==2{print $3 "/" $2 " (" $5 ")"}')"

echo ""
echo "=== 7. ВЕРДИКТ ==="
remaining=$((35 - count))
if [ "$remaining" -eq 0 ]; then
  echo "  ============================================"
  echo "  ВСЕ 35 ЭКСПЕРИМЕНТОВ ЗАВЕРШЕНЫ!"
  echo "  ============================================"
elif [ "$in_progress" -eq 0 ] && [ "$(tmux ls 2>/dev/null | grep -c baseline)" -eq 0 ]; then
  echo "  ВНИМАНИЕ: Нет активных сессий, но $remaining экспов не завершены!"
  echo "  Возможно процессы упали. Проверь логи."
else
  echo "  Осталось: $remaining экспов. Считается: $in_progress."
fi

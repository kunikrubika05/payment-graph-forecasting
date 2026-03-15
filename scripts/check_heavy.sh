#!/bin/bash
WAITING_FOR=(
  "exp_001_link_pred_baselines/period_peak_2018q2_w7_time_weighted_modeA"
  "exp_001_link_pred_baselines/period_mature_2020q2_w14_mean_modeA"
  "exp_001_link_pred_baselines/period_late_2020q4_w7_mean_modeA"
  "exp_001_link_pred_baselines/period_peak_2018q2_w7_mean_modeA"
)

BASE="/tmp/baseline_results"
done_count=0
total=${#WAITING_FOR[@]}

echo "=== Проверка тяжёлых экспериментов ($(date '+%H:%M:%S')) ==="
echo ""

for exp in "${WAITING_FOR[@]}"; do
  name=$(basename "$exp")
  if [ -f "$BASE/$exp/summary.json" ]; then
    echo "  [ГОТОВО] $name"
    done_count=$((done_count + 1))
  else
    last_line=$(grep -v "^$" /tmp/baseline_logs/baseline_*.log 2>/dev/null | grep "$name" | tail -1 | sed 's/.*\] /  /')
    echo "  [......] $name"
    echo "           $last_line"
  fi
done

echo ""
echo "--- Готово: $done_count / $total ---"
echo ""

if [ "$done_count" -eq "$total" ]; then
  echo "============================================"
  echo "  ДОЖДАЛИСЬ ВСЕХ ЭКСПОВ, КОТОРЫХ ЖДАЛИ!"
  echo "  Можно killить: tmux kill-server"
  echo "============================================"
else
  remaining=$((total - done_count))
  echo "  ЕЩЁ СЧИТАЕТСЯ: $remaining эксп."
  echo "  Проверяй снова: bash ~/payment-graph-forecasting/scripts/check_heavy.sh"
fi

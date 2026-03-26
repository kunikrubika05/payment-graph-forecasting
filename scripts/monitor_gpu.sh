#!/bin/bash
###############################################################################
# monitor_gpu.sh — Log GPU utilization to CSV every N seconds.
#
# Usage:
#   bash scripts/monitor_gpu.sh /tmp/gpu_log_baseline.csv &
#   # ... run experiment ...
#   kill %1   # or: kill $(cat /tmp/gpu_monitor.pid)
#
# Output CSV columns:
#   timestamp, gpu_util_pct, mem_util_pct, mem_used_mb, mem_total_mb,
#   gpu_temp_c, power_draw_w
#
# For the comparison experiment:
#   bash scripts/monitor_gpu.sh /tmp/gpu_log_baseline.csv &
#   BASELINE_PID=$!
#   # ... run baseline ...
#   kill $BASELINE_PID
#
#   bash scripts/monitor_gpu.sh /tmp/gpu_log_cuda.csv &
#   CUDA_PID=$!
#   # ... run cuda ...
#   kill $CUDA_PID
###############################################################################

OUTPUT="${1:-/tmp/gpu_utilization.csv}"
INTERVAL="${2:-2}"

echo "timestamp,gpu_util_pct,mem_util_pct,mem_used_mb,mem_total_mb,gpu_temp_c,power_draw_w" > "$OUTPUT"
echo "GPU monitoring -> $OUTPUT (every ${INTERVAL}s). PID=$$"
echo $$ > /tmp/gpu_monitor.pid

while true; do
    nvidia-smi --query-gpu=timestamp,utilization.gpu,utilization.memory,memory.used,memory.total,temperature.gpu,power.draw \
        --format=csv,noheader,nounits 2>/dev/null \
    | while IFS= read -r line; do
        echo "$line" >> "$OUTPUT"
    done
    sleep "$INTERVAL"
done

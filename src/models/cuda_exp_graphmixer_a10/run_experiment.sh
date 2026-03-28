#!/usr/bin/env bash
# run_experiment.sh — GraphMixer sampling benchmark: python vs cpp vs cuda
#
# Runs 3 experiments sequentially, prints timing table at the end.
# Expected total time on A10 (batch=2000, K=20, 3 epochs): ~18 min
# Expected C++→CUDA speedup: ~3-4x
#
# Usage:
#   cd ~/payment-graph-forecasting
#   export YADISK_TOKEN="..."
#   bash src/models/cuda_exp_graphmixer_a10/run_experiment.sh
#
# Override defaults:
#   bash src/models/cuda_exp_graphmixer_a10/run_experiment.sh \
#       --parquet-path stream_graph/2020-06-01_2020-08-31.parquet \
#       --epochs 3 --batch-size 2000

set -uo pipefail

PARQUET_PATH="stream_graph/week.parquet"
EPOCHS=3
BATCH_SIZE=2000
NUM_NEIGHBORS=20
MAX_VAL_EDGES=5000
OUTPUT="/tmp/cuda_exp_a10"
LOG="/tmp/cuda_exp_a10.log"
BACKENDS="python cpp cuda"

for arg in "$@"; do
  case $arg in
    --parquet-path=*)   PARQUET_PATH="${arg#*=}" ;;
    --epochs=*)         EPOCHS="${arg#*=}" ;;
    --batch-size=*)     BATCH_SIZE="${arg#*=}" ;;
    --num-neighbors=*)  NUM_NEIGHBORS="${arg#*=}" ;;
    --output=*)         OUTPUT="${arg#*=}" ;;
    --backends=*)       BACKENDS="${arg#*=}" ;;
  esac
done

echo "========================================"
echo "GraphMixer CUDA Sampler Benchmark (A10)"
echo "  parquet:    $PARQUET_PATH"
echo "  epochs:     $EPOCHS"
echo "  batch_size: $BATCH_SIZE"
echo "  output:     $OUTPUT"
echo "========================================"
echo ""

cd ~/payment-graph-forecasting
source venv/bin/activate

echo "[setup] Compiling C++/CUDA extensions..."
python src/models/build_ext.py --all 2>&1 | tail -5
echo ""

run_backend() {
    local backend=$1
    echo ">>> Backend: $backend  ($(date +'%H:%M:%S'))"
    PYTHONPATH=. python src/models/cuda_exp_graphmixer_a10/launcher.py \
        --parquet-path  "$PARQUET_PATH" \
        --sampling-backend "$backend" \
        --epochs        "$EPOCHS" \
        --batch-size    "$BATCH_SIZE" \
        --num-neighbors "$NUM_NEIGHBORS" \
        --max-val-edges "$MAX_VAL_EDGES" \
        --output        "$OUTPUT" \
        2>&1
    echo ">>> Done: $backend  ($(date +'%H:%M:%S'))"
    echo ""
}

{
    for backend in $BACKENDS; do
        run_backend "$backend"
    done

    echo "========================================"
    echo "ALL BACKENDS COMPLETE"
    echo ""
    echo "--- Timing summary ---"
    printf "%-8s %10s %14s %12s %12s %10s\n" \
        "backend" "epoch(s)" "sampling(s)" "samp%" "fwd(s)" "val_mrr"
    printf "%-8s %10s %14s %12s %12s %10s\n" \
        "-------" "--------" "-----------" "-----" "------" "-------"

    for backend in $BACKENDS; do
        stem=$(basename "${PARQUET_PATH%.parquet}")
        result_dir="$OUTPUT/graphmixer_${backend}_${stem}"
        if [ -f "$result_dir/final_results.json" ]; then
            python3 -c "
import json
with open('$result_dir/final_results.json') as f:
    r = json.load(f)
t = r['timing']
print(f\"{r['sampling_backend']:<8} {t['avg_epoch_sec']:>10.0f} {t['avg_sampling_sec']:>14.1f} {t['sampling_fraction_pct']:>11.0f}% {t['avg_forward_sec']:>12.1f} {r['best_val_mrr']:>10.4f}\")
"
        else
            echo "$backend: results not found at $result_dir"
        fi
    done

    echo ""
    echo "Speedups:"
    python3 -c "
import json, os
stem = os.path.basename('${PARQUET_PATH}'.replace('.parquet',''))
backends = '${BACKENDS}'.split()
results = {}
for b in backends:
    path = f'${OUTPUT}/graphmixer_{b}_{stem}/final_results.json'
    if os.path.exists(path):
        with open(path) as f:
            results[b] = json.load(f)['timing']['avg_epoch_sec']
if 'python' in results and 'cuda' in results:
    print(f'  python → cuda:  {results[\"python\"] / results[\"cuda\"]:.1f}x')
if 'cpp' in results and 'cuda' in results:
    print(f'  cpp    → cuda:  {results[\"cpp\"] / results[\"cuda\"]:.1f}x')
if 'python' in results and 'cpp' in results:
    print(f'  python → cpp:   {results[\"python\"] / results[\"cpp\"]:.1f}x')
"
    echo "========================================"
} 2>&1 | tee "$LOG"

#!/bin/bash
###############################################################################
# run_cuda_comparison.sh — Run exp_005 (baseline) and exp_006 (CUDA) back-to-back.
#
# Prerequisite: run setup_v100.sh first (installs PyTorch, builds extensions).
#
# Usage:
#   source venv/bin/activate
#   bash scripts/run_cuda_comparison.sh [--epochs N] [--batch-size N] 2>&1 | tee /tmp/cuda_comparison.log
#
# Options:
#   --epochs N       Number of training epochs (default: 3)
#   --batch-size N   Training batch size (default: 4000)
#
# Examples:
#   bash scripts/run_cuda_comparison.sh 2>&1 | tee /tmp/cuda_comparison.log
#   bash scripts/run_cuda_comparison.sh --epochs 10 --batch-size 2000 2>&1 | tee /tmp/cuda_comparison.log
#
# What it does:
#   1. Downloads stream graph from Yandex.Disk (if needed)
#   2. Slices 1 week (2020-07-01 to 2020-07-07)
#   3. Runs GLFormer baseline (C++ sampling)
#   4. Runs GLFormer CUDA (CUDA sampling)
#   5. Prints comparison table
###############################################################################
# No set -e: we handle errors explicitly

PARQUET_FULL="/tmp/stream_graph_full.parquet"
PARQUET_1WEEK="/tmp/stream_graph_1week.parquet"
START_DATE="2020-07-01"
END_DATE="2020-07-07"
EPOCHS=3
BATCH_SIZE=4000
NUM_NEIGHBORS=20
SEED=42
OUTPUT_BASELINE="/tmp/exp_005_results"
OUTPUT_CUDA="/tmp/exp_006_results"

# Parse CLI flags
while [[ $# -gt 0 ]]; do
    case "$1" in
        --epochs)    EPOCHS="$2";     shift 2 ;;
        --batch-size) BATCH_SIZE="$2"; shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

log() { echo "[$(date +%H:%M:%S)] $*"; }
log "Config: epochs=$EPOCHS, batch_size=$BATCH_SIZE, K=$NUM_NEIGHBORS"

###############################################################################
# Step 1: Get stream graph data
###############################################################################
log "=== Step 1: Preparing data ==="

if [[ -f "$PARQUET_1WEEK" ]]; then
    log "1-week parquet already exists: $PARQUET_1WEEK"
else
    if [[ -f "$PARQUET_FULL" ]]; then
        log "Full parquet exists, slicing locally..."
        PYTHONPATH=. python scripts/slice_stream_graph.py \
            --input "$PARQUET_FULL" \
            --start "$START_DATE" --end "$END_DATE" \
            --output "$PARQUET_1WEEK"
    else
        if [[ -z "${YADISK_TOKEN:-}" ]]; then
            echo "ERROR: YADISK_TOKEN required to download stream graph"
            echo "Set it: export YADISK_TOKEN=\"...\""
            exit 1
        fi
        log "Downloading from Yandex.Disk and slicing..."
        PYTHONPATH=. python scripts/slice_stream_graph.py \
            --yadisk-path orbitaal_processed/stream_graph/2020-06-01__2020-08-31.parquet \
            --start "$START_DATE" --end "$END_DATE" \
            --output "$PARQUET_1WEEK"
    fi
fi

log "Data ready: $PARQUET_1WEEK"

###############################################################################
# Step 2: Run exp_005 — GLFormer baseline (C++ sampling)
###############################################################################
log ""
log "=== Step 2: exp_005 — GLFormer BASELINE (C++ sampling) ==="
log "Epochs: $EPOCHS, batch_size: $BATCH_SIZE, K: $NUM_NEIGHBORS"

GPU_LOG_BASELINE="/tmp/gpu_log_baseline.csv"
bash scripts/monitor_gpu.sh "$GPU_LOG_BASELINE" 2 &
GPU_MON_PID=$!
log "GPU monitor started (PID=$GPU_MON_PID) -> $GPU_LOG_BASELINE"

PYTHONPATH=. python src/models/GLFormer/glformer_launcher.py \
    --parquet-path "$PARQUET_1WEEK" \
    --epochs "$EPOCHS" \
    --batch-size "$BATCH_SIZE" \
    --num-neighbors "$NUM_NEIGHBORS" \
    --seed "$SEED" \
    --output "$OUTPUT_BASELINE"

kill $GPU_MON_PID 2>/dev/null || true
log "Baseline done. Results: $OUTPUT_BASELINE"
log "GPU log: $GPU_LOG_BASELINE ($(wc -l < "$GPU_LOG_BASELINE") samples)"

sleep 5

###############################################################################
# Step 3: Run exp_006 — GLFormer CUDA (CUDA sampling)
###############################################################################
log ""
log "=== Step 3: exp_006 — GLFormer CUDA (CUDA sampling) ==="
log "Epochs: $EPOCHS, batch_size: $BATCH_SIZE, K: $NUM_NEIGHBORS"

GPU_LOG_CUDA="/tmp/gpu_log_cuda.csv"
bash scripts/monitor_gpu.sh "$GPU_LOG_CUDA" 2 &
GPU_MON_PID=$!
log "GPU monitor started (PID=$GPU_MON_PID) -> $GPU_LOG_CUDA"

PYTHONPATH=. python src/models/GLFormer_cuda/glformer_launcher.py \
    --parquet-path "$PARQUET_1WEEK" \
    --epochs "$EPOCHS" \
    --batch-size "$BATCH_SIZE" \
    --num-neighbors "$NUM_NEIGHBORS" \
    --seed "$SEED" \
    --sampling-backend auto \
    --output "$OUTPUT_CUDA"

kill $GPU_MON_PID 2>/dev/null || true
log "CUDA done. Results: $OUTPUT_CUDA"
log "GPU log: $GPU_LOG_CUDA ($(wc -l < "$GPU_LOG_CUDA") samples)"

###############################################################################
# Step 4: Print comparison
###############################################################################
log ""
log "=== COMPARISON ==="
echo ""

python3 -c "
import json, glob, os

def load_results(base_dir):
    pattern = os.path.join(base_dir, '*', 'final_results.json')
    files = glob.glob(pattern)
    if not files:
        pattern = os.path.join(base_dir, 'final_results.json')
        files = glob.glob(pattern)
    if not files:
        return None
    with open(files[0]) as f:
        return json.load(f)

def load_curves(base_dir):
    pattern = os.path.join(base_dir, '*', 'training_curves.csv')
    files = glob.glob(pattern)
    if not files:
        pattern = os.path.join(base_dir, 'training_curves.csv')
        files = glob.glob(pattern)
    if not files:
        return []
    import csv
    with open(files[0]) as f:
        return list(csv.DictReader(f))

b = load_results('$OUTPUT_BASELINE')
c = load_results('$OUTPUT_CUDA')

if not b or not c:
    print('Could not load results.')
    exit(0)

b_curves = load_curves('$OUTPUT_BASELINE')
c_curves = load_curves('$OUTPUT_CUDA')

b_epochs = [float(r['epoch_time_sec']) for r in b_curves] if b_curves else []
c_epochs = [float(r['epoch_time_sec']) for r in c_curves] if c_curves else []

b_mean = sum(b_epochs) / len(b_epochs) if b_epochs else b['timing']['training_sec'] / b['total_epochs']
c_mean = sum(c_epochs) / len(c_epochs) if c_epochs else c['timing']['training_sec'] / c['total_epochs']

speedup = b_mean / c_mean if c_mean > 0 else 0

print(f'{'Metric':<25s} {'Baseline (C++)':>15s} {'CUDA':>15s} {'Speedup':>10s}')
print('-' * 67)
print(f'{'Mean epoch time (sec)':<25s} {b_mean:>15.1f} {c_mean:>15.1f} {speedup:>9.1f}x')
print(f'{'Total train time (sec)':<25s} {b[\"timing\"][\"training_sec\"]:>15.1f} {c[\"timing\"][\"training_sec\"]:>15.1f} {b[\"timing\"][\"training_sec\"]/max(c[\"timing\"][\"training_sec\"],0.1):>9.1f}x')
print(f'{'Best val MRR':<25s} {b[\"best_val_mrr\"]:>15.4f} {c[\"best_val_mrr\"]:>15.4f} {\"\":>10s}')
print(f'{'Test MRR':<25s} {b[\"test_metrics\"][\"mrr\"]:>15.4f} {c[\"test_metrics\"][\"mrr\"]:>15.4f} {\"\":>10s}')
print(f'{'Test Hits@10':<25s} {b[\"test_metrics\"][\"hits@10\"]:>15.4f} {c[\"test_metrics\"][\"hits@10\"]:>15.4f} {\"\":>10s}')
print(f'{'Sampling backend':<25s} {\"C++/Python\":>15s} {c.get(\"sampling_backend\", \"cuda\"):>15s} {\"\":>10s}')
print()

if abs(b['best_val_mrr'] - c['best_val_mrr']) < 0.01:
    print('Metrics match (delta < 0.01) — correctness verified.')
else:
    print(f'WARNING: MRR difference = {abs(b[\"best_val_mrr\"] - c[\"best_val_mrr\"]):.4f}')
"

###############################################################################
# Step 5: Generate plots
###############################################################################
log ""
log "=== Step 5: Generating plots ==="

pip install matplotlib -q 2>/dev/null || true

PYTHONPATH=. python scripts/plot_cuda_comparison.py \
    --baseline-dir "$OUTPUT_BASELINE" \
    --cuda-dir "$OUTPUT_CUDA" \
    --gpu-log-baseline "$GPU_LOG_BASELINE" \
    --gpu-log-cuda "$GPU_LOG_CUDA" \
    --output /tmp/comparison_plots \
    || log "Plot generation failed (non-critical)"

###############################################################################
# Step 6: Upload to Yandex.Disk
###############################################################################
log ""
log "=== Step 6: Upload results to Yandex.Disk ==="

if [[ -n "${YADISK_TOKEN:-}" ]]; then
    PYTHONPATH=. python -c "
from src.yadisk_utils import upload_directory
import os

token = os.environ['YADISK_TOKEN']
base = 'orbitaal_processed/experiments'

for local, remote in [
    ('$OUTPUT_BASELINE', f'{base}/exp_005_glformer_baseline'),
    ('$OUTPUT_CUDA', f'{base}/exp_006_glformer_cuda'),
    ('/tmp/comparison_plots', f'{base}/exp_005_vs_006_comparison'),
]:
    if os.path.isdir(local):
        try:
            n = upload_directory(local, remote, token)
            print(f'  Uploaded {n} files -> {remote}')
        except Exception as e:
            print(f'  Upload failed for {local}: {e}')

# Upload GPU logs
from src.yadisk_utils import upload_file
for f, name in [
    ('$GPU_LOG_BASELINE', 'gpu_log_baseline.csv'),
    ('$GPU_LOG_CUDA', 'gpu_log_cuda.csv'),
]:
    if os.path.exists(f):
        try:
            upload_file(f, f'{base}/exp_005_vs_006_comparison/{name}', token)
            print(f'  Uploaded {name}')
        except Exception as e:
            print(f'  Upload failed for {name}: {e}')
"
else
    log "YADISK_TOKEN not set — skipping upload"
fi

log ""
log "=== ALL DONE ==="
log "Baseline results: $OUTPUT_BASELINE"
log "CUDA results:     $OUTPUT_CUDA"
log "GPU logs:         $GPU_LOG_BASELINE, $GPU_LOG_CUDA"
log "Plots:            /tmp/comparison_plots/"
log "Full log:         /tmp/cuda_comparison.log"

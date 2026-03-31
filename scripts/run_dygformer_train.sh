#!/bin/bash
###############################################################################
# run_dygformer_train.sh — Обучение DyGFormer (дефолтные параметры из статьи)
#
# Параметры: paper defaults (NeurIPS 2023)
#   d_T=100, d_C=50, d=50, d_out=172, L=2, I=2
#   K=32 соседей, P=1 (без patching'а)
#
# Датасет: 10% stream graph (2020-06-01 — 2020-08-31)
#
# Usage:
#   export YADISK_TOKEN="..."
#   bash scripts/run_dygformer_train.sh 2>&1 | tee /tmp/run_dygformer_train.log
###############################################################################

set -e

log() { echo "[$(date +%H:%M:%S)] $*"; }
die() { echo "[$(date +%H:%M:%S)] ERROR: $*"; exit 1; }

REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_DIR"
source venv/bin/activate

[ -z "$YADISK_TOKEN" ] && die "YADISK_TOKEN не задан. Выполни: export YADISK_TOKEN=\"...\""

SMALL_PARQUET="/tmp/stream_graph_10pct.parquet"
OUTPUT="/tmp/dygformer_train"

if [ ! -f "$SMALL_PARQUET" ]; then
    die "Файл $SMALL_PARQUET не найден. Сначала скачай: PYTHONPATH=. python scripts/download_parquet.py"
fi

log "Запускаю обучение DyGFormer (параметры из статьи NeurIPS 2023)..."
log "Данные: $SMALL_PARQUET"
log "Результаты: $OUTPUT"

PYTHONPATH=. python src/models/DyGFormer/dygformer_launcher.py \
    --parquet-path "$SMALL_PARQUET" \
    --output "$OUTPUT" \
    --exp-name "dygformer_10pct_paper" \
    --num-neighbors 32 \
    --patch-size 1 \
    --time-dim 100 \
    --aligned-dim 50 \
    --num-transformer-layers 2 \
    --num-attention-heads 2 \
    --cooc-dim 50 \
    --output-dim 172 \
    --dropout 0.1 \
    --lr 0.0001 \
    --weight-decay 1e-5 \
    --batch-size 200 \
    --neg-per-positive 5 \
    --edge-feat-dim 2 \
    --node-feat-dim 0 \
    --epochs 100 \
    --patience 20 \
    --max-val-edges 2000

log "=============================================="
log "  ОБУЧЕНИЕ ЗАВЕРШЕНО"
log "  Результаты: $OUTPUT/dygformer_10pct_paper/"
log "  MRR: cat $OUTPUT/dygformer_10pct_paper/final_results.json | python3 -c 'import json,sys; r=json.load(sys.stdin); print(r[\"test_metrics\"][\"mrr\"])'"
log "=============================================="

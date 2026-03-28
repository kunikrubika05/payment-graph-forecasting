#!/bin/bash
###############################################################################
# run_glformer_train.sh — Обучение GLFormer с лучшими гиперпараметрами из HPO
#
# Гиперпараметры: trial 12, val MRR=0.6791 (HPO на 10% датасета)
# Датасет: 10% stream graph (2020-06-01 — 2020-08-31)
#
# Usage:
#   export YADISK_TOKEN="..."
#   bash scripts/run_glformer_train.sh 2>&1 | tee /tmp/run_glformer_train.log
###############################################################################

set -e

log() { echo "[$(date +%H:%M:%S)] $*"; }
die() { echo "[$(date +%H:%M:%S)] ERROR: $*"; exit 1; }

REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_DIR"
source venv/bin/activate

[ -z "$YADISK_TOKEN" ] && die "YADISK_TOKEN не задан. Выполни: export YADISK_TOKEN=\"...\""

SMALL_PARQUET="/tmp/stream_graph_10pct.parquet"
OUTPUT="/tmp/glformer_train"

if [ ! -f "$SMALL_PARQUET" ]; then
    die "Файл $SMALL_PARQUET не найден. Сначала запусти run_glformer_hpo.sh"
fi

log "Запускаю обучение GLFormer (лучшие параметры из HPO trial 12)..."
log "Данные: $SMALL_PARQUET"
log "Результаты: $OUTPUT"

PYTHONPATH=. python src/models/GLFormer/glformer_launcher.py \
    --parquet-path "$SMALL_PARQUET" \
    --output "$OUTPUT" \
    --hidden-dim 100 \
    --num-neighbors 30 \
    --num-glformer-layers 1 \
    --lr 0.00030744654981834505 \
    --weight-decay 1.0550793900692425e-06 \
    --dropout 0.2 \
    --channel-expansion 4.0 \
    --batch-size 4000 \
    --edge-feat-dim 2 \
    --epochs 100 \
    --patience 20

log "=============================================="
log "  ОБУЧЕНИЕ ЗАВЕРШЕНО"
log "  Результаты: $OUTPUT"
log "=============================================="

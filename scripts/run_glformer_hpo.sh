#!/bin/bash
###############################################################################
# run_glformer_hpo.sh — Подбор гиперпараметров GLFormer на 10% датасета
#
# Требования:
#   - bash scripts/setup_v100.sh уже выполнен
#   - YADISK_TOKEN задан
#
# Usage:
#   export YADISK_TOKEN="..."
#   bash scripts/run_glformer_hpo.sh 2>&1 | tee /tmp/run_glformer_hpo.log
###############################################################################

set -e

log()  { echo "[$(date +%H:%M:%S)] $*"; }
die()  { echo "[$(date +%H:%M:%S)] ERROR: $*"; exit 1; }

REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_DIR"
source venv/bin/activate

[ -z "$YADISK_TOKEN" ] && die "YADISK_TOKEN не задан. Выполни: export YADISK_TOKEN=\"...\""

REMOTE_PATH="orbitaal_processed/stream_graph/2020-06-01__2020-08-31.parquet"
FULL_PARQUET="/tmp/stream_graph_full.parquet"
SMALL_PARQUET="/tmp/stream_graph_10pct.parquet"
HPO_OUTPUT="/tmp/glformer_hpo"

###############################################################################
log "=== Шаг 1/3: Скачивание stream graph с Яндекс.Диска ==="
###############################################################################

if [ -f "$FULL_PARQUET" ]; then
    log "Файл уже есть: $FULL_PARQUET"
else
    log "Скачиваю $REMOTE_PATH ..."
    PYTHONPATH=. python - <<EOF
import os
from src.yadisk_utils import download_file
download_file(
    "$REMOTE_PATH",
    "$FULL_PARQUET",
    token=os.environ["YADISK_TOKEN"],
)
print("Скачано: $FULL_PARQUET")
EOF
fi

###############################################################################
log "=== Шаг 2/3: Нарезка 10% датасета ==="
###############################################################################

if [ -f "$SMALL_PARQUET" ]; then
    log "Файл уже есть: $SMALL_PARQUET"
else
    PYTHONPATH=. python - <<EOF
import pandas as pd
df = pd.read_parquet("$FULL_PARQUET")
cut = len(df) // 10
df.iloc[:cut].to_parquet("$SMALL_PARQUET", index=False)
print(f"Нарезано: {len(df):,} -> {cut:,} рёбер -> $SMALL_PARQUET")
EOF
fi

###############################################################################
log "=== Шаг 3/3: Подбор гиперпараметров GLFormer ==="
###############################################################################

log "Запускаю HPO: 30 триалов, 15 эпох каждый, edge-feat-dim=2"
log "Результаты: $HPO_OUTPUT"

PYTHONPATH=. python src/models/GLFormer/glformer_hpo.py \
    --parquet-path "$SMALL_PARQUET" \
    --n-trials 30 \
    --hpo-epochs 15 \
    --edge-feat-dim 2 \
    --output "$HPO_OUTPUT"

log "=============================================="
log "  HPO ЗАВЕРШЁН"
log "  Результаты: $HPO_OUTPUT/hpo_results.json"
log "  Команда обучения: $HPO_OUTPUT/best_train_command.sh"
log "=============================================="

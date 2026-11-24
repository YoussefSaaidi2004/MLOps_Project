#!/usr/bin/env bash
# run_pipeline.sh - lance un flow du pipeline (utilisable à la main ou par cron).
# Usage : ./run_pipeline.sh [flow]      flow : all (défaut), train, evaluate, code, install
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${VENV_DIR:-$PROJECT_DIR/venv}"
FLOW="${1:-all}"
LOG_DIR="$PROJECT_DIR/logs"
LOCK_FILE="/tmp/ml_pipeline_${FLOW}.lock"

mkdir -p "$LOG_DIR"
cd "$PROJECT_DIR"

# Évite deux exécutions simultanées du même flow
exec 9>"$LOCK_FILE"
if ! flock -n 9; then
    echo "$(date '+%F %T') flow '$FLOW' déjà en cours, abandon." >> "$LOG_DIR/cron.log"
    exit 0
fi

# Cron n'a pas d'environnement : on utilise l'interpréteur du venv par chemin absolu
LOG_FILE="$LOG_DIR/pipeline_${FLOW}_$(date +%F_%H%M%S).log"
echo "$(date '+%F %T') début du flow '$FLOW'" >> "$LOG_DIR/cron.log"
if "$VENV_DIR/bin/python" pipeline_prefect.py --flow "$FLOW" > "$LOG_FILE" 2>&1; then
    echo "$(date '+%F %T') flow '$FLOW' terminé avec succès ($LOG_FILE)" >> "$LOG_DIR/cron.log"
else
    code=$?
    echo "$(date '+%F %T') flow '$FLOW' ÉCHEC code $code ($LOG_FILE)" >> "$LOG_DIR/cron.log"
    exit "$code"
fi

#!/usr/bin/env bash
# installer_cron.sh - ajoute (ou retire) automatiquement le pipeline dans le crontab.
# Usage :
#   ./installer_cron.sh                          # flow all tous les jours à 02:00
#   ./installer_cron.sh "30 1 * * *" train       # planification et flow personnalisés
#   ./installer_cron.sh --remove [flow]          # retire l'entrée (défaut : all)
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUNNER="$PROJECT_DIR/run_pipeline.sh"

if [ "${1:-}" = "--remove" ]; then
    FLOW="${2:-all}"
    MARK="# ml-pipeline-$FLOW"
    ( crontab -l 2>/dev/null | grep -vF "$MARK" || true ) | crontab -
    echo "Entrée cron '$FLOW' retirée."
    exit 0
fi

SCHEDULE="${1:-0 2 * * *}"
FLOW="${2:-all}"
MARK="# ml-pipeline-$FLOW"

if [ "$(echo "$SCHEDULE" | wc -w)" -ne 5 ]; then
    echo "Planification invalide : '$SCHEDULE' (attendu : 5 champs cron)" >&2
    exit 1
fi

chmod +x "$RUNNER"
LINE="$SCHEDULE $RUNNER $FLOW $MARK"

# Retire l'ancienne entrée du même flow puis ajoute la nouvelle (idempotent)
( crontab -l 2>/dev/null | grep -vF "$MARK" || true; echo "$LINE" ) | crontab -

echo "Entrée cron installée :"
crontab -l | grep -F "$MARK"

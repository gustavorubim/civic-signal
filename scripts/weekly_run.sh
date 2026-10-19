#!/usr/bin/env bash
# Weekly (or ad hoc) live 2026 forecast: fetch fresh VoteHub/Wikipedia data, run both
# chambers, package, and compare against the previous packaged run.
#
# usage: scripts/weekly_run.sh <kind> <date> <previous-run-dir> [--allow-cached]
#   e.g. scripts/weekly_run.sh weekly 2026-10-26 runs/weekly/2026-10-19
#
# Since the audit fixes:
# - no --as-of: the forecast date defaults to today (capped at election eve);
# - a fetch failure stops the run unless --allow-cached is passed;
# - package_run writes runs/<kind>/<date>/summary.json, with a week-over-week block via --prior.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=src
KIND=$1; DATE=$2; PRIOR=$3; EXTRA=${4:-}
TAG="${KIND}-${DATE}"
OUT="runs/${KIND}/${DATE}"
mkdir -p "${OUT}/logs"
cp "$0" "${OUT}/logs/commands.sh"
for CH in senate house; do
  SC=$([ "$CH" = senate ] && echo senate_2026_state || echo house_2026_district)
  uv run civic-signal forecast run --run-id "${CH}-2026-midterm-live" \
    --scenario "$SC" --sources-config "sources_${CH}_live.yaml" \
    --data-dir "data/${CH}-live-${TAG}" --artifacts-dir "artifacts/${CH}-2026-live-${TAG}" \
    --inference-engine bayes --bayesian-backend analytic ${EXTRA} --quiet \
    2>&1 | tee "${OUT}/logs/${CH}_run.log"
  uv run python scripts/package_run.py --chamber "$CH" --run-date "$DATE" \
    --artifact-root "artifacts/${CH}-2026-live-${TAG}/runs/${CH}-2026-midterm-live" \
    --data-dir "data/${CH}-live-${TAG}" --out "$OUT" --run-kind "$KIND" \
    --command-file "${OUT}/logs/commands.sh" \
    --sources-config "sources_${CH}_live.yaml" --scenario "$SC" --prior "$PRIOR"
done
uv run python scripts/compare_runs.py "$PRIOR" "$OUT" \
  --out "${OUT}/comparison_vs_$(basename "$PRIOR")" --plot "${OUT}/movers_dumbbell.png"

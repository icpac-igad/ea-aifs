#!/bin/bash
# Score a completed cycle's O96 store against AI-WQ observations. Nothing is uploaded.
#
#   ./evaluate_cycle.sh 20260514 [WORKDIR]
#
# 1. O96 -> 1.5 deg for msl/tp/2t, all members   (shared CLI, --source-grid o96)
# 2. weekly aggregate -> quintile probabilities  (shared CLI, untouched)
# 3. RPSS vs ERA5T observations                  (AI_WQ_package's own scoring code)
#
# Only scoreable once init+25 days have passed; score_rpss.py refuses earlier.
set -euo pipefail

DATE="${1:?usage: evaluate_cycle.sh YYYYMMDD [workdir]}"
WORK="${2:-/tank/projects/aifs-run/eval_${DATE}}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PY="${PY:-/tank/projects/micromamba/envs/aifs-gpu/bin/python}"
STORE="${STORE:-/tank/projects/aifs-run/${DATE}_0000/icechunk_o96}"

[ -d "$STORE" ] || { echo "no O96 store at $STORE"; exit 1; }
mkdir -p "$WORK"/{nc_1p5deg,aiwq,cache}

echo "=== 1/3  O96 -> 1.5 deg ==="
"$PY" -u "$REPO/../shared/aifs_n320_grib_1p5defg_nc_cli.py" \
    --date "${DATE}_0000" --members 1-50 --v2 --no-upload \
    --source icechunk --source-grid o96 \
    --icechunk-store "$STORE" --icechunk-tag "cycle-${DATE}_0000" \
    --output-dir "$WORK/nc_1p5deg"

echo "=== 2/3  quintile probabilities ==="
"$PY" -u "$REPO/../shared/ensemble_quintile_analysis_cli.py" \
    --date "$DATE" --v2 --local-nc-dir "$WORK/nc_1p5deg" --work-dir "$WORK/aiwq"

echo "=== 3/3  RPSS ==="
"$PY" "$REPO/O96-icechunk-store/forecast-evaluation/score_rpss.py" \
    --fc "$WORK/aiwq/ensemble_quintile_probabilities_${DATE}_v2.nc" \
    --fc-date "$DATE" --workdir "$WORK/cache" --nc-dir "$WORK/nc_1p5deg" \
    --diagnostics --json "$WORK/rpss_${DATE}.json"

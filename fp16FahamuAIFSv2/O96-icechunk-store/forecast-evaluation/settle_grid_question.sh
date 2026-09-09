#!/bin/bash
# Score 20260813 through BOTH regrid routes from ONE rollout -> grid effect, no draw
# confound. See RPSS_LEADERBOARD_DISCREPANCY.md. Runnable from 2026-08-31.
set -u
D=20260813
B=/tank/projects/aifs-run/${D}_0000
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="${PY:-/tank/projects/micromamba/envs/aifs-gpu/bin/python}"
W=/tank/projects/aifs-run/eval_${D}_grid; mkdir -p $W/cache

for r in "n320:$B/aiwq" "o96:$B/aiwq_via_o96"; do
  name="${r%%:*}"; dir="${r##*:}"
  f="$dir/ensemble_quintile_probabilities_${D}_v2.nc"
  [ -f "$f" ] || { echo "missing $f"; exit 1; }
  echo "===== $name route ====="
  "$PY" "$HERE/score_rpss.py" --fc "$f" --fc-date $D --workdir $W/cache \
        --diagnostics --json "$W/rpss_${D}_${name}.json" || exit 1
done
echo "===== compare ====="
"$PY" - <<'PY'
import json
w="/tank/projects/aifs-run/eval_20260813_grid"
a=json.load(open(f"{w}/rpss_20260813_n320.json"))["results"]
c=json.load(open(f"{w}/rpss_20260813_o96.json"))["results"]
print(f"{'week':7s} {'var':5s} {'N320':>9s} {'O96':>9s} {'delta':>9s}")
for x,y in zip(a,c):
    print(f"{x['week']:7s} {x['variable']:5s} {x['rpss']:9.4f} {y['rpss']:9.4f} "
          f"{y['rpss']-x['rpss']:+9.4f}")
d=[y['rpss']-x['rpss'] for x,y in zip(a,c)]
print(f"\nmean delta (O96 - N320): {sum(d)/len(d):+.4f}")
print("near 0 -> grid is not the cause; materially negative -> O96 costs real skill")
PY

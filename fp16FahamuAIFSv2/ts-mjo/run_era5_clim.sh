#!/usr/bin/env bash
# Build the VPM calendar-day climatology from ARCO-ERA5, one year at a time.
#
# Why per-year and not one call: the link to GCS measured 1.4 MB/s and does NOT
# improve with concurrency (8/16/32 threads all gave 1.4 MB/s), so the full base
# period is a ~11 h stream. This box has lost three overnight jobs to
# `apt-daily-upgrade` swapping glibc and python3.12 under a running process.
# One year per invocation means a kill costs ~23 min, not the whole run, and
# re-running skips the years already on disk.
#
# Nothing large is stored: ~57 GB passes through memory, and what lands on disk
# is one ~600 KB .npz per year plus a ~1.3 MB climatology. Disk is at 97 GB free
# and this does not touch it.
set -u
PY=/tank/projects/micromamba/envs/aifs-gpu/bin/python
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT=${OUT:-/tank/projects/era5_vpm_clim}
Y0=${Y0:-1991}; Y1=${Y1:-2020}          # WMO 1991-2020 standard normal
STRIDE=${STRIDE:-2}                      # every 2nd block; see --stride in the script
mkdir -p "$OUT"

for y in $(seq "$Y0" "$Y1"); do
  f="$OUT/era5_vpm_${y}.npz"
  # The temp name must itself end in .npz: np.savez APPENDS .npz when the path
  # does not have it, so `--out "$f.part"` silently wrote "$f.part.npz" and every
  # `mv "$f.part"` failed. The data was fine but the resume test never matched and
  # the combine glob found nothing -- i.e. the whole point of going year-by-year
  # was quietly defeated. Write to a dotfile so a partial cannot match the glob.
  t="$OUT/.part_${y}.npz"
  if [ -s "$f" ]; then echo "skip $y (have $f)"; continue; fi
  echo "=== $y ==="
  if "$PY" -u "$HERE/era5_vpm_clim.py" --mode series \
        --start "${y}-01-01" --end "$((y+1))-01-01" \
        --block 80 --stride "$STRIDE" --phase $(( y % STRIDE )) --out "$t" \
     && [ -s "$t" ]; then
    mv "$t" "$f"
    echo "  -> $f"
  else
    echo "FAILED $y -- rerun this script to retry only this year"; rm -f "$t"
  fi
done

n=$(ls -1 "$OUT"/era5_vpm_????.npz 2>/dev/null | wc -l)
echo "=== combining $n years ==="
"$PY" -u "$HERE/era5_vpm_clim.py" --mode combine \
    --inputs "$OUT"/era5_vpm_????.npz --harmonics 3 \
    --out "$OUT/vpm_clim_${Y0}_${Y1}.npz"

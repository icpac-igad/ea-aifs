# Forecast evaluation — cycle 20260212 (O96, pre-50r1 donor-sourced)

Local AI-WQ RPSS for the **first pre-Cy50r1 AIFS-ENS v2 cycle**. Scored on this box with
`AI_WQ_package.forecast_evaluation` — the competition's own code. **Nothing was uploaded**;
observations, climatology and the land-sea mask are read-only FTP retrievals.

| | |
|---|---|
| init | **2026-02-12 00z** |
| verification weeks | 2026-03-02 (week1), 2026-03-09 (week2) — init+18 / init+25 |
| source | `/tank/projects/aifs-run/20260212_0000/icechunk_o96`, tag `cycle-20260212_0000` |
| store | 166 GB, 50/50 members, 132/132 steps (full 0–792 h corpus) |
| inputs | 13 of 112 fields donor-sourced (j1r2 waves + ERA5 10 hPa) — see `../../run-pre50r1-dates/` |
| evaluated | 2026-08-20 |

## Result

| week | valid | tas | mslp | pr |
|---|---|---|---|---|
| week1 | 2026-03-02 | −0.086 | **+0.183** | +0.013 |
| week2 | 2026-03-09 | −0.063 | **+0.080** | +0.050 |
| **mean** | | **−0.074** | **+0.132** | **+0.032** |

**Overall mean RPSS +0.030.**

`RPSS = 1 − RPS_fc/RPS_clim` against a uniform-0.2 climatological forecast. Lat-weighted
global mean; `tas`/`pr` land-only, `mslp` global.

### Diagnostics

| var | week | sharpness | hit-rate | reliability | bias | rmse | pattern-r |
|---|---|---|---|---|---|---|---|
| tas | 1 | 0.433 | 0.338 | −0.095 | +0.037 K | 2.64 K | 0.981 |
| tas | 2 | 0.412 | 0.327 | −0.086 | +0.039 K | 2.28 K | 0.990 |
| mslp | 1 | 0.357 | 0.356 | **−0.001** | +12.7 Pa | 487 Pa | 0.881 |
| mslp | 2 | 0.348 | 0.301 | −0.047 | +15.5 Pa | 564 Pa | 0.875 |
| pr | 1 | 0.362 | 0.253 | −0.108 | +0.27 mm | 21.5 mm | 0.654 |
| pr | 2 | 0.354 | 0.270 | −0.084 | −0.01 mm | 23.0 mm | 0.616 |

Biases are negligible in every variable, so the fields are sound — the scores reflect
probabilistic calibration, not a broken pipeline. `mslp` week1 is essentially perfectly
calibrated (reliability −0.001) and scores accordingly; `tas` and `pr` remain
over-confident, which is what drags `tas` below climatology.

## Comparison with 20260514 (operational inputs)

The obvious question for a donor-sourced cycle is whether it scores like a normal one.

| | tas | mslp | pr | overall |
|---|---|---|---|---|
| **20260212** (donor-sourced, pre-50r1) | −0.074 | +0.132 | +0.032 | **+0.030** |
| **20260514** (operational inputs) | −0.167 | +0.233 | +0.005 | **+0.024** |

The pre-50r1 cycle does **not** look degraded — its overall RPSS is marginally higher, its
`tas` deficit is half as large, and its `pr` is better; only `mslp` is lower. Its ensemble
is also less over-confident (`tas` reliability −0.09 vs −0.20).

**Do not read this as evidence the donor route is as good as operational inputs.** These
are two different dates verifying against different weather, and one cycle each. Seasonal
and flow-dependent variation at days 18–33 is far larger than any difference visible here.
The honest claim is narrower: **the donor-sourced cycle produces a forecast in the normal
skill range rather than a broken or obviously degraded one** — consistent with the
`pre50r1b` sensitivity result (20.4 % of the ensemble scale, 5.6× floor) in
`../../run-pre50r1-dates/`.

## Commands

One command runs all three steps:

```bash
cd /home/ezra/ea-aifs/fp16FahamuAIFSv2/O96-icechunk-store/forecast-evaluation
./evaluate_cycle.sh 20260212
```

Expanded, as actually run:

```bash
PY=/tank/projects/micromamba/envs/aifs-gpu/bin/python
DATE=20260212
STORE=/tank/projects/aifs-run/${DATE}_0000/icechunk_o96
WORK=/tank/projects/aifs-run/eval_${DATE}
mkdir -p $WORK/{nc_1p5deg,aiwq,cache}

# 1. O96 -> 1.5 deg, 50 members            (5.1 min, 50/50, 0 failures)
#    --source-grid o96 is the essential flag: the CLI defaults to N320 and an O96
#    store fails with "matmul: dimension mismatch (n,k=542080),(k=40320,1?)".
$PY -u ../../../shared/aifs_n320_grib_1p5defg_nc_cli.py \
    --date ${DATE}_0000 --members 1-50 --v2 --no-upload \
    --source icechunk --source-grid o96 \
    --icechunk-store $STORE --icechunk-tag cycle-${DATE}_0000 \
    --output-dir $WORK/nc_1p5deg

# 2. weekly aggregate -> quintile probabilities   (untouched production CLI)
$PY -u ../../../shared/ensemble_quintile_analysis_cli.py \
    --date $DATE --v2 --local-nc-dir $WORK/nc_1p5deg --work-dir $WORK/aiwq

# 3. RPSS vs ERA5T observations                   (no upload)
$PY score_rpss.py \
    --fc $WORK/aiwq/ensemble_quintile_probabilities_${DATE}_v2.nc \
    --fc-date $DATE --workdir $WORK/cache --nc-dir $WORK/nc_1p5deg \
    --diagnostics --json $WORK/rpss_${DATE}.json
```

Outputs: `$WORK/nc_1p5deg/` (50 × 199 MB), `$WORK/aiwq/…_20260212_v2.nc`,
`$WORK/rpss_20260212.json`. Total 4.0 GB, ~12 min end to end.

## Caveats

1. **This scores the O96 route** (`N320 → O96 → 1.5°`), not a native-N320 submission. The
   A/B in `../evaluation/quintile_yardstick.py` puts that inside the ensemble's own
   bootstrap noise, but it is not bit-identical.
2. **Two of the 13 donor fields are deterministic**, so all 50 members share one ERA5
   10 hPa level. That removes stratospheric initial spread — the spread most likely to
   matter at S2S range, and precisely what an under-dispersion diagnosis is sensitive to.
   Some of the calibration numbers above may be affected by that and this run cannot
   separate the two.
3. **20260212 was never submitted** (it predates AIFS v2's operational life entirely), so
   there is no leaderboard figure to check against.
4. **One cycle, one date.** Nothing here supports a claim about mean skill.

## Fix this run required

The first attempt failed all 50 members with
`matmul: dimension mismatch (n,k=542080),(k=40320,1?)` — the N320 matrix applied to O96
data. `--source-grid` was reaching the parent process but not the per-member workers: the
CLI runs each member as an isolated `--single-member` subprocess and rebuilds that command
line by hand, so any flag not explicitly forwarded reverts to its default in the child.
Fixed in `_run_member_subprocess`; the flag is now forwarded alongside the other
`--icechunk-*` options.

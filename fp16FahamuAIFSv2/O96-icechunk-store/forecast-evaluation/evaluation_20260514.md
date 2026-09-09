# Forecast evaluation — cycle 20260514 (O96, operational inputs)

Local AI-WQ RPSS for the first fully-operational post-Cy50r1 cycle we hold on O96. Scored
with `AI_WQ_package.forecast_evaluation` — the competition's own code. **Nothing was
uploaded**; observations, climatology and the land-sea mask are read-only FTP retrievals.

| | |
|---|---|
| init | **2026-05-14 00z** (the day after the Cy50r1 cutover) |
| verification weeks | 2026-06-01 (week1), 2026-06-08 (week2) — init+18 / init+25 |
| source | `/tank/projects/aifs-run/20260514_0000/icechunk_o96`, tag `cycle-20260514_0000` |
| store | 159 GB, 50/50 members, 132/132 steps (full 0–792 h corpus) |
| inputs | **all 112 fields operational** — no donors |
| evaluated | 2026-08-20 |

## Result

| week | valid | tas | mslp | pr |
|---|---|---|---|---|
| week1 | 2026-06-01 | −0.124 | **+0.228** | +0.007 |
| week2 | 2026-06-08 | −0.211 | **+0.239** | +0.002 |
| **mean** | | **−0.167** | **+0.233** | **+0.005** |

**Overall mean RPSS +0.024.**

`RPSS = 1 − RPS_fc/RPS_clim` against a uniform-0.2 climatological forecast. Lat-weighted
global mean; `tas`/`pr` land-only, `mslp` global.

### Diagnostics

| var | week | sharpness | hit-rate | reliability | bias | rmse | pattern-r |
|---|---|---|---|---|---|---|---|
| tas | 1 | 0.525 | 0.346 | **−0.179** | −0.046 K | 1.84 K | 0.9945 |
| tas | 2 | 0.562 | 0.344 | **−0.219** | −0.059 K | 1.75 K | 0.9950 |
| mslp | 1 | 0.411 | 0.374 | −0.037 | +7.5 Pa | 551 Pa | 0.825 |
| mslp | 2 | 0.415 | 0.365 | −0.050 | +31.0 Pa | 431 Pa | 0.894 |
| pr | 1 | 0.385 | 0.281 | −0.104 | +0.39 mm | 21.6 mm | 0.700 |
| pr | 2 | 0.388 | 0.251 | −0.137 | −0.55 mm | 23.1 mm | 0.652 |

**The field is sound** — `tas` bias is −0.05 K with pattern-r 0.995. There is no units
error and no systematic offset, so the negative `tas` score is a calibration result, not a
pipeline fault. (Pattern-r on absolute values is high by construction: poles cold, tropics
warm. It rules out gross errors; it is not evidence of skill.)

### What the probability distribution actually looks like

Mean probability assigned to each quintile, both weeks pooled — climatology is 0.200 in
every cell:

| var | Q1 | Q2 | Q3 | Q4 | Q5 |
|---|---|---|---|---|---|
| **tas** | 0.231 | **0.130** | **0.132** | 0.151 | **0.356** |
| mslp | 0.188 | 0.175 | 0.180 | 0.187 | 0.270 |
| pr | 0.231 | 0.174 | 0.176 | 0.188 | 0.231 |

`tas` puts **0.587 of its mass in the two outer quintiles** against a climatological 0.400,
and 0.356 on the warmest quintile alone — nearly twice climatology — while hollowing out
the middle. It is right 34–35 % of the time. RPS penalises exactly that: confident and
wrong costs more than the flat 0.2 baseline, which is how a forecast with a −0.05 K bias
ends up below climatological skill.

A hypothesis this run cannot settle: the AI-WQ climatology is a **20-year** distribution,
so a warming trend alone shifts a well-calibrated modern forecast toward Q5. Some of the
Q5 mass may be trend rather than over-confidence. Distinguishing them needs the observed
quintile frequencies over several cycles, not one date.

## Comparison with 20260212 (donor-sourced, pre-50r1)

| | tas | mslp | pr | overall |
|---|---|---|---|---|
| [**20260212**](evaluation_20260212.md) donor-sourced | −0.074 | +0.132 | +0.032 | **+0.030** |
| **20260514** operational | −0.167 | +0.233 | +0.005 | **+0.024** |

The operational cycle scores *lower* overall, with a `tas` deficit twice as large. Its
distribution is also far more extreme (`tas` Q5 0.356 vs 0.267; outer-quintile mass 0.587
vs 0.437).

**This does not mean donor inputs are better.** Two dates, different seasons, different
weather, one cycle each; flow-dependent variability at days 18–33 dwarfs the difference.
What it does support is the narrow claim in the 20260212 report — the donor route lands in
the normal skill range rather than producing something broken.

Note also the direction of the confound: the 20260212 ensemble shares one deterministic
ERA5 10 hPa level across all 50 members, which *removes* initial spread. That should make
it **more** over-confident, not less, yet its reliability is better (−0.09 vs −0.20). So
the reduced spread is not what is driving the difference here.

## Commands

```bash
cd /home/ezra/ea-aifs/fp16FahamuAIFSv2/O96-icechunk-store/forecast-evaluation
./evaluate_cycle.sh 20260514
```

Expanded, as actually run:

```bash
PY=/tank/projects/micromamba/envs/aifs-gpu/bin/python
DATE=20260514
STORE=/tank/projects/aifs-run/${DATE}_0000/icechunk_o96
WORK=/tank/projects/aifs-run/eval_${DATE}
mkdir -p $WORK/{nc_1p5deg,aiwq,cache}

# 1. O96 -> 1.5 deg, 50 members            (4.9 min, 50/50, 0 failures)
#    --source-grid o96 is essential: the CLI defaults to N320 and an O96 store fails
#    with "matmul: dimension mismatch (n,k=542080),(k=40320,1?)".
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

Outputs: `$WORK/nc_1p5deg/` (50 × 199 MB), `$WORK/aiwq/…_20260514_v2.nc`,
`$WORK/rpss_20260514.json`. Total 4.0 GB, ~12 min end to end.

**Reproducibility check:** this cycle was scored twice — once ad hoc during development
and once through `evaluate_cycle.sh` as committed. Both runs agree to four decimal places
on every RPSS value and every diagnostic.

## Caveats

1. **This scores the O96 route** (`N320 → O96 → 1.5°`), not a native-N320 submission. The
   A/B in `../evaluation/quintile_yardstick.py` puts that inside the ensemble's own
   bootstrap noise, but it is not bit-identical. No same-date O96-vs-N320 control exists
   yet: 20260813 has both stores but verifies in the future.
2. **20260514 was never submitted**, so there is no leaderboard figure to check against.
3. **One cycle, one date.** Nothing here supports a claim about mean skill.
4. The trend-vs-over-confidence question above is open and needs several cycles.

# Local AI-WQ forecast evaluation

Score a completed cycle's O96 store against AI-WQ observations, on this box, **without
uploading anything**. Retrieval is read-only; the score comes from
`AI_WQ_package.forecast_evaluation` — the competition's own code, not a reimplementation.

```bash
./evaluate_cycle.sh 20260514
```

Three steps: O96 → 1.5°, weekly aggregate → quintile probabilities, RPSS vs ERA5T
observations. ~15 min for 50 members.

## What is scored

```
RPSS = 1 - RPS_forecast / RPS_climatology
```

against a climatological forecast (uniform 1/5 per quintile). **0 = no better than
climatology, >0 = skill, <0 = worse.** Lat-weighted global mean; `tas` and `pr` are
land-only (the competition masks them), `mslp` is global.

Verification weeks are **init+18 and init+25 days**, so a cycle becomes scoreable only
after init+25 has passed. `score_rpss.py` refuses earlier rather than producing a
misleading number.

## Result: 20260514, O96 route

| week | valid | tas | mslp | pr |
|---|---|---|---|---|
| week1 | 2026-06-01 | −0.124 | **+0.228** | +0.007 |
| week2 | 2026-06-08 | −0.211 | **+0.239** | +0.002 |
| **mean** | | **−0.167** | **+0.233** | **+0.005** |

**Overall mean RPSS +0.024.** `mslp` has real skill at days 18–33; `pr` is at
climatology; `tas` is *worse* than climatology.

### Why tas is negative — and why it is not a pipeline bug

The field itself is sound, which is the first thing to establish before reporting a
negative score as a finding:

| var | week | bias | rmse | pattern r |
|---|---|---|---|---|
| tas | 1 / 2 | −0.046 / −0.059 K | 1.84 / 1.75 K | 0.9945 / 0.9950 |
| mslp | 1 / 2 | +7.5 / +31.0 Pa | 551 / 431 Pa | 0.825 / 0.894 |
| pr | 1 / 2 | +0.39 / −0.55 mm | 21.6 / 23.1 mm | 0.700 / 0.652 |

No units error, no offset. (Pattern-r on absolute values is high by construction — poles
cold, tropics warm. It rules out gross errors; it is not evidence of skill.)

The cause is **over-confidence**:

| var | sharpness | hit-rate | reliability |
|---|---|---|---|
| **tas** | **0.54** | 0.35 | **−0.20** |
| mslp | 0.41 | 0.37 | −0.04 |
| pr | 0.39 | 0.26 | −0.12 |

*sharpness* = mean probability given to the modal quintile; *hit-rate* = how often the
observation landed there. Both are 0.20 for a climatological forecast. `tas` commits ~54 %
to its favoured quintile and is right 35 % of the time — RPS punishes that harder than
saying 0.20 everywhere, hence negative skill. `mslp` is nearly calibrated (−0.04) and
scores accordingly.

This is the same story as the ESS ≈ 1.8–1.9 / redundancy ≈ 0.96 result in
`../../epistemic-reasoning-risk/`: the ensemble spread is low-dimensional, and
calibration is where that surfaces. **The lever on RPSS here is ensemble dispersion, not
forecast accuracy.**

## Three caveats on any number this produces

1. **It scores the O96 route** (`N320 → O96 → 1.5°`), not a native-N320 submission. The
   earlier A/B put that difference inside the ensemble's own bootstrap noise
   (`../evaluation/quintile_yardstick.py`), but it is not bit-identical.
2. **No same-date O96-vs-N320 control exists yet.** 20260813 has both stores but its
   verification weeks are still in the future; 20260604 has past weeks but is incomplete.
   Until one of those closes, the O96 route's effect on RPSS is bounded but unmeasured.
3. **20260514 was never submitted**, so there is no leaderboard figure to check against.
   Useful for comparing routes and cycles; not an official result.

## Files

| file | what it does |
|---|---|
| `evaluate_cycle.sh` | the three steps end to end for one date |
| `score_rpss.py` | RPSS + calibration and field diagnostics; `--json` for machine-readable output |
| `evaluation_<DATE>.md` | one report per scored cycle: result, diagnostics, exact commands |
| `compare_regrid_routes.py` | N320 vs O96 calibration on the same rollout |
| [`regrid_impact_n320_vs_o96.md`](regrid_impact_n320_vs_o96.md) | **does the O96 hop cause the poor `tas` score? No — ~3 %** |

### ⚠ These are O96-route reruns, not the submitted forecasts

**Full analysis: [RPSS_LEADERBOARD_DISCREPANCY.md](RPSS_LEADERBOARD_DISCREPANCY.md)** —
and `./settle_grid_question.sh` runs the experiment that resolves it, from 2026-08-31.


For **20260709** the submitted N320 forecast scored **+0.071 / +0.055 / +0.106** (tas /
mslp / pr) at week1 on the official leaderboard, while the O96 rerun below scores
**−0.035 / +0.023 / +0.060**. Worse on all three.

`score_rpss.py` reproduces the official figure to three decimals when run on the submitted
file, so the scorer is validated — the gap is in the forecast, not the scoring. It
confounds a different regrid route with a different stochastic draw and **is not yet
resolved**; see [evaluation_20260709.md](evaluation_20260709.md). The clean test is
20260813, which holds both products from one rollout and verifies 2026-08-31.

Until then the O96 route is **unvalidated for skill**. The leaderboard reports **week1**,
so compare week1 rows, never the two-week mean.

### Scored cycles

| cycle | inputs | tas | mslp | pr | overall |
|---|---|---|---|---|---|
| [20260212](evaluation_20260212.md) | donor-sourced (pre-50r1) | −0.074 | +0.132 | +0.032 | **+0.030** |
| [20260514](evaluation_20260514.md) | operational | −0.167 | +0.233 | +0.005 | **+0.024** |
| [20260709](evaluation_20260709.md) | operational | −0.115 | +0.060 | +0.032 | **−0.008** |
| 20260820 | operational | — | — | — | *verifies 2026-09-14* |

Two cycles verifying against different weather — not a skill comparison. See the
20260212 report for why the donor-sourced cycle scoring no worse is a weak claim.

`score_rpss.py` runs standalone against any AI-WQ-format quintile file, whatever produced
it — pass `--fc`, `--fc-date`, `--workdir`.

## The one upstream change this needed

`shared/aifs_n320_grib_1p5defg_nc_cli.py` gained **`--source-grid {n320,o96}`**, because
the store being read is O96 and the CLI hardcoded `N320` as the interpolation source.
Default is `n320`, so every existing command line behaves exactly as before — verified by
re-running one member of 20260813 through the default path and diffing against the shipped
production NetCDF: `msl`, `tp`, `2t` all **bit-identical**.

Forking the 50 KB CLI instead was the obvious shortcut and the wrong one; a fork drifts
from the file it was copied from, which is exactly how the slow batched regrid survived in
`transcode_n320_to_o96.py` after `o96_grid.py` was fixed.

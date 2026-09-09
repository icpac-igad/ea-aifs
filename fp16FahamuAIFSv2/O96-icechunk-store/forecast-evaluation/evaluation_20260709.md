# Forecast evaluation — cycle 20260709 (O96, operational inputs)

Local AI-WQ RPSS, scored with `AI_WQ_package.forecast_evaluation` — the competition's own
code. **Nothing uploaded.**

| | |
|---|---|
| init | **2026-07-09 00z** |
| verification weeks | 2026-07-27 (week1), 2026-08-03 (week2) |
| source | `20260709_0000/icechunk_o96`, **compacted**, tag `cycle-20260709_0000-c` |
| store | 104 GB, 50/50 members, 132/132 steps |
| evaluated | 2026-08-22 |

## Result

**Read the per-week rows, not the mean.** The AI-WQ leaderboard reports **week1**, so that
is the row comparable to an official figure.

| week | valid | tas | mslp | pr |
|---|---|---|---|---|
| week1 | 2026-07-27 | −0.035 | +0.023 | +0.060 |
| week2 | 2026-08-03 | −0.196 | +0.098 | +0.004 |
| mean | | −0.115 | +0.060 | +0.032 |

## This does NOT match the submitted forecast, and the gap is unresolved

20260709 was submitted operationally in July from the **N320** route. That store was
deleted, so the O96 numbers above come from a **fresh inference run**, not a re-expression
of what was submitted.

Official leaderboard (`12Fahamu / fp16FahamuAIFSv2`, week1), the surviving submission file
rescored with `score_rpss.py`, and this O96 run:

| week1 | tas | mslp | pr |
|---|---|---|---|
| **official leaderboard** | **0.071** | **0.055** | **0.106** |
| submitted N320 file, rescored here | 0.0715 | 0.0549 | 0.1060 |
| **this O96 run** | **−0.035** | **+0.023** | **+0.060** |

The middle row matters: **`score_rpss.py` reproduces the official score to three decimals**,
so the scoring pipeline is validated against the real leaderboard. Whatever explains the
third row, it is not the scorer.

**The O96 run is worse on all three variables** — tas by 0.106, pr by 0.046, mslp by 0.032.
That is a large, one-directional gap and it is not explained away below.

### What can and cannot be concluded

The comparison confounds two changes at once: a different regrid route *and* a different
stochastic draw (aifs-ens-2.0 is stochastic; re-running the same input gives a different
forecast). Modal-quintile flip rates separate their scale:

| var | regrid alone (20260813, one rollout, two routes) | regrid + different draw (this comparison) |
|---|---|---|
| tas | 13.4 % | 28.1 % |
| **mslp** | **4.3 %** | **29.5 %** |
| pr | 19.0 % | 40.0 % |

`mslp` is the clearest: the regrid moves it 4.3 %, this comparison moves it 29.5 % — nearly
7× more. So most of the movement here is the different draw, not the grid.

That is an argument about *magnitude*, not a clean result. It does **not** show the grid is
harmless, and the fact that all three variables moved the same direction is not what a pure
draw effect would be expected to look like. Two possibilities remain open:

* the gap is mostly an unlucky draw, and O96 costs little; or
* the O96 route does cost real skill, on top of draw noise.

**Nothing here distinguishes them.** The experiment that does is 20260813, which holds
N320-derived and O96-derived products from *one* rollout and verifies **2026-08-31**.
Scoring both then isolates the grid with no draw confound. Until that runs, treat the O96
route as **unvalidated for skill**, and do not use these numbers to compare against
leaderboard figures.

## All scored cycles

| cycle | inputs | tas | mslp | pr | overall |
|---|---|---|---|---|---|
| [20260212](evaluation_20260212.md) | donor-sourced (pre-50r1) | −0.074 | +0.132 | +0.032 | **+0.030** |
| [20260514](evaluation_20260514.md) | operational | −0.167 | +0.233 | +0.005 | **+0.024** |
| **20260709** | operational | −0.115 | +0.060 | +0.032 | **−0.008** |
| 20260820 | operational | — | — | — | verifies 2026-09-14 |

**All three cycles above are O96-route reruns, not the submitted forecasts**, and for
20260709 the submitted N320 forecast scored +0.071/+0.055/+0.106 at week1 against this
run's −0.035/+0.023/+0.060. Treat the table as internally comparable, not as skill
estimates for what was submitted.

Consistent across all three: **`tas` is always negative** (−0.07 to −0.17) and always the
most over-confident variable (reliability −0.09 to −0.22). That is the one finding stable
enough to act on, and it is a dispersion problem, not an accuracy one — `tas` bias never
exceeds 0.06 K and pattern-r is 0.994+ everywhere.

`mslp` and `pr` swing widely between cycles (+0.06→+0.23 and +0.005→+0.032), which three
cycles cannot separate from weather. **Do not read a trend into three points.**

## Commands

```bash
cd .../O96-icechunk-store/forecast-evaluation
./evaluate_cycle.sh 20260709
```

Note this ran against a **compacted** store whose original tag was tombstoned. The
`--icechunk-tag cycle-20260709_0000` in the driver no longer resolves; the reader fell back
to branch `main` — which after compaction is the same snapshot — and reported doing so.
That fallback is what made this evaluation work unchanged.

## Caveats

1. Scores the O96 route (`N320 → O96 → 1.5°`), not a native-N320 submission. The regrid
   contributes ~3 % of the `tas` miscalibration — see
   [regrid_impact_n320_vs_o96.md](regrid_impact_n320_vs_o96.md).
2. Never submitted, so no leaderboard figure to check against.
3. Three cycles, three different seasons. Enough to see the `tas` pattern repeat; not
   enough for any statement about mean skill.

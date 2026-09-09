# The O96 reruns score far below the leaderboard — unresolved

For cycle **20260709** the forecast actually submitted (N320 route) scored **+0.071 /
+0.055 / +0.106** on the AI-WQ leaderboard. Rerunning that cycle through the **O96** route
scores **−0.035 / +0.023 / +0.060** — worse on all three variables, `tas` by 0.106.

This document records what is established, what is not, and the one experiment that
separates them. **Until that experiment runs, the O96 route is unvalidated for skill and
its RPSS values must not be compared against leaderboard figures.**

## 1. The numbers

Week1, `12Fahamu / fp16FahamuAIFSv2`, init 2026-07-09, valid 2026-07-27:

| week1 | tas | mslp | pr |
|---|---|---|---|
| **official leaderboard** | **0.071** | **0.055** | **0.106** |
| submitted N320 file, rescored locally | 0.0715 | 0.0549 | 0.1060 |
| **O96 rerun** | **−0.035** | **+0.023** | **+0.060** |
| Δ (O96 − N320) | **−0.106** | −0.032 | −0.046 |

## 2. Established: the scorer is not the problem

`score_rpss.py` run against the surviving submitted file reproduces the leaderboard **to
three decimal places** on all three variables. It uses
`AI_WQ_package.forecast_evaluation.work_out_RPSS` — the competition's own code — so the
gap is in the forecast, not the measurement.

**Also established: the leaderboard reports week1, not a two-week mean.** Earlier reports
led with the mean, which is comparable to nothing official. Always compare week1.

## 3. Not established: whether the grid is responsible

The July N320 store had been deleted, so the O96 numbers come from a **fresh inference
run**. aifs-ens-2.0 is stochastic — the same input yields a different forecast — so this
comparison changes *two* things at once: the regrid route **and** the ensemble draw.

Modal-quintile flip rates bound their relative scale:

| var | regrid alone (20260813, one rollout, two routes) | regrid + different draw (20260709) |
|---|---|---|
| tas | 13.4 % | 28.1 % |
| **mslp** | **4.3 %** | **29.5 %** |
| pr | 19.0 % | 40.0 % |

`mslp` is the cleanest signal: the regrid moves it 4.3 %, this comparison moves it 29.5 % —
**about 7× more**. So most of the movement is the draw.

That bounds magnitude. It does **not** clear the grid, for two reasons:

* a bound is not a measurement — "most" is not "all", and the residual could still be
  worth 0.05 RPSS;
* **all three variables moved the same direction**, which is not what an unbiased draw
  effect looks like. Three independent draws landing worse on three variables is possible
  by chance, but it is not the null expectation.

Two hypotheses remain live, and nothing collected so far distinguishes them:

1. an unlucky draw, and O96 costs little skill; or
2. the O96 route costs real skill, on top of draw noise.

### Why the earlier calibration result does not settle it either

`regrid_impact_n320_vs_o96.md` measured that the O96 hop accounts for ~3 % of the `tas`
miscalibration and costs 1.3 % of ensemble spread. Those are **calibration** measurements
on one rollout. They are consistent with hypothesis 1 but do not test RPSS, and that
document says so. A small sharpness shift can still move RPSS if it moves probability
across a quintile boundary where the observation actually fell.

## 4. The experiment that settles it

**Cycle 20260813 holds both products from a single rollout**: its O96 store was transcoded
from its N320 store, and both went through the identical downstream chain. Scoring both
gives a grid-vs-grid RPSS difference **with no draw confound** — the exact comparison
missing above.

| | |
|---|---|
| N320 route | `20260813_0000/aiwq/` (from the `icechunk_n320_aiwq` sidecar) |
| O96 route | `20260813_0000/aiwq_via_o96/` |
| verification weeks | **2026-08-31** and 2026-09-07 |
| runnable from | **2026-08-31** (init+18); complete after 09-07 |

Both quintile products already exist on disk, so no inference or regridding is needed —
only observations, which do not exist yet. `score_rpss.py` refuses future dates rather
than producing a number without them.

Run it with:

```bash
cd .../O96-icechunk-store/forecast-evaluation
./settle_grid_question.sh          # refuses politely before 2026-08-31
```

### Reading the outcome

| result | conclusion |
|---|---|
| ΔRPSS ≈ 0 (within ~0.01) | the grid is not the cause; 20260709 was an unlucky draw, and O96 is safe for research archives |
| ΔRPSS materially negative | the O96 route costs real skill; keep the N320 sidecar for anything graded |

Either way the **tier-B sidecar already exists** — 12 GB of N320 holding `msl`/`tp`/`2t`,
verified to reproduce a submission bit-identically. If the grid does cost skill, that is
the answer for graded output, and the O96 corpus stays a research archive.

## 5. Do not do in the meantime

* Do not quote O96 RPSS values against leaderboard figures.
* Do not compare a two-week mean to an official number — the leaderboard is week1.
* Do not conclude anything from `tas` being negative in all three O96 cycles until this
  test runs; that pattern is real but its cause is exactly what is in question.

# Tropical storm days (`TS`) from the AIFS-ENS Icechunk store

The AI Weather Quest **tropical-storm-days** target, built from a per-cycle `icechunk_v2`
store. Detector, tracker, IBTrACS calibration, and the state of a submission.

| | |
|---|---|
| AI-WQ definition | [training_data#tropical-storm-days-data-processing](https://ecmwf-ai-weather-quest.readthedocs.io/en/latest/training_data.html#tropical-storm-days-data-processing) |
| Evaluation | [RPSS on terciles per basin](https://ecmwf-ai-weather-quest.readthedocs.io/en/latest/forecast_evaluation.html#tropical-storm-days-ts) |
| Code | `ts_days.py` (driver), `ts_tracks.py` (tracker), `grid_ops.py`, `store_io.py` |
| Test | `test_tercile_binning.py` |
| Status | **not submittable** — see [§8](#8-the-correction--done-2026-08-23-ts_trackspy) |

```bash
PY=/tank/projects/micromamba/envs/aifs-gpu/bin/python
$PY ts_days.py --store /tank/projects/aifs-run/20260820_0000/icechunk_v2 \
    --tag cycle-20260820_0000 --init 20260820 \
    --aiwq-tercile-dir /tank/projects/ibtracs/clim \
    --out ts_days_probs_20260820_tracked.nc
```

A cycle covering days 18–33 yields **16 days → 2 full 7-day windows**, which is what AI-WQ
scores (its week-3 / week-4 targets).

> The MJO target lives in [`MJO_PHASE.md`](MJO_PHASE.md). Shared readers (`store_io.py`)
> and the reduced-Gaussian operators (`grid_ops.py`) serve both.

---

## The external reference data this target needs

| Needed | Source | Why a forecast alone cannot supply it |
|---|---|---|
| Tercile boundaries (`--aiwq-tercile-dir`, `--tercile-clim`) | **`AI_WQ_package.TS_processing.download_IBTRACS_compute_TStercile_climatology(date, ibtracs_file, savedir)`** — or build from IBTrACS with *this* detector | the package's version is built from **observed** IBTrACS counts; ours are detector counts, and the two only interchange if the detector is bias-corrected ([§8](#8-the-correction--done-2026-08-23-ts_trackspy)) |

Without one, `ts_days.py` falls back to the ensemble's own terciles and prints a loud
warning — those probabilities are self-referential and not submission-grade.

Built and on disk: `/tank/projects/ibtracs/clim/TS_20yrCLIM_WEEKLYTSDAYS_terciles_*.nc`,
from `/tank/projects/ibtracs/IBTrACS.ALL.v04r01.nc` (23 MB, NOAA/NCEI).

---

## The detector and tracker (`ts_days.py`, `ts_tracks.py`)

AI-WQ's observed diagnostic comes from **IBTrACS**, which a forecast obviously
cannot use, so the storms must be detected in the model fields. The target is
the number of unique **`(storm, day)` pairs** in the week
(`TS_processing.count_nstormdays`), so the detection must carry **storm
identity** - two concurrent storms on one day are two storm days, and a week is
not capped at 7. `ts_tracks.py` supplies that identity in three stages:

```
1. candidates   cyclonic zeta850 >= 1e-4 1/s          (--vorticity, 0 disables)
                AND msl <= latitude-band median - 5 hPa      (--msl-drop)
2. centres      candidates reduced to discrete centres, deepest-first,
                >= 500 km apart                      (--centre-separation)
   each centre tested against ITS OWN surroundings:
                msl(centre) <= annulus mean - 5 hPa          (--msl-drop)
                T(200-500 hPa) within 200 km          (--core-radius)
                  exceeds the annulus out to 800 km   (--env-radius)
                  by >= 1.0 K                         (--warm-core, 0 disables)
3. tracks       centres linked step-to-step, <= 25 m/s     (--max-translation)
                tracks shorter than 24 h dropped      (--min-track-hours)
   counted      unique (track, day) pairs whose centre is in the basin box and
                whose max |V10| within 300 km (--wind-radius) >= 17 m/s
                                                       (--wind-threshold)
```

Two details matter more than the threshold values:

**The wind is a neighbourhood maximum, not a co-located value.** IBTrACS
`usa_wind` is the storm's maximum sustained wind, and in a resolved cyclone the
eyewall is 30-100 km from the pressure minimum - so requiring wind, pressure,
warmth and vorticity to co-occur at *one* gridpoint is unsatisfiable for a
well-resolved storm and noise-driven for a marginal one.

**The warm core is measured against the storm's own environment.** A
single-level anomaly against a zonal or latitude-band reference does not
separate a tropical cyclone from an extratropical one: a midlatitude low sitting
in a warm airmass passes it. The core-minus-annulus contrast over the 200-500
hPa layer is the standard discriminator, and switching to it is what removed the
austral-winter over-trigger this file used to lead with (see below).

Sub-TS records stay in the track and are excluded from the count, mirroring
IBTrACS: a storm that weakens below 34 kt and re-intensifies is one storm.

**Vorticity is derived, not read.** `vo` is not an AIFS-ENS-2.0 output, so
`grid_ops.py` computes `zeta = 1/(a cos phi) [dv/dlambda - d(u cos phi)/dphi]`
directly on the N320 *reduced* Gaussian grid: exact circular differences along
each latitude row, and a precomputed nearest-longitude index map between
adjacent rows. That costs ~1.5 s per member, against ~26 min if every field were
regridded to a regular grid purely to differentiate it. Validated against
physics: zonal means are O(1e-6) 1/s, and the deepest 1 % of `msl` carries
+5.3e-5 1/s cyclonic vorticity while the highest 1 % carries -2.0e-5 1/s.
The sign is hemisphere-corrected so one threshold means "cyclonic" in both.

The latitude-band median is still used, but only for the cheap *candidate* net
in stage 1 (the boxes span 0-40 deg, where a whole-domain reference would label
every midlatitude low deep and every tropical column warm). Every criterion that
actually decides whether a centre is kept is local to the centre.

The store's 6-hourly steps are coarser than IBTrACS' 3-hourly records; that can
only miss short qualifying periods, never add them.

### Detector state - measured, not hypothetical

**Corrected 2026-08-23.** The detector previously described here was a
point-wise proxy with no track continuity; it counted *days on which the basin
contained a qualifying gridpoint*, a boolean capped at 7 per week, while the
target counts unbounded `(storm, day)` pairs. It also required the wind maximum
at the pressure minimum, and tested the warm core against a latitude-band
median. All three are fixed above. What that changed, measured on the
`20260820` N320 store, 50 members, verification weeks 2026-09-07 / 09-14:

| | point-wise (old) | tracked (new) | observed IBTrACS |
|---|---|---|---|
| quantity | days with a hit, **max 7** | `(storm, day)` pairs, **unbounded** | `(storm, day)` pairs |
| centres per step | n/a (58 candidate blobs) | **6.9** | - |
| tracks per member | n/a | **18.3** (>= 24 h) | - |
| ATL mean / week | 2.3, 2.6 | **1.9, 2.2** | 8.7, 8.9 |
| NWP mean / week | 6.5, 6.0 (at the ceiling) | **10.3, 9.1** (range 0-22) | 5.8, 6.5 |
| SWIO mean / week | 3.3, 2.8 | **0.6, 0.5** | 0.17, 0.15 |
| SEIO mean / week | 3.3, 1.9 | **0.4, 0.3** | 0.00, 0.00 |

Three things to read out of that table:

**The austral-winter over-trigger is gone.** SWIO 3.3 -> 0.6 and SEIO 3.3 -> 0.4,
against an observed climatology of ~0.15 and exactly 0. This was the headline
failure of the old detector and the reason its thresholds were called
"uncalibrated guesses". It was not a threshold problem: those basins were full
of midlatitude lows that a latitude-band warm-core test cannot reject, and the
core-vs-environment test rejects them.

**The ensemble can now express all three categories.** NWP spans 0-22 storm days
across members, so `P(above)` is a forecast rather than an arithmetic
impossibility. Under the old counter `P(above)` was 0.00 in both scored basins in
both weeks because the observed upper tercile (7 and 8) was at or above the
counter's own ceiling.

**A per-basin amplitude bias remains, and it changes sign.** ATL runs ~4x below
the observed climatology, NWP ~1.6x above. That is a real calibration gap - but
it is a calibration gap, which is a different and far more tractable thing than
the units mismatch it replaced. See "What would actually unblock it" below.

The 6-hourly steps are still coarser than IBTrACS' 3-hourly records; that can
only miss short qualifying periods, never add them.

---

## Outputs

`ts_days_probs.nc`
- `TS_tercile_probability` (week, basin, category) - below/near/above
- `storm_days` (member, week, basin) - the raw per-member counts, always kept
- `tercile_bounds` (week, basin, bound), plus detector settings and `tercile_source`
  in the attributes
- `n_tracks` (member) - tracks surviving the minimum-duration filter

The file records its source store and cycle, so a product can never be mistaken for one
built with a different detector.

---

## Do we need the AIFS ENS v2 BUFR tropical-cyclone tracks? **No**

AIFS ENS v2 now disseminates TC tracks as BUFR
([ECMWF implementation notes](https://confluence.ecmwf.int/display/FCST/Implementation+of+AIFS+ENS+v2#ImplementationofAIFSENSv2-Newparameters)),
which looks like the obvious shortcut - a real tracker, storm identity included.
It does not help this target, for three independent reasons:

1. **Lead time is short by a week.** The tracks run to **step 360 = day 15**.
   AI-WQ's TS target is valid at **day 19 or day 26**, and our store covers days
   18-33. The BUFR tracks stop before the forecast window even begins.
2. **They are ECMWF's forecast, not ours.** They come from ECMWF's operational
   AIFS ENS run (52 members: 50 perturbed + control + AIFS Single). Submitting a
   diagnostic derived from them would be submitting somebody else's forecast.
3. **Access.** Dissemination is described via operational ECPDS for authorised
   users; open-data availability is not stated.

So storms must be detected in our own output - which is what `ts_days.py` does,
and why deriving vorticity was worth doing. The BUFR tracks remain useful as an
**independent check** on the detector for days 1-15 in a hindcast setting, if
they can be obtained; they are not a submission path.

## Can TS be submitted alongside `tas` / `mslp` / `pr`?

**Mechanically yes, today. Scientifically not yet — and the gap is narrower than this
file previously implied.** Evaluated 2026-08-23 against AI-WQ 3.29 and a live run on the
`20260820` N320 store.

### 1. The submission path exists and is small

`TS` is a first-class Edition-2 variable, not something to be bolted on:

```python
check_fc_submission.check_filename_characteristics  ->  ['tas','mslp','pr','MJO','TS']
AI_WQ_forecast_submission(data, variable='TS', fc_start_date, fc_period, ...)
```

The expected object is tiny — `AI_WQ_create_empty_dataarray` (`forecast_submission.py:220`):

| | |
|---|---|
| dims | `('tercile', 'basin')`, shape **(3, 4)** |
| `tercile` | `[1/3, 2/3, 1.0]` — checked by `check_tercile_range` |
| `basin` | `['ATL', 'NWP', 'SWIO', 'SEIO']` |
| `fc_period` `'1'` | days **18–24** |
| `fc_period` `'2'` | days **25–31** |
| per cycle | **2 files**, 12 numbers each |

Our `ts_days_probs.nc` already carries exactly this: `TS_tercile_probability
(week, basin, category)` → transpose per week. The plumbing is a dozen lines; **it is not
the reason to wait.**

### 2. Only two basins are ever scored — and for boreal-season inits they are the two
the detector handles best

`check_fc_submission.py:246-258` builds an `active_mask` from the init month (shifted back
7 days for `fc_period` 2) and requires the probabilities to sum to 1 **only there**:

| init month | basins scored |
|---|---|
| **Jun–Nov** | **ATL, NWP** |
| **Dec–Feb** | SWIO, SEIO |
| Mar–May | *none* — the check passes vacuously |

This matters more than anything else in this file. The headline measured failure above is
that **SWIO/SEIO over-trigger in austral winter** — and for a Jun–Nov init those basins are
not scored at all. The sensitivity table's other half is the reassuring one: every filter
leaves **NWP unchanged at 5.8** while cutting SWIO 5.0 → 3.0, which is the correct physical
signature. The detector's worst known bias lands outside the scored set for exactly the
half of the year we have been running.

That narrows the problem from four basins to two. It does not calibrate them.

### 3. The tercile climatology is obtainable — the notes used to say otherwise

`AI_WQ_package.TS_processing` ships the builder:

```python
download_IBTRACS_compute_TStercile_climatology(date_str, ibtracs_filename, savedir)
#  -> TS_20yrCLIM_WEEKLYTSDAYS_terciles_<date>.nc   (tercile, basin)
```

20 years × 5 day-offsets (−4…+4) = 100 weekly samples per basin, storms filtered at
`usa_wind >= 33.0454` kt (17 m/s), 33rd/67th percentiles. Note `download_IBTRACS` does
**not** download — it opens a local IBTrACS netCDF you supply (free from NOAA/NCEI).

So `--tercile-clim` can be fed. The question is whether it *should* be — §5.

### 4. Measured on 20260820 (N320 store, 50 members, verification 2026-09-07 / 09-14)

```
$PY ts_days.py --store /tank/projects/aifs-run/20260820_0000/icechunk_v2 \
    --tag cycle-20260820_0000 --init 20260820 --out ts_days_probs_20260820.nc
```

~3.5 min for 50 members. `tercile_source = ENSEMBLE_SELF`, so the probabilities below are
self-referential — read the **counts**, not the probabilities:

| week | basin | mean storm days | range | P(below/near/above) |
|---|---|---|---|---|
| 2026-09-07 | **ATL** | 2.3 | 0–7 | 0.42 / 0.34 / 0.24 |
| 2026-09-07 | **NWP** | **6.5** | **2–7** | 0.30 / 0.70 / **0.00** |
| 2026-09-07 | SWIO | 3.3 | 1–7 | 0.38 / 0.34 / 0.28 |
| 2026-09-07 | SEIO | 3.3 | 0–7 | 0.34 / 0.22 / 0.44 |
| 2026-09-14 | **ATL** | 2.6 | 0–7 | 0.44 / 0.22 / 0.34 |
| 2026-09-14 | **NWP** | **6.0** | 1–7 | 0.44 / 0.56 / **0.00** |
| 2026-09-14 | SWIO | 2.8 | 0–7 | 0.48 / 0.34 / 0.18 |
| 2026-09-14 | SEIO | 1.9 | 0–5 | 0.68 / 0.14 / 0.18 |

SWIO at 2.8–3.3 in September repeats the known over-trigger — now harmless, because
September inits score ATL/NWP only.

### 5. The two blockers that remain, both in the scored basins

**(a) NWP saturates against the 7-day ceiling.** A week has 7 days, and the detector marks
6.0–6.5 of them. `P(above) = 0.00` in both weeks is not a forecast — it is the arithmetic
of a variable pinned at its maximum. A saturated predictor cannot discriminate, so RPSS in
NWP would be near zero at best regardless of how good the underlying rollout is. This is
the point-wise-detector problem the caveat above already names: any single 6-hourly wind
maximum anywhere in a 38,000-point basin marks the whole day, and over a basin that size
in peak season something always qualifies. **Track continuity — requiring a coherent centre
to persist across consecutive steps — is the fix, not threshold tuning.**

**(b) Detector counts and IBTrACS counts are not the same quantity.** AI-WQ's terciles come
from observed IBTrACS storm days; ours come from a proxy detector with a different bias. If
the detector's climatological mean sits above the observed one, *every* forecast lands in
"above normal" and RPSS is negative **by construction** — a systematic error that no amount
of ensemble skill recovers. Feeding `--tercile-clim` with the package's IBTrACS file
without first measuring that offset would produce a confidently wrong submission.

### 6. The bounded next step — **done 2026-08-23. Verdict then: do not submit.**

> **Superseded in part by §8.** The units mismatch this section found has since been
> fixed (the tracker in `ts_tracks.py`). What survives is the IBTrACS climatology it
> built and the reason a submission built on the *old* counter would have been wrong.
> Read §8 for the post-correction state.

Run, with the AI-WQ package's own builder against IBTrACS `v04r01` (23 MB, NOAA/NCEI,
downloaded to `/tank/projects/ibtracs/`):

```bash
$PY -c "
from AI_WQ_package import TS_processing as T
ib = T.download_IBTRACS('/tank/projects/ibtracs/IBTrACS.ALL.v04r01.nc')   # opens a LOCAL file
for d in ('20260907','20260914'):
    T.compute_20yr_TStercile_climatology(d, ib, '/tank/projects/ibtracs/clim')"
```

**Observed weekly storm days** (100 samples = 20 yr × 5 day-offsets):

| week | basin | mean | median | p33 | p67 | max | zero-weeks |
|---|---|---|---|---|---|---|---|
| 09-07 | ATL | 8.72 | 8 | 6.00 | 11.00 | 24 | 3 % |
| 09-07 | NWP | 5.81 | 6 | 4.00 | 7.00 | 14 | 5 % |
| 09-07 | SWIO | 0.17 | 0 | 0.00 | 0.00 | 4 | 93 % |
| 09-07 | SEIO | 0.00 | 0 | 0.00 | 0.00 | 0 | 100 % |
| 09-14 | ATL | 8.88 | 8 | 5.00 | 12.00 | 28 | 7 % |
| 09-14 | NWP | 6.49 | 6.5 | 5.00 | 8.00 | 18 | 6 % |
| 09-14 | SWIO | 0.15 | 0 | 0.00 | 0.00 | 4 | 95 % |
| 09-14 | SEIO | 0.00 | 0 | 0.00 | 0.00 | 0 | 100 % |

#### The blocker is not bias — it is a units mismatch

**A week cannot contain 28 storm days.** AI-WQ's observed diagnostic counts unique
**(storm, day) pairs** (`TS_processing.count_nstormdays`: `np.unique(stack([storm_ids, day]))`),
so two concurrent hurricanes on the same day contribute **two** storm days. The quantity is
*storm*-days summed across storms, and it is **unbounded**.

`ts_days.py` *then* counted **basin-days** — `by_day[day] = step_hit.any()`, a boolean per
day, summed over 7. It was **capped at 7 by construction**. (Fixed in §8; the current
counter is unbounded.)

These are different quantities. No threshold tuning, and no bias correction, makes a
0–7 counter comparable to a 0–28 one. The earlier framing in this file — "the detector is
uncalibrated" — understated it.

#### What that does to a submission, measured

Applying the real IBTrACS terciles to the 50 member counts from `20260820`:

| week | basin | fc mean | fc max | obs p33 | obs p67 | P(below) | P(near) | P(above) |
|---|---|---|---|---|---|---|---|---|
| 09-07 | **ATL** | 2.34 | 7 | 6.00 | 11.00 | **0.96** | 0.04 | **0.00** |
| 09-07 | **NWP** | 6.46 | 7 | 4.00 | 7.00 | 0.02 | 0.98 | **0.00** |
| 09-14 | **ATL** | 2.58 | 7 | 5.00 | 12.00 | **0.86** | 0.14 | **0.00** |
| 09-14 | **NWP** | 6.00 | 7 | 5.00 | 8.00 | 0.24 | 0.76 | **0.00** |
| 09-07 | SWIO | 3.34 | 7 | 0.00 | 0.00 | 0.00 | 0.00 | 1.00 |
| 09-14 | SEIO | 1.94 | 5 | 0.00 | 0.00 | 0.24 | 0.00 | 0.76 |

(bold = the basins actually scored for an August init.)

- **`P(above) = 0.00` in NWP is structural, not a forecast.** The observed upper tercile is
  7.0 and 8.0 storm days; our counter's maximum *is* 7, so `c > hi` is unsatisfiable. The
  ensemble is incapable of predicting an above-normal North-West Pacific week.
- **ATL is a near-deterministic "below normal"** (0.96 / 0.86) for the same reason: the
  ceiling of 7 sits barely above the observed p33 and far below p67.
- SWIO/SEIO invert to `P(above) ≈ 1` against an observed climatology that is **zero in
  93–100 % of weeks** — the over-trigger, now quantified. Unscored for these inits, but it
  is the same defect.

Submitting this would score a strongly negative RPSS in every basin, and would do so
whatever the underlying rollout looked like. **Do not wire it up.**

#### What would actually unblock it

Counting (storm, day) pairs requires **storm identity** — which storm, persisting across
steps. That is a tracker, not a threshold. So track continuity moves from "the next
structural fix" (as this file had it) to **a prerequisite for submitting at all**:

1. group qualifying points into connected centres per step;
2. link centres across consecutive 6-hourly steps into tracks;
3. count `(track_id, day)` pairs — the same quantity AI-WQ counts;
4. *then* compare against the IBTrACS terciles, which are already built and on disk at
   `/tank/projects/ibtracs/clim/`.

Only after step 4 shows comparable distributions does calibration become the question.

**All four were done on 2026-08-23 — see §8.**

A smaller gap fixed on the way: `--tercile-clim` expected an `.npz` with a `bounds`
`(basin, 2)` array applied to **all** weeks, while AI-WQ writes one `.nc` per valid week
with dims `(tercile, basin)`. `--aiwq-tercile-dir` now reads the official files and
`tercile_bounds` carries a `week` dimension.

### 7. A constraint worth knowing before planning this

The detector needs `10u`, `10v`, `msl`, `t_300`, `u_850`, `v_850` over days 18–33, and
`grid_ops.py` differentiates on the **N320 reduced Gaussian** rows. Only **two N320 stores
now exist** — `20260806` and `20260820`; the rest of the archive was released or is O96-only.
Two cycles is not enough to characterise the detector's own distribution, so a
detector-native climatology is not currently reachable from what is on disk. The O96 corpora
hold all six variables over the *full* 0–792 h, which is tempting, but O96 is ~112 km and
under-resolves TC winds further than N320 already does — it would most likely move the bias,
not remove it. **Untested; do not assume it substitutes.**

### 8. The correction — **done 2026-08-23. `ts_tracks.py`.**

The three defects §6 identified are fixed: the counter measures the right quantity, the
wind test matches `usa_wind`'s semantics, and the warm-core test discriminates tropical
from extratropical. Run:

```bash
$PY ts_days.py --store /tank/projects/aifs-run/20260820_0000/icechunk_v2 \
    --tag cycle-20260820_0000 --init 20260820 \
    --aiwq-tercile-dir /tank/projects/ibtracs/clim \
    --out /tank/projects/aifs-run/20260820_0000/ts_days_probs_20260820_tracked.nc
```

~11 min for 50 members. 18.3 tracks/member survive the 24 h minimum, 6.9 centres per
step — against 1090 raw and 234 surviving under the old warm-core test, whose ≥24 h
population averaged **|lat| 32.6°**, i.e. midlatitude cyclones, not tropical ones. The
tracked population sits at a median vmax of 25 m/s and yields 2–4 qualifying storms per
basin per member, which is the right order for mid-September.

Probabilities against the **official IBTrACS terciles**, 50 members:

| week | basin | mean | range | terciles | P(below/near/above) |
|---|---|---|---|---|---|
| 09-07 | **ATL** | 1.9 | 0–10 | 6 / 11 | 0.92 / 0.08 / 0.00 |
| 09-07 | **NWP** | 10.3 | 2–22 | 4 / 7 | 0.06 / 0.10 / **0.84** |
| 09-07 | SWIO | 0.6 | 0–5 | 0 / 0 | *degenerate — unscoreable* |
| 09-07 | SEIO | 0.4 | 0–3 | 0 / 0 | *degenerate — unscoreable* |
| 09-14 | **ATL** | 2.2 | 0–11 | 5 / 12 | 0.82 / 0.18 / 0.00 |
| 09-14 | **NWP** | 9.1 | 0–21 | 5 / 8 | 0.14 / 0.32 / **0.54** |
| 09-14 | SWIO | 0.5 | 0–4 | 0 / 0 | *degenerate — unscoreable* |
| 09-14 | SEIO | 0.3 | 0–2 | 0 / 0 | *degenerate — unscoreable* |

Binned as AI-WQ's scorer bins the observation (see below); regenerated 2026-08-27,
18.3 tracks/member. The counts are unchanged from the first tracked run — the tracker is
deterministic — so only the probabilities moved.

#### Verdict: still do not submit — but for a different, smaller reason

`P(above) = 0.00` in NWP was previously an arithmetic certainty; it is now 0.84 and 0.54,
a genuine forecast from a distribution spanning 0–22. The structural blocker is gone.

What remains is a **per-basin amplitude discrepancy of opposite sign**: against the
observed climatology ATL runs ~4× low (4.59× / 4.11×) and NWP ~1.6× high (0.56× / 0.71×
obs/fc). Fed the observed terciles, that puts ATL at a near-deterministic "below normal"
(0.92 / 0.82) — a confident forecast of what is, climatologically, the less likely
outcome. If it is wrong, RPSS in ATL is strongly negative, and one basin scoring badly is
enough to make the submission a net loss.

> **"Discrepancy", not "bias" — the distinction is the whole of §8's open question.**
> This is **one cycle** compared against a **20-year climatology**, which cannot separate a
> detector defect from a genuine forecast of a quiet Atlantic fortnight. The ensemble is
> not blind to Atlantic storms — members range 0–10 and 0–11, and 2/50 and 1/50 reach the
> climatological mean; the distribution is skewed low, which is equally what a quiet
> forecast looks like.
>
> The opposite signs matter here. A **resolution** bias depresses every basin. ATL low and
> NWP high is basin-dependent, so it is not the single multiplicative resolution factor
> that the ERA5 argument below implicitly assumes. Quantile mapping would still work
> mechanically — it is rank-based — but the *mechanism* named below is not established.

#### The candidate fix — and why it is not yet the *named* fix

The obvious move is terciles built **with this detector** rather than from IBTrACS's
agency-assigned intensities — what `--tercile-clim` was always for. A detector-native
climatology absorbs a detector-native error; an observed one cannot.

The ensemble cannot supply it: 50 members of one cycle over two weeks is 100 samples, the
same sample size AI-WQ uses, but they share initial conditions and calendar weeks, so they
are one draw, not a climatology. That is the `ENSEMBLE_SELF` path, correctly flagged as
self-referential.

**Three reasons not to start building it yet.** An earlier draft of this section called
ERA5 terciles "the one bounded job standing in the way". That was overstated:

1. **It may be aimed at the wrong term.** If the ATL deficit is the *model's* forecast
   rather than the *detector's* miss, a detector climatology corrects nothing. Nothing
   collected so far distinguishes those (see the note above).
2. **ERA5 is an analysis climatology; the forecast is day 18–33.** It would correct the
   detector, not the model's behaviour at that lead. Standard S2S practice calibrates
   against a **model climatology at matching lead** — hindcasts — precisely because
   long-lead attenuation does not exist in analyses. That alternative is unscoped.
3. **It is not currently startable.** No `~/.cdsapirc` and no `cdsapi` installed, so a CDS
   account and licence acceptance are unmet *external* prerequisites, not compute. And
   disk is **203 GB free (91 %)** as of 2026-08-29 with two 583 GB N320 stores resident
   (`20260820`, `20260827`); a seasonal 20-year ERA5 archive is order 100–200 GB even
   restricted to the TC belt, and does not fit alongside them.

**If it is built, ERA5 is the right source.** At ~31 km it is close to N320's ~28 km, so
the same tracker run over 20 years of 6-hourly ERA5 for the target calendar weeks would
produce terciles carrying the same *detector* error as the forecast — which is what would
make that term cancel. It needs `10u`, `10v`, `msl`, `t_200/300/500`, `u_850`, `v_850`,
the same six-ish fields the store carries, and the tracker functions take flat
`(time, values)` arrays plus `lat`/`lon`, so they should port to a regular lat/lon grid —
**untested**, and the radius index assumes row-major storage, which ERA5 satisfies but has
not been exercised.

What it would *not* correct is the model's own behaviour at day 18–33 (reason 2 above).
So it is a necessary step if the detector turns out to be at fault, not a sufficient one,
and not the whole distance to a submission.

Until then, `--aiwq-tercile-dir` is a diagnostic, not a submission path.

#### The next checkpoint costs nothing: 2026-09-14

`20260820`'s **week 1 verifies 2026-09-07 … 09-13**, so the observation is available from
about **2026-09-14** — and it settles the question above without building anything:

```bash
$PY -c "
from AI_WQ_package import TS_processing as T
ib = T.download_IBTRACS('/tank/projects/ibtracs/IBTrACS.ALL.v04r01.nc')   # refresh first
print(T.compute_single_week_numTSdays('20260907', ib))"
```

Compare the observed ATL count against our forecast distribution (mean 1.9, range 0–10,
`P(below) = 0.92`):

| observed ATL storm days | reading |
|---|---|
| **≲ 6** (below the lower tercile) | the forecast was right; there is no ATL bias to correct, and the ERA5 job is aimed at the wrong term |
| **≈ 6–11** | inconclusive from one week; wait for `20260827` |
| **≫ 11** | the forecast was confidently wrong — the discrepancy is real, and a detector-native climatology becomes the justified next job |

`20260827` gives a second point a week later (week 1 = 2026-09-14 … 09-20). Two cycles is
still not a bias estimate, but it is the difference between a measured claim and an
inferred one — and it arrives for free.

**If a cheap probe is wanted before then**, run the tracker over ERA5 for a *handful* of
past weeks — tens of GB, not 100–200 — and compare against IBTrACS for those same weeks.
That measures detector-versus-observation with no forecast involved, which is the question
actually blocking everything, and it decides whether the full 20-year archive is worth
building.

#### The tercile edges are AI-WQ's to define, and we had both of them wrong

The probabilities above were binned `c <= lo` / `lo < c <= hi` / `c > hi`. That looked
like a house convention to pin down — AI-WQ distributes only the two bounds — but the
convention is not ours. `forecast_evaluation.conditional_obs_probs` decides which category
the **observation** falls in, and a forecast partitioned any other way is graded against a
different partition:

```python
q == 0      ->  obs <  bounds[0]                    # below
0 < q < n   ->  bounds[q-1] <= obs <  bounds[q]     # near, and only when lower != upper
q == n      ->  obs >= bounds[-1]                   # above
```

Both edges differed from what we had, not one. With integer storm-day counts against
integer-valued terciles, ties at a bound are common, so this is not academic:

| week | basin | as written | scorer-matched | Δ above |
|---|---|---|---|---|
| 09-07 | **NWP** | 0.06 / 0.24 / 0.70 | 0.06 / 0.10 / **0.84** | **+0.14** |
| 09-14 | **NWP** | 0.22 / 0.30 / 0.48 | 0.14 / 0.32 / **0.54** | +0.06 |
| 09-14 | **ATL** | 0.86 / 0.14 / 0.00 | 0.82 / 0.18 / 0.00 | 0.00 |

The upper edge (`c > hi` → `c >= hi`) carries most of it and the lower edge none in NWP
week 1 — so fixing only the lower edge, the obvious half of the correction, would have
left the larger error in place. The direction matters too: matching the scorer moves mass
**up**, partly offsetting the NWP high bias rather than compounding the ATL low bias.

`tercile_probs()` in `ts_days.py` now implements the predicate, and
`test_tercile_binning.py` asserts agreement with the installed package over counts that
span both bounds — a conformance test, so it fails if AI-WQ changes its edges rather than
silently drifting:

```bash
$PY test_tercile_binning.py
# -> PASS -- binning matches AI_WQ_package.forecast_evaluation.conditional_obs_probs
```

This has to be settled **before** the ERA5 terciles are built, not after: the bias only
cancels if the climatology is partitioned the same way as the forecast, so an edge
mismatch would corrupt the very correction §8 is waiting on.

##### Degenerate terciles are unscoreable, not merely awkward

When `lower == upper` — SWIO/SEIO in September, where 93–100 % of observed weeks have zero
storm days — AI-WQ's `all_equal` mask sets the **observation** to NaN and the basin-week
cannot be scored at all. Under its partition every count then falls in "above", so the
file now emits `0 / 0 / 1` there and says so loudly. That is arithmetic, not a forecast; it
is kept rather than NaN only because it satisfies `check_data_characteristics`'
sum-to-1 test. Harmless for Jun–Nov inits, where `active_mask` already excludes those two
basins — but a **Dec–Feb init can hit a degenerate bound in a scored basin**, and that is
the case to watch.

---

## Not done

- The detector/tracker is unvalidated against observed tracks: no cycle has been verified
  storm-by-storm against IBTrACS positions, only distribution-against-climatology.
- **Still not submittable**: a per-basin amplitude discrepancy of opposite sign remains
  (ATL ~4× low, NWP ~1.6× high) against the observed IBTrACS terciles. Whether that is a
  **detector** error or a genuine forecast is **not yet established** — it rests on one
  cycle against climatology. The free test is `20260820`'s week-1 verification from
  **2026-09-14** ([§8](#the-next-checkpoint-costs-nothing-2026-09-14)); a detector-native
  ERA5 climatology is a *candidate* fix, not a confirmed one, and has unmet prerequisites
  (no CDS credentials, ~200 GB free disk).
- Tercile binning matches AI-WQ's scorer exactly (`tercile_probs`, guarded by
  `test_tercile_binning.py`); §8's tables were regenerated against it on 2026-08-27.
- Tested against `20260730` (old detector) and `20260820` (both).

# The MJO product: what was tried, what was settled, and what it needs

`MJO_PHASE.md` is the working log — every attempt in the order it happened, including the
wrong turns. **This document is the summary**: the method as it now stands, the data it
depends on, and how it will be verified. Read this first; go there for why.

**Status: the observed pipeline is validated; the forecast product is not submittable.**
One thing blocks it, and it is named in §5.

---

## 1. Why this is hard at all

AI-WQ's MJO target is **RMM** (Wheeler & Hendon 2004): project
`[OLR, U850, U200]`, each reduced to a 144-point longitude band, onto published combined
EOFs. The package ships `WH04_combinedEOFs.nc` and `WH04_RMM_stddevs.nc`, and scores against
RMM `phase`/`amplitude`.

**AIFS-ENS 2.0 cannot produce OLR.** Its single-level output is entirely surface; there is no
top-of-atmosphere term, so `ttr` is outside the checkpoint's output space and cannot be added
by re-running inference. ERA5 cannot fill the gap either — the forecast days are in the future.

Dropping the OLR block and projecting the remaining 288 elements onto the surviving EOF rows
is **not a valid projection**: the truncated vector is not an EOF of the wind-only space, so
the PCs are not RMM1/RMM2 and the published standard deviations do not apply.

So the third field has to come from somewhere else.

## 2. What was tried

| # | approach | outcome |
|---|---|---|
| 1 | Truncate WH04 to the two wind blocks | **Rejected on principle** — not an EOF of the wind-only space; `mjo_index.py` refuses it |
| 2 | Emulate OLR from model fields | **Not attempted** — would need its own validation, and any error enters the index unquantified |
| 3 | **VPM**: replace OLR with **χ₂₀₀**, diagnosed exactly from the forecast wind | **Adopted.** A published index (Ventrice et al. 2013), and χ₂₀₀ is derived, not emulated |
| 4 | Get VPM's EOFs from NOAA PSL | **Failed.** PSL does not publish them; its `eof1/eof2` are **OMI's** (2448 values = 144×17, one file per day-of-year, a single OLR field) |
| 5 | Fit a projection onto PSL's published **VPM index** series, then rotate VPM→RMM | **Worked, then superseded.** 89.9% octant, but needed a ~166° reflection to reach RMM's convention |
| 6 | **Fit directly onto AI-WQ's official RMM series** | **Adopted.** 98.4% octant out of sample, correct handedness by construction, no convention step |

Two findings from the discarded branches are worth keeping:

- **PSL's `vpm.1x.txt` is stored mirrored.** On active consecutive days it advances **−7.6 °/day
  (westward)**; an MJO index must advance eastward through phases 1→8. Anything fitted against
  that file inherits the mirror.
- **Routing through VPM cost most of the accuracy.** The "64% ceiling" measured in approach 5
  was not a property of using wind + χ₂₀₀ instead of OLR — it was the cost of fitting to one
  index and then correcting the convention, which compounds two errors.

## 3. The method as it stands

```
1. D    = divergence(u200, v200)          on a 1.5° regular grid
2. chi  : solve laplacian(chi) = D        real FFT in longitude, tridiagonal per wavenumber
3. band : cosine-weighted mean over ±15°  -> 144 longitudes, for chi200, U850, U200
4. daily means from the 6-hourly steps
5. subtract the ERA5 calendar-day climatology, 1991-2020, 3 annual harmonics
6. [120-day trailing mean -- NOT APPLIED, see §6]
7. divide by FIXED normalisation factors -- the ones the basis was fitted with
8. project onto a basis fitted to AI-WQ's official RMM
9. RMM1/RMM2 -> amplitude, phase -> 9 categories -> (9,4) at init +7/14/21/28
```

**Step 8 is a regression, not an EOF.** It is not orthonormal and not variance-maximising,
and it inherits ERA5, our climatology and our grid. Nothing downstream needs otherwise — the
projection only forms `x · e1`, `x · e2`. It is *not* "the VPM basis applied to our fields";
it is a linear map fitted to reproduce the official RMM from our predictors. Anything shipped
should be labelled that way.

Two consistency rules are load-bearing, both learned the hard way:

- **Everything is computed the same way on both sides.** Divergence is taken on the regular
  grid for the forecast *because* that is what the climatology does. Taking it on the native
  reduced-Gaussian grid is ~2× cheaper and moves the wavenumber-1 crest **22.8° ± 36.9°**,
  which is several days of MJO propagation.
- **Normalisation factors are fixed, not per-forecast.** RMM ships observed standard
  deviations for exactly this reason. A forecast's own spread runs 1.2–1.6× ERA5's, *unequally
  across the three fields*, which would reweight them against each other inside the projection.

## 4. Input data, and where each piece comes from

| input | source | size | used for |
|---|---|---|---|
| `u_200`, `v_200`, `u_850` forecast | `icechunk_o96` (O96, ~112 km, 120 vars) | 164 GB/cycle | steps 1–4 |
| ERA5 `u,v` at 200/850 hPa | ARCO-ERA5 `ar/…240x121…` (1.5°, 6-hourly, 1959–2021) | ~57 GB streamed, **20 MB kept** | step 5 |
| Official RMM daily series | AI-WQ FTP `/training_data/MJO_DAILY_{1979..2026}.nc` | ~2 MB | step 8 (fitting target) |
| MJO observations | AI-WQ FTP `/observations/<Monday>/MJO_obs_DAILY_*.nc` | ~30 KB/week | verification |
| WH04 EOFs + stddevs | AI-WQ FTP `/training_data/MJO_reference_data/` | 22 KB | reference; not used in the fit |

**The MJO reads the O96 corpus, not the N320 sidecar.** The sidecar carries no
`u_200`/`v_200`, and the MJO does not need 28 km — it is zonal wavenumber 1–3, so ~112 km is
ample. This is the mirror of the TS product, where 28 km is essential and O96 measurably
degrades the wind maximum. `--native-vars-mjo` can add the pair for +13.2 GB/cycle if a
cycle's O96 corpus is going to be purged while the MJO must still run.

**AI-WQ FTP access note.** `ftp_or_ecbox_loading(remote, local, password)` takes one
`password` argument and uses it as **two different credentials** — the FTP password, then on
failure as an ecbox Branca token. A `403 "Not a Branca token"` therefore means *the FTP leg
failed first*; read that error, not the fallback's. `AIWQ_PASSWORD` works over FTP.

## 5. What blocks submission

**A model climatology at matching lead.** Step 5 removes an *observed* climatology from a
*model* field, which leaves the model's own mean-state bias and lead-dependent drift inside
the anomaly. Measured: a systematic per-longitude offset with `|offset|/std ≈ 0.67` for every
field in every cycle. Removing it takes χ₂₀₀ from 7.79e6 to 4.22e6 against ERA5's 4.91e6 —
from 1.59× the observed variance to 0.86×.

The consequence in the product:

| | forecast | observed RMM |
|---|---|---|
| mean amplitude | 2.5–2.7 | 1.30 |
| P(inactive) | 0.01–0.02 | 0.37 |
| PC drift | +0.2…+1.1 °/day | +5.49 |

Operational MJO forecast indices remove a **model** climatology per lead time for exactly this
reason. This is the same pattern as the TS product, where the useful climatology also turned
out to be model-derived rather than observational.

Two things make it tractable, and one makes it slow:

- The offset is **stable across cycles** (0.66 / 0.67 / 0.68), so few cycles are needed — the
  TS climatology was built from five.
- It is **large and systematic**, so removing it should be most of the fix.
- But three September cycles give a *September* climatology. Year-round coverage needs cycles
  spread through the year, and the purged cycles' O96 corpora are gone, so this **accumulates
  going forward** rather than being recoverable from what is on disk.

A rescale of amplitudes to match the observed inactive fraction is **not** an acceptable
shortcut: it would hide a mean-state bias behind a variance correction, and the phases would
still carry the offset.

## 6. Secondary gaps

- **The 120-day trailing mean (step 6) is not applied.** It removes interannual variability,
  chiefly ENSO, and is a *trailing* mean rather than a symmetric 20–100 day bandpass because a
  bandpass needs future data and so cannot run on a forecast. Measured worth: 7–11% of
  variance, but specifically it lifts amplitude *r* from 0.626 to 0.697 — and amplitude is what
  decides the inactive category. Buildable now (`raw/date-variable-pressure_level/` reaches
  2026, ~15 GB, ~3 h); deferred behind §5.
- **The ERA5 series is stride-2**, so it cannot supply a contiguous 120-day history or a
  properly conditioned basis. Fixing that means the in-cloud stride-1 run (`MJO_PHASE.md` §7.2).

## 7. How it will be verified

Three checks, in increasing strength.

**(a) The basis, against held-out years — done.** Fit 1991–2010, test 2011–2020 (1828 days),
ERA5 predictors against the official RMM:

| | test |
|---|---|
| RMM1 *r* | 0.922 |
| RMM2 *r* | 0.936 |
| amplitude *r* | 0.859 |
| median phase error | 8.9° |
| **correct octant, active days** | **98.4%** |

ERA5 through this basis gives mean amplitude 1.36 and P(amp<1) = 0.34 against the official
1.30 and 0.37, advancing **+5.49 °/day eastward**. The observed half of the pipeline is sound.

**(b) The forecast, against AI-WQ observations — not yet possible.** `retrieve_daily_MJO_obs()`
returns a one-hot `(9,)` per day; scoring is Brier per AI-WQ's
`calculate_MJO_brier_score`, against the published `MJO_20yrCLIM_DAILYprobs_*.nc` as the
reference. The obstacle is latency, not access: observations run to **2026-08-24** and the
official RMM series to **2026-04-30**, while the earliest valid time in hand is **2026-09-10**.
Cycle 20260903 becomes verifiable first, as observations catch up.

**(c) The whole chain, against a known event — done once.** Running the identical chain on
ERA5 over the DYNAMO period (2011-10-01…2012-01-31, documented strong MJO events) recovers
**+6.54 °/day eastward with lag-1 correlation 0.90**, while day-shuffled nulls show no
coherent propagation.

One limit applies to any forecast-side propagation check and is not fixable by better
references: **a 34-day window cannot resolve MJO propagation.** The same ERA5 series, chopped
into 34-day windows, gives +11.9 … −9.7 °/day, with one window in nine landing in the MJO
band and two running westward. Forecast phase speeds must not be read as evidence of MJO
presence or absence.

## 8. Files

| file | role |
|---|---|
| `era5_vpm_clim.py` | ARCO-ERA5 stream; `--mode series\|clim\|combine` |
| `run_era5_clim.sh` | per-year driver, restartable |
| `era5_vpm_colab.py` | generated by `make_colab_bundle.py`, for running inside GCP |
| `velocity_potential.py` | Poisson solve on the sphere; `divergence_latlon` |
| `grid_ops.py` | divergence/vorticity on the reduced Gaussian grid |
| `vpm_index.py` | the forecast pipeline, steps 1–9 |
| `vpm_basis_fit.py` | fits the projection; `--target-rmm` for the official series |
| `mjo_submission.py` | reduces the daily product to AI-WQ's `(9,4)` |
| `mjo_propagation.py` | lag-longitude phase-speed diagnostic |
| `measure_phase_offset.py` | VPM→RMM convention transform (only for a VPM-fitted basis) |
| `MJO_PHASE.md` | the full working log |

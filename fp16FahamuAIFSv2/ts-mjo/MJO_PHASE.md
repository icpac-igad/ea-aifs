# MJO phase from the AIFS-ENS Icechunk store

The AI Weather Quest **MJO phase** target. `aifs-ens-2.0` does not output OLR, so the
canonical RMM index cannot be computed from the store — this file records why, what the
submission actually demands, and the route that is still open.

| | |
|---|---|
| AI-WQ definition | [training_data#mjo-data-processing](https://ecmwf-ai-weather-quest.readthedocs.io/en/latest/training_data.html#mjo-data-processing) |
| Evaluation | [BSS over 9 categories](https://ecmwf-ai-weather-quest.readthedocs.io/en/latest/forecast_evaluation.html#mjo-phase-probability-forecasts) |
| Code | `mjo_index.py`, `grid_ops.py`, `store_io.py` |
| Research notes | `vpm-mjo.md` — exploratory VPM notes, kept **outside the repo**; §4 below is the committed summary |
| Status | **not submittable** — no valid index without OLR or a substitute basis |

> The tropical-storm-days target lives in [`TS_STORM_DAYS.md`](TS_STORM_DAYS.md).

---

## 1. The model does not forecast OLR

The canonical RMM combines **OLR + U850 + U200**. `U850`/`U200` are present and
verified; OLR is **not obtainable**, and this is not a writer setting we can flip.

AIFS-ENS-2.0's complete single-level output is 35 variables, **all surface**:

```
100u 100v 10u 10v 2d 2t cdww cp h1012 h1214 h1417 h1721 h2125 h2530 hcc lcc
mcc msl mwp ro sd sf skt snowc sp ssrd stl1 stl2 strd swh swvl1 swvl2 tcc tcw tp
```

`ssrd`/`strd` are *downward surface* fluxes; there is no top-of-atmosphere term
of any kind. `ttr` is outside the checkpoint's output space, so it cannot be
added by re-running inference. ERA5 cannot substitute either: the forecast days
(18-33) are in the future, and ERA5 only covers dates already past.

**Wind-only is not a fallback - it is not a valid projection.** The RMM projects
onto Wheeler & Hendon's *combined* EOFs: a single 3 x 144 = 432-element state
vector ordered `[OLR, U850, U200]`, with `WH04_RMM_stddevs.nc` normalising the
PCs *that basis* produces. Dropping the OLR block and projecting the 288-element
wind vector onto what remains gives something that is not an EOF of the
wind-only space - not orthonormal there, not variance-maximising - so its PCs are
not RMM1/RMM2, the published standard deviations do not apply, and the
`amplitude < 1` inactive test loses its meaning.

`mjo_index.py` therefore **refuses** to project without OLR:

- `--olr-source FILE` - the full, valid 3-field projection
  (`index_kind = "rmm3"`, `submittable = yes`). For *past* dates that file can be
  ERA5 (`olr = -ttr / accumulation_seconds`); for forecast dates only an emulator
  can supply it.
- `--olr-source none` (default) - stops after the normalised band anomalies
  (steps 1-5) and says why. Adding `--allow-wind-only` emits the truncated-EOF
  index anyway, but every variable is renamed `windproxy_*` and the file carries
  `submittable = NO`, so it cannot be mistaken for an MJO product.

**Way forward - only two honest options:**

1. **Statistical OLR emulator.** Tropical OLR is dominated by deep convection,
   and the model does output `hcc`, `tcc`, `cp`, `tp`, `tcw`. Fit `ttr` on those
   predictors in ERA5 (where both sides exist), apply to the forecast fields,
   then run the standard 3-field projection. This keeps the official EOF basis
   intact - the only route that can yield a real RMM - but it is an emulator and
   must be validated against ERA5-derived RMM before any submission.
2. **Do not submit the MJO target**, and submit only what this model supports.

The wind-only mode is *not* a third option: it is a diagnostic aid, kept for
inspecting zonal wind structure, and is barred from producing an MJO file.


---

## 2. What the submission actually demands — and why it changes the plan

Read from `AI_WQ_package.forecast_submission.AI_WQ_create_empty_dataarray` and
`check_fc_submission.check_data_characteristics`:

| | |
|---|---|
| dims | `('MJO_phase', 'valid_time')`, shape **(9, 4)** |
| `MJO_phase` | `0..8` — **0 = inactive** (amplitude < 1), 1–8 = octants |
| `valid_time` | issue + **7, 14, 21, 28 days** — the code's own comment calls these *day 8, 15, 22 and 29* |
| check | every column must sum to 1.0 (`atol=0.2`) |

**All four lags are mandatory.** NaN passes the range check, but `da.sum(axis=0)` treats
NaN as 0, so a column left empty fails the sum test. There is no partial MJO submission:
day 8 and day 15 cannot be dropped in favour of the two lags a 432–792 h store happens to
cover.

That decides which store an MJO product must be built from:

| store | hours | day 8 | day 15 | **day 22** | **day 29** | usable for MJO? |
|---|---|---|---|---|---|---|
| `icechunk_v2` (N320, windowed) | 432–792 | ✗ | ✗ | ✓ | ✓ | **no** — two of four lags missing |
| `icechunk_o96` (full corpus) | **0–792** | ✓ | ✓ | ✓ | ✓ | **yes** |

So the O96 corpus — built as a cheap research archive, and the thing the N320 rerun was
meant to supersede — is the **only** store on disk that can serve an MJO submission at
all. The 432–792 h window is an N320-era compromise (see
[`../O96-icechunk-store/README.md`](../O96-icechunk-store/README.md) §7, "never window
O96"), and MJO is the clearest case for why the full corpus earns its disk.

Resolution is not the obstacle here that it is for tropical cyclones. The MJO is a
planetary-scale, zonal-wavenumber-1–3 phenomenon and the index bins onto **144 × 2.5°
longitude bins** in a ±15° belt; O96 (~112 km) is far finer than that binning. Whether the
*forecast* is as good at O96 is a separate question, and untested.

---

## 3. What `mjo_index.py` implements today

Implements the documented chain on the store's native N320 points, with no
regridding step: points within +-15 deg are binned straight into **144 x 2.5 deg
longitude bins** with cosine-latitude weighting (steps 1-2), then daily-averaged
from the 6-hourly steps.

Steps 3-4 are applied only if the corresponding reference file is given; step 5
uses the Wheeler & Hendon (2004) factors (OLR 15.1 W m-2, U850 1.81 m s-1,
U200 4.81 m s-1). Phases follow AI-WQ's 9 categories: octant 1-8 from
`arctan2(RMM2, RMM1)`, **phase 0 when amplitude < 1**. A member's weekly phase is
its modal daily phase; the 50 members are then counted into the 9 categories.

Verified on `cycle-20260730_0000`: 16 days x 144 bins, all bins populated
(135 680 N320 points fall inside +-15 deg).


Verified on `cycle-20260730_0000`: 16 days × 144 bins, all bins populated (135 680 N320
points fall inside ±15°).

---

## 4. The way forward: VPM, and what it does and does not solve

Summarised here with what is **established from this repository's own artifacts** kept
separate from what is proposal. The longer exploratory notes (`vpm-mjo.md`) are
deliberately not committed — they are first-person and unreviewed, and everything in them
that survives scrutiny is below.

### 4.1 The index: velocity potential in place of OLR

The **VPM index** (Ventrice et al. 2013, *Mon. Wea. Rev.*) is structurally the RMM with
**χ200 — 200 hPa velocity potential — replacing OLR**, combined with U850 and U200. It was
designed for this situation: velocity potential captures the divergent circulation that
deep convection drives, and it is a *dynamical* field rather than a radiative one.

**It is computable from the store — verified.** The 20260820 N320 store carries 124
variables over 14 pressure levels, and all three VPM inputs are present:

```
u_200  PRESENT      v_200  PRESENT      u_850  PRESENT
```

χ200 follows from the 200 hPa wind: form the divergence δ = ∇·V₂₀₀, then invert
∇²χ = δ. `grid_ops.py` already differentiates on the reduced Gaussian grid for relative
vorticity — exact circular differences along each latitude row, with a precomputed
nearest-longitude index map between adjacent rows — and divergence is the same operator
with the components exchanged. **The Poisson inversion is the new piece, and it is not
written.**

That is a real change of position: `ttr` is outside the checkpoint's output space and can
never be added by any amount of work, whereas χ200 needs only code.

### 4.2 What VPM does *not* solve

Two things, and both belong in front of any decision to adopt it:

**(a) The EOF basis is not distributed.** AI-WQ ships `WH04_combinedEOFs.nc` and
`WH04_RMM_stddevs.nc` (`retrieve_MJO_projection_data`) — the Wheeler & Hendon
**OLR/U850/U200** basis. The package has no VPM equivalent. A VPM projection needs
Ventrice's published EOFs or a basis computed from ERA5 — and §2 of the old blocker list
already records why self-computed EOFs are the commonest cause of an RMM that will not
reproduce the official files: they differ in sign and mode order, silently rotating every
phase.

**(b) AI-WQ scores against RMM, not VPM.** The observed truth is the WH04 RMM phase
(`MJO_processing.compute_20yr_MJOprob_climatology` consumes an RMM phase time series). VPM
phases track RMM closely but not identically, and the disagreement cases are real rather
than hypothetical. A VPM forecast graded against RMM truth therefore carries a
systematic phase error that ensemble skill cannot remove. It must be **measured** against
the observed RMM record before submission, not assumed small.

So VPM converts an **impossible** problem — no OLR, ever — into a **bounded but
unfinished** one: write the Poisson inversion, obtain or build the basis, quantify the
VPM-vs-RMM phase offset.

### 4.3 The 120-day filter still bites — and the full corpus helps here too

Step 4 removes the preceding 120-day mean. A 0–792 h store holds 33 days, so the trailing
window still gaps between init and the forecast days, and ERA5 up to init is still
required (`--lowfreq`). The full O96 corpus shrinks the gap — days 1–17 are present rather
than absent — but does not close it. Same conclusion as §2 from another direction: the
full corpus is **necessary** for MJO, and **not sufficient**.

### 4.4 The options, ranked honestly

| option | verdict |
|---|---|
| **RMM from the store** | **impossible** — `ttr` is outside the checkpoint's output space, and ERA5 cannot substitute because the forecast days are in the future |
| **VPM (χ200 + U850 + U200)** | **the only route that yields a real index from these fields.** Needs the Poisson inversion (code), an EOF basis (external), and the VPM-vs-RMM offset (research) |
| Statistical OLR emulator — fit `ttr` on `hcc`/`tcc`/`cp`/`tp`/`tcw` in ERA5 | keeps the **official** WH04 basis intact, which VPM cannot. More work, and an emulator to validate, but the only path to a *true* RMM |
| Wind-only truncated projection | **not an option** — not a valid projection; `mjo_index.py` refuses it and renames every output `windproxy_*` with `submittable = NO` |
| Do not submit MJO | the honest default until 4.2 is resolved |

---

## Not done

- No `ttr` in the store, so no true RMM without an emulator (§1).
- χ200 is derivable, but the divergence + Poisson inversion is **not written**.
- No VPM EOF basis obtained or built; the VPM-vs-RMM phase offset is **not measured**.
- No EOF / climatology / low-frequency reference files are bundled here — they are
  observational products and belong with the AI-WQ package data, not in this repo.
- `mjo_index.py` has run only against `20260730`, and only as far as step 5 (normalised
  band anomalies, `--dump-bands`).
- The O96 full-corpus route (§2) is reasoned from the store schema and the submission
  checks, **not exercised** — no MJO product has been built from `icechunk_o96`.

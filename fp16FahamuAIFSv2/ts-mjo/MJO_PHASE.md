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

## 5. VPM is now implemented — everything except the basis

Built and verified 2026-09-13. `vpm_index.py` runs the whole VPM pipeline on the store;
`velocity_potential.py` supplies the one genuinely new capability.

```bash
$PY vpm_index.py --store /tank/projects/aifs-run/<DATE>_0000/icechunk_o96 \
    --tag cycle-<DATE>_0000 --init <DATE> --dump-bands vpm_bands.npz
# add --eofs VPM_EOFs.npz to finish; without it the script stops after step 8
```

**~2 min for 50 members** over the full 132-step corpus.

### How chi200 is obtained

`chi` solves the Poisson equation on the sphere, `laplacian(chi) = D`. The implementation:

1. **`grid_ops.divergence()`** — new, on the **native reduced Gaussian grid**. The same two
   operators as `relative_vorticity` with `u`/`v` exchanged and the sign flipped, now shared
   between the two so they cannot drift apart.
2. **regrid `D` only** to a regular 1.5 deg grid — one field per step instead of two, because
   `meridional_band` already works on the flat native vector, so `U850`/`U200` never need a
   regular grid at all.
3. **`velocity_potential.solve_poisson_sphere()`** — a real FFT in longitude is *exact* on a
   periodic grid, turning the problem into one tridiagonal system per zonal wavenumber. Cell-
   centred latitudes keep `cos(phi)` non-zero, and the half-level cosines vanish at the poles,
   which imposes no-flux automatically. `m = 0` is singular (chi is defined up to a constant),
   so it is pinned and the area-weighted mean removed — physically meaningless and invariant
   under every downstream step.

Regridding is not a compromise here: VPM consumes chi200 as a cosine-weighted mean over
+-15 deg reduced to 144 longitudes, and the MJO is zonal wavenumber 1–3. Nothing that survives
that averaging is resolution-limited at 1.5 deg.

### Verification

| test | result |
|---|---|
| Poisson vs analytic `Y_1^1`, `Y_1^0`, `Y_2^2`, `Y_3^1` | rel. err **4e-5 … 4e-4** |
| round trip chi → divergent wind → `D` → solve → chi | rel. err **6.9e-4** (global and in-band) |
| `divergence` on a solid-body rotation (non-divergent by construction) | **exactly 0.00e+00**, while vorticity stays 6.3e-6 |
| `D200` global mean (mass balance) | **+4.5e-07 1/s** |
| `relative_vorticity` after the shared-operator refactor | unchanged: zonal mean −3.6e-08, p1/p99 ∓8e-5 |
| chi200 magnitude on the real store | **2.3e7 m²/s** — literature scale O(1e6–1e7) |
| chi200 tropical band, zonal spectrum | **k=1 dominant** — the Walker/MJO signature |

### What is still missing — and it is not the EOFs

An earlier version of this section said the blocker was "acquisition, not computation:
obtain the NOAA VPM EOFs". **That framing was too narrow**, and `vpm-mjo.md` is right to
push back on it.

The submitted object is `P(RMM phase = 0..8)`, not VPM1/VPM2. So the architecture is

```
AIFS -> chi200,U850,U200 -> [basis] -> state -> P(RMM phase | state) -> 9x4
```

and the `P(RMM phase | state)` step is **learned empirically**. That means **the basis does
not have to be NOAA's**. Any *fixed* 2-D projection of `[chi200, U850, U200]` serves as a
state representation provided the **same** basis is used for the historical calibration and
the forecast — the calibration absorbs the choice. Build the basis from ERA5 and the result
is a legitimate index; it simply must be called `AIFS-MJO` rather than VPM, because it does
not reproduce Ventrice's published preprocessing.

The labels are free. AI-WQ ships all three reference pieces:

| call | gives |
|---|---|
| `retrieve_daily_MJO_obs(date, password, phase_probs=True)` | **observed RMM phases** — the calibration labels |
| `retrieve_20yr_MJO_clim(...)` | climatological phase probabilities — the BSS reference |
| `retrieve_MJO_projection_data(...)` | WH04 EOFs — OLR space, unusable here |

So there is **no need to reproduce the competition's RMM calculation**, which also avoids
subtle mismatches in EOF sign, normalisation, filtering and phase convention.

**The one real dependency is a historical record of `chi200`, `U850`, `U200`** — needed once,
to build the basis and learn the conditional probabilities.

---

## 6. ERA5 without downloading it: ARCO-ERA5

Verified reachable from this box 2026-09-21, **over plain HTTPS, with no new dependency and
no credentials** — `gcsfs` is not installed and is not needed:

```python
import xarray as xr
U = ("https://storage.googleapis.com/gcp-public-data-arco-era5/ar/"
     "1959-2022-6h-240x121_equiangular_with_poles_conservative.zarr")
ds = xr.open_zarr(U, chunks={}, consolidated=True)          # lazy, opens in seconds
```

That dataset is an unusually good fit:

| | |
|---|---|
| grid | **240 x 121 = 1.5 deg** — the exact grid step 3a already produces |
| cadence | **6-hourly** — the store's cadence |
| levels | **50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000** — identical to the store's `q` levels |
| span | 1959-01-02 .. 2021-12-31 (92 040 steps) |
| fields present | `u/v_component_of_wind`, `temperature`, `specific_humidity`, `10m_u/v`, `mean_sea_level_pressure` |
| absent | `total_precipitation` (in the 0.25 deg product instead) |

### The chunking decides the cost — measure it before planning around it

```
u_component_of_wind   chunks (8 time, 13 level, 240, 121)   compressor: None
```

**All 13 levels sit in one chunk, uncompressed.** Selecting two levels therefore transfers
all thirteen, and a one-week slice touches 4 time-chunks. Measured: a request for `u,v` at
200 and 850 hPa for one week returned 13 MB of data after moving **~48 MB** over the wire, in
**185 s**.

Two consequences, both of which change how the routine should be written:

1. **Ask for every level you might want** — you are paying for all 13 regardless. Subsetting
   levels saves nothing and costs clarity.
2. **Budget on chunks, not on the size of the array you asked for.** A 20-year `u`+`v`
   calibration is roughly **90 GB of transfer**, not the 2.5 GB the selected array would
   suggest.

### Why this is still the right design

The 90 GB is a **one-off stream**, and nothing is kept:

```
ARCO-ERA5 (streamed once)  ->  basis + P(RMM|state) lookup  ->  a few MB on disk
                                                                      |
weekly run:  AIFS chi200/U850/U200  ->  state  ->  lookup  ->  9x4    v
             ZERO bytes of ERA5, no CDS account, no archive to maintain
```

Against the alternative — a CDS account, licence acceptance, bulk retrieval and a local ERA5
archive to keep current — this is strictly better: no credentials, no storage, and the weekly
path never touches ERA5 at all. The one-off is a background job, not an interactive wait.

If the 90 GB matters, a 10-year calibration halves it, and the MJO literature generally uses
20-40 years because the index is defined that way rather than because the conditional
`P(RMM | state)` needs it.

**Currency is the one limitation**: this product ends **2021-12-31**. Fine for building a
basis and a conditional lookup — both are climatological — but it cannot supply anything
about recent or current conditions.

### This route does *not* solve the TS requirement

Worth stating plainly, because the two targets look similar and are not. The TS tracker
detects cyclone centres and needs resolution comparable to the forecast it runs on
(N320, ~28 km). This dataset is **1.5 deg (~165 km)** — coarser than the O96 corpus, and far
too coarse to resolve a tropical cyclone.

ARCO's 0.25 deg product would match, but at 1440x721x37 levels the same chunk arithmetic puts
a 20-year, 8-variable stream near **1 TB**. That is not a weekly-routine problem, it is a
different project. See [`TS_STORM_DAYS.md`](TS_STORM_DAYS.md).

---

## 7. What remains, in order

1. **Stream ARCO-ERA5** for `u200, v200, u850` over the calibration period — background job,
   ~90 GB of transfer, nothing retained.
2. **Build the basis** — EOFs of `[chi200, U850, U200]`. Reuse `velocity_potential.py`
   unchanged: it takes `(..., nlat, nlon)` on a regular grid, which is exactly ARCO's layout.
3. **Learn `P(RMM phase | state)`** against `retrieve_daily_MJO_obs()` labels. `vpm-mjo.md`
   recommends a binned lookup with Dirichlet smoothing before any ML, shrunk toward
   `retrieve_20yr_MJO_clim()` where support is thin — interpretable and hard to overfit.
4. **Wire into `vpm_index.py`** — `--eofs` and the lookup; the pipeline already stops exactly
   where these plug in.

Steps 2-4 are determined work. Step 1 is a download that needs no permission.

### One caution `vpm-mjo.md` raises that our own results argue against

It advises against rolling AIFS past D+15 and propagating statistically to D+22/29 instead
(its §"There is a larger AIFS problem at D+22 and D+29"). The caution is reasonable a priori
— ECMWF documents AIFS-ENS v2 as a 15-day system. But we submit days 18-31 every week, and
cycle `20260709` scored **+0.071 / +0.055 / +0.106** on the official leaderboard: real skill
beyond climatology, well outside the documented horizon. The rollout holds up for the gridded
variables. Whether it holds for MJO specifically is **untested**, and worth testing rather
than assuming in either direction.

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

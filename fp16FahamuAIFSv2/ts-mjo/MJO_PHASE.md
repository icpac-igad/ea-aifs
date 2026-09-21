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
u_component_of_wind   chunks (8 time, 13 level, 240, 121)   compressor: Blosc(lz4, clevel=5)
```

*(Corrected: this section first recorded `compressor: None`. The arrays are Blosc/lz4, but
the ratio is only **1.16×** — 10.39 MB on the wire per 12.08 MB chunk — so the conclusions
below, which assume you pay nearly the raw size, all still hold. §7.1 has the measured rates.)*

**All 13 levels sit in one chunk.** Selecting two levels therefore transfers
all thirteen, and a one-week slice touches 4 time-chunks. Measured: a request for `u,v` at
200 and 850 hPa for one week returned 13 MB of data after moving **~48 MB** over the wire, in
**185 s**.

Two consequences, both of which change how the routine should be written:

1. **Ask for every level you might want** — you are paying for all 13 regardless. Subsetting
   levels saves nothing and costs clarity.
2. **Budget on chunks, not on the size of the array you asked for.** A 20-year `u`+`v`
   calibration is roughly **90 GB of transfer**, not the 2.5 GB the selected array would
   suggest. (§7.1 refines this against a measured 1.4 MB/s link: 30 years at stride 2 is
   ~57 GB and ~13 h. Point 1 above needs one caveat — you pay for all 13 levels *per read*,
   so asking for two levels in two separate calls pays twice. Read the variable once.)

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

## 6b. Run on three cycles — the signal is buried, and that is measurable

Run 2026-09-21 on `20260903`, `20260910`, `20260917`.

**From the O96 corpus, not the N320 sidecar.** The sidecar carries `t_200` but **not
`u_200`/`v_200`**, so it cannot supply chi200. This is the exact mirror of the TS target: MJO
is planetary-scale and 112 km is ample, where the TS tracker needs 28 km and the sidecar is
the only store that has it. Both targets are served, from different stores, for the same
reason — resolution matched to the quantity.

Normalisations are stable across cycles, which is the first sign the diagnostic is behaving:

| cycle | chi200 | U850 | U200 |
|---|---|---|---|
| 20260903 | 1.344e7 | 2.771 | 7.030 |
| 20260910 | 1.340e7 | 2.636 | 6.940 |
| 20260917 | 1.304e7 | 2.573 | 6.985 |

### The MJO is not visible, and the reason is the missing climatology

| cycle | k=1 power | k=1 drift | excursion over 34 d |
|---|---|---|---|
| 20260903 | **1.00** (others ≤ 0.21) | **+0.3 °/day** | +32° |
| 20260910 | **1.00** (≤ 0.17) | **−0.9 °/day** | −5° |
| 20260917 | **1.00** (≤ 0.24) | **+0.5 °/day** | −8° |

**The MJO propagates eastward at 4–8 °/day. This is stationary.**

That is not a defect in `velocity_potential.py` or `vpm_index.py` — both are verified (§5).
It is the expected consequence of steps 6 and 7 not being run. **Wavenumber-1 dominance in
*raw* chi200 is the Walker circulation**, the time-mean divergent overturning, which is
stationary by construction. The MJO anomaly is order 1e6 m²/s beneath a climatological mean
state of ~1e7 — precisely the 1.3e7 standard deviation measured above. The signal is present
and buried by roughly a factor of ten.

```
6. day-of-year climatology removed     [--clim]     <- removes the Walker mean state
7. preceding 120-day rolling mean      [--lowfreq]  <- removes low-frequency background
```

The script already says so on every run:

```
!! no --clim: anomalies are not referenced to an observed climatology, so these are NOT VPM-comparable
!! no --lowfreq: the preceding 120-day mean is not removed
```

Those warnings were correct. This section makes them quantitative.

### What this changes about the plan

§6 treated the ERA5 climatology as one of several remaining pieces and the **basis** as the
interesting blocker. That ordering was wrong. **The climatology is prior**: without it there
is no MJO anomaly to project, so no basis — NOAA's, ours, or anyone's — can recover a signal
from these fields. Acquiring the ARCO-ERA5 history is therefore not optional polish; it is
the step that makes the target exist at all.

It also means a useful intermediate test looked available before any EOF work: remove a
climatology, recompute the drift, and check it moves toward 4–8 °/day eastward.

**That test does not work, and §6c measures why.** It was run, and the reasoning behind it
was wrong in a way worth keeping on the record: a 34-day window cannot define its own MJO
anomaly, so an absent MJO and a broken pipeline leave the same incoherent residual. The
sentence this section originally ended on — "if the drift does not appear then something is
wrong upstream of the projection rather than in it" — **is false**, and §6c shows ERA5
failing that exact test eight times out of nine on data containing a documented strong MJO.

---

## 6c. The chain is correct — validated against ERA5, and the window is the problem

§6b's test was run, and then a control was run that §6b did not think to propose. The
control is what settled the question.

### First, the free version of §6b's test

Removing a stationary field (the forecast-window mean per longitude) from χ200 does move the
drift off zero — a stationary field cannot *create* propagation, so anything that appears was
already there. The wavenumber-1 amplitude falls from ~0.9 sd to ~0.33 sd, confirming that
about two thirds of the raw wavenumber 1 is the stationary mean state.

But the resulting drifts were **−18.1, +3.2, +12.1 °/day** across the three cycles. That is
not a measurement. The cause is the diagnostic: fitting the drift of a wavenumber-1 crest
needs phase unwrapping, and once the wave is weak the phase is noise, so the fit returns a
confident wrong number rather than a wide error bar. `mjo_propagation.py` replaces it with a
**lag-longitude correlation**, which never unwraps anything and degrades to a low correlation
instead.

### Second, the control: does the chain recover a *known* MJO?

`era5_vpm_clim.py --mode series` runs the identical chain — divergence → Poisson solve → ±15°
band mean → 144 longitudes — on ERA5 over **2011-10-01 … 2012-01-31**, the DYNAMO field
campaign, which contains documented strong MJO events. 123 days, 18 min of streaming.

| series | phase speed | lag-1 corr | |
|---|---|---|---|
| ERA5, 123 days, raw | **+6.54 °/day** | 0.90 | **eastward, squarely in the MJO band** |
| ERA5, 20–60 d bandpass | +8.97 | 0.98 | eastward, just above the band |
| ERA5, 20–80 d bandpass | +8.60 | 0.99 | eastward, just above the band |
| ERA5, days shuffled ×3 | *no coherent propagation* | — | the null control fires correctly |

**The chain is correct.** The Poisson solver, the polar handling, the band reduction and the
daily means recover the MJO from observations at the textbook speed, and the shuffled-day
nulls confirm the diagnostic is not biased toward finding eastward motion.

### Third: the forecasts, and why their numbers mean less than they look like

| cycle | phase speed | lag-1 corr |
|---|---|---|
| 20260903 | +9.85 °/day | 0.68 |
| 20260910 | +21.47 °/day | 0.50 |
| 20260917 | −10.26 °/day | 0.74 |

Scattered, and all with correlations well below ERA5's 0.90. The obvious reading is that
these cycles have no MJO. **That reading is not supported**, because the forecasts differ
from the ERA5 control in one more way: 34 days against 123.

So the ERA5 series — the *same data*, known to contain a strong MJO, measured at +6.54 °/day
over its full length — was chopped into 34-day windows and put through the identical
diagnostic:

| window start | 2011-10-01 | 10-11 | 10-21 | 10-31 | 11-10 | 11-20 | 11-30 | 12-10 | 12-20 |
|---|---|---|---|---|---|---|---|---|---|
| °/day | +11.9 | +10.1 | +9.9 | +10.4 | **+7.4** | +8.8 | +8.5 | **−9.7** | **−9.4** |

**One window in nine lands in 4–8 °/day. Two of them run westward.** The spread is −9.7 to
+11.9 on data whose 123-day answer is +6.5.

The three forecast numbers sit *inside that distribution*. They are therefore not evidence of
a missing MJO, and not evidence of a broken pipeline — they are what this diagnostic does to
a 34-day window regardless of what is in it.

### Why, and what it costs

The MJO period is 30–60 days. A 34-day window is about one cycle, so its own time mean
contains a large part of the oscillation: subtracting it removes the Walker cell *and* much
of the signal. ERA5's 123 days span three to four cycles, which is why it can self-reference
and the forecast cannot.

This is the concrete, measured statement of why the climatology is prior. It is not that an
anomaly is conceptually required — it is that **the forecast window is too short to supply
its own reference, and no amount of care downstream can recover what the window mean removed.**
An external climatology is the only way a 34-day series gets an anomaly at all.

### Two code fixes this work turned up

- **`vpm_index.py` mis-used `phase_from_pcs`.** It returns `(phase, amplitude)` and is already
  vectorised, but the `--eofs` branch mapped it element-wise and `np.stack`ed the result,
  silently building a `(member, day, 2)` array where a `(member, day)` phase was intended.
  It had never fired, because that branch needs a basis and there is none — it would have
  failed on the *first* run that had one. `mjo_index.py` at the equivalent line unpacks the
  tuple correctly; the two had drifted.
- **New: `mjo_propagation.py`** — the lag-longitude diagnostic, with the bandpass refusing a
  window shorter than twice its longest retained period rather than returning something that
  looks filtered and is mostly edge effect.
- **New: `era5_vpm_clim.py` / `run_era5_clim.sh`** — the ARCO-ERA5 stream, in `series`,
  `clim` and `combine` modes.

---

## 7. What remains, in order

1. **Stream ARCO-ERA5** for `u200, v200, u850` — **running now**; see §7.1 for the sizing.
   **This is step 1 on evidence, not by convention**: §6b shows the signal is invisible
   until a climatology is removed, and §6c shows the 34-day window cannot supply one itself.
2. **Build the basis** — EOFs of `[chi200, U850, U200]`. Reuse `velocity_potential.py`
   unchanged: it takes `(..., nlat, nlon)` on a regular grid, which is exactly ARCO's layout.
3. **Learn `P(RMM phase | state)`** against `retrieve_daily_MJO_obs()` labels. `vpm-mjo.md`
   recommends a binned lookup with Dirichlet smoothing before any ML, shrunk toward
   `retrieve_20yr_MJO_clim()` where support is thin — interpretable and hard to overfit.
4. **Wire into `vpm_index.py`** — `--eofs` and the lookup; the pipeline already stops exactly
   where these plug in.

Steps 2-4 are determined work. Step 1 is a download that needs no permission.

### 7.1 Collecting the climatology: the grid, the cost, and what was measured

**It does not need the O96 grid, and it never touches one.** That was the natural assumption
— the forecast corpus is O96, so a matching climatology sounds like it should be O96 too —
but the χ200 path regrids to a **1.5° regular grid before the Poisson solve** (`REGRID_DEG`
in `vpm_index.py`); O96 is only where the *forecast wind* happens to live. What a climatology
must match is the grid the solve happens on, and ARCO-ERA5 publishes exactly it:

```
ar/1959-2022-6h-240x121_equiangular_with_poles_conservative.zarr
```

240×121 is 1.5°, 6-hourly, levels 50…1000 including **200 and 850**, spanning 1959–2021. No
regridding, no `earthkit` call, no interpolation error on the climatology side at all. The
polar rows are *kept* rather than trimmed, because `vpm_index.py` keeps them — `cos φ` is
6e-17 there, not 0, so the `m²/cos φ` term pins χ≈0 at the poles. That is inelegant, but a
climatology must be computed the same way as the field it will be subtracted from, and the
±15° band is 75° away from either choice.

**The cost is set by bandwidth, and bandwidth does not improve with concurrency.** Measured
against the bucket: one chunk is 10.39 MB on the wire (12.08 MB raw, lz4 ratio only 1.16×),
and 8, 16 and 32 concurrent readers all returned **1.4 MB/s**. That is the link, not latency,
so there is no parallel speed-up to buy. End-to-end the pipeline runs at **2.18 s per
6-hourly step**.

Two things followed from measuring rather than assuming:

- Chunks are `(8, 13, 240, 121)` — **all 13 levels in one chunk**. So `.sel(level=200)` and
  `.sel(level=850)` on the same variable fetch *the same chunks twice*. Reading each variable
  once and slicing levels in memory cut a third off the job.
- `--stride` skips blocks, and blocks are counted from the start of each year, so without
  `--phase` every year would sample **the same calendar days** — half the calendar covered 30
  times and half never. The driver sets `phase = year % stride`.

| base period | stride | steps | wall clock | transferred | samples/calendar day |
|---|---|---|---|---|---|
| 1991–2020 | 1 | 43,832 | ~26 h | ~114 GB | ~30 |
| **1991–2020** | **2** | **21,916** | **~13 h** | **~57 GB** | **~15** |
| 1991–2020 | 3 | 14,611 | ~9 h | ~38 GB | ~10 |

Stride 2 over the WMO 1991–2020 normal is what is running. Thinning is safe here because the
climatology is **smoothed onto 3 annual harmonics** (7 parameters per longitude per field),
as MJO climatologies conventionally are — that does not need every calendar day sampled in
every year, and it fills any day that drew none.

**Nothing large is retained.** ~57 GB passes through memory; what lands on disk is one
~600 KB `.npz` per year and a ~1.3 MB climatology. Disk is at 97 GB free and this does not
touch it — a useful property, given that the standing constraint on this box is storage.

**It is restartable by year.** `run_era5_clim.sh` runs one year per invocation and skips
years already on disk, then folds them with `--mode combine`. That is not fastidiousness:
this box has lost three overnight rollouts to `apt-daily-upgrade` replacing glibc and
python3.12 underneath a running process, and a 13-hour single process is exactly the shape of
job that loses. A killed year costs ~27 min, not the run.

Two guards sit on the fold, because a climatology that is quietly wrong is worse than one
that fails: it refuses to fit if fewer than 4× the parameter count of calendar days carry
samples, and it refuses if the fitted curve has more than 3× the variance of the data it was
fitted to — a smoother cannot amplify, so if it does, it is extrapolating through gaps.

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
- **No VPM EOF basis obtained or built**; the VPM-vs-RMM phase offset is **not measured**.
  This is now the only structural gap: §6c validated everything upstream of the projection.
- No `P(RMM phase | state)` lookup (§7 step 3). Nothing has been submitted for MJO.
- The climatology is **running, not finished** — 1991–2020 at stride 2, ~13 h
  (§7.1). Until it lands, `vpm_index.py` still stops at step 8.
- The **120-day low-frequency filter** (§4.3) is still unsourced. The climatology stream
  gives the machinery for it but the preceding-120-day means are a separate product.
- No MJO result has been checked against `retrieve_daily_MJO_obs()` labels for any date.
  §6c validates the chain against ERA5 *propagation*, which is weaker than matching an
  observed RMM phase.
- The ERA5 control is **one 123-day window over one season**. It shows the chain recovers a
  strong MJO; it does not establish behaviour in a weak or inactive period, which would need
  a second control window chosen for the opposite reason.
- §6c's 34-day windows come from a single ERA5 season, so "1 in 9" is an illustration of
  the spread, **not a calibrated false-negative rate**.

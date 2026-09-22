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

## 6d. The climatology landed — and exposed that the two sides computed different quantities

The 1991–2020 climatology finished: 30 years, 5483 daily samples, **365/365 calendar days
covered**, 7–23 samples per day, smoothed on 3 annual harmonics. 20 MB retained.

Applying it found two bugs and one real inconsistency.

**`--clim` had never run.** It did `stacked[k] - z[k][None, :, :]`, subtracting a `(365, 144)`
calendar-day climatology from a `(member, 34, 144)` forecast — shapes that cannot even
broadcast. It now indexes by each forecast day's own date, through a `calendar_index()` shared
with the builder, because a climatology subtracted under a different day convention than it
was built with is silently wrong by a day in three years out of four.

**The forecast and the climatology were not the same quantity.** `vpm_index.py` took the
divergence on the native reduced Gaussian grid and regridded only `D` — justified in its
docstring on cost, since that halves the interpolation. `era5_vpm_clim.py` regrids `u,v` and
differentiates on the regular grid. Measured on one member, 20 steps:

| | native path / regular path |
|---|---|
| χ₂₀₀ band std | 1.60 |
| band correlation | 0.64 |
| k=1 amplitude | 1.38 |
| k=2 amplitude | 3.41 |
| **k=1 crest longitude** | **+22.8° mean offset, 36.9° std** |

The last row is decisive: the drift diagnostic reads the motion of the k=1 crest, and a 37°
path-dependent scatter is several days of propagation at 4–8 °/day. The cheaper path was
chosen on cost before the difference was measured, and the difference is not small.

`--div-path` now selects it and **defaults to `regular`**, matching the climatology, with a
warning if `--clim` is combined with `native`.

### The result, and what it does and does not settle

| cycle | stage | k=1 power | speed | lag-1 corr |
|---|---|---|---|---|
| 20260903 | raw | 0.86 | +9.85 | 0.68 |
| | clim, mixed paths | 0.71 | +1.97 | 0.67 |
| | **clim, consistent** | 0.70 | **+8.08** | **0.87** |
| 20260910 | raw | 0.88 | +21.47 | 0.50 |
| | clim, mixed paths | 0.76 | +9.22 | 0.51 |
| | **clim, consistent** | 0.67 | +12.95 | **0.79** |
| 20260917 | raw | 0.83 | −10.26 | 0.74 |
| | clim, mixed paths | 0.62 | −12.50 | 0.73 |
| | **clim, consistent** | 0.60 | −12.53 | **0.82** |

Two things improved and one did not.

**Magnitudes now agree with observations.** The forecast χ₂₀₀ anomaly std is 7.8/7.1/7.3e6
against ERA5's ~8e6; under the mixed paths it was 1.21e7.

**Coherence improved, and that is the real evidence the fix was right.** Lag-1 correlation went
0.68/0.50/0.74 → **0.87/0.79/0.82**, toward ERA5's 0.90. A physical field should be temporally
coherent; the inconsistent path was destroying that, and no amount of tuning the speed estimate
would have revealed it.

**The speeds are still scattered, and §6c says they must be.** +8.08, +12.95, −12.53 on 34-day
windows — and ERA5's own strong MJO gave +11.9 … −9.7 across nine such windows. **So these
numbers still cannot be read as MJO presence or absence.** 20260903 at +8.08 with corr 0.87 is
the most MJO-like of the three, and that is as much as a 34-day window supports saying.

### 6e. What the 120-day filter is for, and how much it actually matters

*(This replaces a claim made here first — that the 120-day filter "is now the next blocker,
ahead of the basis". That was asserted without measuring it, and it is wrong on both counts:
it is second-order, and it does not gate a submission. Measured below.)*

**What it does.** After the seasonal climatology is removed, RMM and VPM both subtract the mean
of the **preceding 120 days**. The purpose is to remove interannual and low-frequency
variability — principally **ENSO**, which shifts the Walker circulation persistently and
projects onto the same planetary-scale χ₂₀₀/U200 patterns the MJO lives in. Without it, an El
Niño reads as a large standing "MJO anomaly".

**Why a trailing mean and not a bandpass.** A symmetric 20–100 day bandpass is the better
filter but needs *future* data, so it cannot be used in real time or on a forecast. A trailing
120-day mean is **causal** — past data only — which is exactly why WH04 chose it. That is the
whole reason this awkward-looking step exists.

**How much it matters, measured** on the 30-year ERA5 band series already on disk, comparing
the climatology-removed anomaly against its own trailing 120-day mean:

| field | anomaly std | 120-day mean std | ratio | variance removed |
|---|---|---|---|---|
| χ₂₀₀ | 5.09e6 | 2.15e6 | 0.42 | **7.0%** |
| U850 | 2.05 | 0.96 | 0.47 | **10.8%** |
| U200 | 5.78 | 2.68 | 0.47 | **9.6%** |

The removed field is slowly varying and wavenumber-1 dominated (k1 = 0.54 of its power), which
is the ENSO/Walker signature the step is designed for — so it is doing what it should. But it
accounts for only **7–11% of variance**. It is a **refinement required for strict VPM
comparability, not a missing ingredient.**

**It is buildable now** — this was also checked rather than assumed. `ar/` stops at
2021-12-31, but ARCO's other trees are kept current: `co/single-level-reanalysis.zarr-v2`
returns valid data for **2026-05-06**, and
`raw/date-variable-pressure_level/2026/09/02/u_component_of_wind/` has all 20 levels. The
120 days before 20260903 are 2026-05-06 … 2026-09-02, so the window exists.

Cost, from `raw/`: 3 fields × ~134 days × ~37 MB per day-variable-level file ≈ **15 GB, ~3 h**
at the measured 1.4 MB/s. It is 0.25° hourly, so it must be conservatively regridded to 1.5°
and subsampled to 6-hourly to match the climatology — and §6d is the reason to take that
seriously rather than casually.

**Recommendation: not now.** It is 7–11% of variance, it does not gate a submission, and the
thing that does gate one is the **basis** (§7 step 2), which cannot be self-built without
producing a different index. Doing the filter first would be optimising an input to a
projection that does not exist yet. `--lowfreq` already exists in `vpm_index.py` and now
validates alignment, so this is a data-acquisition task that can be picked up unchanged later.

§6c's window limit is also **not** something the filter would fix: 34 days is too short to
resolve propagation regardless of how well the anomaly is referenced.

---

## 6f. NOAA PSL does not publish the VPM EOFs — but the index is enough

Checked, and the answer is a negative result plus a way round it.

**PSL does not publish the VPM EOF patterns.** Its
`ftp2.psl.noaa.gov:/Datasets.other/MJO/` has `eof1/` and `eof2/` directories, but
they are **OMI's**, not VPM's: 366 files each (one per day of year), 2448 values per
file = **144 longitudes × 17 latitudes**, a single OLR field. VPM would be one fixed
pair of 3×144 = 432-element vectors. The `MJO` note in that directory reads *"EOF
basis patterns. could be deleted most likely."* The `README` reads *"eof patterns.
See George Kiladis."* Neither is the VPM basis.

**But PSL publishes the VPM index itself** — [`vpm.1x.txt`](https://psl.noaa.gov/mjo/mjoindex/vpm.1x.txt),
daily VPM1, VPM2 and amplitude, **1979-04-30 … 2026-03-17**, 17,124 days.

That is enough, and it dissolves the objection this document has been making since
§4.2. The objection was never to fitting coefficients — it was that a *self-computed
EOF* is a different index with no published phase convention. So do not compute an
EOF: **fit the linear functional that reproduces PSL's published PCs from our own
band anomalies.** It inherits VPM's phase convention by construction, because it is
fitted to VPM's own series.

`vpm_basis_fit.py` does this. It is a **regression, not an EOF** — not orthonormal,
not variance-maximising, and it inherits our reanalysis, climatology and grid.
Nothing downstream needs otherwise; `vpm_index.py` only forms `x @ e1`, `x @ e2`.

### How well it works, out of sample

Fit on 1991–2010, tested on the held-out 2011–2020 (1828 days):

| | climatology only | + 120-day mean |
|---|---|---|
| VPM1 test *r* | 0.794 | 0.778 |
| VPM2 test *r* | 0.868 | **0.903** |
| amplitude test *r* | 0.626 | **0.697** |
| median phase error (active days) | 16.4° | **15.7°** |
| **correct octant (active days)** | **90.2%** | 89.9% |

**The octant hit rate is the number that matters**, because the AI-WQ MJO product is
exactly those nine categories: the right octant ~90% of the time on active days, from
a basis we do not have and fields the model was never asked to produce.

Amplitude is the weak component (*r* 0.63–0.70), and amplitude is what decides
**category 0, inactive** — so category 0 will be the noisiest of the nine.

### This also settles §6e empirically

The 120-day filter lifts amplitude *r* from 0.626 to **0.697** and leaves the octant
rate flat. So §6e's "7–11% of variance" understates its value: the filter buys
**amplitude**, which is precisely the weak link and precisely what the inactive
category depends on. §6e's recommendation to defer it stands on priority, but the
reason to build it is now specific and measured rather than "strict comparability".

(Measured with a crude trailing mean over the stride-2 gappy series, so a contiguous
one should do slightly better — another argument for the in-cloud stride-1 run, §7.2.)

### Who supplies what, and what the 90% therefore means

Three datasets are involved and they play different roles. Conflating them would
overstate the result, so this is the provenance explicitly:

| dataset | role here | do we touch it? |
|---|---|---|
| **NCEP/NCAR R1, 1979–2012** | what Ventrice et al. built the **real VPM EOFs** from | **no — never** |
| **NOAA PSL `vpm.1x.txt`** | the published **VPM index series**: the regression's *target*, and the source of the phase convention | yes, as labels |
| **ERA5 (ARCO)** | the **predictor fields** — χ₂₀₀, U850, U200 band anomalies — and the climatology removed from them | yes, as inputs |
| **AIFS-ENS 2.0** | the forecast the fitted map is finally applied to | yes, at inference |

So the chain is: *ERA5 fields → a functional fitted to reproduce → PSL's VPM index,
which was itself built from NCEP R1 fields we never see.*

**The claim that is supported:** our ERA5-derived band anomalies, mapped through a
functional fitted against PSL's published VPM, reproduce that index on held-out
years at *r* = 0.78 / 0.90, with 90% octant agreement on active days.

**The claim that is _not_ supported:** "we applied the VPM basis to our fields."
We never had the VPM basis. We have a different object — an ERA5-specific linear map
that happens to output numbers agreeing with VPM.

### Why the distinction has teeth

**Reanalysis differences are absorbed, not exposed.** Wherever ERA5 and NCEP R1 disagree
on tropical upper-level divergence — and the tropical upper troposphere is among the
least observationally constrained parts of any reanalysis — that disagreement is
silently folded into the fitted coefficients. The fit *cannot* distinguish "this is how
VPM weights χ₂₀₀" from "this is how ERA5 differs from NCEP R1 in χ₂₀₀". Both appear as
coefficient values. A genuine VPM basis would keep them separate; this cannot.

**That is not purely a drawback here.** AIFS-ENS 2.0 is *trained on ERA5*, so its output
lives in ERA5's representation. A map fitted on ERA5 is therefore better matched to this
forecast than the true NCEP-R1-based VPM EOFs would be. Fitting against ERA5 is the
right choice for this model — but that is a *different* justification from "we used
VPM's basis", and it should be argued on its own terms.

**The errors are correlated in a way we cannot see.** If ERA5 and AIFS share a bias
relative to NCEP R1, the fit absorbs it and the held-out score will not reveal it — the
held-out years are also ERA5. The 90% is agreement with PSL's index **over observed
days**, not forecast skill, and not independent validation against VPM's own construction.

**What would settle it**, in increasing cost: obtain the actual VPM EOF vectors from the
authors (PSL does not host them — §6f); or rebuild VPM from NCEP R1 directly, which is a
separate acquisition and a separate index computation; or verify against AI-WQ's own
`retrieve_daily_MJO_obs()` labels, which is the operationally relevant comparison and is
the cheapest of the three.

**Practical consequence for the product.** Anything shipped from this basis should be
labelled as a *VPM-like index fitted to PSL's VPM*, not as VPM. The distinction matters
if the forecast is ever scored against an official VPM or RMM series, or compared with
another group's VPM.

### What is still not established

- The fit is against **ERA5**; VPM was built on **NCEP R1 1979–2012**. The out-of-sample
  numbers above already absorb that, but they are agreement with PSL's index, not with
  a VPM basis applied to our fields — those are not the same claim.
- It has **not been applied to a forecast.** `vpm_index.py --eofs` will now run, but no
  cycle has been put through it, and §6c's 34-day window limit is untouched by any of this.
- The **VPM-vs-RMM offset is measured and applied** (§6h, §6i): **+166.3° with det = −1, a
  reflection**, via `--rmm-rotation`. It corrects a real mirror: `vpm.1x.txt`, and so our
  basis, ran **westward** at −7.6°/day; it now runs +7.6°/day. Measured against PSL's RMM*,
  **not** BOM's official RMM — re-measure before submitting.
- AI-WQ scores MJO against **RMM**, not VPM (§6h). Ceiling for a perfect VPM: 64% exact
  octant, 98.7% within one.
- **AI-WQ retrieval is blocked**: the ECBox token returns 403 "Not a Branca token". This
  blocks both the official-RMM offset check and any forecast verification. It does not need to be for a VPM
  product, but it does if anything is ever compared to RMM phases.

---

## 6g. The pipeline completes — and the forecast amplitudes are wrong, for a locatable reason

`vpm_index.py --eofs` now runs end to end on all three cycles and writes a correctly
shaped AI-WQ product: `(34 days, 9 categories)`, summing to 1 on every day. This is the
first MJO output this project has produced.

**It is not submittable.** The reason is worth the detail, because the failure is
specific and the diagnosis separates what works from what does not.

### The basis and the ERA5 side are validated

Pushing **ERA5 itself** through the fitted basis reproduces the published index's
distribution:

| | mean amplitude | P(amplitude < 1) |
|---|---|---|
| ERA5 through our basis | **1.20** | **0.42** |
| PSL published VPM | 1.26 | 0.38 |

So the basis is correctly calibrated, and the climatology, band reduction, χ₂₀₀ solve
and normalisation are all sound on observed data.

### The forecast side is not

| cycle | mean amplitude | P(amplitude < 1) | modal category, day 1 / 8 / 18 / 31 |
|---|---|---|---|
| 20260903 | 2.22 | 0.05 | 6 (1.00) · 6 (0.60) · 6 (0.58) · 5 (0.42) |
| 20260910 | 2.05 | 0.06 | 6 (0.84) · 5 (0.92) · 6 (0.52) · 5 (0.42) |
| 20260917 | 2.04 | 0.07 | 5 (1.00) · 5 (0.74) · 6 (0.74) · 5 (0.36) |

Amplitude ~2× too large, and **P(inactive) of 0.05 against an observed 0.38–0.42**. The
MJO is inactive about two days in five; this says one day in twenty. That single number
disqualifies the product — category 0 would be systematically starved.

The phases are also nearly stationary (stuck in 5–6), which §6c already predicts a
34-day window cannot resolve either way.

### Cause: a mean-state bias, measured

A wrong first guess is worth recording. The excess was initially attributed to U850/U200
being banded on the **native O96 grid** while ERA5 bands on the regular grid — plausible,
since U850 had the largest ratio (1.62×). Banding both on the regular grid **changed
nothing** (1.62 → 1.65): reducing to 2.5° bins already smooths away the grid difference.
`--band-path` was kept for consistency, but it fixed nothing.

The actual cause is a **systematic per-longitude offset** between the forecast mean state
and ERA5's calendar-day climatology:

| cycle | field | anomaly std | \|offset\|/std | std after removing the offset |
|---|---|---|---|---|
| 20260903 | χ₂₀₀ | 7.79e6 | 0.68 | **4.22e6** |
| 20260910 | χ₂₀₀ | 7.12e6 | 0.66 | 3.92e6 |
| 20260917 | χ₂₀₀ | 7.26e6 | 0.67 | 3.97e6 |
| 20260917 | U850 | 3.15 | 0.63 | 1.71 |
| 20260917 | U200 | 7.43 | 0.69 | 4.63 |

ERA5's own factors are **4.91e6 / 1.94 / 5.49**. So removing the offset takes the
forecast from **1.59×** ERA5's variance to **0.86×** — i.e. essentially all the excess is
a constant-in-time offset, not extra variability. The ratio is ~0.67 for every field in
every cycle, which is a bias, not noise.

### Why this was predictable, and what fixes it

Subtracting an **observed** climatology from a **model** field leaves the model's own
bias and lead-dependent drift in the anomaly. Operational MJO forecast indices therefore
remove a **model climatology at matching lead**, not an observed one. This pipeline
removes ERA5's, so the model's departure from ERA5 is being read as MJO signal — and
then divided by ERA5's standard deviations, which converts it directly into amplitude.

This is the **same pattern as the TS product**, where the useful climatology turned out
to be detector-native and model-derived rather than observational.

Two facts make it tractable:

- The offset is **stable across cycles** (0.66, 0.67, 0.68), so it is estimable from a
  small number of cycles — as the TS climatology was from five.
- It is **large and systematic**, so removing it should be most of the fix: forecast
  variance lands at 0.86× ERA5's, which is the right order.

The obstacle is seasonal coverage. Three cycles give a *September* model climatology.
A year-round product needs cycles spread through the year, or hindcasts — and the O96
corpus of purged cycles is gone, so this accumulates going forward rather than being
recoverable from what is on disk.

**Until then the MJO product should not be submitted.** A tempting shortcut — rescaling
amplitudes to match the observed inactive fraction — is not defensible: it would hide a
mean-state bias behind a variance correction, and the phases would still carry the
offset.

---

## 6h. AI-WQ scores against RMM, not VPM — and the offset is ~4 octants

Two findings, one of which would have silently ruined any submission.

### AI-WQ's MJO target is RMM

`AI_WQ_package.retrieve_training_data.retrieve_MJO_projection_data()` fetches
**`WH04_combinedEOFs.nc`** and **`WH04_RMM_stddevs.nc`**, and
`retrieve_daily_MJO_obs()` returns `phase` / `amplitude` from those observations.
So the verification target is **Wheeler & Hendon's RMM**. Our index is VPM,
adopted because the model cannot produce OLR (§1). **A VPM product is therefore
scored against an RMM truth**, and the correspondence between them is not a
detail — it is the ceiling on the whole approach.

### The convention offset is ~166°, and ignoring it is catastrophic

Comparing PSL's two published series directly — `vpm.1x.txt` against
`rmm_star_data.txt`, 7613 common days, 1979–2021 — the components do not line up:

```
VPM1 vs RMM*1   r = -0.804      <- sign flip
VPM2 vs RMM*2   r = +0.837
```

The best rigid rotation between the two 2-D series is **+166.3°, i.e. +3.70
octants**, after which components correlate **+0.822 / +0.866**.

| | same octant | within ±1 octant | median phase error |
|---|---|---|---|
| **without** the rotation | **11.5%** | 44.7% | 79.8° |
| **with** the rotation | **64.0%** | **98.7%** | **13.1°** |

Submitting VPM phases as if they were RMM phases would have been **wrong by
about four octants, systematically** — worse than climatology, and it would have
looked like a modelling failure rather than a convention error. `MJO_PHASE.md`
has flagged this offset as unmeasured since §4.2; it is now measured, and it is
nowhere near zero.

*(Recorded because it nearly went unnoticed: the first pass reported 11.5%
same-octant and read it as "VPM is a poor RMM surrogate". That conclusion was
wrong. The number was an unmodelled convention difference, and the check that
exposed it was correlating the components rather than comparing the derived
phases.)*

### The realistic ceiling

RMM\*'s components are not unit-variance (mean amplitude 0.885, P(amp<1) = 0.62);
rescaling by 0.704 gives mean 1.26 and P(<1) = 0.40, matching VPM's 1.26 / 0.38.
With that and the rotation applied:

**64% exact octant, 98.7% within one octant** is what a *perfect* VPM would score
against RMM. For a deterministic label that is mediocre; for the **probabilistic
nine-category product AI-WQ actually wants it is workable**, because the honest
response to a 64/98.7 split is to spread probability across adjacent octants
rather than to concentrate it — which our ensemble does naturally.

### Caveats

- `rmm_star_data.txt` is **PSL's** RMM realisation, not BOM's official RMM, which
  is what AI-WQ most likely distributes. The offset against the official series
  must be re-measured before anything is submitted — the *existence* and rough
  size of the offset is established, its exact value is not.
- The check that would settle it, `retrieve_daily_MJO_obs()`, is **blocked**: the
  ECBox token returns `403 {"message":"Not a Branca token"}`. Rotating it also
  unblocks real forecast verification against observed days.

---

## 6i. The offset applied — and it is a reflection, not a rotation

`measure_phase_offset.py` measures the transform and `vpm_index.py --rmm-rotation`
applies it. Getting it right needed one correction that is worth keeping.

### It is a reflection, and forcing it to be a rotation destroys the agreement

The first implementation guarded the orthogonal Procrustes solution with "keep it
a rotation, not a reflection" — flipping the sign when `det < 0`. That guard looks
like defensive hygiene and is **wrong here**:

| | angle | det | same octant | component *r* |
|---|---|---|---|---|
| forced to a rotation | +36.4° | +1 | **15.9%** | −0.731 / +0.824 |
| the actual solution | **+166.3°** | **−1** | **64.0%** | **+0.822 / +0.866** |

The reflection is not a numerical accident. The two PSL files carry **opposite sign
conventions**, and it shows up physically — median phase advance on active
consecutive days:

```
vpm.1x.txt        -7.56 deg/day     WESTWARD
rmm_star.txt      +6.84 deg/day     eastward
```

Same magnitude (360/6.8 ≈ 53 days, squarely the MJO period), opposite sign. **An MJO
index must advance eastward through phases 1→8**, so `vpm.1x.txt` is the mirrored
one — and because `vpm_basis_fit.py` was fitted against that file, **our index
inherited the mirror**. After the transform ours runs **+7.56 °/day**, eastward.

`measure_phase_offset.py` now **refuses to write** a transform whose output runs
westward. That check, not the correlation, is what makes a reflection safe to apply.

### Effect on the product

The modal categories move by about four octants, as expected:

| cycle | before (VPM convention) | after (RMM convention) |
|---|---|---|
| 20260903 | 6 · 6 · 6 · 5 | **7 · 6 · 7 · 8** |
| 20260910 | 6 · 5 · 6 · 5 | **6 · 7 · 7 · 7** |
| 20260917 | 5 · 5 · 6 · 5 | **8 · 7 · 7 · 7** |

The transform is orthogonal, so **amplitude is unchanged** — mean 2.0–2.2,
P(inactive) 0.05–0.07. §6g's bias is untouched, as it must be.

### What the phase progression now shows

Ensemble-mean PCs advance at **+1.44, +0.11, −0.19 °/day** across the three cycles,
against an observed +6.8. Near-stationary, and consistent with two things already
established: §6c's finding that a 34-day window cannot resolve propagation, and
§6g's mean-state offset, which is large enough to pin the PCs near a fixed
direction — which is also why the product concentrates 0.7–1.0 probability on a
single category at short lead. **A confident, nearly stationary phase is what a bias
looks like**, not what skill looks like.

So the ordering is unchanged: the offset had to be applied, and it was; but §6g
remains the blocker.

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

A naming bug in the first launch is worth recording, because it defeated exactly the property
the per-year split exists for. `np.savez` **appends `.npz`** when the path lacks it, so
`--out "$f.part"` wrote `$f.part.npz` and every `mv "$f.part" "$f"` failed. The band series
were correct, but the `[ -s "$f" ]` resume test never matched and the `era5_vpm_????.npz`
combine glob found nothing — a crash at year 20 would have redone all twenty. Partials now go
to a dotfile that cannot match the glob.

### 7.2 A cluster does not help; moving the compute does

The obvious response to a 12-hour job is to parallelise it. **That is the wrong lever here,
and the split is worth measuring before spending effort on it.** For one 80-step block:

| | |
|---|---|
| divergence | 0.09 s |
| Poisson solve | 0.08 s |
| band mean | 0.00 s |
| **compute total** | **0.17 s — 2 ms/step, 0.1% of wall clock** |
| **transfer** | **148 s — 85% of wall clock** |

Cores are already 99.9% idle. And the link does not respond to concurrency either: 8, 16 and
32 readers all returned 1.4 MB/s, so the constraint is bandwidth out of the building, not
latency and not parallelism. A Dask cluster, local or remote, multiplies the resource that
is not scarce.

The lever that does work is **moving the reduction to where the bytes already are.**
ARCO-ERA5 is in GCS, so compute inside GCP (Colab, or a VM in the bucket's region) reads it
at hundreds of MB/s, and only the *reduced* band series — a few tens of MB — crosses the slow
link. Same code, same answer, transfer cut by ~1500×.

`era5_vpm_colab.py` is that job. It is **generated** by `make_colab_bundle.py` from
`velocity_potential.py` and `era5_vpm_clim.py` rather than written by hand, so the cloud copy
cannot drift from the local one; the generated solver is verified bit-identical to the repo's
on random fields. Regenerate it whenever either source changes.

**And it buys more than speed.** Once bandwidth stops being the constraint the right setting
is `STRIDE = 1` — a *contiguous daily* series, which the local run cannot afford. That
matters because two remaining pieces need unbroken daily history and a stride-2 series cannot
supply it:

- the **120-day low-frequency filter** (§4.3), which is a rolling mean over preceding days;
- **building a VPM EOF basis**, which is computed on anomalies after both references are
  removed.

So the local stride-2 run can only ever produce the climatology. The in-cloud stride-1 run
produces the climatology *and* the input for §7 steps 2 and 3.

### A rebuild brief, for doing this properly elsewhere

[`ERA5_VPM_CLOUD_BRIEF.md`](ERA5_VPM_CLOUD_BRIEF.md) is a self-contained handover for an agent
with a cloud VM and no access to this machine: what the forecast stores contain, why VPM
rather than RMM, why the climatology is prior (§6c's numbers), what is wrong with each public
ERA5 copy, and a tiered rebuild plan.

Two findings from writing it are worth surfacing here:

- **ERA5's native grid *is* the AIFS N320 grid.** `co/single-level-reanalysis.zarr-v2` has
  542,080 points with latitude `89.78487690721863 … -89.78487690721863` and longitude
  `0 … 340` — identical to our `icechunk_n320_aiwq`. Not a coincidence (AIFS is trained on
  ERA5), but it means a native ERA5 archive needs **no regridding on either side** and every
  diagnostic can run through identical code on both. The catch: `co/` is surface-only, and
  pressure-level wind at native N320 is **not in ARCO** at all.
- **The record reaches 1940**, not 1980 — `raw/date-variable-pressure_level/` has directories
  from `1940/`, one file per date/variable/**level** (so no level penalty). It is 0.25°, not
  native. The long record matters for the **EOF basis**, not for the climatology: a normal is
  supposed to be a fixed reference, so 1991–2020 stays 1991–2020 and the long record is what
  conditions the basis.

### What does not need keeping

Caching the ~57 GB of raw wind locally is not worth it — and would not fit comfortably beside
97 GB free. The reusable artifact is the **band series**, 643 KB per year and ~19 MB for the
full period, and that is already what lands on disk. Raw wind would only be needed to change
the band definition itself (the ±15° window, the 144 longitudes, the two levels), which is
fixed by the VPM index definition and is not a free parameter.

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
- The climatology is **done** (§6d): 1991–2020, 365/365 calendar days, 20 MB. `vpm_index.py`
  now completes steps 5 and 7; steps 6 and 8 remain.
- The **120-day low-frequency mean is not built.** §6e measures it at 7–11% of variance and
  recommends deferring it: the **basis** is what gates a submission. The data exists through
  2026-09 (`raw/`, ~15 GB), so this is acquisition work, not a blocker.
- `era5_vpm_colab.py` (§7.2) is generated and its solver verified bit-identical, but it has
  **not been run** — the in-GCP speed-up is inferred from where the data sits, not measured.
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

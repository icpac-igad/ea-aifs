# From skill scores to understanding: S2S diagnostics on the AIFS-ENS Icechunk store

> *"The operational value of machine-learning weather prediction cannot be assessed
> through verification skill alone. Acceptance depends on support for the core functions of
> operational meteorology: **understanding, translating and communicating**. Current MLWP
> systems show strong performance but lack the interpretability, diagnostic depth, and
> stability required for trust in safety-critical contexts."*

This note takes that argument seriously for the East-Africa S2S problem and asks a concrete
question: **what does the per-cycle Icechunk store let a forecaster *understand* that the
AI-WQ submission pipeline throws away — and how would we compute it?**

The submission pipeline is a *skill* pipeline: it reduces the whole forecast to `tp/msl/2t`
regridded to 1.5° and scored as weekly quintile probabilities. Everything an
ingredients-based forecaster reasons with — moisture transport, jets, the vertical
structure, the large-scale modes — is discarded. **The Icechunk store keeps all of it.** It
is therefore the natural substrate for the "understanding / translating / communicating"
functions the talk says are missing.

---

## 1. What the store actually contains (verified)

Per cycle, `…/aifs-run/<date>_0000/icechunk_v2` (tag `cycle-<date>_0000`):

- **120 variables**, native **N320** (542 080 reduced-Gaussian cells — no regridding loss),
  **50 members**, chunked `(1, 1, values)` so both a single member and an
  all-member statistic are cheap.
- **Full 3D dynamics:** `u,v,w,t,z` on **14 levels** (1000→10 hPa), `q` on 13 (no 10 hPa).
- **Moisture:** `q` (3D), `tcw` (total column water = IWV), `2d`, `sp`.
- **Circulation at the surface:** `10u/10v`, `100u/100v`, `msl`.
- **Convection / cloud proxies:** `cp` (convective precip), `tp`, `tcc/hcc/mcc/lcc`,
  `ssrd/strd` (radiation — an OLR-adjacent signal).
- **Lower boundary:** `skt` (skin temp ≈ **SST over ocean**), `stl1/2`, `swvl1/2`, `sd`.
- **Ocean surface waves:** `swh, mwp, cdww, cos_mwd/sin_mwd`, the `hNNNN` partitions.

> ⚠️ **Two honest constraints, up front.**
> 1. **Only hours 432–792 (days 18–33) are stored** — we discard 0–432 h to save disk (a
>    submission-driven choice). Diagnostics that need *propagation from initialization*
>    (MJO phase evolution, AR life-cycle tracking from genesis) are therefore limited to the
>    day-18→33 window. If diagnostics become a primary product, store a **reduced full-window
>    field set** (e.g. just `u850,u200,tcw,z500,q*` every 12 h for 0–792 h) alongside the
>    current window — cheap relative to the 583 GB full store.
> 2. **No OLR and no SST *analysis*** in the set. Use `skt` over ocean as the SST proxy and
>    `strd`/`cp`/`tcw` as convection/OLR proxies; state the proxy wherever it is used.

Everything below is `xarray.open_zarr(session.store)` over a tag — lazy, dask-backed, so a
diagnostic sweeps all 50 members × 61 steps without loading 583 GB into RAM.

---

## 2. The talk's three functions, mapped to the store

| Function (talk) | What it needs | What the store provides |
|---|---|---|
| **Understand** | diagnostic depth, the *physics* not the score | full 3D state → IVT, moisture-flux convergence, jets, static stability, PV — the ingredients a forecaster reasons with |
| **Translate** | raw fields → forecaster objects & modes | AR objects, teleconnection indices (MJO/IOD/Somali-jet), weather regimes, analogues — derived *from* the store |
| **Communicate** | calibrated uncertainty, scenarios, provenance | 50 members → probabilities & spread; **immutable tags** → the exact state a product was made from is reproducible (the "stability/trust" the talk demands) |

The Icechunk design maps to **trust** specifically: every diagnostic is run against an
**immutable tag** (`cycle-<date>_0000`), `repo.ancestry()` gives provenance, and re-running a
product months later reproduces it bit-for-bit. That is the difference between a plot and an
auditable diagnostic — exactly the "stability required for safety-critical contexts."

---

## 3. Method 1 — Atmospheric rivers / tropical moisture plumes

**Why it matters for East Africa:** extreme rainfall over the Horn and the highlands is often
delivered by **tropical moisture exports** — filaments of high vapour transport off the
Indian Ocean and from the Congo basin. These are ingredients-visible long before a QPF
verifies, which is the whole point of the talk.

**The store has exactly the fields for the standard IVT diagnostic** (no proxy needed):

```
IVT = (1/g) ∫ q·u dp ,  (1/g) ∫ q·v dp        over p = 1000…50 hPa
|IVT| = sqrt(IVTx² + IVTy²)                     [kg m⁻¹ s⁻¹]
```

We have `q,u,v` on the shared 13 levels → a clean vertical integral. Detection then follows
the standard **magnitude-plus-geometry** rule (e.g. Guan & Waliser): `|IVT| > threshold`
(regionally tuned, ~250–500 kg m⁻¹ s⁻¹), connected object with length ≳ 2000 km and
length:width ≳ 2:1, and a coherent transport direction. `tcw` gives an independent IWV
cross-check.

**Ensemble translation:** run detection per member → an **AR probability field** (fraction of
50 members with an AR over each cell) and **landfall-timing spread** for the East-African
coast. That is a communicable, calibrated product, not a single deterministic streak.

Sketch (runnable against the store):

```python
import icechunk, zarr, xarray as xr, numpy as np
LEV = [50,100,150,200,250,300,400,500,600,700,850,925,1000]   # shared q/u/v levels
repo = icechunk.Repository.open_or_create(
    icechunk.local_filesystem_storage("…/20260730_0000/icechunk_v2"))
ds = xr.open_zarr(repo.readonly_session(tag="cycle-20260730_0000").store,
                  consolidated=False, zarr_format=3)          # lazy, all 50 members

g = 9.80665
q = xr.concat([ds[f"q_{l}"] for l in LEV], "plev").assign_coords(plev=LEV)
u = xr.concat([ds[f"u_{l}"] for l in LEV], "plev").assign_coords(plev=LEV)
v = xr.concat([ds[f"v_{l}"] for l in LEV], "plev").assign_coords(plev=LEV)
dp = xr.DataArray(np.gradient(np.array(LEV)*100.0), coords={"plev":LEV}, dims="plev").abs()
IVTx = (q*u*dp).sum("plev")/g
IVTy = (q*v*dp).sum("plev")/g
IVT  = np.hypot(IVTx, IVTy)                                     # (member, time, values)
ar_prob = (IVT > 250).mean("member")                           # AR probability per cell/step
```

(Detection/geometry + coastline masking to N320 cells is the next ~40 lines; the physics
above is the part the store uniquely enables.)

---

## 4. Method 2 — ENSO / IOD fingerprints (the boundary-forced modes)

**Be precise about what a single 40-day ensemble can and cannot say.** ENSO and the Indian
Ocean Dipole are *slowly-varying, ocean-forced* modes. One forecast cycle **cannot establish
the ENSO/IOD state** — that lives in the initial ocean/atmosphere state the model was handed.
What the store *can* show is the **atmospheric fingerprint** the forecast carries, with
ensemble confidence:

- **IOD is the dominant driver of the East-African short rains.** Its atmospheric signature
  is in the **equatorial Indian-Ocean low-level zonal wind** (`u_850` averaged 5°S–5°N,
  ~60–90°E — easterly anomaly = positive IOD, suppressed rains) and the **west-minus-east
  SST-proxy gradient** using `skt` over the two IOD boxes. Both are in the store.
- **ENSO/Walker-cell fingerprint:** the **200 hPa divergent circulation**. Compute velocity
  potential χ from `u_200,v_200` (Helmholtz split); an eastward-shifted divergent centre over
  the Pacific with subsidence over the Maritime Continent is the El-Niño signature that
  teleconnects to East Africa. `skt` over the Niño-3.4 box gives the SST-proxy anchor.
- **Translate to a scenario, not a label:** across the 50 members, report the *distribution*
  of the IOD-wind index and the χ200 pattern — "N of 50 members carry a positive-IOD /
  drying circulation over the Horn," which is a communicable risk statement.

**What is missing and how to close it:** a proper index needs an **anomaly baseline**. We
only hold the AI-WQ 20-yr climatology for `tp/tas/mslp`. For `u850/χ200/skt` build a **model
climatology** by accumulating these few fields across cycles (cheap), or difference against
**ERA5**. Until then these are *relative* fingerprints within the ensemble, and should be
labelled as such — the talk's point about not overclaiming.

---

## 5. Method 3 — Large-scale / synoptic patterns over East Africa

The store supports the standard synoptic-to-planetary ingredient set for the region:

| Feature | Fields in store | Diagnostic |
|---|---|---|
| **MJO** (top S2S driver) | `u_850`, `u_200`, `tcw`/`cp` (OLR proxy) | project onto RMM-like EOFs; per-member phase/amplitude → an **MJO phase-probability fan**. *Caveat: robust propagation needs the full-window field set (§1).* |
| **Somali / East-African low-level jet** | `10u/10v`, `u_850/v_850` | cross-equatorial southerly flux 40–55°E; jet index time series + spread — the moisture pump for the long rains |
| **Tropical Easterly Jet, subtropical jet** | `u_200`, `u_150` | jet-core latitude/strength — modulates convection over the Sahel/Horn |
| **ITCZ / moisture convergence** | `q_850,u_850,v_850`, `tp`, `cp` | 850 hPa moisture-flux convergence → ITCZ position as an *ingredient*, not just a rain total |
| **Weather regimes** | `z_500` (+ `msl`) | k-means / EOF clustering of the ensemble into a handful of regimes → "which large-scale pattern, with what probability" |
| **Turkana / low-level channelling** | `z_700/z_850`, `10u/10v` | flow-terrain interaction (with `z`, `slor`, `sdor` constants) |

**The unifying "translate" product: weather regimes + analogues.** Cluster the 50 members'
`z500` (day 18–33) into K regimes, report the regime-occupancy probability, and for each
regime pull the **nearest historical analogue** from accumulated past cycles. That converts a
120-variable, 50-member, 61-step tensor into *"the atmosphere is most likely in regime B (Horn
ridge / suppressed convection), p≈0.6; the closest past analogue was <date>"* — a sentence a
duty forecaster can act on and challenge. This is precisely the understanding→translate→
communicate chain the talk says MLWP lacks.

---

## 6. Why the *Icechunk store*, specifically, makes this possible

1. **Diagnostic depth is retained.** The GRIB→submission path kept 3 of 120 variables; the
   store keeps all 120 at native N320. You cannot diagnose IVT or χ200 from `tp/msl/2t`.
2. **Ensemble-native.** `(1,1,values)` chunking → per-member detection *and* cross-member
   probability both stream cheaply; no 583 GB load.
3. **Reproducible & auditable.** Immutable **tags** + `ancestry()` mean every diagnostic
   pins the exact state it used — the "stability/trust for safety-critical" the talk names.
   A plot from six months ago re-renders bit-identically.
4. **One analysis substrate.** `xarray.open_zarr` over the tag is the same entry point for
   AR detection, regime clustering, and the AI-WQ regrid — no format-shifting between the
   skill pipeline and the understanding pipeline.
5. **Lineage to the score.** Because the *same* store feeds both the quintile submission and
   these diagnostics, a forecaster can ask "*why* did the model favour this quintile?" and
   answer it from the ingredients — closing the loop the talk says is open.

---

## 7. Suggested build order (honest, incremental)

1. **IVT / AR probability** (§3) — highest value, needs no new data, no climatology. Ship the
   `(member,time,values)` IVT field + AR-probability product first.
2. **Regime clustering on `z500`** (§5) — one flagship "translate" product; interpretable,
   communicable.
3. **Somali-jet & IOD-wind index time series** (§4/§5) — cheap, regionally decisive; label
   as *relative* until a baseline exists.
4. **Model climatology accumulation** — start storing `u850,u200,χ200,skt,z500,tcw` per cycle
   (a few MB) so §4 becomes true anomalies within a season.
5. **Full-window reduced field set** (§1 caveat) — unlocks MJO propagation and AR tracking
   from genesis.

Each step is a diagnostic that *adds understanding* over the quintile score — which is the
acceptance criterion the talk actually proposes.

---

*Grounded in the verified 20260730 store: 120 vars, 50 members, N320, hours 432–792. All
methods use fields confirmed present; every proxy (skt→SST, cp/strd/tcw→OLR) and every
limitation (window, baseline) is flagged where used.*

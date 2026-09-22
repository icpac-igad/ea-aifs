# ERA5 for the VPM MJO index: a rebuild brief

**Audience: an agent with a cloud VM, no access to this machine, and no prior context.**
Everything needed to build the dataset is here — what the forecast system produces, why the
MJO index needs ERA5 at all, exactly which fields, which grid, and what is wrong with the
existing public copies. Nothing below is assumed; measured numbers are marked as measured.

Companion documents in this repo, for depth rather than for doing the job: `MJO_PHASE.md`
(the index, §6c the validation), `TS_STORM_DAYS.md` (the other product from the same stores),
`vpm-mjo.md` (the literature review).

---

## 1. What the forecast system is, and what it emits

**AIFS-ENS 2.0**, ECMWF's data-driven ensemble, run locally on one GPU. 50 members, 6-hourly
steps, rolled out to **792 h (33 days)** — well past the 15 days ECMWF documents, which is
deliberate: the target is the sub-seasonal AI Weather Quest, whose windows are forecast days
18–31.

Output is written to two Icechunk (Zarr) stores per cycle. Both are on **reduced Gaussian**
grids, stored as a flat vector of points with `latitude`/`longitude` companion arrays — not a
2-D lat/lon mesh:

| store | grid | points | data arrays | steps | size (measured) |
|---|---|---|---|---|---|
| `icechunk_o96` | O96 octahedral, ~112 km | 40,320 | **120** | all 132 (0–792 h) | 164 GB |
| `icechunk_n320_aiwq` | **N320, ~28 km** | **542,080** | 10 | last 61 (432–792 h) | 53 GB |

Array shape is `(member, step, points)` = `(50, 132, 542080)`, chunked `(1, 1, npoints)` —
one member, one step per chunk. `float32`.

The split exists because a full-N320 corpus is 583 GB per cycle and this box has 2 TB total.
The O96 corpus carries every variable at coarse resolution; the N320 sidecar carries only what
genuinely needs 28 km, over only the lead range that gets submitted.

**The N320 sidecar's 10 variables:**

```
10u  10v  2t  msl  t_200  t_300  t_500  tp  u_850  v_850
```

`t_200/t_300/t_500` are there for the tropical-cyclone warm-core test; `u_850/v_850` for the
low-level wind. **`u_200` and `v_200` are absent.** That single fact is why §5 of this brief
matters.

The O96 corpus **does** have `u_200`, `v_200`, `u_850`.

---

## 2. The MJO index, and why it needs ERA5

### 2.1 The index is VPM, not RMM, and that is forced

The standard MJO index is Wheeler & Hendon's **RMM**: project `[OLR, U850, U200]`, each
reduced to a 144-point longitude band, onto published combined EOFs.

**AIFS-ENS 2.0 cannot produce OLR.** Its single-level output is all surface; there is no
top-of-atmosphere term, so `ttr` is outside the checkpoint's output space and cannot be added
by re-running inference. Dropping the OLR block and projecting the remaining 288-element
vector onto the surviving EOF rows is **not a valid projection** — the truncated vector is not
an EOF of the wind-only space, so the PCs are not RMM1/RMM2 and the published standard
deviations do not apply.

So the index used is **VPM** (Ventrice et al. 2013), an established RMM-like index in which
**200 hPa velocity potential replaces OLR**:

```
RMM :  [ OLR   , U850, U200 ]
VPM :  [ chi200, U850, U200 ]
```

χ₂₀₀ is *diagnosed exactly* from the forecast wind rather than statistically emulated, which
is why this is the one MJO index this model can support on its own output.

### 2.2 The pipeline

```
1. D    = divergence(u200, v200)                on the model's own grid
2. chi  : solve  laplacian(chi) = D             on the sphere
3. band : cosine-weighted mean over +-15 deg -> 144 longitudes (2.5 deg)
          for each of chi200, U850, U200
4. daily means from the 6-hourly steps
5. subtract the calendar-day climatology         <-- NEEDS ERA5
6. subtract the preceding 120-day mean           <-- NEEDS ERA5
7. normalise each field by its own std
8. project onto the VPM combined EOFs            <-- NEEDS ERA5 (or the published basis)
9. VPM1/VPM2 -> amplitude, phase -> 9 categories
```

Steps 1–4 and 7 are implemented and validated. **Steps 5, 6 and 8 all require an observational
record, and that is the entire reason ERA5 is needed.**

### 2.3 Why the climatology is not optional — this was measured, not assumed

Raw χ₂₀₀ is ~86% zonal wavenumber 1. That looks like the MJO and **is not** — it is the
stationary **Walker circulation**. ERA5's own raw χ₂₀₀ is 74% wavenumber 1, confirming this is
the real atmosphere and not a model artifact. The MJO anomaly is ~10⁶ m²/s beneath a mean
state of ~10⁷.

An obvious shortcut is to use the forecast window's own time mean as the reference. **It does
not work, and the failure is quantified.** Running the identical diagnostic on ERA5 over
2011-10-01…2012-01-31 (DYNAMO, documented strong MJO events):

| | phase speed | lag-1 corr |
|---|---|---|
| ERA5, full 123-day window | **+6.54 °/day eastward** | 0.90 |
| ERA5, 20–60 d bandpass | +8.97 | 0.98 |
| ERA5, days shuffled (null) | *no coherent propagation* | — |

The chain recovers the MJO at textbook speed. But the **same** ERA5 series, chopped into
34-day windows matching the forecast length, gives:

```
+11.9  +10.1  +9.9  +10.4  +7.4  +8.8  +8.5  -9.7  -9.4   deg/day
```

One window in nine lands in the MJO band 4–8 °/day; two run **westward**. The MJO period is
30–60 days, so a 34-day window's own mean contains much of the oscillation and subtracting it
removes the signal along with the Walker cell.

**Conclusion to carry forward: a 34-day forecast cannot supply its own anomaly reference. An
external climatology is not refinement, it is the thing that makes the target exist.**

---

## 3. Exactly what is needed from ERA5

Three fields, nothing else:

| field | level | used for |
|---|---|---|
| `u_component_of_wind` | 200 hPa | χ₂₀₀ (with v), and the U200 band directly |
| `v_component_of_wind` | 200 hPa | χ₂₀₀ |
| `u_component_of_wind` | 850 hPa | the U850 band |

- **Cadence:** 6-hourly is sufficient (the pipeline forms daily means). Hourly is wasted.
- **Coverage:** `u200`/`v200` must be **global** — the Poisson solve is elliptic, so χ at the
  equator depends on divergence everywhere. `u850`/`u200` are only read in ±15°, but they come
  from the same files, so there is nothing to gain by subsetting latitude.
- **Levels:** exactly 200 and 850. No other level is touched.

### 3.1 Period — 1940, not 1954, and the choice matters differently per product

ERA5 begins at **1940** (back extension 1940–1978, main record 1979–present). ARCO's `raw/`
tree confirms directories from `1940/`. So the archive extends further than the 1954 suggested
— but the three consumers want different things:

| consumer | period wanted | why |
|---|---|---|
| calendar-day climatology (step 5) | a fixed 30-yr normal, e.g. **1991–2020** | convention; a normal is *supposed* to be a fixed reference, and extending it does not make it better |
| 120-day low-frequency mean (step 6) | contiguous daily, any period | needs unbroken history, not length |
| **EOF basis (step 8)** | **as long as possible — 1940 on** | this is where 85 years pays: more independent MJO events, a better-conditioned basis |

So: build the long record for the **basis**, and derive the conventional normal as a slice of
it. Do not replace a 1991–2020 normal with a 1940–2024 one; compute both from the same store.

**Caveat to respect:** the 1940–1978 back extension is a separate, lower-confidence product
with sparser observations, and tropical upper-level wind is exactly where reanalysis is least
constrained before satellites. Keep the pre-1979 era **flagged in the store** so a consumer
can weight or exclude it. Do not silently blend it into a single array.

---

## 4. What is wrong with the public copies — measured

ARCO-ERA5 is at `gs://gcp-public-data-arco-era5`, readable over plain HTTPS as Zarr, no
credentials. Three trees: `ar/` (analysis-ready, regridded), `co/` (climate-optimised, native
grid), `raw/` (per-date netCDF).

### 4.1 `ar/1959-2022-6h-240x121_equiangular_with_poles_conservative.zarr`

240×121 = **1.5° regular, 6-hourly, 1959–2021**, levels including 200 and 850. This is what
the current local run uses. Two problems:

**Chunking is fatal for this use case:** `(8 time, 13 level, 240, 121)` — **all 13 levels in
one chunk**. Reading 2 levels transfers all 13, a **6.5× penalty**. Worse, `.sel(level=200)`
and `.sel(level=850)` on the *same* variable fetch the same chunks **twice** unless the
variable is read once and sliced in memory.

Compression is Blosc/lz4 but the ratio is only **1.16×** (10.39 MB on the wire per 12.08 MB
chunk), so you pay nearly raw size.

**It is regridded**, so it cannot be compared point-for-point with the N320 forecast store.

### 4.2 `co/` — native grid, and a significant find

`co/single-level-reanalysis.zarr-v2`: `values: 542080`, chunks `(1, 542080)`.

**Its grid is point-for-point identical to the AIFS N320 store.** Verified:

```
ERA5 co/  latitude (542080,)  89.78487690721863 .. -89.78487690721863   longitude 0.0 .. 340.0
AIFS N320 latitude (542080,)  89.784874        .. -89.784874            longitude 0.0 .. 340.0
```

**ERA5's native grid *is* AIFS-ENS 2.0's output grid** — both are N320 reduced Gaussian.
That is not a coincidence (AIFS is trained on ERA5) and it is the single most useful fact in
this brief: a native-grid ERA5 archive needs **no regridding on either side**, and every
diagnostic can run on ERA5 and on the forecast through identical code.

But `co/single-level-*` is **surface only** — no pressure-level wind.

`co/model-level-wind.zarr-v2` has `d` (divergence!), `vo`, `t`, `w` on **137 hybrid model
levels**, chunked `(1, 1, 410240)` — one level per chunk, which is the chunking we want. Two
obstacles before relying on it:

- `values: 410240` is **not** the 542,080-point N320 grid, and the store exposes **no
  latitude/longitude arrays at all**. The grid must be identified from ECMWF grid definitions
  before any of it can be used. *Do this first if pursuing this route — do not assume.*
- Hybrid model levels are not pressure levels. Getting 200 hPa needs surface pressure and
  vertical interpolation, which introduces its own error.

The prize if it works: **`d` is divergence directly**, so step 1 disappears, and on the native
spectral-adjacent representation the Poisson solve is far better conditioned than a finite
difference.

### 4.3 `raw/date-variable-pressure_level/`

`YYYY/MM/DD/<variable>/<level>.nc` — **one file per date, per variable, per level**, from
1940. Measured: 49.85 MB per file, which is 24 hourly steps × 721 × 1440 packed to int16.

**This is the best existing source for a rebuild**: per-level granularity means no 6.5×
level penalty, and it reaches 1940. But it is **0.25° regular, not native N320**, and hourly
(4× more than needed).

### 4.4 Why this is not a compute problem

Measured on the local box, one 80-step block:

| | |
|---|---|
| divergence | 0.09 s |
| Poisson solve | 0.08 s |
| band mean | 0.00 s |
| **compute total** | **0.17 s — 2 ms/step, 0.1% of wall clock** |
| **transfer** | **148 s — 85% of wall clock** |

And 8, 16 and 32 concurrent readers all returned **1.4 MB/s** — the limit is the link, not
latency or cores. **A cluster is the wrong tool.** The only lever is locality: run inside
GCP, in the bucket's region, where the same reads are hundreds of MB/s.

---

## 5. The rebuild: what to produce

The goal is a **rechunked, appropriately-chunked copy** that fixes §4.1's level penalty,
reaches back as far as the record allows, and — critically — is on the **native N320 grid** so
it matches the forecast store point-for-point.

### 5.1 Chunking rules, which is the whole point of rebuilding

1. **One level per chunk.** Never bundle 200 and 850 with eleven levels nobody reads.
2. **One variable per array.** `u` and `v` separately.
3. **Contiguous in time within a chunk**, ~24–48 steps, so a sequential year-long read is
   sequential on the object store. A contiguous daily series is required by step 6 and by any
   honest EOF basis; a strided/gappy store forecloses both.
4. Keep `(time, points)` for a reduced Gaussian store — the flat point vector is the native
   layout and matches the forecast store's `(member, step, points)` minus the member axis.
5. Compress properly. `ar/` gets 1.16× from lz4; zstd on float32 wind should do much better,
   and int16 packing with per-level scale/offset (what `raw/` already does) is another ~2×.

### 5.2 Three tiers — pick by what will actually be asked of it

Sizes are computed from 542,080 points × 4 B = 2.17 MB per field per step, 1460 steps/year:

| tier | contents | per year | 1979–2024 (46 y) | 1940–2024 (85 y) |
|---|---|---|---|---|
| **1 — reduced** | daily 144-longitude band series for χ₂₀₀, U850, U200 | **0.6 MB** | **29 MB** | **54 MB** |
| **2 — 1.5° global** | u200, v200, u850 on 240×121, rechunked per level | 0.5 GB | 23 GB | 43 GB |
| **3 — native N320** | u200, v200, u850 on 542,080 points | 9.5 GB | 437 GB | **807 GB** (≈400 GB packed) |

**Tier 1 is all the VPM index itself needs** — steps 5, 6 and 8 consume the band series, not
the fields. It is 54 MB for the entire 85-year record and can be attached to this repo.
**Produce tier 1 unconditionally.** Everything else is optional.

**Tier 3 is what unlocks §5.3**, and is the only tier that matches the forecast grid.

### 5.3 Why native N320 matters: adding `u_200`/`v_200` to the sidecar

The N320 sidecar has 10 variables and **lacks `u_200`/`v_200`** (§1). Consequences today:

- The MJO must be computed from the **O96 corpus** at ~112 km, not from N320.
- That is *acceptable* for MJO — it is a planetary-scale, wavenumber-1–3 phenomenon, and
  112 km is ample. The measured resolution sensitivity is the mirror of tropical cyclones,
  where 28 km is essential and O96 demonstrably degrades the wind maximum.

So adding `u_200`/`v_200` to the sidecar is **not** needed to make the MJO work. What it buys:

1. **One store serves both products.** Today TS reads N320 and MJO reads O96, so a cycle
   cannot be reduced to a single store without losing one.
2. **A point-for-point ERA5 comparison at 28 km**, because the grids are identical (§4.2) —
   no interpolation anywhere in the comparison.
3. **Headroom for anything else at 200 hPa** — upper-level divergence and outflow diagnostics
   are relevant to convective organisation generally, not only to the MJO.

**The cost, computed:** 2 arrays × 50 members × 61 written steps × 542,080 points × 4 B =
**13.2 GB per cycle**, taking the sidecar from 53 GB to ~66 GB (+25%). The rollout does not
need re-running for a *future* cycle — these are ordinary output variables and adding them is
a configuration change in the writer. Past cycles would need the O96 corpus, which has them.

**Recommendation — now done.** `u_200`/`v_200` were added to `run_local_icechunk_v2.DOWNSTREAM_VARS`,
which is also now the `--native-vars` default, so the next cycle picks them up without anyone
retyping the list. (They stayed missing for five cycles precisely because it was retyped.)
Original reasoning: It is 25%
more storage for the ability to retire the O96 corpus from the MJO path, and it makes the
ERA5 comparison exact rather than interpolated. Storage on the local box is the binding
constraint, so this is a real trade and not free — but 13.2 GB against a 583 GB full-N320
alternative is the cheap side of it.

### 5.4 Where to put the result

Tier 1 → this repo. Tier 2/3 → object storage, Zarr v3 + Icechunk to match the forecast
stores, public-read if possible so no credentials are needed on the consuming side (this is
exactly why ARCO is usable at all). `source.coop` is already used by this project for
published artifacts.

---

## 6. Concrete task list

1. **Identify the `co/model-level-wind` grid** (410,240 points, no coordinate arrays). If it
   resolves to a usable reduced Gaussian and `d` is trustworthy at 200 hPa after vertical
   interpolation, it is the cheapest native route and removes the divergence step. **If it
   does not resolve, say so and fall back** — do not guess a grid.
2. **Decide the native-N320 source for pressure-level wind.** It is *not* in ARCO (§4.2/4.3
   — `co/` is surface-only, `raw/` is 0.25°). Either CDS/MARS `retrieve` with
   `grid: N320`/native, or accept 0.25° from `raw/` and note the store is not natively N320.
   **State clearly in the output metadata which was done.**
3. **Build tier 1 for 1940–2024**, flagging pre-1979 separately (§3.1). 54 MB, and it makes
   the index work. Do this before tiers 2/3 whatever else is decided.
4. **Build tier 2 or 3** per §5.1's chunking rules.
5. **Derive and ship**: the 1991–2020 calendar-day climatology (smoothed on 3 annual
   harmonics), the 120-day low-frequency means, and a VPM EOF basis from the long record —
   noting that a self-computed basis is **not** VPM and has no published phase convention
   (see `MJO_PHASE.md`); the VPM-vs-RMM phase offset must be measured before any phase label
   is used.
6. **Validate by reproducing §2.3**: the DYNAMO window must give ~+6.5 °/day eastward with
   lag-1 correlation ~0.9, and shuffled days must give no coherent propagation. If a rebuild
   cannot reproduce that, the rebuild is wrong, not the reference.

## 7. Reference implementation in this repo

- `velocity_potential.py` — `solve_poisson_sphere` (real FFT in longitude, tridiagonal per
  zonal wavenumber), `divergence_latlon`. Verified against analytic spherical harmonics to
  4e-5…4e-4. Takes `(..., nlat, nlon)` on a regular grid.
- `grid_ops.py` — `divergence`/`relative_vorticity` **on the reduced Gaussian grid directly**,
  which is what a native-N320 store would use.
- `era5_vpm_clim.py` — the ARCO stream; `--mode series|clim|combine`.
- `era5_vpm_colab.py` — generated by `make_colab_bundle.py`, self-contained, for running
  **inside** GCP. Its solver is verified bit-identical to the repo's.
- `mjo_propagation.py` — the validation diagnostic of §2.3.
- `vpm_index.py` — the forecast-side pipeline; stops after step 7 without a basis, by design.

**One trap, load-bearing:** the 1.5° ERA5 grid includes the poles, where `cos φ = 6e-17`, so
the `m²/cos φ` term makes the diagonal enormous and the solver effectively pins χ≈0 there.
The forecast path does the same, and **a climatology must be computed the same way as the
field it will be subtracted from**. Do not "fix" one side alone. The ±15° band is 75° away
and cannot see either choice.

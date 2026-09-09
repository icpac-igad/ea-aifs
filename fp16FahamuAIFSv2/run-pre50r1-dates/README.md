# Running AIFS-ENS v2 for dates before IFS Cy50r1

**Question.** We wanted a v2 cycle for **12 February 2026** — a MAM 2026 forecast window
(days 18–33 → 2 Mar – 17 Mar). Step 1 failed immediately:

```
LookupError: 0 msgs for {'param': 'wmb', 'levtype': 'sfc', 'number': 1} (expected 1)
```

**Answer, first pass: it cannot be done, from any source.** Thirteen of the model's 112
input fields are outputs of IFS Cycle 50r1, which went operational with the 06 UTC run on
**12 May 2026 — jointly with AIFS v2 itself**. Before that date those fields were not
withheld from *open data*, they *did not exist* there. The earliest v2 cycle
initialisable from open data alone is **2026-05-13 00z**.

**That answer was wrong, and §2 now says why.** Both halves of it have since been
falsified by direct retrieval:

* The eight wave fields **are published**, in ECMWF research experiment
  [`j1r2`](https://apps.ecmwf.int/ifs-experiments/rd/j1r2/) (DOI
  [10.21957/1tzs-0946](https://doi.org/10.21957/1tzs-0946)) — the very ecWAM hindcast
  AIFS v2's wave inputs were trained on. All 13 wave params were **retrieved for
  2026-02-12**, on the N320 model grid.
* The 10 hPa level was never absent from **ERA5**. It was absent from the 0.25°
  *open-data dissemination set*, which is not the same statement. All five fields were
  **retrieved from CDS for 2026-02-12**.

So the *data* blocker is gone. What replaces it is a narrower and better question: a
substitute is only usable if the forecast built on it is. §3 answered that for the
substitutes available at the time and the answer was a hard no — but the 10 hPa
substitute it condemned (`l10copy`, 50 hPa carried up) is **1140× further from the truth
than ERA5** in input space (§2.4). That measurement has not been redone. **§5 says how.**

The point of this folder is to record what was checked. Including when the record was
wrong.

---

## 1. What is missing, and why

| group | fields | first in operational IFS | obtainable earlier from |
|---|---|---|---|
| Period-band significant wave height | `h1012` `h1214` `h1417` `h1721` `h2125` `h2530` | Cy50r1 | `j1r2` (§2.2) |
| Wave drag / bathymetry | `cdww` `wmb` | Cy50r1 | `j1r2` (§2.2) |
| **New 10 hPa level** | `t_10` `u_10` `v_10` `w_10` `z_10` | Cy50r1 | ERA5 (§2.3) |

"First in operational IFS" is the accurate claim; "first produced anywhere" is not. The
fourth column is what §2 established after this table was first written.

The wave bands are significant wave heights within fixed period bands — 10–12 s, 12–14 s,
14–17 s, 17–21 s, 21–25 s, 25–30 s. The AIFS ocean paper describes them as "the
decomposition of significant wave height into six distinct period bands for all waves
with periods larger than 10 seconds."

The 10 hPa level is easy to miss and matters more: the model card notes v2 has "14
pressure levels, now including a new 10 hPa level for improved stratospheric
representation." Stratospheric initial state is a recognised S2S predictability source,
so this is a more serious gap than the ocean-wave fields.

Measured cutover, by bisection over the open-data indexes — exactly consistent with a
12 May 06z implementation making 13 May the first complete 00z cycle. Read this table as
what it is: a statement about **open data**, not about the fields' existence anywhere:

| date | waef params | 10 hPa level |
|---|---|---|
| 2026-02-12 … 2026-05-12 | 5 | absent |
| **2026-05-13 →** | **13** | **present** |

## 2. Every source checked

| source | wave bands | `cdww` | `wmb` | 10 hPa | verdict |
|---|---|---|---|---|---|
| ECMWF open data `waef` (ensemble), pre-13 May | ✗ | ✗ | ✗ | ✗ | only `mwd mwp swh pp1d mp2` |
| ECMWF open data `wave` (deterministic), pre-13 May | ✗ | ✗ | ✗ | ✗ | identical 5 params — not an ensemble-only gap |
| ECMWF open data `enfo` / `oper` | ✗ | ✗ | ✗ | ✗ | no wave fields at all |
| MARS operational archive `class=od` | ✗ | ✗ | ✗ | **unreachable, not absent** | the IFS did not compute the wave fields before 50r1; the 10 hPa question is blocked on licensing, see §2.8 |
| ERA5 (CDS), wave fields | ✗ | ✓ | ✓ | — | Cy41r2 (2016); swell *partitions*, not period bands |
| **ERA5 (CDS), 10 hPa `t/u/v/w/z`** | — | — | — | **✓** | **retrieved for 2026-02-12; see §2.3** |
| **ECMWF research `class=rd expver=j1r2`** | **✓** | **✓** | **✓** | ✗ | **retrieved for 2026-02-12; see §2.2** |
| ERA5 on Google Earth Engine | ✗ | ✗ | ✗ | ✗ | atmospheric-only subset, no wave variables |
| ERA5 on AWS (`s3://nsf-ncar-era5`) | ✗ | ✗ | ✗ | ✗ | 8 streams, 754 keys checked — **zero** GRIB table-140 files |
| NOAA GEFS-Wave / GFS-Wave (WW3) | ✗ | ✗ | ✗ | ✗ | `SWELL 1/2/3` partitions; also no `cdww`/`wmb` |

### The wave stream was stable for two years, then stepped on one day

The `wave` (deterministic) param list, straight from the open-data indexes — independently
reproduced with `Herbie(model="ifs", product="wave")` for 2024-03-01, which returns the
same 5 params and the same 8 missing:

| date | n | params |
|---|---|---|
| 2024-03-01 | 5 | `mp2 mwd mwp pp1d swh` |
| 2025-03-01 | 5 | `mp2 mwd mwp pp1d swh` |
| 2026-03-01 | 5 | `mp2 mwd mwp pp1d swh` |
| 2026-05-12 | 5 | `mp2 mwd mwp pp1d swh` |
| **2026-05-13** | **13** | **+ `cdww h1012 h1214 h1417 h1721 h2125 h2530 wmb`** |

Two years unchanged, then a step change on exactly the Cy50r1 implementation date. In
**open data** there is no earlier date to fall back to and no partial availability to
exploit.

Reproduce the open-data half of this table for any date:

```bash
PY=/tank/projects/micromamba/envs/aifs-gpu/bin/python
$PY check_open_data_inputs.py --date 20260212      # -> v2 NOT runnable, v1 RUNNABLE
$PY check_open_data_inputs.py --scan 20260510 20260515   # -> finds the cutover
```

### 2.1 "But v2 was trained on ERA5 — surely the fields are in ERA5?"

No, and the reason turned out to be the thread worth pulling. ECMWF built a purpose-made
dataset rather than using ERA5, precisely because ERA5 lacks these fields. From the AIFS
ocean paper:

> "To train the additional wave-related fields, a **dedicated hindcast dataset covering
> the period 1979–2025** was produced using ECMWF's most recent wave model (ecWAM) and
> altimeter wave height Data Assimilation (DA) system at a resolution of approximately
> 9 km."

This folder previously concluded: *"That hindcast is not published… February 2026 is
after the hindcast ends and before 50r1 went live — plausibly a period for which nobody,
ECMWF included, has these fields."*

Both halves of that are wrong.

### 2.2 The hindcast **is** published, and it covers the gap

<https://apps.ecmwf.int/ifs-experiments/rd/j1r2/> — DOI
[10.21957/1tzs-0946](https://doi.org/10.21957/1tzs-0946):

> **TCo1279 CY50R1 wave hindcast using CY50R1 winds and ORAS6 currents 2025-2026 for AIFS
> training.** Standalone ecWAM wave model CY50R1 with altimeter wave height assimilation
> forced by CY50R1 e-suites and operational lwda 6-hourly stress equivalent neutral 10m
> winds, air density, gustiness and sea ice fraction AND surface currents from ORAS6.
> Native grid is Tco1279 (9km), 36 directions, 36 frequencies, current refraction. […]
> **!!! AIFS v2 was trained with**
> `140229/140230/140232/140233/140114/140115/140116/140117/140118/140119`

It is a **continuation of the training hindcast into the gap period** — same model, same
DA, same 9 km grid. Every paramId on that page resolved against `codes.ecmwf.int`:

| needed by AIFS v2 | paramId | in `j1r2` |
|---|---|---|
| `h1012 h1214 h1417 h1721 h2125 h2530` | 140114–140119 | ✓ |
| `cdww` | 140233 | ✓ |
| `wmb` | 140219 | ✓ (bathymetry — static, any date will do) |
| `swh mwd mwp` (exist pre-50r1 too) | 140229 / 140230 / 140232 | ✓ |

**All 8 missing fields are present**, and were retrieved live for the blocked date:

```
retrieve: class=rd, dataset=research, expver=j1r2, stream=wave, type=an,
          date=20260212, time=00/06/12/18, param=<13 params>, grid=N320
-> 52 fields retrieved, no errors    (13 params × 4 cycles)
h1012  0.453–0.705   h1214  0.258–0.471   h1417  0.169–0.321
h1721  0.076–0.162   h2125  0.026–0.117   h2530  0.012–0.066
swh 1.21–2.19   mwp 6.82–9.03   cdww 9.5e-4–1.6e-3   wmb 22.6–999
```

Monotonically decreasing with period band, as physics requires. Four practical notes:

* **MARS delivers `grid=N320` directly** — 542080 points, exactly the AIFS state grid. No
  local regridding, and no grid-ordering risk.
* **Coverage, measured by bisection** (a date outside it fails cleanly with
  `ERROR 89 (MARS_EXPECTED_FIELDS): Expected 1, got 0`):

  | probe | result |
  |---|---|
  | 2024-05-01 | ✗ (description gives the start as 2024-05-02) |
  | 2024-06-01 … **2026-06-20** | ✓ |
  | 2026-06-25 → | ✗ |

  **February 2026 is comfortably inside.** The archive stops around 2026-06-20 — after
  which operational 50r1 provides the fields anyway. The two sources overlap by ~5 weeks,
  which is what makes §5's validation possible.
* It is **deterministic** — one analysis, not 50 perturbed members. Every ensemble member
  would share it, so wave initial-condition spread is lost.
* Part of it is **on tape** (~294 MB per 52-field request, ~2 min). Pull whole days at a
  time.

### 2.3 The 10 hPa level was never absent from ERA5

The §1 table says "10 hPa: absent". That was measured in the **open-data index** and is
true there. It says nothing about ERA5, which has carried 10 hPa since 1940. Retrieved
from CDS for the blocked date with the token already in `.env`:

```
reanalysis-era5-pressure-levels, 2026-02-12 00z, pressure_level 10 -> successful
t 10 isobaricInhPa 20260212 0        w 10 isobaricInhPa 20260212 0
u 10 isobaricInhPa 20260212 0        z 10 isobaricInhPa 20260212 0
v 10 isobaricInhPa 20260212 0
```

Units need no conversion: `fp16_multi_run_AIFS_ENS_v2.get_input_fields` builds `z_<level>`
as open-data `gh` × g, i.e. geopotential in m²/s² — which is what ERA5 `z` already is.
`w` is Pa/s in both.

ERA5 arrives on a 0.25° regular lat-lon grid and must be regridded to N320 locally.
`fetch_era5_l10.py --validate` checks the point ordering rather than trusting it, by
pulling ERA5 **50 hPa** for the same cycle and correlating against the state's own `z_50`:

| field | r | bias | rms |
|---|---|---|---|
| `z_50` | 0.99997 | −40.0 m²/s² | 62.7 m²/s² |
| `t_50` | 0.99646 | −0.028 K | 0.699 K |
| `u_50` | 0.99599 | +0.025 m/s | 1.76 m/s |

Ordering confirmed. The residual is the real ERA5-vs-operational difference — a different
model cycle at a different resolution — and that residual is the thing §5 measures.

### 2.4 How good are the two donors, measured in input space?

This is the number that reopens the question. §3 condemned `l10copy` — 50 hPa carried up
to 10 hPa — because it was the only reconstruction available without a 50r1 analysis.
Compare both candidate donors against the *real* 50r1 10 hPa analysis, in input space,
on 2026-08-13 00z (no GPU time needed):

| field | ERA5 rms error | `l10copy` rms error | ERA5 better by |
|---|---|---|---|
| `t_10` | 0.946 K | 16.4 K | **17×** |
| `u_10` | 1.85 m/s | 15.1 m/s | **8.2×** |
| `v_10` | 1.68 m/s | 6.62 m/s | **3.9×** |
| `w_10` | 0.0087 Pa/s | 0.0264 Pa/s | **3.0×** |
| `z_10` | 88.7 m²/s² | 101 100 m²/s² | **1140×** |

`l10copy`'s `z_10` error is *three orders of magnitude* larger, which is not a subtle
point: 50 hPa and 10 hPa geopotential differ by ~1×10⁵ m²/s² by definition, so `l10copy`
was never a reconstruction of the field — it was a different field with the right name.
§3 measured what happens when you feed the model that. It did not measure ERA5.

**And the wave donor is close to exact.** `j1r2` overlaps operational 50r1 for ~5 weeks,
so it can be checked against the real thing. On 2026-06-04 00z, `j1r2` against the
operational analysis the pipeline actually used:

| field | operational scale | `j1r2` − operational rms | relative | r |
|---|---|---|---|---|
| `h1012` | 1.218 m | 0.070 m | 5.7 % | 0.9956 |
| `h1214` | 1.088 m | 0.070 m | 6.4 % | 0.9953 |
| `h1417` | 1.027 m | 0.073 m | 7.1 % | 0.9951 |
| `h1721` | 0.719 m | 0.057 m | 7.9 % | 0.9948 |
| `h2125` | 0.318 m | 0.031 m | 9.7 % | 0.9932 |
| `h2530` | 0.109 m | 0.016 m | 14.8 % | 0.9916 |
| `cdww` | 1.31e-3 | 7.4e-5 | 5.7 % | 0.9620 |
| `wmb` | 951.1 | 32.3 | 3.4 % | 0.9911 |

Every field correlates at **r ≥ 0.96**, most at ≥ 0.99. The error grows with period band
(5.7 % → 14.8 %) because the long-period bands are small and sparse — the absolute error
*falls*. This is a genuinely good substitute, not a stand-in.

One wrinkle: ecWAM's TCo1279 land-sea mask interpolated to N320 marks **29.3 %** of points
as land against the operational **30.3 %**, disagreeing on **5798** coastal points.
`fetch_j1r2.py --patch-pkl-dir` reconciles this to the operational mask, which is the
mask the model was trained to expect.

### 2.5 Pre-50r1 the wave ensemble has no spread at t=0 — so a deterministic donor costs nothing

The obvious objection to `j1r2` is that it is one analysis, not 50, so every member would
share it and the wave initial spread would be lost. Measured on the `waef` index directly,
that spread does not exist to lose:

| date | max \|`swh`(member n) − `swh`(member 1)\|, n = 2, 10, 25, 50 |
|---|---|
| 2026-02-12 (pre-50r1), t−6h and t | **0** |
| 2026-06-04 (operational), t−6h | 0.128 m |
| 2026-06-04 (operational), t | 0.066 m |

Distinct GRIB messages at distinct byte offsets, decoded independently — and bit-identical
values. Before Cy50r1 the wave ensemble is initialised from a **single** wave analysis and
its spread grows only from the perturbed winds; Cy50r1 is what gave the members perturbed
wave initial states. `mwp` and `mwd` behave the same way.

So substituting one deterministic `j1r2` analysis across all 50 members is exactly what
open data already does at these dates. **The deterministic-donor caveat applies only to
the ERA5 10 hPa level.**

### 2.6 The route that is now closed rather than open

The old §2 floated deriving the six period bands from ERA5 2D wave spectra, since the
band edges (10, 12, 14, 17, 21, 25, 30 s) fall on ERA5's 1.1-ratio spectral frequency
grid:

```
Hs_band = 4 · √( Σ_bins Σ_directions E(f,θ)·Δf·Δθ )
```

**Don't.** It was always expensive (tape-backed spectra on their own grid) and it would
have reconstructed a 9 km ecWAM product from a coarse 2016-cycle reanalysis. `j1r2` *is*
that 9 km ecWAM product, already computed, on the model grid, one MARS request away.

### 2.7 Credentials: two separate systems

This tripped up the first attempt and is worth stating plainly.

| | `.env` `ak=<uuid>` | `.ecmwfapirc` |
|---|---|---|
| issued by | [CDS profile](https://cds.climate.copernicus.eu/profile) | [api.ecmwf.int/v1/key](https://api.ecmwf.int/v1/key/) |
| gets you | ERA5, CAMS, C3S catalogue | MARS, `class=rd` research experiments |
| gets you **not** | `j1r2` | ERA5 (`Resource not found: datasets/era5`) |

Verified: the CDS token against `api.ecmwf.int` returns `403 Invalid token`; the ECMWF key
against `datasets/era5` returns `Resource not found`. You need **both**, and neither
substitutes for the other. `fetch_j1r2.py` reads `.ecmwfapirc` from this folder (falling
back to `~/.ecmwfapirc`); `fetch_era5_l10.py` reads `ak=` from `.env`.

### 2.8 Why not take the 10 hPa level from 49r1 itself?

The obvious question, and the better answer if it were available. ERA5 is a *different
model cycle at a different resolution*, and deterministic — the two caveats §4 still
carries. Operational 49r1 would have neither: same cycle as the other 107 fields, same
resolution, and `stream=enfo type=pf` would give the level **per member**, restoring the
stratospheric initial spread that the shared ERA5 level removes.

The 10 hPa level is very probably in MARS. Cy50r1 added it to the **0.25° dissemination
set**; the IFS model top is ~0.01 hPa and MARS has always archived more pressure levels
than are disseminated. So "absent from open data" says nothing about MARS.

It could not be tested here. The account reaches `datasets/research` (which is how `j1r2`
works) but not the operational archive:

```
class=od stream=oper type=an levtype=pl levelist=10 date=20260212
  -> User 'nkalladath@icpac.net' has no access to services/mars
class=od stream=enfo type=cf levtype=pl levelist=10 date=20260212
  -> User 'nkalladath@icpac.net' has no access to services/mars
```

**This is a licensing wall, not a data-availability finding.** Record it as unresolved, not
as closed. With a full MARS licence the first thing to try is
`class=od stream=enfo type=pf levelist=10 param=t/u/v/w/z number=1/to/50 grid=N320`
for a pre-50r1 date — if it returns, it supersedes ERA5 for this purpose and retires both
remaining caveats in §4. ERA5 is the best route *available to this account*, and it was
measured to work (14.8 % of the ensemble scale at day 5); it is not the best route that
could exist.

## 3. Does filling the gap actually work? (`wave_sensitivity.py`)

> **Scope note added after §2.** Everything in §3 is measured and stands. But every
> treatment here **invents** the missing fields — zeros, NaNs, or 50 hPa carried up. None
> of it uses a donor, because when it was run none was known to exist. So §3 answers
> "what does an invented field cost?" (answer: for 10 hPa, everything) and **not** "what
> does ERA5 cost?", which is a question with a 1140× smaller input error and no
> measurement yet. §5 runs it.

Rather than argue about it, measure it: take **20260813** (a date where all 13 fields are
real), and re-run member 1 with them replaced by what a pre-50r1 run would have to invent.

**Design note — why this is a paired test.** aifs-ens-2.0 is stochastic: two runs from the
same input differ by 2t rms ≈ 2.8 K. That would drown any signal. But the noise comes
from torch's global RNG, so seeding pairs the runs — at 24 h, same seed gives
max|d| = 0.16 K against 12.8 K for different seeds, an 80× separation. Every
configuration is therefore run under identical seeds, and two baselines make the numbers
readable:

* **floor** — `controlB` vs `control`, same seed: irreducible FP16/cuDNN
  non-determinism. Chaos amplifies it over a 33-day rollout, so at day 18–33 this, not
  zero, is the honest baseline.
* **ensemble scale** — `control` seed 0 vs seed 1: as different as drawing another member.

A fill is usable only if it sits near the floor *and* well below the ensemble scale.
Landing between them means the fill changes the forecast in a way one member cannot
separate from noise — which is not the same as harmless.

| mode | what it does |
|---|---|
| `control` / `controlB` | real fields (identical runs; their difference is the floor) |
| `zero` / `nan` | the 8 wave fields set to 0 / NaN |
| `wmb` | only `wmb` zeroed — the runner reports it as an unsupported coupled forcing |
| `l10copy` | 10 hPa level carried up from 50 hPa (the realistic reconstruction) |
| `l10zero` | 10 hPa level zeroed (pessimistic bound) |
| `pre50r1` | wave NaN + 10 hPa from 50 hPa — the honest best-effort pre-50r1 input |

```bash
$PY wave_sensitivity.py --out RESULTS --seeds 0,1 \
      --modes control,controlB,zero,nan,wmb,l10copy,l10zero,pre50r1
$PY analyse_sensitivity.py --dir RESULTS
```

### Result 1 (day 18–33, the product window): the 10 hPa level is fatal; the waves are unmeasurable

14 rollouts — 7 configurations × 2 seeds. 2t global rms against the control at the same
seed, as a multiple of the non-determinism floor and as a percentage of the ensemble
scale:

| treatment | 2t rms (K) | × floor | % of ensemble | reading |
|---|---|---|---|---|
| `controlB` — identical run | 2.86 / 3.06 | 1.0 | 88 % / 95 % | **the floor** |
| `wmb` alone | 2.96 / 3.16 | 1.0 / 1.1 | 92 % / 98 % | at floor |
| `nan` — 8 wave fields → NaN | 2.95 / 3.33 | 1.0 / 1.2 | 91 % / 103 % | at floor |
| `zero` — 8 wave fields → 0 | 3.31 / 3.15 | 1.2 / 1.1 | 102 % / 98 % | at floor |
| **`l10copy` — 10 hPa from 50 hPa** | **9.05 / 8.69** | **3.2 / 3.0** | **280 % / 269 %** | **far above** |
| **`pre50r1` — waves + 10 hPa** | **8.75 / 8.47** | **3.1 / 3.0** | **271 % / 262 %** | **far above** |

**This test is one-sided.** Because the floor sits at 85–97 % of the ensemble scale, it
cannot prove a perturbation is *safe* — anything near 100 % is simply below the
resolution. But it can prove one is *unacceptable*, and it does.

**The wave fields are at the floor.** All three wave treatments are indistinguishable
from re-running the identical input. Note this does not prove they are harmless; it
proves the day-18–33 single-member test cannot see them (Result 2 looks where it can).

**The 10 hPa substitution is not survivable.** Carrying 50 hPa up to 10 hPa — the
realistic reconstruction, and the only one available without a 50r1 analysis — puts the
forecast **~2.7× outside the model's own ensemble spread**. A merely decorrelated
forecast saturates at ~100 %; 270 % means a *systematic* shift, not chaotic divergence.
That is physically unsurprising: 50 hPa temperatures (~210–220 K) and winds are simply
wrong at 10 hPa, and the stratospheric bias propagates down over 33 days.

`pre50r1` ≈ `l10copy` (271 % vs 280 %) confirms the decomposition: **the stratospheric
level accounts for essentially the whole effect, and the wave fields add nothing
measurable.** The irony is that the run originally failed on `wmb`, a wave field — and
the wave fields turn out to be the part that does not matter.

A perturbation the model ignores producing the same signal as one that should matter is
the signature of **chaotic saturation**. The duplicate-control run proves it outright —
two **byte-identical** rollouts under the **same seed**, differing only by FP16/cuDNN
non-determinism, diverge to almost the full ensemble spread by day 18:

| var | floor (`controlB` vs `control`, same seed) | ensemble scale | floor / ensemble |
|---|---|---|---|
| 2t | 2.86 K | 3.23 K | **88 %** |
| tp | 0.00315 | 0.00324 | **97 %** |
| msl | 766 Pa | 904 Pa | **85 %** |

The floor is at the ceiling: every treatment (91–107 %) is indistinguishable from
changing nothing at all. The `wmb` run was included as an internal control and it did its
job — without it and without `controlB`, these numbers read as "the wave fields matter
enormously," which is the opposite of what they show.

Two consequences worth carrying beyond this experiment:

1. **A single-member day-18–33 difference is not a sensitivity measurement.** It measures
   chaos, whatever it is nominally comparing.
2. **Single-member AIFS-ENS forecasts at S2S range are not reproducible on the same
   hardware.** Bit-identical inputs and a fixed seed still diverge to ensemble spread. The
   ensemble is the product; no individual member is.

**Do not read a single-member day-18–33 comparison as a sensitivity measurement.** It
measures chaos. Two ways out: measure at short lead where the signal is still traceable
(Result 2), or compare *distributions* with many members per configuration (not done —
would take ~40 rollouts).

### Result 2 (6–120 h): everything separates, and the ranking is unambiguous

At 5-day lead the non-determinism floor collapses to **3 % of the ensemble scale** (2t
0.037 K vs 1.11 K), so every treatment is resolvable. 16 rollouts, 8 configurations ×
2 seeds:

| treatment | 2t × floor | 2t % of ensemble | msl × floor | verdict |
|---|---|---|---|---|
| `controlB` — identical run | 1.0 | 3 % | 1.0 | the floor |
| `wmb` alone | 8.6 / 8.8 | 29 % | 9.0 | **used after all** |
| `nan` — 8 wave fields → NaN | 12.5 / 11.6 | 42 % / 39 % | 26 / 22 | real, sub-ensemble |
| `zero` — 8 wave fields → 0 | 14.1 / 14.2 | 47 % | 23 / 22 | real, sub-ensemble |
| **`l10copy` — 10 hPa from 50 hPa** | **164 / 162** | **547 % / 541 %** | **371 / 377** | **catastrophic** |
| `l10zero` — 10 hPa → 0 | 237 / 236 | 792 % / 789 % | 306 | catastrophic |
| **`pre50r1` — waves + 10 hPa** | **164 / 163** | **548 % / 544 %** | **370 / 373** | **catastrophic** |

Three things this settles that Result 1 could not:

1. **The wave fields *are* used** — 12–14× the floor, ~40–47 % of the ensemble scale.
   They were not "unimportant" at day 18–33, merely saturated. Faking them costs
   something real, roughly half an ensemble member's worth of perturbation.
2. **`wmb` is used too** — 8.6–8.8× the floor on its own. The runner's
   `Coupled forcings are not supported by this runner: [... 'wmb' ...]` message refers to
   *updating* the forcing through the rollout, **not** to ignoring the field at input.
   Do not read that message as "this field is free."
3. **The 10 hPa level is ~12× more damaging than all eight wave fields combined**, and
   its best available substitute already produces **5.5× the ensemble spread at day 5**.
   `l10copy` (547 %) beats `l10zero` (792 %) as expected — carrying 50 hPa up is better
   than zeros — but both are far outside anything usable. `tp` is the least affected
   field (101 % of ensemble), consistent with a stratospheric temperature/height bias
   hitting `2t` and `msl` hardest.

`pre50r1` ≈ `l10copy` at both leads: the stratospheric level dominates completely.

## 4. Conclusion

**AIFS-ENS v2 can be run for dates before 2026-05-13.** The data blocker is resolved and
the accuracy question has now been measured.

All 13 Cy50r1-only fields are obtainable for 2026-02-12, from two sources, both retrieved
and verified here — and a full 112-field input state has been built from them (§5):

| what | from | input-space error | forecast cost at day 5 |
|---|---|---|---|
| 8 wave fields | `class=rd expver=j1r2`, `grid=N320` | r ≥ 0.96, 3–15 % rms | 17.7 % of ensemble |
| 10 hPa `t/u/v/w/z` | ERA5 on CDS, regridded to N320 | 3–1140× better than `l10copy` | 14.8 % of ensemble |
| **both, as actually built** | | | **20.4 % of ensemble, 5.6× floor** |

The old conclusion — *"AIFS-ENS v2 cannot be run for dates before 2026-05-13"* — rested on
the 10 hPa level having no acceptable substitute. That was true of the substitutes then
available and false of ERA5, whose `z_10` error is three orders of magnitude smaller than
the one §3 condemned. In forecast space the difference is **491 % → 14.8 %** of the
ensemble scale.

Two of the three risks flagged before the experiment were retired by measurement:

1. ~~ERA5's cycle/resolution mismatch may be fatal~~ — measured at 14.8 % of the ensemble
   scale at day 5, and at the floor at day 18–33. It is not.
2. ~~Deterministic donors lose the wave initial spread~~ — pre-50r1 the wave analysis has
   **no member spread at t=0** to lose (§2.5), confirmed independently in the pkls (§5).
   The caveat survives only for the ERA5 10 hPa level, which all 50 members share.
3. **`j1r2` covers 2024-05-02 → ~2026-06-20**, and operational 50r1 covers 2026-05-13 →.
   Together they leave no gap — but only because they overlap by five weeks. Outside that
   union the route does not exist, and the builder refuses the date rather than guessing.

What is still **not** measured, and should be stated whenever this output is used:

* The whole result rests on **one member, one date, two seeds**. It measures the mean
  offset a donor introduces, not the effect on ensemble *distribution* — which would take
  ~40 rollouts per configuration and has not been done.
* `tp` is the marginal field at 42 % of the ensemble scale; `2t` and `msl` are far inside.
* The 10 hPa level being shared across all 50 members removes stratospheric initial spread
  from the ensemble. At S2S range that is the spread most likely to matter, and this
  experiment cannot see it.

Honest status: **defensible to run, with those three caveats stated in whatever it feeds.**

## 5. What to do for pre-50r1 dates

### Today, for a product you have to ship: use AIFS v1

v1 predates the coupled wave inputs and the 10 hPa level, so it needs none of the missing
fields. Verified against the 12 Feb 2026 open-data index:

```
AIFS v1: RUNNABLE
    sfc     8/8      pl      6/6
    sol     1/1      levels 13/13
```

`s3_grib_pkl/s3_grib_pkl_input_aifsens.py` builds v1 inputs from open data (50 members,
IFS ENS), and `era5tFp16FahamuAIFSv1/` holds the downstream chain. Note the v1 runner is
separate from `run_local_icechunk_v2.py`, so O96 output would need wiring in.

### RESULT: the donor experiment has been run, and it passes

Executed at `20260604` exactly as specified below. 12 rollouts, 6 configurations x 2 seeds,
5-day lead. `2t` global rms against the control at the same seed:

| treatment | 2t x floor | 2t % of ensemble | msl % of ens | tp % of ens | reading |
|---|---|---|---|---|---|
| `controlB` — identical run | 1.0 | 3.6 % / 2.5 % | 1.4 % | 14 % | the floor |
| **`l10era5` — ERA5 10 hPa** | **4.1 / 4.1** | **14.8 %** | 8.7 % | 37 % | **usable** |
| **`wavej1r2` — j1r2 waves** | **4.9 / 4.9** | **17.7 % / 17.9 %** | 13.1 % | 39 % | **usable** |
| **`pre50r1b` — both donors** | **5.6 / 5.7** | **20.4 % / 20.6 %** | 14.1 % | 42 % | **usable** |
| `l10copy` — 50 hPa carried up | 135 | 491 % | 502 % | 104 % | catastrophic (control) |

**The positive control worked.** `l10copy` came back at 491 % against §3's 547 % on a
different date — close enough to confirm the harness measures the same thing §3 did, which
is what makes the other four rows trustworthy.

Against the decision table below, `pre50r1b` lands in the first row on both criteria:
**5.6x the floor** (threshold 10x) and **20 % of the ensemble scale** (threshold 40 %).

Three things this settles:

1. **ERA5 is a 33x better 10 hPa donor in forecast space**, not just input space: 14.8 %
   against `l10copy`'s 491 %. The 1140x input-space gap in `z_10` (§2.4) translates into a
   forecast that sits well inside the ensemble instead of 5x outside it.
2. **The 10 hPa level is no longer the dominant error term.** With `l10copy` it was ~12x
   the wave fields; with ERA5 it is *smaller* than the wave donor (14.8 % vs 17.7 %). The
   two donor errors are now comparable and add roughly in quadrature
   (√(17.7² + 14.8²) = 23 % against the measured 20.4 %), i.e. they are largely independent.
3. **The reconstructed input beats the fakes §3 measured.** `pre50r1b` at 20.4 % is better
   than §3's `nan` (39–42 %) and `zero` (47 %) wave fills — those only ever faked the 8
   wave fields and still cost more than sourcing all 13 properly.

`tp` is the marginal field at 42 % of the ensemble scale, consistent with §3 Result 2 where
`tp` was also the least well-behaved. Everything else is comfortably inside.

#### Product window (day 18–33): no red flag, and by construction no green one either

§5 requires the product-window leg before trusting a MAM-style forecast. 10 rollouts,
5 configurations × 2 seeds, `--lead-time 792 --write-hours 432-792`:

| treatment | 2t × floor | 2t % of ensemble | msl × floor | tp × floor |
|---|---|---|---|---|
| `controlB` — identical run | 1.0 | 88.7 % / 85.1 % | 1.0 / 1.1 | 1.0 |
| `l10era5` | 1.0 / 1.1 | 88.4 % / 94.4 % | 1.0 / 1.1 | 1.0 |
| `wavej1r2` | 1.0 / 1.0 | 91.8 % / 89.0 % | 1.2 / 1.1 | 1.0 |
| `pre50r1b` | 1.1 / 0.9 | 93.9 % / 83.7 % | 1.1 | 1.0 |

Every donor treatment sits at **0.9–1.2× the non-determinism floor**, and the floor is
itself at 81–96 % of the ensemble scale. This is §3's chaotic saturation exactly as
documented, so read the table the way §3 says to read it: **this test is one-sided.** It
cannot show a perturbation is safe, because nothing below ~100 % is resolvable here.

What it *can* do is fail, and it does not. `l10copy` sat at 271–280 % at this window in §3
— well clear of the floor — so a donor with that character would have shown up. None does.
The day-5 numbers above are what carry the information; this leg rules out the
`l10copy`-style failure at the product window and adds nothing else.

Raw rollouts: `/tank/projects/aifs-run/wave_sensitivity_20260604_donor_short/` (5-day,
12 runs) and `…_donor_long/` (day 18–33, 10 runs).

### The experiment, for reference and re-running

The test must run on a date where the **real** fields exist, so the donors can be scored
against ground truth. That requires a date inside `j1r2` (≤ 2026-06-20) **and** after the
50r1 cutover (≥ 2026-05-13). `20260604` satisfies both and already has input states.
Note this is *not* 20260813 — §3's date — which `j1r2` does not cover.

```bash
PY=/tank/projects/micromamba/envs/aifs-gpu/bin/python
D=/tank/projects/aifs-run/20260604_0000/input_states
export HF_HOME=/tank/projects/hf_cache HF_HUB_OFFLINE=1   # see note below

# donors (needs BOTH credentials -- see §2.7)
$PY fetch_j1r2.py     --date 20260604 --time 00 --out j1r2_20260604_00.npz
$PY fetch_era5_l10.py --date 20260604 --time 00 --out era5_l10_20260604_00.npz \
      --validate $D/input_state_member_001.pkl        # do not skip --validate

# short lead first: at day 18-33 the non-determinism floor hides everything (§3)
$PY wave_sensitivity.py --out RESULTS_donor_short --input-dir $D \
      --lead-time 120 --write-hours 6-120 --seeds 0,1 \
      --modes control,controlB,l10copy,l10era5,wavej1r2,pre50r1b \
      --era5-l10 era5_l10_20260604_00.npz --wave-donor j1r2_20260604_00.npz
$PY analyse_sensitivity.py --dir RESULTS_donor_short
```

`l10copy` is included deliberately as a **positive control**: §3 measured it at 547 % of
the ensemble scale at day 5, so if the new run does not reproduce roughly that, something
is wrong with the harness, not with the donors. It came back at 491 %.

**Set `HF_HOME` and `HF_HUB_OFFLINE`.** The checkpoint lives in the shared cache at
`/tank/projects/hf_cache` (the same export the `run_commands_*.md` files use). Without
`HF_HOME` the runner starts a fresh 2.4 GB download into `~/.cache`; without
`HF_HUB_OFFLINE=1` it still contacts the Hub to revalidate and can stall indefinitely on
unauthenticated rate limits, at `Fetching 14 files: 21%|██▏ | 3/14` with the GPU idle at
0 %. Both failure modes look like a hung job rather than an error.

Read the result against §3's yardsticks — at 5-day lead the floor is 3 % of the ensemble
scale, so everything is resolvable:

| if `pre50r1b` lands… | then |
|---|---|
| near the floor (≲ 10× floor, ≲ 40 % of ensemble) | comparable to what faking the wave fields alone costs — a pre-50r1 v2 run is defensible, with the deterministic-donor caveat stated |
| 50–150 % of the ensemble scale | one member is indistinguishable from another draw; usable only as an ensemble, and the lost spread must be quantified first |
| ≳ 300 % (i.e. anywhere near `l10copy`) | ERA5's cycle/resolution mismatch is fatal too — close the question for good |

If and only if the short-lead result passes, repeat at the product window
(`--lead-time 792 --write-hours 432-792`) before trusting a MAM-style day-18–33 forecast.
Both legs have been run; results above.

### To actually build a pre-50r1 cycle

`ecmwf_opendata_pkl_input_aifsens_v2_pre50r1.py` is a copy of the operational builder with
one change: the 13 fields open data lacks are taken from the donors instead. Everything
else — the other 99 fields, the member perturbations, the transforms, both retrieval
backends, the GCS upload — is untouched, so a pre-50r1 pkl differs from an operational one
in 13 fields and nothing else.

```bash
PY=/tank/projects/micromamba/envs/aifs-gpu/bin/python
$PY ecmwf_opendata_pkl_input_aifsens_v2_pre50r1.py       --date 20260212 --members 1-50 --no-upload --keep-local       --out-dir input_states_v2_20260212 --save-donors donors_20260212.npz

# donors are per-cycle, so reuse them rather than re-hitting MARS/CDS
$PY ecmwf_opendata_pkl_input_aifsens_v2_pre50r1.py --date 20260212 --members 1-50       --j1r2-npz donors_20260212.npz --era5-l10-npz donors_20260212.npz
```

**Verified for 2026-02-12** — the date that originally failed at Step 1:

```
donor fields (shared across all members) ...
  waves  <- MARS class=rd expver=j1r2 (grid=N320)
  10 hPa <- ERA5 on CDS (regridded to N320)
  13 donor fields (147.1s)
    mask repair: 1976 donor points filled from nearest valid neighbour
  done in 122.9s — 99 fields, shape (2, 542080)
  verify member 1: 112 fields (expected 112)
    ✅ all fields present
```

~2 min/member for the open-data fields (unchanged from operational) plus a one-off ~2.5 min
for the donors. Checks the output passes: field names identical to an operational pkl,
shapes and dtypes identical, the 8 wave fields' NaN mask **byte-identical** to the member's
own `swh`, the 10 hPa fields everywhere finite, `z_10 > z_50` at every point, and
`t_10` (229 K) > `t_50` (210 K) as the stratospheric profile requires. The February 10 hPa
zonal wind reaches −92/+88 m/s against −24/+89 m/s in June — a strong boreal-winter polar
vortex, which is the whole reason the level matters at S2S range.

Two guard rails, because the window has hard edges on both sides: a date on or after
2026-05-13 is refused (use the operational builder), and so is one outside `j1r2`'s
2024-05-02 → 2026-06-20 coverage.

The one repair worth knowing about: ecWAM's land-sea mask and open data's disagree on a
coastal fringe. Where open data says land the donor is NaN-ed to match; the other direction
leaves ~120 points per field-timestep where the member's own `swh` has a value but the
donor does not — a combination that never occurs in training data and is exactly the shape
of input that seeds NaN through a rollout. Those are filled from the nearest valid donor
point on the sphere (1976 points in total for 2026-02-12), and the result is asserted
byte-identical to `swh`'s mask before the pkl is written.

### Is a pre-50r1 pkl the same object as an operational one?

`check_pkl_structure.py` compares input-state directories field-by-field across dates.
Run on members 1–5 of 20260212 (donor-built) against 20260514 and 20260604 (operational):

```
  20260212     OK
  20260514     OK
  20260604     OK
VERDICT: all dates structurally interchangeable
```

Identical top-level keys, 112 field names, `(2, 542080)` shapes, `float64` dtypes, and the
same 15 fields carrying NaN. Two differences are real, expected, and reported rather than
hidden:

| | 20260212 (donor-built) | operational |
|---|---|---|
| fields varying by member | **91** / 112 | 106 / 112 |
| `wmb`/`cdww` mask fringe vs `swh` | **0** pts | 454 pts |
| `swh` NaN fraction | 33.49 % | 30.29 % |

* **91 vs 106 varying fields.** The 15 extra shared fields are the 5 ERA5 10 hPa fields
  plus 10 wave fields. Only 5 of those 10 are donor-sourced sharing — the other
  5 (`swh`, `mwp`, `cos_mwd`, `sin_mwd`, and the bands' open-data counterparts) are shared
  because **open data itself has no wave spread at t=0 pre-50r1** (§2.5). This is the
  §2.5 finding reappearing independently at the pkl level.
* **The `wmb`/`cdww` mask fringe.** Operationally these two deviate from `swh`'s mask by
  454 points (0.04 %, symmetric); the six band heights, `mwp` and `mwd` match `swh`
  exactly at all 1 084 160 points. The donor path forces all eight to `swh`, making this
  cycle marginally *more* uniform than operational rather than matching it. The
  alternative — copying an operational cycle's absolute `wmb`/`cdww` mask — would import
  that cycle's sea-ice edge into a February state, which is worse. 0.04 % of points, and
  the checker prints the fringe size for every date so it stays visible.
* **The `swh` NaN fraction difference is sea ice**, not a defect: February has more
  ice-covered ocean than June. The operational 20260514 and 20260604 masks are
  bit-identical to each other, so the mask does track the season rather than the cycle.

```bash
$PY check_pkl_structure.py     --dir 20260212=/tank/projects/aifs-run/20260212_0000/input_states     --dir 20260514=/tank/projects/aifs-run/20260514_0000/input_states     --dir 20260604=/tank/projects/aifs-run/20260604_0000/input_states     --members 1-5 --reference 20260604
```

**Building a pkl is not permission to trust the forecast** — but as of §5's result, the
forecast has now been measured, and it passes at 5-day lead.

## 6. Files

| file | what it is |
|---|---|
| `check_open_data_inputs.py` | is a date runnable for v1 / v2 from **open data**? also `--scan` a range to locate a cutover |
| `fetch_j1r2.py` | pull the 8 Cy50r1-only wave fields from the `j1r2` hindcast, on N320; `--patch-pkl-dir` to inject them |
| `fetch_era5_l10.py` | pull ERA5 10 hPa `t/u/v/w/z` from CDS, regrid to N320; `--validate` checks grid ordering at 50 hPa |
| `ecmwf_opendata_pkl_input_aifsens_v2_pre50r1.py` | builds the 112-field input pkls for a pre-50r1 date: 99 fields from open data as usual, 13 from the donors |
| `check_pkl_structure.py` | compare input-state pkl dirs across dates: schema, NaN structure, member variation, physical invariants |
| `wave_sensitivity.py` | the paired sensitivity experiment; donor modes `l10era5` / `wavej1r2` / `pre50r1b` |
| `analyse_sensitivity.py` | scores treatments against the floor and the ensemble scale |
| `.env` | `ak=` CDS token (ERA5) |
| `.ecmwfapirc` | ECMWF Web API key (MARS, `j1r2`) |

Raw rollouts (30 × 61 or 20 captured steps of `2t`/`tp`/`msl` on N320) are kept outside
the repo — 3.3 GB in `/tank/projects/aifs-run/wave_sensitivity_20260813/` and 1.2 GB in
`…_short/`. Delete them once the numbers above are trusted; `wave_sensitivity.py` skips
configurations whose `.npz` already exists, so a partial re-run is cheap.

## 7. If ECMWF changes the inputs again

The same trap will recur at the next cycle upgrade, and the failure mode is misleading:
Step 1 dies on whichever missing field it happens to look up first, which need not be the
one that matters. Before assuming a substitute is viable:

1. `check_open_data_inputs.py --date <D>` — enumerate **every** missing field, not just
   the one in the traceback.
2. Measure at **short lead** (5 days). At S2S range the non-determinism floor reaches
   ~90 % of the ensemble spread and hides everything below it.
3. Always include a **duplicate control** and a **should-be-inert treatment**. Both
   internal controls earned their place here — `controlB` exposed the saturation, and
   `wmb` disproved the assumption that the runner's "coupled forcings are not supported"
   message meant the field was unused.
4. **Separate "absent from open data" from "does not exist."** This folder conflated them
   and reached a confident wrong conclusion twice: the 10 hPa level was in ERA5 the whole
   time, and the training hindcast was published under a DOI. Both were one HTTP request
   away. Before writing "from any source", enumerate the sources — MARS `class=rd`
   research experiments are searchable at <https://apps.ecmwf.int/ifs-experiments/>, and
   a model's own training data is a source.
5. **Score a substitute in input space before spending GPU time on it.** The 1140× gap
   between `l10copy` and ERA5 took seconds to compute from data already on disk, and it
   is what showed the rollout needed redoing. A treatment that is three orders of
   magnitude off in the input is not a reconstruction, whatever it is named.

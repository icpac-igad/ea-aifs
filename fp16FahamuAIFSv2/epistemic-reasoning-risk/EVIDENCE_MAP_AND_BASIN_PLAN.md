# Evidence nodes → AIFS-ENS variables, and the HydroBASINS level-3 plan

Maps the S2S/SEAS5 evidence-node architecture of `s2s-seas5.txt` onto **what the AIFS-ENS v2
Icechunk store actually contains** (verified, not assumed), then sets out the next steps for the
extended East Africa + SW Indian Ocean domain at **HydroBASINS level 3**.

Companion to `S2S_BN_ONTOLOGY.md` (store inventory, state definitions) and
`DIRECTION_EXPLANATION_FIRST.md` (why explanation leads and risk follows).

---

## 1. The six fundamental evidence nodes

| Evidence node | Recommended diagnostic | AIFS-ENS v2 variables (verified in store) | Status |
|---|---|---|---|
| **Moisture** | column / low–mid moisture availability | `tcw` (total column water — **includes condensate**, so not strictly `tcwv`); `q` at **13 levels** (50→1000, no 10) | ✅ available |
| **Low-level transport / convergence** | moisture flux **and its convergence** | flux: `q`,`u`,`v` at 925/850/700 — implemented as the full 13-level **IVT** | ✅ flux · ⚠️ **convergence** — see §4 |
| **Disturbance circulation** | ζ₈₅₀, SLP / Z850 anomaly | `u_850`,`v_850`,`z_850`,`msl` | ⚠️ **ζ needs a derivative** — see §4 |
| **Vertical dynamical support** | deep-layer shear, upper divergence, trough/ridge | `u_200`,`v_200`,`z_200`,`z_500`,`u_850`,`v_850` | ✅ **shear `\|V₂₀₀−V₈₅₀\|` is a difference — computable today**; TEJ core implemented; divergence see §4 |
| **Convective environment** | instability + boundary-layer moisture | **`cape` ABSENT. No OLR — the store has no TOA radiation at all** (`ssrd`/`strd` are *surface downward*, not OLR). Substitutes: `w_500`,`w_700`, `t_850−t_500`, `q_700`+`t_700` (RH), `2d`+`2t` (VPD), `cp/tp`, `hcc` | ⚠️ substitute index only |
| **Predicted precipitation** | ensemble probability / percentile / tail | `tp`, `cp` — both **interval per-6 h** (verified; the checkpoint declares six accumulations) | ✅ implemented |

### Beyond the six

| Node | AIFS-ENS variables | Status |
|---|---|---|
| **Ocean / boundary forcing** (SST, IOD, Niño-3.4) | **none.** `sst` is absent from the checkpoint under *any* name; `skt` is a skin temperature whose member spread on the IO dipole is 0.32–0.39 K, as large as a weak IOD event | ❌ **impossible in-store** → external OISST/OSTIA (`loc.sstOISST.v1`) |
| **Antecedent land state** | `swvl1` (0–7 cm), `swvl2` (7–28 cm), `stl1/2`, `ro` | ⚠️ **only to 28 cm** — no equivalent of S2S `smv100`; and it is the *same model run* as the forecast, so not independent evidence |
| **Convective organisation** | `cp`/`tp` ratio, `hcc` (no OLR) | ✅ computable, **unbuilt** |
| **Signal persistence** | any node across the **7 cycles on disk** (20260625 → 20260806) | ✅ **newly feasible**; within-window duration already implemented |

### What AIFS adds that neither S2S nor SEAS5 offers here

`ro` (runoff), `100u`/`100v`, and **11 wave fields** (`swh`, `mwp`, `cdww`, `cos_mwd`/`sin_mwd`,
`h1012`…`h2530`) — swell arrival timing, coastal hazard, and a marine TC signature via
`swh` + `msl` minima.

---

## 2. The extended-domain hierarchy

`s2s-seas5.txt` §EXTENT proposes **15°E–80°E, 40°S–25.25°N** to capture Madagascar, the
Mozambique Channel, the SWIO cyclone corridor and the Mascarene High.

| Diagnostic | AIFS-ENS variables | Status |
|---|---|---|
| Low-level moisture flow | `tcw`, `q_850/700`, `u_850`,`v_850` | ✅ |
| Low-level convergence | `u_850`,`v_850` | ⚠️ §4 |
| **Cyclonic disturbance / TC influence** | `u_850`,`v_850`,`u_700`,`v_700`,`msl` (+ `swh` marine signature) | ⚠️ §4 · **unbuilt** |
| **Mozambique Channel Trough** | `msl`, `u_850`,`v_850`, `tcw`, `w_700` | ⚠️ §4 · **unbuilt** |
| Mascarene High | `msl`, `u_850`,`v_850` | ✅ **implemented** (`loc.ujh.v1`) |
| Somali / Findlater flow | `u_850`,`v_850` | ✅ **implemented** (`loc.llj.v1`) |
| Upper-level support | `u_200`,`v_200`,`z_500` | ✅ TEJ implemented; shear unbuilt |
| Ascent | `w_850`,`w_700`,`w_500` | ✅ in the R index |
| Moisture availability | `tcw`, `q`, RH from `q`+`t` | ✅ |
| Convection | **no MUCAPE** — temperature profile only | ⚠️ substitute |
| Rainfall | `tp` | ✅ |

### The domain constraint does not bind us

The extent note trades domain size against storage for a *regional IFS archive*, and on that
basis recommends dropping St Helena. **Our store is already global** — N320, 120 variables, all
542 080 cells — so that trade-off does not apply. The proposed domain is only **9.5 % of the
globe** (51 718 cells, 44 % land). St Helena is therefore kept at zero marginal cost, and it is
already implemented and working. Verified cell counts for the new boxes:

| Box | cells | land |
|---|---|---|
| EXTENT domain 15–80 E, 40 S–25.25 N | 51 718 | 22 869 (44 %) |
| Mozambique Channel 35–50 E, 25–10 S | 2 764 | 1 292 |
| Madagascar 43–51 E, 26–11 S | 1 528 | 750 |
| SWIO corridor 40–80 E, 30–5 S | 12 330 | 861 (7 %) |
| Mascarene (implemented) 45–90 E, 35–20 S | 7 952 | 238 |

*Seasonality caveat:* the MCT is an austral-summer feature peaking around February, and the TEJ
is boreal-summer. The cycles tested so far are late July/August, so a near-null MCT is the
**correct** result there, not a failure — the same rule already recorded for the TEJ.

---

## 3. Why HydroBASINS **level 3** fits — and where it strains

Level-3 basins are sub-continental (10⁵–10⁶ km²). That is a much better match to day-18–33
predictability than admin-1 units: Part 0 of `CPT_BUILD_CRITIQUE_AND_PLAN.md` says a locus is
admissible only where its scale retains predictability, and admin-1 pooling was projecting a
planetary-scale signal onto polygons where predictability is weakest. **Level 3 moves the unit
of analysis back up to the scale where the ensemble carries information.**

The tension, stated plainly: level-3 basins may be *too coarse to act on*. If they are, the
fallback is level 4/5 — but that trades predictability for actionability, which is the admin-1
problem again in a new coordinate system. Decide it with a duty forecaster, not a modeller.

**Two rules carry over unchanged.** Pool *then* classify — never classify per cell and vote. And
use **percentile / areal-fraction metrics, not basin means**: a mean over a 10⁵ km² basin dilutes
exactly what the jet work showed must be preserved.

---

## 4. The unblocking idea: boundary integrals instead of gradients

> **⚠️ Superseded in part — see
> [`SHORT_RANGE_EVIDENCE_MAP.md`](SHORT_RANGE_EVIDENCE_MAP.md) §1.** The premise below is
> wrong: `ts-mjo/grid_ops.py` computes the vorticity **field** directly on the reduced
> Gaussian axis, and `ts-mjo/ts_tracks.py` does object detection there too. It was committed
> 2026-08-07, five days *before* this file. Vorticity is done and re-verified on O96;
> divergence is a near-copy of the same method. Boundary integrals remain the cheaper route
> to an area mean, but they are now an optimisation, not the only option.

Three nodes above are marked ⚠️ for the same reason — vorticity, moisture-flux convergence and
upper-level divergence all want ∇· or ∇×, undefined on the unstructured reduced-Gaussian
`values` axis. This has been registered as *blocked pending regrid or spherical harmonics*.

**For basin-level nodes that blocker can be removed without regridding**, via Stokes and Gauss:

```
area-mean vorticity      ζ̄ = (1/A) ∮ V · dl          (circulation around the basin boundary)
area-mean divergence     D̄ = (1/A) ∮ V · n̂ dl
area-mean MFC           −(1/A) ∮ (qV) · n̂ dl
```

A line integral needs only cells in a thin band along the boundary — no gradient operator, no
regridding, and it works on any polygon including a HydroBASINS outline. It yields **area means,
not fields**, so object detection (tracking an individual cyclone centre) still needs the regrid.
But area means are precisely what a basin-level evidence node consumes.

Two cautions. **Southern-Hemisphere sign convention:** cyclonic rotation is *negative* ζ, so the
cyclonic-disturbance node must test ζ̄ < threshold, not |ζ̄|. And a boundary integral is only as
good as the boundary sampling — with N320 (~31 km) cells, a basin needs a perimeter of at least
a few hundred kilometres for the band to be populated, which level 3 comfortably satisfies and
level 6+ would not.

If this works, it converts **E2 (regrid / spherical harmonics)** from a prerequisite into an
optimisation, and unblocks four diagnostics at once. It should be tested before any regrid work
is funded.

---

## 5. Subsequent steps

**B1 — Basin ingest.** HydroBASINS level 3 (`af` region, plus Madagascar) clipped to
15°E–80°E / 40°S–25.25°N. `geopandas`/`fiona` are **not installed**; the existing pure-numpy
point-in-polygon path (`--adm1`, wrap-aware since the 360° bug) consumes GeoJSON, so convert the
shapefile with `ogr2ogr` or `pyshp`. Register each basin as a region with its `HYBAS_ID`, record
the cell count, and drop basins below the minimum-cell threshold.

**B2 — Boundary-integral operators (§4).** Implement ζ̄, D̄ and MFC as perimeter line integrals;
validate against a known case before trusting them (a Mozambique Channel cyclone in an austral-
summer cycle is the natural test, and would need a DJF cycle rather than the July/August ones on
disk).

**B3 — Cyclonic-disturbance node.** Deliberately *not* a "tropical cyclone" node — classification
belongs to the operational cyclone centres. Evidence: ζ̄₈₅₀ and ζ̄₇₀₀ (SH sign), `msl` minimum and
its position, wind-speed percentile, `swh` marine signature, and persistence across steps and
cycles. The proposition is *"is a coherent cyclonic disturbance present, and is it modifying the
moisture flow relevant to East Africa?"* — which does not require landfall.

**B4 — Mozambique Channel Trough node.** `msl` + ζ̄₈₅₀ + winds + `tcw` + `w_700`, with the
austral-summer seasonality declared in the registry so an out-of-season null reads as correct.

**B5 — Two precipitation nodes, not one.** *Expected* precipitation and the *extreme tail*, per
basin, as areal fractions and exceedance counts above named physical thresholds — a moderate
ensemble mean can coexist with a small but decisive probability of extreme rainfall, and the
existing state-resolution work shows the 4-state split will not survive on its own.

**B6 — Persistence node.** Across the seven cycles on disk. This is the epistemic node the CRMA
framing most wants and the cheapest one still unbuilt: a signal appearing in three consecutive
initialisations is stronger evidence than the same signal in one.

**B7 — Deep-layer shear.** `|V₂₀₀ − V₈₅₀|` — a vector difference, no derivative, computable
today. The cheapest unbuilt atmospheric diagnostic on the list.

**B8 — E4 model climatology** remains the gate for every anomaly-based state (the source note's
states are all defined *"relative to the lead-time- and calendar-specific hindcast
climatology"*, which we do not have). Three independent lines now point at it: the M/R/P
threshold mismatch, the subtropical-high saturation, and this.

**Order.** B7 and B6 are cheap and immediate. B1 then B2 unlock everything else, and B2 should be
tested before any regrid work is funded. B3–B5 follow the basins. B8 runs in the background and
gates the anomaly work regardless.

---

*Verified against `cycle-20260730_0000` and `cycle-20260723_0000`. Every "absent" in this
document was probed, and the checkpoint's own variable list was read to confirm that `sst` does
not exist in the model rather than merely being unfetched.*

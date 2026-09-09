# The six evidence nodes across the 33-day forecast

**Companion to [`EVIDENCE_MAP_AND_BASIN_PLAN.md`](EVIDENCE_MAP_AND_BASIN_PLAN.md), not a
replacement.** That document maps the six evidence nodes onto verified store contents at S2S
lead. This one resolves them against **the range the cycle actually runs — 0 to 792 h, 33
days** — and records three things it could not:

1. which nodes are available **at which lead and which resolution**, now that tier B writes
   two stores with different windows and different variable sets;
2. that §4's derivative blocker **was already solved when §4 was written**;
3. what the short-range end of the range costs, if it is wanted.

Verified 2026-09-09 against `20260903_0000` (tier B).

---

## 1. What the 33 days actually contain

A tier-B cycle writes two stores. They differ in resolution, in lead coverage, **and in
variable set** — the third difference is the one that decides what the nodes can do.

| | `icechunk_o96` | `icechunk_n320_aiwq` |
|---|---|---|
| cells | 40 320 — **~112 km** | 542 080 — **~28 km** |
| leads | **6–792 h** (days 0.25–33), 132 steps | **432–792 h** (days 18–33), 61 steps |
| variables | **all 124** | **10** |
| size | 158 GB | 51 GB |

The 10 in the sidecar are `msl` `tp` `2t` `10u` `10v` `t_200` `t_300` `t_500` `u_850` `v_850`
— chosen for the AI-WQ gridded submission and the TS tracker, **not** for these six nodes.

### Consequence: the six nodes are an O96 product across the whole range

Tested variable by variable against the sidecar:

| node | at 28 km (days 18–33) | why |
|---|---|---|
| 1 Moisture | ❌ **none** | no `q` at any level, no `tcw` |
| 2 Transport / convergence | ⚠️ partial | `u/v_850` only — **no `q`, so no moisture flux at all** |
| 3 Disturbance circulation | ⚠️ partial | ζ₈₅₀ and `msl` yes; `z_850` no |
| 4 Vertical dynamics | ❌ **none** | no `w`, no `z`, no 200-hPa winds |
| 5 Convective environment | ⚠️ partial | `t_500`, `2t` only — no `t_850/t_700`, no `q`, no `2d`/`sp` |
| 6 Precipitation | ⚠️ partial | `tp` yes, `cp` no |

**Two of the six are entirely unavailable at 28 km, and the remaining four are partial.** The
sidecar cannot carry this product at any lead. Everything below therefore runs on the **O96
corpus at ~112 km — but across the full 33 days, with all 124 variables and no gaps.**

That is the reframing this document exists to make: the range is 0–792 h, the resolution is
112 km, and both are properties of how the cycle is stored rather than of the model.

### Where that bites

- **Area means survive 112 km; fields do not.** A 112 km cell will not resolve a mesoscale
  convective system, a sharp moisture front, or a compact vortex. Basin-integrated nodes are
  fine; object-scale diagnosis is not.
- **The basin-size floor roughly quadruples.**
  [`EVIDENCE_MAP_AND_BASIN_PLAN.md`](EVIDENCE_MAP_AND_BASIN_PLAN.md) §4 notes a boundary
  integral needs a perimeter of a few hundred kilometres to populate the band at N320's
  ~31 km. At 112 km that threshold strains HydroBASINS level 3 rather than comfortably
  fitting it.
- **Lead range is not the constraint — resolution is.** All 132 steps are present, so
  days 0–7, the AI-WQ target weeks at days 18–31, and the tail to day 33 are equally covered.

---

## 2. The nodes, resolved against the O96 corpus

| # | node | specified | AIFS ENS v2, all 132 steps |
|---|---|---|---|
| **1** | Moisture | `tcwv` | ⚠️ `tcw` is total column **water** (includes condensate). True TCWV = (1/g)∫q dp over the 13 `q` levels |
| | | `q925/850/700/500` | ✅ direct |
| | | `r850/r700` | 🔧 **exactly derivable** — RH = f(q, t, p); only the saturation formula is a choice. `2d` gives near-surface humidity directly |
| **2** | Transport / convergence | `q850·V850` | ✅ direct product of `q_850`, `u_850`, `v_850` |
| | | `d850` | 🔧 needs the divergence operator (§3) |
| | | `u/v` 925/850/700 | ✅ direct |
| **3** | Disturbance circulation | `vo850/700/500` | ✅ **implemented** (§3) — `u`,`v` exist at all three |
| | | `msl` | ✅ direct |
| | | `gh850` | 🔧 `z_850 / 9.80665` |
| **4** | Vertical dynamics | `w850/700/500` | ✅ direct — **omega, Pa/s, negative = ascent** |
| | | `d200` | 🔧 divergence operator (§3) |
| | | `gh500` | 🔧 `z_500 / 9.80665` |
| | | `u/v200` | ✅ direct |
| **5** | Convective environment | `mucape` | ❌ **absent — the one real gap.** See §4 |
| | | `t850/700/500` | ✅ direct |
| | | `r700` | 🔧 derive as above |
| **6** | Predicted precipitation | `tp` ensemble | ✅ direct. `cp` also present → convective fraction `cp/tp` |

### Store inventory, verified

**Pressure levels** — `q` `t` `u` `v` `w` `z` on 1000, 925, 850, 700, 600, 500, 400, 300,
250, 200, 150, 100, 50 hPa (`t/u/v/w/z` add 10; `q` starts at 1000). 83 arrays.

**Surface (37)** — 18 atmospheric (`msl` `sp` `2t` `2d` `10u` `10v` `100u` `100v` `tcw` `tp`
`cp` `sf` `tcc` `hcc` `mcc` `lcc` `ssrd` `strd`), 8 land (`skt` `stl1` `stl2` `swvl1` `swvl2`
`sd` `snowc` `ro`), **11 ocean wave** (`swh` `mwp` `cdww` `cos_mwd` `sin_mwd` `h1012` `h1214`
`h1417` `h1721` `h2125` `h2530`).

**120 variables total, of which 101 are atmospheric.** The wave and land fields are not
atmospheric and the six nodes do not consume them — worth stating because "120 atmospheric
variables" is a claim that appears elsewhere and is wrong on both counts.

### Units, checked rather than assumed

Sampled from member 001; standard IFS conventions, but the sign of `w` inverts node 4 if
taken backwards:

| var | range | reading |
|---|---|---|
| `w_500` | -4.2 … +2.0 | **omega, Pa/s — negative is ascent** |
| `z_500` | 4.50e4 … 5.87e4 | geopotential m2 s-2, so `gh` ~ 5856 m |
| `tp`, `cp` | 0 … 0.074 | **metres** — x1000 for mm |
| `msl`, `sp` | ~1.01e5 | Pa |
| `q_850` | 0 … 0.018 | kg kg-1 |
| `t_850`, `2d` | ~288 | K |
| `tcw` | 0 … 96 | kg m-2 |

---

## 3. The blocker in §4 is resolved — including the part it said would not be

[`EVIDENCE_MAP_AND_BASIN_PLAN.md`](EVIDENCE_MAP_AND_BASIN_PLAN.md) §4 registers vorticity,
moisture-flux convergence and upper divergence as

> undefined on the unstructured reduced-Gaussian `values` axis … *blocked pending regrid or
> spherical harmonics*

and adds that boundary integrals give

> **area means, not fields**, so object detection (tracking an individual cyclone centre)
> still needs the regrid.

Both halves are out of date. `ts-mjo/grid_ops.py` computes the **vorticity field** directly
on the reduced Gaussian axis — exact circular differences along each latitude row, plus a
precomputed nearest-longitude index map between adjacent rows:

```
zeta = 1/(a cos phi) * [ dv/dlambda - d(u cos phi)/dphi ]
```

~1.5 s per member against ~26 min to regrid every field purely to differentiate it. And
object detection on the native grid is not hypothetical: `ts-mjo/ts_tracks.py` finds cyclone
centres and links them into tracks, entirely on the unstructured axis.

**Why the documents disagreed:** `grid_ops.py` was committed **2026-08-07** (`0d8e402`), the
evidence map **2026-08-12** (`98a5681`). The solution sat in the repository for five days
before the blocker was written down — a process fact, not a blame one; the TS/MJO and risk-BN
workstreams had no reason to read each other's files.

### Re-verified on O96, since that is where this product runs

| check | result |
|---|---|
| grid parsed | 40 320 points → **192 rows** (correct for O96) |
| zonal mean zeta850 | **+5.2e-08 s^-1** — machine-zero, as a global mean must be |
| deepest 1 % of `msl` | **+5.80e-05 s^-1** cyclonic |
| highest 1 % of `msl` | **-2.71e-05 s^-1** anticyclonic |

The southern-hemisphere sign trap §4 flags is already handled — `grid_ops.cyclonic()`
multiplies by `sign(-lat)`, so one threshold means "cyclonic" in both hemispheres.

**Divergence is not implemented** — only vorticity is. It is the same machinery with the terms
swapped (`D = 1/(a cos phi)[du/dlambda + d(v cos phi)/dphi]`). That one addition closes nodes
2 and 4, delivers moisture-flux convergence `-div(qV)`, and supplies the `chi200` that the VPM
MJO index needs (`../ts-mjo/vpm-mjo.md`) — one method, three uses.

**Boundary integrals are not obsolete.** They remain the cheaper route to a basin area mean
and need no differential operator. They are now an optimisation rather than the only option.

---

## 4. CAPE is the one genuine gap — and MU-CAPE is the wrong ask

AIFS ENS emits **no CAPE-family field at all**, consistent with
[`S2S_BN_ONTOLOGY.md`](S2S_BN_ONTOLOGY.md). From 13–14 pressure levels, ranked honestly:

**Surface-based CAPE is well-posed.** `2t`, `2d` and `sp` define the parcel exactly; lift it
through the `t`/`q` profile. The parcel origin is not an approximation.

**MU-CAPE is not.** "Most unstable" means scanning parcels through the lowest few hundred hPa,
and the store offers only 1000 / 925 / 850 there before a 150 hPa gap to 700. LCL/LFC
placement is crude at that spacing and CAPE is acutely sensitive to exactly the low-level
detail that is missing. A number would be produced; its ranking across members would be
dominated by discretisation.

**Prefer the stability indices.** Lifted index, K-index and Total Totals come from `t`/`q` at
standard levels and are far more robust to coarse spacing than any CAPE integral. They answer
node 5's actual question — *can it convect vigorously?* — without implying a precision the
levels do not support.

---

## 5. What this changes, and the one decision it forces

| node | in `EVIDENCE_MAP_AND_BASIN_PLAN.md` §1 | now |
|---|---|---|
| Disturbance circulation | ⚠️ zeta needs a derivative | ✅ **implemented, re-verified on O96** |
| Transport / convergence | ⚠️ convergence blocked | 🔧 one near-copy method away |
| Vertical dynamical support | ⚠️ divergence blocked | 🔧 same |
| Convective environment | ⚠️ substitute index only | ⚠️ **unchanged** — §4 argues the substitute is the right call, not a concession |

Three of the six ⚠️ marks were pessimistic by five days. The fourth is real.

### The decision: 112 km everywhere, or pay for 28 km

The six nodes run at **112 km across all 33 days** today. Raising any of them to 28 km is an
**inference-time** choice — it cannot be recovered afterwards without re-running the rollout
(~4½ h per cycle). Measured from the current sidecar (51 GB / 10 vars / 61 steps =
**83.6 MB per variable per step**), serving all six nodes at 28 km needs roughly 28 variables:

| option | cost |
|---|---|
| current sidecar (10 vars, days 18–33) | **51 GB** |
| six nodes at 28 km, days 18–33 only | **~143 GB** |
| six nodes at 28 km, full 0–792 h | **~309 GB** |
| *reference: tier-B pair today* | 209 GB |
| *reference: old full-N320 store* | 583 GB |

So the full-range 28 km option costs about **1.5× the whole tier-B cycle** and roughly half
the old N320 store it replaced. That is a real trade, not an obvious yes — and it competes
with disk that is already the binding constraint on this box.

**Recommended order:** settle this first, because every cycle run under the current
`--native-vars` forecloses it for that cycle. Then add `divergence()` — its value is the same
either way, and it serves three consumers. Node 5's CAPE question can wait; §4 already says
what to do.

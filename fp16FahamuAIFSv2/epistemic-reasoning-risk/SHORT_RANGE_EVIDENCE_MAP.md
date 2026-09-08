# The six evidence nodes at 0–7 days — and a blocker that is no longer one

**Companion to [`EVIDENCE_MAP_AND_BASIN_PLAN.md`](EVIDENCE_MAP_AND_BASIN_PLAN.md), not a
replacement.** That document already maps the six evidence nodes onto verified store
contents; this one records two things it could not:

1. what changes when the same six nodes are asked at **0–7 days** rather than S2S lead,
   where the question becomes concrete — *is the atmosphere assembling the ingredients for
   heavy rainfall over this basin?* — and the diagnostics are named as specific fields
   (`vo850`, `d850`, `r700`, `gh500`, `mucape`) rather than as concepts;
2. that **§4's central blocker was already solved when §4 was written**, which changes the
   status of three of the six nodes.

Verified 2026-09-08 against `20260903_0000/icechunk_o96` (tier B, 124 arrays, full 0–792 h).

---

## 1. The blocker in §4 is resolved — including the part it said would not be

[`EVIDENCE_MAP_AND_BASIN_PLAN.md`](EVIDENCE_MAP_AND_BASIN_PLAN.md) §4 states that vorticity,
moisture-flux convergence and upper divergence are

> undefined on the unstructured reduced-Gaussian `values` axis … registered as *blocked
> pending regrid or spherical harmonics*

and proposes boundary integrals as a workaround, with the caveat that they yield

> **area means, not fields**, so object detection (tracking an individual cyclone centre)
> still needs the regrid.

Both halves are now out of date. `ts-mjo/grid_ops.py` computes the **vorticity field**
directly on the reduced Gaussian axis — exact circular differences along each latitude row,
plus a precomputed nearest-longitude index map between adjacent rows for the meridional term:

```
zeta = 1/(a cos phi) * [ dv/dlambda - d(u cos phi)/dphi ]
```

~1.5 s per member, against ~26 min if every field were regridded purely to differentiate it.
And object detection on the native grid is not hypothetical: `ts-mjo/ts_tracks.py` detects
cyclone centres and links them into tracks across 6-hourly steps, entirely on the
unstructured axis.

**Why the two documents disagreed:** `grid_ops.py` was committed **2026-08-07** (`0d8e402`),
the evidence map **2026-08-12** (`98a5681`). The solution was in the repository for five days
before the blocker was written down. Worth recording as a process fact, not a blame one — the
TS/MJO and the risk-BN workstreams simply had no reason to read each other's files.

### Re-verified here, on O96 rather than N320

`grid_ops.py` was validated on N320. The basin work is at O96, so it was re-run:

| check | result |
|---|---|
| grid parsed | 40 320 points → **192 rows** (correct for O96) |
| zonal mean zeta850 | **+5.2e-08 s^-1** — machine-zero, as a global mean must be |
| deepest 1 % of `msl` | **+5.80e-05 s^-1** cyclonic — sign-corrected via `cyclonic()` |
| highest 1 % of `msl` | **-2.71e-05 s^-1** anticyclonic |

The southern-hemisphere sign trap §4 flags is already handled: `grid_ops.cyclonic()` multiplies
by `sign(-lat)`, so one threshold means "cyclonic" in both hemispheres and the node does *not*
need the `zeta < threshold` special case.

**Divergence is not implemented** — only vorticity is. It is the same machinery with the terms
swapped (`D = 1/(a cos phi)[du/dlambda + d(v cos phi)/dphi]`), a near-copy of the existing
method rather than new work. That single addition would also give moisture-flux convergence,
`-div(qV)`, which is a better node-2 diagnostic than `d850` alone.

**Boundary integrals are not obsolete.** They remain the cheaper route to a basin *area mean*,
and they need no differential operator at all. What has changed is that they are now an
optimisation rather than the only option, and the "fields need a regrid" caveat is gone.

---

## 2. The 0–7 day field list, resolved against the store

The short-range specification names exact fields. Status against AIFS ENS v2:

| # | node | specified | AIFS ENS v2 |
|---|---|---|---|
| **1** | Moisture | `tcwv` | ⚠️ `tcw` is total column **water** (includes condensate). True TCWV = (1/g)∫q dp over the 13 `q` levels |
| | | `q925/850/700/500` | ✅ direct |
| | | `r850/r700` | 🔧 **exactly derivable** — RH = f(q, t, p); only the saturation formula is a choice. `2d` gives near-surface humidity directly |
| **2** | Transport / convergence | `q850·V850` | ✅ direct product of `q_850`, `u_850`, `v_850` |
| | | `d850` | 🔧 needs the divergence operator (§1) |
| | | `u/v` 925/850/700 | ✅ direct |
| **3** | Disturbance circulation | `vo850/700/500` | ✅ **implemented** (§1) — `u`,`v` exist at all three |
| | | `msl` | ✅ direct |
| | | `gh850` | 🔧 `z_850 / 9.80665` |
| **4** | Vertical dynamics | `w850/700/500` | ✅ direct — **omega, Pa/s, negative = ascent** |
| | | `d200` | 🔧 divergence operator (§1) |
| | | `gh500` | 🔧 `z_500 / 9.80665` |
| | | `u/v200` | ✅ direct |
| **5** | Convective environment | `mucape` | ❌ **absent — the one real gap.** See §3 |
| | | `t850/700/500` | ✅ direct |
| | | `r700` | 🔧 derive as above |
| **6** | Predicted precipitation | `tp` ensemble | ✅ direct. `cp` also present → convective fraction `cp/tp` |

### Store inventory, verified

**Pressure levels** — `q` `t` `u` `v` `w` `z` on 1000, 925, 850, 700, 600, 500, 400, 300,
250, 200, 150, 100, 50 hPa (`t/u/v/w/z` add 10; `q` starts at 1000).

**Surface (37)** — `msl` `sp` `2t` `2d` `10u` `10v` `100u` `100v` `skt` `tcw` `tp` `cp` `sf`
`ro` `tcc` `hcc` `mcc` `lcc` `ssrd` `strd` `stl1` `stl2` `swvl1` `swvl2` `sd` `snowc` `swh`
`mwp` `cdww` `cos_mwd` `sin_mwd` `h1012` `h1214` `h1417` `h1721` `h2125` `h2530`.

### Units, checked rather than assumed

Sampled from member 001 at a mid-forecast step; all are standard IFS conventions, but the
sign of `w` in particular is worth stating because getting it backwards inverts node 4:

| var | range | reading |
|---|---|---|
| `w_500` | -4.2 … +2.0 | **omega, Pa/s — negative is ascent** |
| `z_500` | 4.50e4 … 5.87e4 | geopotential m2 s-2, so `gh` ≈ 5856 m |
| `tp`, `cp` | 0 … 0.074 | **metres** — x1000 for mm |
| `msl`, `sp` | ~1.01e5 | Pa |
| `q_850` | 0 … 0.018 | kg kg-1 |
| `t_850`, `2d` | ~288 | K |
| `tcw` | 0 … 96 | kg m-2 |

---

## 3. CAPE is the one genuine gap — and MU-CAPE is the wrong ask

AIFS ENS emits **no CAPE-family field at all**, consistent with what
[`S2S_BN_ONTOLOGY.md`](S2S_BN_ONTOLOGY.md) records. What can be computed from 13–14 pressure
levels, ranked honestly:

**Surface-based CAPE is well-posed.** `2t`, `2d` and `sp` define the parcel exactly; lift it
through the `t`/`q` profile. The parcel origin is not an approximation.

**MU-CAPE is not.** "Most unstable" means scanning candidate parcels through the lowest few
hundred hPa, and the store offers only 1000 / 925 / 850 there, with a 150 hPa gap to 700.
LCL/LFC placement is crude at that spacing and CAPE is acutely sensitive to exactly the
low-level detail that is missing. A number would be produced; its ranking across members
would be dominated by discretisation.

**Prefer the stability indices at this vertical resolution.** Lifted index, K-index and Total
Totals all come from `t`/`q` at standard levels and are far more robust to coarse spacing than
any CAPE integral. They answer node 5's actual question — *can it convect vigorously?* —
without implying a precision the levels do not support.

This is the same discipline the older documents apply elsewhere: prefer the diagnostic the
data can carry over the one the textbook names.

---

## 4. What this changes

| node | status in `EVIDENCE_MAP_AND_BASIN_PLAN.md` §1 | status now |
|---|---|---|
| Disturbance circulation | ⚠️ zeta needs a derivative | ✅ **implemented and re-verified on O96** |
| Transport / convergence | ⚠️ convergence blocked | 🔧 one near-copy method away |
| Vertical dynamical support | ⚠️ divergence blocked | 🔧 same |
| Convective environment | ⚠️ substitute index only | ⚠️ **unchanged** — and §3 argues the substitute is the right call, not a concession |

Three of the six ⚠️ marks were pessimistic by five days. The fourth is real and stays.

**Next, in order of value per unit work:** add `divergence()` beside `relative_vorticity()` in
`ts-mjo/grid_ops.py`, which closes nodes 2 and 4 together and delivers MFC; then decide between
SB-CAPE and the stability indices for node 5 — §3 recommends the indices.

#!/usr/bin/env python3
"""
S2S circulation diagnostics from the AIFS-ENS v2 Icechunk store.

PRIMARY OUTPUT (as of 2026-08-06, see DIRECTION_EXPLANATION_FIRST.md): the **circulation
explanation record** (`--explain-out`) — ensemble counts over circulation states per region per
lead window, with the effective sample size attached to every count. That is the product the
store uniquely enables: the submission pipeline keeps 3 of 120 variables, the store keeps the
full 3-D state, and at days 18-33 the model's real information is planetary/synoptic
circulation, not admin-1 impact.

SECONDARY OUTPUT: the Bayesian-network evidence CSVs (`--out`, `--member-out`, `--cpt-json`)
consumed by `s2s_water_balance_bn.jl`. Risk is now a DOWNSTREAM CONSUMER of the same loci,
switched on when the observation side exists — it is not the goal of this script.

Note the deliberate asymmetry between the two: within-ensemble relative framing ("n of 50
members place the atmosphere in regime B") is NATIVE to the explanation product and fatal to a
calibrated risk state — which is exactly the bug recorded in the registry header.

Counterpart of bn-ibf's `flood_data_prep.py`, rebuilt for this store and for the *grouped
process ontology* of `qd-1.md.txt`:

    C circulation regime -> M moisture transport -> R rainfall-generation environment
                                                 -> P precipitation forcing
    A antecedent wetness ------------------------------------------> W water-balance pressure
                                                    RO runoff (evidence CHILD of W)

What is different from the bn-ibf original, and why (see S2S_BN_ONTOLOGY.md):

  * Grid.   The store is native N320 REDUCED GAUSSIAN on an unstructured `values` axis —
            not a lat/lon mesh. regionmask/xarray 2-D zonal statistics do not apply. Cells
            are quasi-equal-area, so a plain cell mean IS the area-weighted mean; no
            cos(lat) weights. Spatial DERIVATIVES are undefined here, so every diagnostic
            is integral/mean-based (no moisture-flux convergence — see blocked_loci).
  * Window. Only hours 432-792 (days 18-33) are written. There is no event-scale lead time;
            the target node is a water-balance TENDENCY, not a flood.
  * Fields.  cape is ABSENT -> the rainfall-environment index substitutes ascent+lapse+RH700.
            lsm is ABSENT   -> the land mask is finite(swvl1) (input-side NaN over sea).
  * Sample. One state per member per lead window BEFORE counting (A1 of the critique):
            the 50 members give 50 JOINT draws over the whole chain, which is what makes the
            counted CPTs legitimate instead of pseudo-replicated over 61 correlated steps.

Outputs
  --explain-out  PRIMARY. JSON circulation record per (lead window x region): regime occupancy,
                 |IVT| distribution and threshold-exceedance counts, Somali-jet index,
                 rainfall-environment components, precipitation-rate distribution, ESS, and a
                 generated narrative sentence per region. No thresholds on impact, no exposure,
                 no elicitation.
  --out          one row per (unit x lead window): ensemble state fractions as BN soft evidence
                 `{node}_p{k}`, plus the raw ensemble-mean diagnostics.
  --member-out   one row per (unit x window x member): per-member states for storylines.
  --cpt-out /--cpt-json
                 joint member counts for C->M, M->R, (M,R)->P with the ESS discount and full
                 provenance (the B7 artifact schema).

Usage
  # explanation product over several circulation regions (the primary use)
  python s2s_bn_evidence_prep.py \
      --store /tank/projects/aifs-run/20260730_0000/icechunk_v2 \
      --tag cycle-20260730_0000 \
      --regions IGAD_EA,EQ_INDIAN,CONGO --explain-out explain_20260730.json

  # plus the downstream BN evidence
      --out evidence_20260730.csv --member-out evidence_members_20260730.csv \
      --cpt-out process_cpt_20260730.nc --cpt-json process_cpt_20260730.json
  # optional admin-1 units (no geopandas needed — pure-numpy point-in-polygon):
      --adm1 icpac_adm1v3.geojson --adm1-id GID_1 --adm1-name NAME_1
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os

import numpy as np
import pandas as pd
import xarray as xr
import icechunk
import yaml
from scipy.cluster.vq import kmeans2, whiten

G = 9.80665
SHARED_LEV = [50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000]
HERE = os.path.dirname(os.path.abspath(__file__))

# Node state vocabularies — MUST match s2s_water_balance_bn.jl and the registry.
STATES = {
    # k=2, NOT k=4. The k-regimes test (k_regimes_test.py, artifact
    # k_regimes_test_20260730.json) showed k=4 clustering is UNSTABLE on this spread
    # (subsample ARI 0.557, silhouette 0.24) while k=2 is stable (ARI 0.817). PC1 alone
    # carries 72-88% of the ensemble variance. Four regimes were manufacturing structure
    # from ~1 dominant direction; the ensemble carries two circulation stories, not four.
    "C":  ["unfavourable", "convergent"],
    "M":  ["deficient", "normal", "enhanced", "extreme_persistent"],
    "R":  ["suppressed", "weakly_supportive", "supportive", "strongly_supportive"],
    "P":  ["below_normal", "normal", "heavy_episodic", "heavy_persistent"],
    "A":  ["very_dry", "dry", "normal", "wet", "saturated"],
    "RO": ["nil", "low", "moderate", "high"],
}


# --------------------------------------------------------------------------------------
# store access
# --------------------------------------------------------------------------------------
def load_registry():
    with open(os.path.join(HERE, "discretization_registry.yaml")) as f:
        return yaml.safe_load(f)


def open_cycle(store, tag):
    repo = icechunk.Repository.open_or_create(icechunk.local_filesystem_storage(store))
    ds = xr.open_zarr(repo.readonly_session(tag=tag).store, consolidated=False, zarr_format=3)
    return repo, ds


def window_steps(ds, hours, m_idx=0):
    """Written (finite) step indices whose lead hour falls in [h0, h1]. Step t is (t+1)*6 h."""
    h0, h1 = hours
    fin = np.isfinite(ds["2t"].isel(member=m_idx, values=0).values)
    return [t for t in range(ds.sizes["time"]) if fin[t] and h0 <= (t + 1) * 6 <= h1]


def land_mask(ds, step):
    """lsm is ABSENT from the store; the input-side LSM mask survives as NaN over sea in
    swvl1, so finite(swvl1) is the land mask."""
    return np.isfinite(ds["swvl1"].isel(member=0, time=step, drop=True).values)


# --------------------------------------------------------------------------------------
# spatial units — box from the registry, or admin-1 polygons (no geopandas dependency)
# --------------------------------------------------------------------------------------
def _ring_contains(px, py, ring):
    """Vectorised ray-casting point-in-ring test."""
    x, y = ring[:, 0], ring[:, 1]
    inside = np.zeros(px.shape, dtype=bool)
    j = len(ring) - 1
    for i in range(len(ring)):
        yi, yj = y[i], y[j]
        straddles = (yi > py) != (yj > py)
        if straddles.any():
            xint = (x[j] - x[i]) * (py - yi) / (yj - yi + 1e-300) + x[i]
            inside ^= straddles & (px < xint)
        j = i
    return inside


def _polygon_contains(px, py, polygon):
    """polygon = [exterior_ring, hole1, ...] in GeoJSON coordinate order."""
    ext = np.asarray(polygon[0], dtype=float)
    inside = _ring_contains(px, py, ext)
    for hole in polygon[1:]:
        inside &= ~_ring_contains(px, py, np.asarray(hole, dtype=float))
    return inside


def adm1_units(path, lon, lat, id_field, name_field):
    """Return [(unit_id, unit_name, bool_mask_over_cells), ...] from a GeoJSON file."""
    with open(path) as f:
        gj = json.load(f)
    lon180 = ((lon + 180.0) % 360.0) - 180.0          # GeoJSON is -180..180
    units = []
    for feat in gj["features"]:
        props, geom = feat.get("properties", {}), feat["geometry"]
        polys = geom["coordinates"] if geom["type"] == "MultiPolygon" else [geom["coordinates"]]
        flat = [pt for poly in polys for ring in poly for pt in ring]
        arr = np.asarray(flat, dtype=float)
        # bbox pre-filter keeps the vertex loop cheap
        cand = ((lon180 >= arr[:, 0].min()) & (lon180 <= arr[:, 0].max()) &
                (lat >= arr[:, 1].min()) & (lat <= arr[:, 1].max()))
        mask = np.zeros(lon.shape, dtype=bool)
        if cand.any():
            px, py = lon180[cand], lat[cand]
            hit = np.zeros(px.shape, dtype=bool)
            for poly in polys:
                hit |= _polygon_contains(px, py, poly)
            mask[cand] = hit
        units.append((str(props.get(id_field, "?")), str(props.get(name_field, "?")), mask))
    return units


def box_unit(ds, box):
    from low_level_jets import box_cells          # wrap-aware; see the note there
    lat = ds["latitude"].values
    m = box_cells(ds, box)
    name = box.get("note", box["id"]).split(";")[0].split("—")[0].strip().rstrip(".")
    return [(box["id"], name, m)]


# --------------------------------------------------------------------------------------
# per-member diagnostics (all region-pooled BEFORE classification — Part-0 admissibility)
# --------------------------------------------------------------------------------------
def _region_step_mean(ds, var, steps, cells):
    """(member, step) region-mean of `var`."""
    da = ds[var].isel(time=steps, values=np.where(cells)[0])
    return np.nanmean(da.values, axis=2)


def _region_window_mean(ds, var, steps, cells):
    """(member,) region+window mean of `var`."""
    return np.nanmean(_region_step_mean(ds, var, steps, cells), axis=1)


def circulation(ds, steps, cells, k=2, seed=1234):
    """C: k-means on the z500 anomaly PATTERN, clusters ordered by a circulation-only index.

    The ordering index deliberately excludes humidity so that C stays independent of M's
    moisture content — otherwise the C->M table would be partly circular."""
    z = np.nanmean(ds["z_500"].isel(time=steps, values=np.where(cells)[0]).values, axis=1)
    anom = z - z.mean(axis=0, keepdims=True)                      # (member, cell)
    _, labels = kmeans2(whiten(anom.astype(float)), k, seed=seed, minit="++", missing="warn")

    msl = _region_window_mean(ds, "msl", steps, cells)
    v850 = _region_window_mean(ds, "v_850", steps, cells)
    z_ = lambda a: (a - a.mean()) / (a.std() + 1e-12)
    idx = z_(-(msl - msl.mean())) + z_(v850)                      # low pressure + southerly
    order = np.argsort([idx[labels == c].mean() if (labels == c).any() else np.inf
                        for c in range(k)])
    remap = np.empty(k, dtype=int)
    remap[order] = np.arange(k)                                   # ascending -> state 0..k-1
    return remap[labels.astype(int)], anom, idx


def ivt(ds, steps, cells):
    """M: |IVT| per (member, step), region-mean. Vertical integral of q*(u,v) over 13 levels."""
    sel = np.where(cells)[0]
    dp = np.abs(np.gradient(np.array(SHARED_LEV, dtype=float) * 100.0))     # Pa
    ivtx = ivty = None
    for k, lev in enumerate(SHARED_LEV):
        q = ds[f"q_{lev}"].isel(time=steps, values=sel).values
        u = ds[f"u_{lev}"].isel(time=steps, values=sel).values
        v = ds[f"v_{lev}"].isel(time=steps, values=sel).values
        if ivtx is None:
            ivtx, ivty = np.zeros_like(q), np.zeros_like(q)
        ivtx += q * u * dp[k] / G
        ivty += q * v * dp[k] / G
    return np.nanmean(np.hypot(ivtx, ivty), axis=2)               # (member, step)


def rainfall_environment(ds, steps, cells):
    """R: the CAPE substitute — ascent + lapse rate + mid-level RH, standardised and summed."""
    w500 = _region_window_mean(ds, "w_500", steps, cells)
    w700 = _region_window_mean(ds, "w_700", steps, cells)
    t850 = _region_window_mean(ds, "t_850", steps, cells)
    t500 = _region_window_mean(ds, "t_500", steps, cells)
    q700 = _region_window_mean(ds, "q_700", steps, cells)
    t700 = _region_window_mean(ds, "t_700", steps, cells)

    ascent = -(w500 + w700) / 2.0                                 # Pa/s; w<0 is ascent
    lapse = t850 - t500                                           # K
    es = 610.94 * np.exp(17.625 * (t700 - 273.15) / (t700 - 30.11))   # Tetens
    qs = 0.622 * es / (70000.0 - 0.378 * es)
    rh700 = np.clip(q700 / np.maximum(qs, 1e-12), 0.0, 1.5)

    # Physically anchored sub-scores — NOT z-scores. A z-score is a within-ensemble rank and
    # would make the index (and therefore the node's states) informationless in aggregate.
    # Anchors: 0.01 Pa/s ~ modest large-scale ascent; 26 K ~ typical tropical 850-500 lapse;
    # RH700 0.50 ~ the dry/moist mid-level divide. Scales are the spread each term should have
    # to count as one "unit" of support.
    idx = (ascent / 0.01) + ((lapse - 26.0) / 3.0) + ((rh700 - 0.50) / 0.15)
    return idx, ascent, lapse, rh700


def precipitation(ds, steps, cells):
    """P: tp is INTERVAL per-6h in metres (verified). -> total / peak-24h / wet fraction, mm."""
    tp = _region_step_mean(ds, "tp", steps, cells) * 1000.0       # (member, step) mm/6h
    total = tp.sum(axis=1)
    if tp.shape[1] >= 4:
        roll24 = np.stack([tp[:, i:i + 4].sum(axis=1) for i in range(tp.shape[1] - 3)], axis=1)
        peak24 = roll24.max(axis=1)
    else:
        peak24 = total
    wetfrac = (tp > 0.2).mean(axis=1)
    return total, peak24, wetfrac


def runoff(ds, steps, cells):
    """RO: ro is INTERVAL per-6h in m over land -> window total in mm."""
    return _region_step_mean(ds, "ro", steps, cells).sum(axis=1) * 1000.0


def low_level_jets(ds, steps, reg, n_mem):
    """Somali (Findlater) and Turkana jets by PHYSICAL core speed — loc.llj.v1.

    FIXED geographic indices, not properties of the analysis region: they describe the moisture
    pump feeding the whole Horn, so they belong to the lead window, not to a unit.

    Retires the old `loc.somalijet.v1` sign test (box MEAN of v_850, classified against the
    ensemble median). A box mean dilutes a jet CORE, the meridional component is not the
    literature criterion (that is speed |V|), and an ensemble-median binary is a RANK. Here the
    core is the 95th percentile of |V| over the box and the thresholds are physical:
    Somali 15/25 m/s at 850 hPa, Turkana 12/16.8/30 m/s at 925 hPa. See low_level_jets.py."""
    from low_level_jets import box_cells, core_speed, summarise
    llj = reg["process_loci"]["low_level_jets"]
    hours = np.array([(t + 1) * 6 for t in steps])
    noct = ((hours + 3) % 24 == 3)                    # 00 UTC == 0300 East Africa Time
    out = {}
    for jname, key, rkey, direction in (
            ("somali_findlater", "somali_jet", "SOMALI_JET", ("v", "pos")),
            ("turkana", "turkana_jet", "TURKANA", ("u", "neg"))):
        spec, box = llj[key], reg["regions"][rkey]
        cells = box_cells(ds, box)
        core, mx, ubar, vbar = core_speed(ds, spec["level"], steps, cells)
        thr = {k: float(v) for k, v in spec["thresholds_m_s"].items()}
        rec = summarise(core, mx, ubar, vbar, thr,
                        noct if jname == "turkana" else None, direction, n_mem)
        rec.update({"box": box["id"], "level_hPa": spec["level"],
                    "n_cells": int(cells.sum())})
        out[jname] = rec
    return out


def soil_water(ds, step, cells):
    """A (proxy): depth-weighted swvl1/swvl2 at the first step of the window, land cells."""
    sel = np.where(cells)[0]
    s1 = ds["swvl1"].isel(time=step, values=sel).values           # (member, cell)
    s2 = ds["swvl2"].isel(time=step, values=sel).values
    return np.nanmean((0.07 * s1 + 0.21 * s2) / 0.28, axis=1)


# --------------------------------------------------------------------------------------
# state assignment — ABSOLUTE PHYSICAL THRESHOLDS, versioned in the registry
# --------------------------------------------------------------------------------------
# Why not within-ensemble quantiles: a quantile state is a RANK, so the fraction of members
# in each state is fixed by the cut points no matter what the forecast says (quartiles always
# give [.25,.25,.25,.25]). The unit-level soft evidence would then be a constant vector and
# carry zero information — verified empirically on cycle-20260730_0000, where every
# quantile-based node returned identical fractions for both lead windows. States must be
# defined against a FIXED reference. Until a model climatology exists (Phase 4 / Route B),
# that reference is these physically-anchored thresholds. They are provisional-by-construction
# and versioned: changing any number mints a NEW registry id, and counts never pool across ids.
THRESH = {
    # region-mean |IVT|, kg m-1 s-1. Anchored to the standard AR magnitude criterion
    # (Guan & Waliser, 250) at 0.6x / 1.0x / 1.4x — NOT to this sample's quantiles.
    "ivt": [150.0, 250.0, 350.0],
    "ivt_persistence_min": 0.5,          # extreme_persistent additionally requires duration
    # rainfall-environment index (sum of three physically-anchored sub-scores; see below)
    "rgen": [-1.0, 0.5, 2.0],
    # precipitation RATE, mm/day — window-length independent so W1 and W2 are comparable
    "tp_rate": [1.0, 4.0],
    "tp_episodic_share": 0.5,            # one 24 h peak delivering >=50% of the window total
    # volumetric soil water, m3/m3. Anchored to loam wilting point (~0.12) and field
    # capacity (~0.30), not to the ensemble.
    "swvl": [0.12, 0.20, 0.28, 0.35],
    # runoff RATE, mm/day
    "ro_rate": [0.05, 0.25, 0.75],
}


def _cut(x, edges):
    """Assign state 0..len(edges) by fixed threshold edges."""
    return np.searchsorted(np.asarray(edges, dtype=float), x, side="right").astype(int)


def states_M(ivt_win, persistence):
    s = _cut(ivt_win, THRESH["ivt"])
    s[(s == 3) & (persistence < THRESH["ivt_persistence_min"])] = 2   # extreme needs duration
    return s


def states_R(idx):
    return _cut(idx, THRESH["rgen"])


def states_P(total_mm, peak24_mm, days):
    """Severity from the rate; the heavy class is split by SHAPE (scale-free ratio)."""
    s = _cut(total_mm / max(days, 1e-9), THRESH["tp_rate"])           # 0,1,2
    share = peak24_mm / np.maximum(total_mm, 1e-9)
    out = s.copy()
    heavy = s == 2
    out[heavy] = np.where(share[heavy] > THRESH["tp_episodic_share"], 2, 3)
    return out


def states_A(sw):
    return _cut(sw, THRESH["swvl"])


def states_RO(ro_mm, days):
    return _cut(ro_mm / max(days, 1e-9), THRESH["ro_rate"])


def fractions(states, k):
    """Ensemble state fractions = the node's soft evidence."""
    return np.bincount(states, minlength=k)[:k] / float(len(states))


# --------------------------------------------------------------------------------------
# explanation record — the primary product
# --------------------------------------------------------------------------------------
def state_resolution(x, cuts, k):
    """How much of a k-state partition this ensemble actually resolves.

    mrp_stability_test.py found the 4-state partitions of M/R/P are mostly NOT resolved: on
    average 1.3 (M) to 2.0 (R, P) of 4 states are occupied, and the cuts that fall inside the
    ensemble range usually land on the density PEAK (density-at-cut up to 0.99), so the split
    is decided by where the line sits rather than by the forecast. Thresholds are deliberately
    NOT re-tuned to this cycle's density valleys — that is the threshold-re-tuning failure the
    critique names (C3). Instead every record carries this flag, so a fragile split is visible.
    """
    lab = np.searchsorted(np.asarray(cuts, float), x, side="right")
    occ = np.bincount(lab, minlength=k)[:k]
    n_occ = int((occ > 0).sum())
    spread = float(x.max() - x.min())
    worst_density, worst_cut = 0.0, None
    if spread > 0:
        bw = 1.06 * max(float(np.std(x, ddof=1)), 1e-12) * len(x) ** (-1 / 5)
        grid = np.linspace(x.min(), x.max(), 256)
        dens = lambda t: np.exp(-0.5 * ((np.asarray(t)[..., None] - x) / bw) ** 2).sum(-1)
        peak = float(dens(grid).max())
        for c in cuts:
            if x.min() <= c <= x.max() and peak > 0:
                d = float(dens(np.array([float(c)]))[0] / peak)
                if d > worst_density:
                    worst_density, worst_cut = d, float(c)
    flag = ("degenerate" if n_occ <= 1 else
            "fragile" if worst_density > 0.5 else "resolved")
    return {"occupancy": occ.tolist(), "n_states_occupied": n_occ, "n_states_defined": k,
            "worst_cut": worst_cut, "density_at_worst_cut": round(worst_density, 3),
            "flag": flag}


def dist(x):
    """Compact distribution summary of a per-member quantity."""
    q = np.percentile(x, [0, 10, 50, 90, 100])
    return {"min": round(float(q[0]), 3), "p10": round(float(q[1]), 3),
            "median": round(float(q[2]), 3), "p90": round(float(q[3]), 3),
            "max": round(float(q[4]), 3), "mean": round(float(np.mean(x)), 3)}


def narrate(unit_name, hours, n, ess, C, ivt_win, persistence, r_idx, rate, jet, res=None):
    """Generated explanation sentence: counts over members, ESS always attached.

    Deliberately plain and quantitative — this is the 'translate/communicate' step the MLWP
    acceptance argument asks for, and every clause is traceable to a locus in the registry."""
    kC = len(STATES["C"])
    k_top = int(np.argmax(np.bincount(C, minlength=kC)))
    n_top = int(np.bincount(C, minlength=kC)[k_top])
    n_ar = int((ivt_win > THRESH["ivt"][1]).sum())
    n_supp = int((r_idx > THRESH["rgen"][1]).sum())
    n_heavy = int((rate >= THRESH["tp_rate"][1]).sum())
    jet_txt = None
    if jet:
        bits = []
        for jname, label, thr_key in (("somali_findlater", "Somali jet", "present_15.0m_s"),
                                      ("turkana", "Turkana jet", "present_12.0m_s")):
            r = jet.get(jname)
            if not r:
                continue
            wm = r["core_speed_m_s"]["window_mean_of_core"]
            t = r["threshold_counts"].get(thr_key, {})
            bits.append(f"{label} core {wm['min']}-{wm['max']} m/s at {r['level_hPa']} hPa, "
                        f"{t.get('n_members_reaching', 0)} of {n} above "
                        f"{thr_key.split('_')[1]} for "
                        f"{t.get('mean_duration_frac', 0)*100:.0f}% of steps")
        jet_txt = "; ".join(bits) if bits else None
    s = (f"{unit_name}, hours {hours[0]}-{hours[1]}: "
         f"{n_top} of {n} members place the circulation in regime {k_top} "
         f"({STATES['C'][k_top].replace('_', ' ')}); "
         f"region-mean |IVT| {ivt_win.min():.0f}-{ivt_win.max():.0f} kg/m/s with "
         f"{n_ar} of {n} above the 250 corridor threshold "
         f"(mean duration above it {persistence.mean()*100:.0f}% of steps); "
         f"rainfall environment supportive in {n_supp} of {n}; "
         f"precipitation {rate.min():.2f}-{rate.max():.2f} mm/day, {n_heavy} of {n} heavy")
    if jet_txt:
        s += f"; {jet_txt}"
    s += (f". Ensemble spread has ESS ~ {ess:.1f} of {n}, so treat every count above as "
          f"roughly {ess:.0f} independent draws, not {n}.")
    if res:
        weak = [f"{node} {r['n_states_occupied']}/{r['n_states_defined']} states"
                f"{' (cut through the density peak)' if r['flag'] == 'fragile' else ''}"
                for node, r in res.items() if r["flag"] != "resolved"]
        if weak:
            s += (" State resolution is limited — " + "; ".join(weak) +
                  " — so read the ranges and threshold counts above, not a multi-way "
                  "state split (mrp_stability_test.py).")
    return s


# --------------------------------------------------------------------------------------
# ESS (B3) — reused from cpt_build.py: participation ratio of the member spread spectrum
# --------------------------------------------------------------------------------------
def ensemble_ess(anom):
    x = anom.astype(float)
    lam = np.linalg.eigvalsh(x @ x.T)
    lam = lam[lam > 1e-10]
    if lam.size == 0:
        return 1.0, 0.0
    pr = float((lam.sum() ** 2) / (lam ** 2).sum())
    ceiling = min(anom.shape[0] - 1, anom.shape[1])
    return pr, (float(1.0 - pr / ceiling) if ceiling > 0 else 0.0)


def counted_cpt(parent, child, kp, kc, ess, n):
    """P(child | parent) from the JOINT member sample, counts down-weighted to the ESS."""
    counts = np.zeros((kp, kc))
    for m in range(len(parent)):
        counts[parent[m], child[m]] += 1.0
    counts_eff = counts * (ess / n)
    alpha = max(1.0, ess) / counts.size
    post = counts_eff + alpha
    return counts, post / post.sum(axis=1, keepdims=True)


# --------------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--store", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--explain-out", default=None,
                    help="PRIMARY: JSON circulation explanation record")
    ap.add_argument("--regions", default="IGAD_EA",
                    help="comma-separated registry region keys, or 'all'")
    ap.add_argument("--k-regimes", type=int, default=2,
                    help="circulation clusters. Default 2: k_regimes_test.py found k>=3 "
                         "UNSTABLE on this spread (ARI 0.56 at k=4 vs 0.82 at k=2). Any other "
                         "value is experimental and outside the registry contract.")
    ap.add_argument("--out", default=None, help="downstream BN evidence CSV")
    ap.add_argument("--member-out", default=None, help="per-member storyline sidecar CSV")
    ap.add_argument("--cpt-out", default=None, help="counted process CPT artifact (netCDF)")
    ap.add_argument("--cpt-json", default=None,
                    help="same counted CPTs as JSON, for the Julia BN's --cpt-json override")
    ap.add_argument("--adm1", default=None, help="admin-1 GeoJSON (optional)")
    ap.add_argument("--adm1-id", default="GID_1")
    ap.add_argument("--adm1-name", default="NAME_1")
    ap.add_argument("--min-cells", type=int, default=20,
                    help="skip units with fewer N320 cells (Part-0: no sub-grid units)")
    args = ap.parse_args()

    reg = load_registry()
    kC = args.k_regimes
    if kC != len(STATES["C"]):
        print(f"[warn] --k-regimes={kC} is outside the registry contract (k=2, "
              f"loc.circ.v2); using generic labels. See k_regimes_test.py.")
        STATES["C"] = [f"regime_{i}" for i in range(kC)]

    repo, ds = open_cycle(args.store, args.tag)
    lat, lon = ds["latitude"].values, ds["longitude"].values
    n_mem = ds.sizes["member"]

    if not (args.explain_out or args.out):
        ap.error("nothing to do: pass --explain-out (primary) and/or --out (BN evidence)")

    keys = (list(reg["regions"]) if args.regions == "all"
            else [k.strip() for k in args.regions.split(",")])
    for k in keys:
        if k not in reg["regions"]:
            ap.error(f"unknown region '{k}'; registry has {list(reg['regions'])}")
    boxes = [reg["regions"][k] for k in keys]

    if args.adm1:
        # admin-1 units are clipped to the FIRST named region
        units = adm1_units(args.adm1, lon, lat, args.adm1_id, args.adm1_name)
        clip = box_unit(ds, boxes[0])[0][2]
        units = [(i, n, m & clip) for i, n, m in units]
    else:
        units = [u for b in boxes for u in box_unit(ds, b)]
    units = [(i, n, m) for i, n, m in units if m.sum() >= args.min_cells]
    print(f"[units] {len(units)} spatial unit(s) with >= {args.min_cells} N320 cells")

    rows, mrows, cpt_blocks = [], [], []
    explain = {"meta": {
        "source_store": args.store, "source_tag": args.tag,
        "registry_version": reg["registry_version"],
        "product": reg["explanation_products"]["circulation_record"]["id"],
        "typing": "within-cycle ensemble counts over circulation states; n_cycles=1; "
                  "NOT a calibrated probability and NOT a risk statement",
        "honesty_rule": reg["explanation_products"]["circulation_record"]["honesty_rule"],
        "created": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "generator": "s2s_bn_evidence_prep.py",
    }, "windows": {}}
    for wname, w in reg["lead_windows"].items():
        steps = window_steps(ds, w["hours"])
        days = len(steps) * 6.0 / 24.0        # rate normalisation makes W1/W2 comparable
        land = land_mask(ds, steps[0])
        print(f"\n[{wname}] {w['id']} hours {w['hours']}: {len(steps)} steps "
              f"({(steps[0]+1)*6}-{(steps[-1]+1)*6} h)")

        jet = low_level_jets(ds, steps, reg, n_mem)     # fixed indices, per window
        explain["windows"][wname] = {
            "id": w["id"], "hours": w["hours"], "n_steps": len(steps), "days": round(days, 2),
            "low_level_jets": jet,
            "regions": {},
        }

        for uid, uname, umask in units:
            cells = umask
            lcells = umask & land
            if lcells.sum() < args.min_cells:      # land nodes need land
                lcells = cells

            C, z_anom, circ_idx = circulation(ds, steps, cells, k=kC)
            ess, redundancy = ensemble_ess(z_anom)

            ivt_step = ivt(ds, steps, cells)                       # (member, step)
            ivt_win = ivt_step.mean(axis=1)
            # duration above the same ABSOLUTE 'enhanced' threshold — not an ensemble quantile
            persistence = (ivt_step > THRESH["ivt"][1]).mean(axis=1)
            M = states_M(ivt_win, persistence)

            r_idx, ascent, lapse, rh700 = rainfall_environment(ds, steps, cells)
            R = states_R(r_idx)

            total, peak24, wetfrac = precipitation(ds, steps, cells)
            P = states_P(total, peak24, days)

            sw = soil_water(ds, steps[0], lcells)
            A = states_A(sw)

            ro_mm = runoff(ds, steps, lcells)
            RO = states_RO(ro_mm, days)
            rate = total / days

            # How much of each 4-state partition this ensemble actually resolves (see
            # mrp_stability_test.py). Carried into the record so a fragile or degenerate
            # split is visible in the product rather than discovered later.
            resolution = {
                "M": state_resolution(ivt_win, THRESH["ivt"], len(STATES["M"])),
                "R": state_resolution(r_idx, THRESH["rgen"], len(STATES["R"])),
                "P": state_resolution(rate, THRESH["tp_rate"], len(STATES["P"])),
            }

            # ---- PRIMARY: the circulation explanation record ----
            explain["windows"][wname]["regions"][uid] = {
                "name": uname, "n_cells": int(cells.sum()),
                "n_land_cells": int(lcells.sum()), "n_members": n_mem,
                "ess": round(float(ess), 2), "redundancy": round(float(redundancy), 3),
                "regime_states": STATES["C"],
                "regime_occupancy": np.bincount(C, minlength=kC).tolist(),
                "regime_circulation_index": [
                    round(float(circ_idx[C == k].mean()), 3) if (C == k).any() else None
                    for k in range(kC)],
                "k_regimes": kC,
                "k_justification": "k=2: subsample-ARI 0.82 vs 0.56 at k=4 "
                                   "(k_regimes_test.py); PC1 holds 72-88% of the variance",
                "ivt_kg_m_s": {**dist(ivt_win),
                               "n_above_250": int((ivt_win > THRESH["ivt"][1]).sum()),
                               "n_above_350": int((ivt_win > THRESH["ivt"][2]).sum()),
                               "persistence_mean": round(float(persistence.mean()), 3)},
                "rainfall_environment": {
                    "index": dist(r_idx), "ascent_pa_s": dist(ascent),
                    "lapse_850_500_K": dist(lapse), "rh700": dist(rh700),
                    "n_supportive": int((r_idx > THRESH["rgen"][1]).sum())},
                "precipitation_mm_day": {**dist(rate),
                                         "n_heavy": int((rate >= THRESH["tp_rate"][1]).sum()),
                                         "wet_step_frac_mean": round(float(wetfrac.mean()), 3)},
                "state_resolution": resolution,
                "narrative": narrate(uname, w["hours"], n_mem, ess, C, ivt_win,
                                     persistence, r_idx, rate, jet, resolution),
            }

            row = {
                "unit_id": uid, "unit_name": uname, "lead_window": wname,
                "lead_window_id": w["id"], "lead_hours": f"{w['hours'][0]}-{w['hours'][1]}",
                "n_cells": int(cells.sum()), "n_land_cells": int(lcells.sum()),
                "n_members": n_mem, "ess": round(float(ess), 3),
                "redundancy": round(float(redundancy), 3),
                "ivt_mean_kg_m_s": round(float(ivt_win.mean()), 1),
                "ivt_persistence": round(float(persistence.mean()), 3),
                "rgen_index": round(float(r_idx.mean()), 3),
                "ascent_pa_s": round(float(ascent.mean()), 5),
                "lapse_850_500_K": round(float(lapse.mean()), 2),
                "rh700": round(float(rh700.mean()), 3),
                "window_days": round(days, 2),
                "tp_window_mm": round(float(total.mean()), 2),
                "tp_rate_mm_day": round(float(total.mean()) / days, 3),
                "tp_peak24_mm": round(float(peak24.mean()), 2),
                "tp_wet_step_frac": round(float(wetfrac.mean()), 3),
                "soil_water_m3m3": round(float(sw.mean()), 4),
                "runoff_window_mm": round(float(ro_mm.mean()), 3),
                "runoff_rate_mm_day": round(float(ro_mm.mean()) / days, 4),
                "antecedent_source": "MODEL_SOIL_PROXY(swvl1,swvl2)@window_start",
            }
            for node, s in (("C", C), ("M", M), ("R", R), ("P", P), ("A", A), ("RO", RO)):
                f = fractions(s, len(STATES[node]))
                for i, p in enumerate(f):
                    row[f"{node}_p{i+1}"] = round(float(p), 4)
            rows.append(row)

            if args.member_out:
                for m in range(n_mem):
                    mrows.append({
                        "unit_id": uid, "unit_name": uname, "lead_window": wname,
                        "member": int(ds["member"].values[m]) if "member" in ds.coords else m,
                        "C_state": STATES["C"][C[m]], "M_state": STATES["M"][M[m]],
                        "R_state": STATES["R"][R[m]], "P_state": STATES["P"][P[m]],
                        "A_state": STATES["A"][A[m]], "RO_state": STATES["RO"][RO[m]],
                        "C_idx": int(C[m]) + 1, "M_idx": int(M[m]) + 1, "R_idx": int(R[m]) + 1,
                        "P_idx": int(P[m]) + 1, "A_idx": int(A[m]) + 1, "RO_idx": int(RO[m]) + 1,
                        "ivt_kg_m_s": round(float(ivt_win[m]), 1),
                        "ivt_persistence": round(float(persistence[m]), 3),
                        "rgen_index": round(float(r_idx[m]), 3),
                        "tp_window_mm": round(float(total[m]), 2),
                        "tp_peak24_mm": round(float(peak24[m]), 2),
                        "runoff_mm": round(float(ro_mm[m]), 3),
                        "soil_water_m3m3": round(float(sw[m]), 4),
                    })

            if args.cpt_out or args.cpt_json:
                cm_c, cm_p = counted_cpt(C, M, kC, 4, ess, n_mem)
                mr_c, mr_p = counted_cpt(M, R, 4, 4, ess, n_mem)
                mp_c, mp_p = counted_cpt(M, P, 4, 4, ess, n_mem)
                rp_c, rp_p = counted_cpt(R, P, 4, 4, ess, n_mem)
                cpt_blocks.append({
                    "unit_id": uid, "lead_window": wname, "ess": ess,
                    "CM_counts": cm_c, "CM_post": cm_p, "MR_counts": mr_c, "MR_post": mr_p,
                    "MP_counts": mp_c, "MP_post": mp_p, "RP_counts": rp_c, "RP_post": rp_p,
                })

            print(f"  {uid:12s} ESS={ess:4.1f} | C={fractions(C,kC).round(2).tolist()} "
                  f"M={fractions(M,4).round(2).tolist()} P={fractions(P,4).round(2).tolist()} "
                  f"| tp={total.mean():5.1f}mm ro={ro_mm.mean():5.2f}mm")

    if args.explain_out:
        with open(args.explain_out, "w") as f:
            json.dump(explain, f, indent=1)
        print(f"\n[explain] {args.explain_out}")
        for wname, wrec in explain["windows"].items():
            for rec in wrec["regions"].values():
                print(f"  * {rec['narrative']}")

    if args.out:
        pd.DataFrame(rows).to_csv(args.out, index=False)
        print(f"\n[out] {args.out}  rows={len(rows)}")
    if args.member_out:
        pd.DataFrame(mrows).to_csv(args.member_out, index=False)
        print(f"[out] {args.member_out}  rows={len(mrows)}")

    if args.cpt_out and cpt_blocks:
        keys = [(b["unit_id"], b["lead_window"]) for b in cpt_blocks]
        pack = lambda k: np.stack([b[k] for b in cpt_blocks])
        out = xr.Dataset(
            {name: (("block", "parent", "child"), pack(name)) for name in
             ["MR_counts", "MR_post", "MP_counts", "MP_post", "RP_counts", "RP_post"]},
            coords={"block": [f"{u}|{w}" for u, w in keys],
                    "parent": [f"s{i+1}" for i in range(4)],
                    "child": [f"s{i+1}" for i in range(4)],
                    "c_state": STATES["C"]},
        )
        # C has kC states (2), the rest have 4 — separate dims rather than padding
        out["CM_counts"] = (("block", "c_state", "child"), pack("CM_counts"))
        out["CM_post"] = (("block", "c_state", "child"), pack("CM_post"))
        out["ess"] = ("block", np.array([b["ess"] for b in cpt_blocks]))
        out.attrs.update({
            "title": "Process-chain CPTs counted from the JOINT 50-member sample (one state "
                     "per member per window)",
            "network_direction": "chain is mechanism; see NETWORK_DIRECTION.md and "
                                 "S2S_BN_ONTOLOGY.md",
            "typing": "within-cycle joint structure, n_cycles=1 — NOT a P(E|H) calibration",
            "n_cycles": 1, "source_store": args.store, "source_tag": args.tag,
            "registry_version": reg["registry_version"],
            "loci": ",".join(reg["process_loci"][k]["id"] for k in reg["process_loci"]),
            "counting_rule": "one state per member per lead window; count over member dim (A1)",
            "k_regimes": kC,
            "k_justification": "k=2 selected by k_regimes_test.py: subsample-ARI 0.82 (stable) "
                               "vs 0.56 at k=4 (unstable); PC1 holds 72-88% of ensemble variance",
            "ess_rule": "participation ratio of the member covariance eigenspectrum; counts "
                        "down-weighted to ESS; raw counts retained (B3/B7)",
            "grid": "native N320 reduced Gaussian, quasi-equal-area cells; unweighted cell mean",
            "absent_fields": "cape, tcwv, lsm, z/slor/sdor (orography) — see blocked_loci",
            "created": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
            "generator": "s2s_bn_evidence_prep.py",
        })
        out.to_netcdf(args.cpt_out)
        print(f"[out] {args.cpt_out}  blocks={len(cpt_blocks)}")

    if args.cpt_json and cpt_blocks:
        # Parent-major rows: obj[parent][child]. The Julia loader hcat's them back to
        # (child, parent), which is the orientation RxInfer's DiscreteTransition wants.
        payload = {
            "meta": {
                "source_tag": args.tag, "registry_version": reg["registry_version"],
                "n_cycles": 1,
                "typing": "within-cycle joint member counts — NOT a P(E|H) calibration",
                "counting_rule": "one state per member per lead window (A1)",
                "k_regimes": kC, "c_states": STATES["C"],
            },
            "blocks": {
                f"{b['unit_id']}|{b['lead_window']}": {
                    "ess": float(b["ess"]),
                    "CM_post": b["CM_post"].tolist(), "CM_counts": b["CM_counts"].tolist(),
                    "MR_post": b["MR_post"].tolist(), "MR_counts": b["MR_counts"].tolist(),
                    "MP_post": b["MP_post"].tolist(), "RP_post": b["RP_post"].tolist(),
                }
                for b in cpt_blocks
            },
        }
        with open(args.cpt_json, "w") as f:
            json.dump(payload, f, indent=1)
        print(f"[out] {args.cpt_json}  blocks={len(cpt_blocks)}")


if __name__ == "__main__":
    main()

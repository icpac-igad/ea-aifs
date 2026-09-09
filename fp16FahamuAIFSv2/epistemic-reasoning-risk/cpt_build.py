#!/usr/bin/env python3
"""
Phase-1 CPT builder — the corrected Route-A counter (CPT_BUILD_CRITIQUE_AND_PLAN.md).

Builds the within-cycle co-occurrence structure of forecast-side loci from ONE Icechunk
cycle tag, honouring the critique's fixes:

  A1  one state per member per lead window BEFORE counting  -> n <= 50 (not 50*61)
  A3  tp verified interval (no de-accumulation); grade-A loci use NO climatology
  A5  derivation-independence: ONE moisture representative (IVT), not IVT+tcw+mfc+tp
  B1  lead-window stratification (every table carries exactly one window)
  B3  ESS = participation ratio of ensemble spread; counts down-weighted to ESS; raw kept
  B7  CPT artifact with the full provenance schema (raw counts, tags, registry ids, grades)
  A2  typed as within-cycle soft evidence (n_cycles=1), NOT the P(E|H) calibration

Parent (synoptic state) = z500 ensemble regime.  Child = IVT moisture state (high/low).
Somali-jet index is recorded as a RELATIVE diagnostic. NOTHING here is a grid-cell node
(Part-0 admissibility): every locus is region-pooled before classification.

Usage:
    python cpt_build.py --store /tank/projects/aifs-run/20260730_0000/icechunk_v2 \
        --tag cycle-20260730_0000 --out cpt_artifact_20260730.nc
"""
import argparse
import datetime as dt
import json
import os

import numpy as np
import xarray as xr
import icechunk
import yaml
from scipy.cluster.vq import kmeans2, whiten

G = 9.80665
SHARED_LEV = [50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000]
HERE = os.path.dirname(os.path.abspath(__file__))


def load_registry():
    with open(os.path.join(HERE, "discretization_registry.yaml")) as f:
        return yaml.safe_load(f)


def open_cycle(store, tag):
    repo = icechunk.Repository.open_or_create(icechunk.local_filesystem_storage(store))
    sess = repo.readonly_session(tag=tag)
    ds = xr.open_zarr(sess.store, consolidated=False, zarr_format=3)
    return repo, ds


def verify_tp_interval(ds, m_idx=0):
    """A3 gate: confirm tp is interval (per-step), not accumulated-since-init."""
    tp = ds["tp"].isel(member=m_idx)
    fin = np.where(np.isfinite(tp.isel(values=0).values))[0]
    a = tp.isel(time=int(fin[1])).values
    b = tp.isel(time=int(fin[2])).values
    frac_decrease = float(np.mean(b < a - 1e-9))
    interval = frac_decrease > 0.05   # accumulated fields never decrease
    return {"tp_is_interval": bool(interval), "frac_cells_decreasing": frac_decrease,
            "de_accumulation_required": bool(not interval)}


def region_mask(ds, box):
    lat = ds["latitude"].values
    lon = ds["longitude"].values % 360.0
    lo, hi = box["lat"]
    wlo, whi = box["lon"]
    return (lat >= lo) & (lat <= hi) & (lon >= wlo % 360) & (lon <= whi % 360)


def window_steps(ds, hours, m_idx=0):
    """Stored step indices whose lead hour is in [h0,h1] and actually written (finite)."""
    h0, h1 = hours
    nt = ds.sizes["time"]
    fin = np.isfinite(ds["2t"].isel(member=m_idx, values=0).values)
    steps = [t for t in range(nt) if fin[t] and h0 <= (t + 1) * 6 <= h1]
    return steps


def member_window_field(ds, var, steps, cells):
    """(member, cell) window-time-mean of `var` over the region cells."""
    da = ds[var].isel(time=steps, values=np.where(cells)[0]).mean("time")
    return da.values  # (member, ncell)


def ivt_region_mean(ds, steps, cells):
    """|IVT| per member, window-mean, region-mean. Vertical integral of q*(u,v)."""
    sel = np.where(cells)[0]
    dp = np.abs(np.gradient(np.array(SHARED_LEV) * 100.0))  # Pa
    ivtx = np.zeros((ds.sizes["member"], len(sel)))
    ivty = np.zeros_like(ivtx)
    for k, lev in enumerate(SHARED_LEV):
        q = ds[f"q_{lev}"].isel(time=steps, values=sel).mean("time").values
        u = ds[f"u_{lev}"].isel(time=steps, values=sel).mean("time").values
        v = ds[f"v_{lev}"].isel(time=steps, values=sel).mean("time").values
        ivtx += q * u * dp[k] / G
        ivty += q * v * dp[k] / G
    mag = np.hypot(ivtx, ivty)              # (member, cell)
    return np.nanmean(mag, axis=1)          # (member,)


def ensemble_ess(anom):
    """Effective # of independent members = participation ratio of the member covariance
    eigenspectrum:  ESS = (Σλ)² / Σλ².

    This is the honest estimator (B3). Mean-pairwise-correlation of ensemble-anomalies is
    NOT usable: subtracting the ensemble mean forces Σ_members anom = 0, so the average
    pairwise correlation is pinned near −1/(n−1) ≈ 0 regardless of real redundancy. The
    participation ratio instead measures how many distinct directions the members span:
    diverse members → ESS≈n−1; a few dominant patterns → ESS small.
    """
    # member x member Gram matrix of the regional anomaly maps
    x = anom.astype(float)
    G_ = x @ x.T
    lam = np.linalg.eigvalsh(G_)
    lam = lam[lam > 1e-10]
    if lam.size == 0:
        return 1.0, 0.0
    pr = float((lam.sum() ** 2) / (lam ** 2).sum())     # participation ratio
    # redundancy fraction: how far below the n-1 ceiling (rank after mean removal)
    ceiling = min(anom.shape[0] - 1, anom.shape[1])
    redundancy = float(1.0 - pr / ceiling) if ceiling > 0 else 0.0
    return pr, redundancy


def build_window(ds, reg, box_cells, hours, k_regimes, seed_offset):
    """One lead window -> parent(regime) x child(ivt) count tensor over members (A1)."""
    steps = window_steps(ds, hours)
    n = ds.sizes["member"]

    # --- parent: z500 ensemble regime (anomaly from ensemble mean, region-pooled) ---
    z = member_window_field(ds, "z_500", steps, box_cells)     # (member, cell)
    anom = z - z.mean(axis=0, keepdims=True)                   # anomaly from ENS mean
    e, redundancy = ensemble_ess(anom)                         # participation-ratio ESS (B3)
    feat = whiten(anom.astype(float))
    # deterministic seeding (no Math.random reliance): fixed seed per window
    _, labels = kmeans2(feat, k_regimes, seed=1234 + seed_offset, minit="++", missing="warn")
    regime = labels.astype(int)                               # (member,) one state/member

    # --- child: IVT moisture state, binary vs ensemble median (A5: single moisture locus) ---
    ivt = ivt_region_mean(ds, steps, box_cells)               # (member,)
    ivt_hi = (ivt > np.median(ivt)).astype(int)               # 0=low,1=high

    # --- relative diagnostic: Somali jet (labelled relative) ---
    v850 = None
    jet_box = (ds["latitude"].values >= -5) & (ds["latitude"].values <= 5) & \
              ((ds["longitude"].values % 360) >= 40) & ((ds["longitude"].values % 360) <= 55)
    if jet_box.any():
        vj = member_window_field(ds, "v_850", steps, jet_box).mean(axis=1)  # (member,)
        v850 = (vj > np.median(vj)).astype(int)

    # --- count over MEMBERS only (A1): each member contributes ONE (regime, ivt) pair ---
    counts = np.zeros((k_regimes, 2), dtype=float)
    for m in range(n):
        counts[regime[m], ivt_hi[m]] += 1.0

    # B3, done correctly: the ESS enters by DOWN-WEIGHTING the counts to the effective
    # sample (low ESS -> weak data -> wide posterior), not by inflating the prior. Raw
    # counts are still stored (B7) so any other policy can be reapplied later.
    counts_eff = counts * (e / n)                            # sums to ESS, not n
    alpha0 = max(1.0, e)                                     # prior ~ as strong as eff. data
    alpha = alpha0 / counts.size                             # symmetric Dirichlet per cell
    post = counts_eff + alpha
    post = post / post.sum(axis=1, keepdims=True)            # P(ivt | regime), ESS-discounted

    return {
        "steps": steps, "hours": hours, "n_members": n,
        "counts": counts, "counts_eff": counts_eff,
        "posterior_ivt_given_regime": post,
        "rho_bar": redundancy, "ess": e, "alpha0": alpha0,
        "regime_occupancy": np.bincount(regime, minlength=k_regimes).astype(int),
        "ivt_region_mean": ivt,
        "somali_jet_strong_frac": (float(v850.mean()) if v850 is not None else None),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--store", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--k-regimes", type=int, default=4)
    args = ap.parse_args()

    reg = load_registry()
    box = reg["regions"]["IGAD_EA"]
    repo, ds = open_cycle(args.store, args.tag)

    tp_check = verify_tp_interval(ds)
    print(f"[A3] tp interval: {tp_check['tp_is_interval']} "
          f"({tp_check['frac_cells_decreasing']*100:.1f}% cells decreasing) "
          f"-> de-accumulation required: {tp_check['de_accumulation_required']}")
    assert tp_check["tp_is_interval"], "tp appears accumulated; de-accumulate before rainfall loci"

    cells = region_mask(ds, box)
    print(f"[region] {box['id']}  N320 cells in box: {int(cells.sum())}")

    windows = reg["lead_windows"]
    results = {}
    for i, (name, w) in enumerate(windows.items()):
        r = build_window(ds, reg, cells, w["hours"], args.k_regimes, seed_offset=i)
        results[name] = (w, r)
        print(f"\n[{name}] {w['id']} hours {w['hours']} | {len(r['steps'])} steps | "
              f"n={r['n_members']} redundancy={r["rho_bar"]:.2f} ESS={r['ess']:.1f} "
              f"alpha0={r['alpha0']:.1f}")
        print(f"    regime occupancy (of {r['n_members']}): {r['regime_occupancy'].tolist()}")
        print(f"    P(IVT_high | regime): "
              f"{np.round(r['posterior_ivt_given_regime'][:,1],2).tolist()}")
        if r["somali_jet_strong_frac"] is not None:
            print(f"    Somali-jet strong fraction (RELATIVE): {r['somali_jet_strong_frac']:.2f}")

    # ---- CPT artifact (B7): raw counts + posterior + full provenance ----
    K = args.k_regimes
    wnames = list(windows.keys())
    counts = np.stack([results[w][1]["counts"] for w in wnames])          # (lw, regime, ivt)
    counts_eff = np.stack([results[w][1]["counts_eff"] for w in wnames])
    post = np.stack([results[w][1]["posterior_ivt_given_regime"] for w in wnames])
    occ = np.stack([results[w][1]["regime_occupancy"] for w in wnames])
    out = xr.Dataset(
        {
            "counts": (("lead_window", "regime", "ivt_state"), counts),
            "counts_eff": (("lead_window", "regime", "ivt_state"), counts_eff),
            "posterior_ivt_given_regime": (("lead_window", "regime", "ivt_state"), post),
            "regime_occupancy": (("lead_window", "regime"), occ),
            "ess": (("lead_window",), np.array([results[w][1]["ess"] for w in wnames])),
            "redundancy": (("lead_window",), np.array([results[w][1]["rho_bar"] for w in wnames])),
            "alpha0": (("lead_window",), np.array([results[w][1]["alpha0"] for w in wnames])),
        },
        coords={
            "lead_window": wnames,
            "regime": [f"regime_{i}" for i in range(K)],
            "ivt_state": ["ivt_low", "ivt_high"],
        },
    )
    out["lead_window_id"] = ("lead_window", [windows[w]["id"] for w in wnames])
    out["lead_window_hours"] = ("lead_window", [str(windows[w]["hours"]) for w in wnames])
    out.attrs.update({
        "title": "Phase-1 within-cycle CPT structure (forecast loci) — NOT a P(E|H) calibration",
        "network_direction": "generative: H latent, evidence children P(E|H); see NETWORK_DIRECTION.md v0.1.0",
        "n_cycles": 1,
        "typing": "within-cycle soft evidence about H for this cycle (A2). n_cycles=1.",
        "source_store": args.store,
        "source_tag": args.tag,
        "registry_version": reg["registry_version"],
        "parent_locus_id": reg["loci"]["z500_regime"]["id"],
        "child_locus_id": reg["loci"]["ivt_moisture"]["id"],
        "parent_grade": reg["loci"]["z500_regime"]["grade"],
        "child_grade": reg["loci"]["ivt_moisture"]["grade"],
        "region_id": box["id"],
        "counting_rule": "one modal state per member per lead window; count over member dim; n<=50 (A1)",
        "ess_rule": "ESS = participation ratio (sum lam)^2/sum lam^2 of member covariance eigenspectrum; redundancy = 1 - ESS/(n-1) (B3)",
        "alpha_rule": "counts down-weighted to ESS (counts_eff=counts*ESS/n); symmetric Dirichlet alpha0=max(1,ESS); RAW counts retained so any policy can be reapplied (B3/B7)",
        "derivation_independence": "IVT is the SOLE moisture representative; tcw/mfc/tp excluded (A5)",
        "tp_semantics": json.dumps(tp_check),
        "admissibility": "all loci region-pooled before classification; no grid-cell node (Part 0)",
        "created": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "generator": "cpt_build.py",
    })
    out.to_netcdf(args.out)
    print(f"\n[artifact] wrote {args.out} ({os.path.getsize(args.out)/1024:.0f} KB)")
    print("  typed: within-cycle structure, n_cycles=1 (NOT P(E|H)); raw counts retained.")


if __name__ == "__main__":
    main()

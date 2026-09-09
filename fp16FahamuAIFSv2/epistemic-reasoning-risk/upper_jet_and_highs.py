#!/usr/bin/env python3
"""
Tropical Easterly Jet and the two subtropical highs — by physical thresholds.

Applies the recipe that worked for the low-level jets (localised feature + PERCENTILE metric +
ABSOLUTE physical threshold) to the upper-level and boundary drivers, and is therefore also a
test of whether that recipe generalises or whether the LLJ result was luck.

  TEJ   `u_200` (secondary `u_150`), African sector 30-55E/0-20N and Indian sector 55-90E/5-20N.
        core = 95th percentile of the EASTERLY component (-u) over the box. NOT |V|: the TEJ is
        defined by easterly flow, so a speed magnitude would score westerly outliers as jet.
        Thresholds 15 / 25 / 35 m/s. Boreal-summer feature — absence in a DJF cycle is correct.

  HIGHS `msl` over the Mascarene box (20-35S, 45-90E) and St Helena box (20-35S, 25W-0).
        central pressure = box MAX of msl (hPa); the position of that maximum is the ridge
        centre and is reported as coordinates. Thresholds 1020 / 1024 / 1028 hPa.
        These EXPLAIN the low-level jets rather than duplicating them: the Mascarene High sets
        the cross-equatorial gradient that drives the Somali jet, St Helena drives the Congo
        westerly inflow. Their positions are the mechanism linking the two.

Reporting follows the two stability tests: counts above named physical thresholds with the ESS
attached, positions as coordinates, never a multi-way state split.

Usage:
    python upper_jet_and_highs.py --store /tank/projects/aifs-run/20260730_0000/icechunk_v2 \
        --tag cycle-20260730_0000 --out upper_jet_and_highs_20260730.json
"""
import argparse
import datetime as dt
import json

import numpy as np

from low_level_jets import box_cells


def easterly_core(ds, level, steps, cells, pct=95):
    """(member, step) TEJ core = pct-th percentile of the easterly component (-u)."""
    sel = np.where(cells)[0]
    u = ds[f"u_{level}"].isel(time=steps, values=sel).values      # (member, step, cell)
    east = -u                                                     # easterly positive
    return np.percentile(east, pct, axis=2), east.max(axis=2), np.nanmean(u, axis=2)


def high_metrics(ds, steps, cells, lat, lon, extent_thr_hpa=1020.0):
    """Box max msl (hPa) per (member, step), the lat/lon of that max, and the area fraction
    above `extent_thr_hpa`."""
    sel = np.where(cells)[0]
    p = ds["msl"].isel(time=steps, values=sel).values / 100.0     # (member, step, cell) hPa
    imax = np.argmax(p, axis=2)
    return (p.max(axis=2), p.mean(axis=2), np.percentile(p, 95, axis=2),
            lat[sel][imax], lon[sel][imax], (p >= extent_thr_hpa).mean(axis=2))


def threshold_block(per_member_peak, series, thresholds, n_mem):
    out = {}
    for name, thr in thresholds.items():
        sustained = (series >= thr).mean(axis=1)
        out[f"{name}_{thr}"] = {
            "n_members_reaching": int((per_member_peak >= thr).sum()), "n_members": n_mem,
            "mean_duration_frac": round(float(sustained.mean()), 3),
            "max_duration_frac": round(float(sustained.max()), 3),
        }
    return out


def rng3(x):
    return {"min": round(float(np.min(x)), 2), "median": round(float(np.median(x)), 2),
            "max": round(float(np.max(x)), 2)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--store", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    from s2s_bn_evidence_prep import load_registry, open_cycle, window_steps, ensemble_ess
    reg = load_registry()
    ujh = reg["process_loci"]["upper_jet_and_highs"]
    repo, ds = open_cycle(args.store, args.tag)
    lat = ds["latitude"].values
    lon = ds["longitude"].values % 360.0
    n_mem = ds.sizes["member"]

    tej_thr = {k: float(v) for k, v in ujh["tej"]["thresholds_m_s"].items()}
    high_thr = {k: float(v) for k, v in ujh["subtropical_highs"]["thresholds_hPa"].items()}
    lev = ujh["tej"]["level"]

    results = {"meta": {"locus": ujh["id"], "tag": args.tag,
                        "registry_version": reg["registry_version"],
                        "created": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
                        "generator": "upper_jet_and_highs.py"},
               "windows": {}}

    for wname, w in reg["lead_windows"].items():
        steps = window_steps(ds, w["hours"])
        print(f"\n=== {wname} {w['hours']} — {len(steps)} steps")
        wrec = {"hours": w["hours"], "n_steps": len(steps), "tej": {}, "highs": {}}

        for key in ("TEJ_AFRICA", "TEJ_INDIAN"):
            box = reg["regions"][key]
            cells = box_cells(ds, box)
            core, mx, ubar = easterly_core(ds, lev, steps, cells)
            ess, _ = ensemble_ess(core - core.mean(axis=0, keepdims=True))
            peak = core.max(axis=1)
            rec = {"box": box["id"], "level_hPa": lev, "n_cells": int(cells.sum()),
                   "ess": round(float(ess), 2),
                   "core_easterly_m_s": {"window_mean": rng3(core.mean(axis=1)),
                                         "peak_any_step": rng3(peak),
                                         "box_max_any_step": round(float(mx.max()), 2)},
                   "threshold_counts": threshold_block(peak, core, tej_thr, n_mem),
                   "direction": {"required": "easterly (u<0)",
                                 "n_members_satisfying": int((ubar.mean(axis=1) < 0).sum()),
                                 "box_mean_u_m_s": round(float(ubar.mean()), 2)}}
            wrec["tej"][key] = rec
            cm = rec["core_easterly_m_s"]
            print(f"  TEJ {key:11s} {box['id']:14s} {int(cells.sum()):5d} cells  ESS={ess:4.1f}")
            print(f"      core (p95 of -u): window-mean {cm['window_mean']['min']}-"
                  f"{cm['window_mean']['max']} m/s | peak {cm['peak_any_step']['min']}-"
                  f"{cm['peak_any_step']['max']} | box max {cm['box_max_any_step']}")
            for k, v in rec["threshold_counts"].items():
                print(f"      {k:14s} reached by {v['n_members_reaching']:2d}/{n_mem}, "
                      f"mean duration {v['mean_duration_frac']*100:3.0f}% of steps")
            print(f"      easterly in {rec['direction']['n_members_satisfying']}/{n_mem} "
                  f"(box-mean u = {rec['direction']['box_mean_u_m_s']} m/s)")

        for key in ("MASCARENE", "ST_HELENA"):
            box = reg["regions"][key]
            cells = box_cells(ds, box)
            pmax, pmean, p95, plat, plon, extent = high_metrics(ds, steps, cells, lat, lon)
            ess, _ = ensemble_ess(pmax - pmax.mean(axis=0, keepdims=True))
            peak = pmax.max(axis=1)
            rec = {"box": box["id"], "n_cells": int(cells.sum()), "ess": round(float(ess), 2),
                   "central_pressure_hPa": {"window_mean": rng3(pmax.mean(axis=1)),
                                            "peak_any_step": rng3(peak)},
                   "ridge_strength_hPa": {"box_mean": rng3(pmean.mean(axis=1)),
                                          "box_p95": rng3(p95.mean(axis=1))},
                   "threshold_caveat": "the 1020/1024/1028 hPa thresholds are CENTRAL-pressure "
                                       "values from the literature, but the box MAX over a "
                                       "large winter-hemisphere box exceeds them almost always "
                                       "- see the saturation in threshold_counts. Metric and "
                                       "threshold are mismatched; the fix is a climatology "
                                       "(E4), NOT re-tuning the numbers to fit (C3).",
                   "ridge_centre": {"lat_mean": round(float(plat.mean()), 2),
                                    "lat_sd_across_members": round(float(plat.mean(axis=1).std()), 3),
                                    "lon_mean": round(float(plon.mean()), 2),
                                    "lon_sd_across_members": round(float(plon.mean(axis=1).std()), 3)},
                   "extent_frac_above_1020": round(float(extent.mean()), 3),
                   "threshold_counts": threshold_block(peak, pmax, high_thr, n_mem)}
            wrec["highs"][key] = rec
            cp = rec["central_pressure_hPa"]
            print(f"  HIGH {key:10s} {box['id']:14s} {int(cells.sum()):5d} cells  ESS={ess:4.1f}")
            rs = rec["ridge_strength_hPa"]
            print(f"      central pressure (box max): window-mean {cp['window_mean']['min']}-"
                  f"{cp['window_mean']['max']} hPa | peak {cp['peak_any_step']['max']}")
            print(f"      ridge strength: box-mean {rs['box_mean']['min']}-"
                  f"{rs['box_mean']['max']} | box-p95 {rs['box_p95']['min']}-"
                  f"{rs['box_p95']['max']} hPa")
            print(f"      ridge centre {rec['ridge_centre']['lat_mean']}N "
                  f"{rec['ridge_centre']['lon_mean']}E "
                  f"(member sd {rec['ridge_centre']['lat_sd_across_members']}/"
                  f"{rec['ridge_centre']['lon_sd_across_members']} deg) | "
                  f"extent>1020 = {rec['extent_frac_above_1020']*100:.0f}% of box")
            for k, v in rec["threshold_counts"].items():
                print(f"      {k:18s} reached by {v['n_members_reaching']:2d}/{n_mem}, "
                      f"mean duration {v['mean_duration_frac']*100:3.0f}% of steps")

        results["windows"][wname] = wrec

    if args.out:
        with open(args.out, "w") as f:
            json.dump(results, f, indent=1)
        print(f"\n[out] {args.out}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Low-level jet identification — Somali (Findlater) and Turkana — by PHYSICAL core speed.

Replaces the weak `loc.somalijet.v1` index and unblocks the Turkana jet. What was wrong before
(and why this file exists):

  * the old index took a MEAN of `v_850` over a 15x10 deg box and classified it binary against
    the ENSEMBLE MEDIAN. A box mean dilutes a jet CORE — the same scale artefact that made the
    moisture node degenerate; the meridional component alone is not the literature criterion,
    which is wind SPEED |V| = sqrt(u^2+v^2); and an ensemble-median binary is a RANK, the
    informationless state definition the registry header condemns.
  * the Turkana jet was registered BLOCKED "needs orography". Too pessimistic: identifying the
    jet by a geographic box plus a speed threshold needs no orography whatsoever. Orography
    (slor/sdor/z-surface, genuinely absent from the store) is only needed to ATTRIBUTE the jet
    to terrain channelling and to compute gap geometry.

Method (registry `loc.llj.v1`):
  core speed = 95th PERCENTILE of |V| over the box cells, per member per step — a percentile,
  not a mean, so the core survives the pooling. Box max is reported alongside. Duration is the
  fraction of window steps whose core exceeds each named threshold.

  Somali / Findlater   box 40-55E, 5S-10N   850 hPa (~1.5 km, the Findlater core altitude)
                       thresholds 15 m/s present, 25 m/s strong; southerly check v > 0
  Turkana              box 34.5-38.5E, 1-6N  925 hPa (~0.75 km, the channel jet sits low)
                       thresholds 12 present, 16.8 typical nocturnal core, 30 extreme;
                       easterly check u < 0

  NOCTURNAL: the Turkana jet peaks around 0300 local. Local time is UTC+3 and the cycle is
  initialised 00Z, so every step whose lead hour is a multiple of 24 is 00 UTC = 0300 LOCAL.
  Those steps are reported separately — the store lands exactly on the campaign hour.

Reporting follows the lesson of k_regimes_test.py and mrp_stability_test.py: counts above NAMED
PHYSICAL thresholds, with the ESS attached — never a multi-way state split, which this ensemble
has been shown not to resolve.

Usage:
    python low_level_jets.py --store /tank/projects/aifs-run/20260730_0000/icechunk_v2 \
        --tag cycle-20260730_0000 --out low_level_jets_20260730.json
"""
import argparse
import datetime as dt
import json

import numpy as np

# NB: the s2s_bn_evidence_prep imports live inside main() so that the prep script can import
# the jet functions from here without a circular import.
UTC_OFFSET_H = 3          # East Africa Time, for the nocturnal (0300 local) subset


def box_cells(ds, box):
    """Wrap-aware box selection on the unstructured `values` axis.

    Two traps handled: an eastern edge of exactly 360 would become 0 under a naive `% 360`
    (selecting nothing — this silently emptied the St Helena box), and a box straddling the
    prime meridian (e.g. 340->20) needs an OR, not an AND."""
    lat = ds["latitude"].values
    lon = ds["longitude"].values % 360.0
    lo0, lo1 = float(box["lon"][0]) % 360.0, float(box["lon"][1]) % 360.0
    if lo1 == 0.0 and float(box["lon"][1]) != 0.0:
        lo1 = 360.0                                    # 360 is the end of the range, not 0
    inlon = (lon >= lo0) & (lon <= lo1) if lo0 <= lo1 else ((lon >= lo0) | (lon <= lo1))
    return (lat >= box["lat"][0]) & (lat <= box["lat"][1]) & inlon


def core_speed(ds, level, steps, cells, pct=95):
    """(member, step) core speed = pct-th percentile of |V| over the box; plus box max and
    the box-mean components for the direction check."""
    sel = np.where(cells)[0]
    u = ds[f"u_{level}"].isel(time=steps, values=sel).values      # (member, step, cell)
    v = ds[f"v_{level}"].isel(time=steps, values=sel).values
    spd = np.hypot(u, v)
    return (np.percentile(spd, pct, axis=2), spd.max(axis=2),
            np.nanmean(u, axis=2), np.nanmean(v, axis=2))


def summarise(core, mx, ubar, vbar, thresholds, noct_mask, direction, n_mem):
    """Per-member window statistics, then counts across members above each named threshold."""
    per_member_mean = core.mean(axis=1)                 # (member,)
    per_member_max = core.max(axis=1)
    box_max = mx.max(axis=1)

    out = {
        "core_speed_m_s": {
            "window_mean_of_core": {"min": round(float(per_member_mean.min()), 2),
                                    "median": round(float(np.median(per_member_mean)), 2),
                                    "max": round(float(per_member_mean.max()), 2)},
            "peak_core_any_step": {"min": round(float(per_member_max.min()), 2),
                                   "median": round(float(np.median(per_member_max)), 2),
                                   "max": round(float(per_member_max.max()), 2)},
            "box_max_any_step": round(float(box_max.max()), 2),
        },
        "threshold_counts": {},
        "direction": {},
    }

    for name, thr in thresholds.items():
        ever = int((per_member_max >= thr).sum())            # members reaching it at any step
        sustained = (core >= thr).mean(axis=1)               # duration fraction per member
        out["threshold_counts"][f"{name}_{thr}m_s"] = {
            "n_members_reaching": ever, "n_members": n_mem,
            "mean_duration_frac": round(float(sustained.mean()), 3),
            "max_duration_frac": round(float(sustained.max()), 3),
        }

    comp, sign = direction
    bar = ubar if comp == "u" else vbar
    ok = (bar.mean(axis=1) < 0) if sign == "neg" else (bar.mean(axis=1) > 0)
    out["direction"] = {
        "component": comp, "required": "easterly (u<0)" if sign == "neg" else "southerly (v>0)",
        "n_members_satisfying": int(ok.sum()), "n_members": n_mem,
        "box_mean_component_m_s": round(float(bar.mean()), 2),
    }

    if noct_mask is not None and noct_mask.any():
        nc = core[:, noct_mask]
        out["nocturnal_0300_local"] = {
            "n_steps": int(noct_mask.sum()),
            "core_mean": round(float(nc.mean()), 2),
            "core_max": round(float(nc.max()), 2),
            "n_members_reaching_typical_core": int(
                (nc.max(axis=1) >= thresholds.get("typical_core", np.inf)).sum()),
        }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--store", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    from s2s_bn_evidence_prep import load_registry, open_cycle, window_steps, ensemble_ess
    reg = load_registry()
    llj = reg["process_loci"]["low_level_jets"]
    repo, ds = open_cycle(args.store, args.tag)
    n_mem = ds.sizes["member"]

    jets = {
        "somali_findlater": (llj["somali_jet"], reg["regions"]["SOMALI_JET"], ("v", "pos")),
        "turkana": (llj["turkana_jet"], reg["regions"]["TURKANA"], ("u", "neg")),
    }

    results = {"meta": {
        "locus": llj["id"], "tag": args.tag, "registry_version": reg["registry_version"],
        "metric": llj["metric"], "reporting_rule": llj["reporting_rule"],
        "created": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
    }, "windows": {}}

    for wname, w in reg["lead_windows"].items():
        steps = window_steps(ds, w["hours"])
        hours = np.array([(t + 1) * 6 for t in steps])
        noct = ((hours + UTC_OFFSET_H) % 24 == UTC_OFFSET_H)     # 00 UTC == 0300 local
        print(f"\n=== {wname} {w['hours']} — {len(steps)} steps, "
              f"{int(noct.sum())} at 0300 local (00 UTC)")
        results["windows"][wname] = {"hours": w["hours"], "n_steps": len(steps),
                                     "n_nocturnal_steps": int(noct.sum()), "jets": {}}

        for jname, (spec, box, direction) in jets.items():
            cells = box_cells(ds, box)
            lev = spec["level"]
            core, mx, ubar, vbar = core_speed(ds, lev, steps, cells)
            ess, _ = ensemble_ess(core - core.mean(axis=0, keepdims=True))
            thr = {k: float(v) for k, v in spec["thresholds_m_s"].items()}
            rep = summarise(core, mx, ubar, vbar, thr,
                            noct if jname == "turkana" else None, direction, n_mem)
            rep.update({"box": box["id"], "n_cells": int(cells.sum()), "level_hPa": lev,
                        "ess": round(float(ess), 2)})
            results["windows"][wname]["jets"][jname] = rep

            cs = rep["core_speed_m_s"]
            print(f"  {jname:17s} {box['id']:15s} {lev} hPa  {int(cells.sum()):4d} cells  "
                  f"ESS={ess:.1f}")
            print(f"      core (p95 of |V|): window-mean {cs['window_mean_of_core']['min']}-"
                  f"{cs['window_mean_of_core']['max']} m/s | "
                  f"peak-any-step {cs['peak_core_any_step']['min']}-"
                  f"{cs['peak_core_any_step']['max']} | box max {cs['box_max_any_step']}")
            for k, v in rep["threshold_counts"].items():
                print(f"      {k:24s} reached by {v['n_members_reaching']:2d}/{v['n_members']} "
                      f"members, mean duration {v['mean_duration_frac']*100:.0f}% of steps")
            d = rep["direction"]
            print(f"      direction {d['required']:18s} satisfied by "
                  f"{d['n_members_satisfying']}/{d['n_members']} "
                  f"(box-mean {d['component']} = {d['box_mean_component_m_s']} m/s)")
            if "nocturnal_0300_local" in rep:
                nn = rep["nocturnal_0300_local"]
                print(f"      NOCTURNAL 0300 local ({nn['n_steps']} steps): core mean "
                      f"{nn['core_mean']}, max {nn['core_max']} m/s; "
                      f"{nn['n_members_reaching_typical_core']}/{n_mem} members reach the "
                      f"16.8 m/s campaign core")

    if args.out:
        with open(args.out, "w") as f:
            json.dump(results, f, indent=1)
        print(f"\n[out] {args.out}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
M / R / P stability test — are 4 threshold-defined states resolvable on this ensemble?

The follow-up the k-regimes test forced (HANDOVER.md §5.1c): k=4 circulation clustering was
shown unstable, and the same guilty-until-tested standard has to reach the other 4-state nodes:

    M  moisture transport        (region-mean |IVT|, cuts 150/250/350 kg m-1 s-1)
    R  rainfall environment      (index, cuts -1.0/0.5/2.0)
    P  precipitation forcing     (rate mm/day, cuts 1.0/4.0 + a shape split)

BUT THE K-TEST'S PROTOCOL DOES NOT TRANSFER, and saying why is half the point. Circulation
states come from k-means: the partition is *fitted to the members*, so re-drawing members can
change every label — which is exactly what the ARI measured. M/R/P states come from ABSOLUTE
thresholds fixed in the registry: a member's state is a deterministic function of its own value
and nothing else. Subsample re-labelling is therefore trivially perfect (ARI = 1) and measures
nothing. Applying the k-test here would produce a reassuring number with no content.

What can actually go wrong with a fixed-threshold partition on 50 low-DOF members:

  1 DEGENERACY      every member lands in one state -> the node carries no information this
                    cycle (already seen for M over the IGAD box).
  2 CUT FRAGILITY   a cut falls through a dense part of the ensemble distribution, so members
                    pile up on either side of an arbitrary line. Measured two ways: the kernel
                    density AT the cut relative to the distribution's peak, and how many members
                    sit within a narrow margin of it. This is the threshold analogue of the
                    silhouette score - a good cut sits in a valley, not through a peak.
  3 THRESHOLD       the reported state distribution is an artefact of where the line was drawn.
    SENSITIVITY     Measured by shifting every cut by +/-10% and +/-25% of the ensemble spread
                    and taking the total-variation distance between state distributions. The
                    thresholds are judgement calls (HANDOVER.md §6), so this asks how much the
                    product depends on that judgement.
  4 SAMPLING        "36 of 50 supportive" is an estimate. Bootstrap CIs are reported at BOTH
    ERROR           n = 50 (members as independent draws - the optimistic bound) and
                    n = round(ESS) (members as ~2 independent atmospheres - the pessimistic
                    bound). The honest interval lies between, and WHICH applies is an open
                    question, not something this script can settle: the ESS is the participation
                    ratio of the z500 spatial-pattern spectrum, which bounds the dimensionality
                    of the spread, not the exchangeability of members as draws.

Reads the per-member sidecar written by `s2s_bn_evidence_prep.py --member-out` (and the
unit-level CSV for ESS and window length), so it needs no store access and is cheap to re-run.

Usage:
    python mrp_stability_test.py --member-csv ev_members_k2.csv --unit-csv ev_k2.csv \
        --out mrp_stability_20260730.json
"""
import argparse
import datetime as dt
import json

import numpy as np
import pandas as pd

from s2s_bn_evidence_prep import THRESH, STATES

BOOT = 4000
RNG = np.random.default_rng(20260807)

# node -> (member-CSV column, cut list, needs rate normalisation by window days)
NODES = {
    "M": ("ivt_kg_m_s", THRESH["ivt"], False),
    "R": ("rgen_index", THRESH["rgen"], False),
    "P": ("tp_window_mm", THRESH["tp_rate"], True),
}


def kde_density(x, at, bw=None):
    """Gaussian KDE evaluated at `at`, normalised by the peak density over the data range."""
    x = np.asarray(x, dtype=float)
    if bw is None:                                  # Silverman
        bw = 1.06 * max(x.std(ddof=1), 1e-12) * len(x) ** (-1 / 5)
    grid = np.linspace(x.min(), x.max(), 512)
    dens = lambda t: np.exp(-0.5 * ((np.asarray(t)[..., None] - x) / bw) ** 2).sum(-1)
    peak = dens(grid).max()
    return float(dens(np.array([at]))[0] / peak) if peak > 0 else 0.0


def state_dist(x, cuts, k):
    lab = np.searchsorted(np.asarray(cuts, float), x, side="right")
    return np.bincount(lab, minlength=k)[:k] / float(len(x)), lab


def tvd(p, q):
    return float(0.5 * np.abs(np.asarray(p) - np.asarray(q)).sum())


def boot_ci(x, cuts, k, n, reps=BOOT, lo=5, hi=95):
    """Percentile CI on each state fraction, resampling `n` members with replacement."""
    fr = np.empty((reps, k))
    for b in range(reps):
        s = RNG.choice(x, n, replace=True)
        fr[b] = state_dist(s, cuts, k)[0]
    return (np.percentile(fr, lo, axis=0).round(3).tolist(),
            np.percentile(fr, hi, axis=0).round(3).tolist())


def analyse(x, cuts, k, ess):
    spread = float(x.max() - x.min())
    frac, lab = state_dist(x, cuts, k)
    occupied = int((np.bincount(lab, minlength=k)[:k] > 0).sum())

    cut_rep = []
    for c in cuts:
        margin = 0.05 * spread
        cut_rep.append({
            "cut": float(c),
            "inside_range": bool(x.min() <= c <= x.max()),
            "density_ratio_at_cut": round(kde_density(x, c), 3) if spread > 0 else 0.0,
            "n_within_5pct_spread": int((np.abs(x - c) < margin).sum()),
        })

    sens = {}
    for pct in (0.10, 0.25):
        worst = 0.0
        for sign in (-1, 1):
            shifted = [c + sign * pct * spread for c in cuts]
            worst = max(worst, tvd(frac, state_dist(x, shifted, k)[0]))
        sens[f"tvd_shift_{int(pct*100)}pct"] = round(worst, 3)

    n_eff = max(1, int(round(ess)))
    lo50, hi50 = boot_ci(x, cuts, k, 50)
    loE, hiE = boot_ci(x, cuts, k, n_eff)
    widest50 = float(max(h - l for l, h in zip(lo50, hi50)))
    widestE = float(max(h - l for l, h in zip(loE, hiE)))

    if occupied == 1:
        verdict = "DEGENERATE"
    elif sens["tvd_shift_10pct"] > 0.20:
        verdict = "FRAGILE"
    elif any(c["density_ratio_at_cut"] > 0.5 and c["inside_range"] for c in cut_rep):
        verdict = "FRAGILE(cut-through-peak)"
    else:
        verdict = "robust"

    return {
        "n_members": int(len(x)), "ess": round(float(ess), 2),
        "value_range": [round(float(x.min()), 3), round(float(x.max()), 3)],
        "state_fractions": frac.round(3).tolist(),
        "occupancy": np.bincount(lab, minlength=k)[:k].tolist(),
        "n_states_occupied": occupied, "n_states_defined": k,
        "cuts": cut_rep, "threshold_sensitivity": sens,
        "boot_ci90_n50": [lo50, hi50], "boot_ci90_n_eff": [loE, hiE],
        "widest_ci_n50": round(widest50, 3), "widest_ci_n_eff": round(widestE, 3),
        "verdict": verdict,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--member-csv", required=True)
    ap.add_argument("--unit-csv", required=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    mem = pd.read_csv(args.member_csv)
    unit = pd.read_csv(args.unit_csv).set_index(["unit_id", "lead_window"])

    results = {"meta": {
        "question": "are the 4 threshold-defined states of M/R/P resolvable on this ensemble?",
        "protocol_note": "the k-regimes ARI test does NOT transfer: threshold states are a "
                         "deterministic function of each member's own value, so subsample "
                         "re-labelling is trivially stable and measures nothing. Tested instead: "
                         "degeneracy, cut fragility, threshold sensitivity, sampling error.",
        "member_csv": args.member_csv, "unit_csv": args.unit_csv,
        "created": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
    }, "cases": {}}

    for (uid, win), g in mem.groupby(["unit_id", "lead_window"]):
        row = unit.loc[(uid, win)]
        ess = float(row["ess"])
        days = float(row["window_days"])
        print(f"\n=== {uid} / {win}   (n={len(g)}, ESS={ess:.2f}, {days:.2f} d)")
        case = {}
        for node, (col, cuts, is_rate) in NODES.items():
            x = g[col].to_numpy(dtype=float)
            if is_rate:
                x = x / days
            k = len(STATES[node])
            rep = analyse(x, cuts, k, ess)
            case[node] = rep
            print(f"  {node}: {rep['value_range']} -> occ {rep['occupancy']} "
                  f"({rep['n_states_occupied']}/{k} states)  "
                  f"TVD@10%={rep['threshold_sensitivity']['tvd_shift_10pct']:.2f} "
                  f"TVD@25%={rep['threshold_sensitivity']['tvd_shift_25pct']:.2f}  "
                  f"CI90 width n=50 {rep['widest_ci_n50']:.2f} / n_eff {rep['widest_ci_n_eff']:.2f}"
                  f"  -> {rep['verdict']}")
            for c in rep["cuts"]:
                if c["inside_range"]:
                    print(f"       cut {c['cut']:>8.2f} IN RANGE  density@cut="
                          f"{c['density_ratio_at_cut']:.2f}  "
                          f"{c['n_within_5pct_spread']} members within 5% of spread")
        results["cases"][f"{uid}|{win}"] = case

    # ---- aggregate verdict ----
    print("\n" + "=" * 78)
    print("VERDICT by node (over all cases):")
    summary = {}
    for node in NODES:
        reps = [c[node] for c in results["cases"].values()]
        verdicts = [r["verdict"] for r in reps]
        occ = [r["n_states_occupied"] for r in reps]
        summary[node] = {
            "verdicts": verdicts,
            "mean_states_occupied": round(float(np.mean(occ)), 2),
            "max_states_occupied": int(max(occ)),
            "mean_tvd_10pct": round(float(np.mean(
                [r["threshold_sensitivity"]["tvd_shift_10pct"] for r in reps])), 3),
            "mean_widest_ci_n50": round(float(np.mean([r["widest_ci_n50"] for r in reps])), 3),
            "mean_widest_ci_n_eff": round(float(np.mean([r["widest_ci_n_eff"] for r in reps])), 3),
        }
        s = summary[node]
        print(f"  {node}: states occupied {s['mean_states_occupied']} of "
              f"{len(STATES[node])} (max {s['max_states_occupied']})  "
              f"| mean TVD@10% {s['mean_tvd_10pct']}  "
              f"| CI90 width n=50 {s['mean_widest_ci_n50']} / n_eff {s['mean_widest_ci_n_eff']}")
        print(f"     {verdicts}")
    results["meta"]["summary_by_node"] = summary

    if args.out:
        with open(args.out, "w") as f:
            json.dump(results, f, indent=1)
        print(f"\n[out] {args.out}")


if __name__ == "__main__":
    main()

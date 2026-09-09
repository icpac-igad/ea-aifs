#!/usr/bin/env python3
"""
k-regimes test — is k=4 over-partitioning a rank-~2 ensemble spread?

The falsifier named in DIRECTION_EXPLANATION_FIRST.md ("what would show this decision to be
wrong", item 3) and HANDOVER.md §5.1. The explanation product reports circulation-regime
occupancy counts from k-means over the z500 ensemble anomaly. But `cpt_build.py` found ESS ~
1.8 (participation ratio of the member-spread eigenspectrum): the 50 members span only ~2
effective directions. Partitioning ~2 dimensions into 4 clusters may be manufacturing structure.

This does NOT just re-run at k=2. For k = 2..6 it measures:

  stability   Ben-Hur-style: cluster two independent 80% subsamples, compare labels on their
              intersection with the Adjusted Rand Index. ARI is chance-corrected (expected 0
              for random labellings), so it needs no null model. Convention: >0.75 stable,
              0.6-0.75 marginal, <0.6 unstable.
  silhouette  mean over members, on the whitened anomaly features.
  var_expl    1 - within-cluster SS / total SS. Always rises with k, so it is only useful for
              spotting where the *gain* flattens.

and reports the spread's actual dimensionality (PC variance shares, ESS) alongside, plus the
occupancy the narrative would quote at each k.

Usage:
    python k_regimes_test.py --store /tank/projects/aifs-run/20260730_0000/icechunk_v2 \
        --tag cycle-20260730_0000 --regions IGAD_EA,EQ_INDIAN,CONGO --out k_test.json
"""
import argparse
import datetime as dt
import json

import numpy as np
from scipy.cluster.vq import kmeans2, whiten
from scipy.special import comb

from s2s_bn_evidence_prep import (load_registry, open_cycle, window_steps, box_unit,
                                  ensemble_ess)

K_RANGE = [2, 3, 4, 5, 6]


def adjusted_rand(a, b):
    """ARI between two labellings. Chance-corrected: 0 = random agreement, 1 = identical."""
    n = len(a)
    cont = np.zeros((int(a.max()) + 1, int(b.max()) + 1))
    for i in range(n):
        cont[int(a[i]), int(b[i])] += 1
    sum_c = comb(cont, 2).sum()
    sum_a = comb(cont.sum(axis=1), 2).sum()
    sum_b = comb(cont.sum(axis=0), 2).sum()
    total = comb(n, 2)
    expected = sum_a * sum_b / total
    maximum = 0.5 * (sum_a + sum_b)
    return float((sum_c - expected) / (maximum - expected)) if maximum > expected else 0.0


def cluster(feat, k, seed):
    _, lab = kmeans2(feat, k, seed=seed, minit="++", missing="warn")
    return lab.astype(int)


def silhouette(feat, labels):
    """Mean silhouette over members. Euclidean on the whitened features."""
    d = np.sqrt(((feat[:, None, :] - feat[None, :, :]) ** 2).sum(-1))
    n = len(labels)
    out = np.zeros(n)
    for i in range(n):
        same = labels == labels[i]
        same[i] = False
        if not same.any():
            out[i] = 0.0
            continue
        a = d[i, same].mean()
        b = min(d[i, labels == c].mean()
                for c in np.unique(labels) if c != labels[i])
        out[i] = (b - a) / max(a, b)
    return float(out.mean())


def var_explained(feat, labels):
    total = ((feat - feat.mean(0)) ** 2).sum()
    within = sum(((feat[labels == c] - feat[labels == c].mean(0)) ** 2).sum()
                 for c in np.unique(labels) if (labels == c).sum() > 0)
    return float(1.0 - within / total) if total > 0 else 0.0


def stability(feat, k, rng, n_rep=40, frac=0.8):
    """Mean ARI between clusterings of two independent subsamples, on their intersection."""
    n = feat.shape[0]
    m = int(round(frac * n))
    scores = []
    for r in range(n_rep):
        s1 = rng.choice(n, m, replace=False)
        s2 = rng.choice(n, m, replace=False)
        both = np.intersect1d(s1, s2)
        if len(both) < k + 2:
            continue
        l1 = cluster(whiten(feat[s1]), k, seed=1000 + r)
        l2 = cluster(whiten(feat[s2]), k, seed=2000 + r)
        p1 = {v: i for i, v in enumerate(s1)}
        p2 = {v: i for i, v in enumerate(s2)}
        scores.append(adjusted_rand(np.array([l1[p1[v]] for v in both]),
                                    np.array([l2[p2[v]] for v in both])))
    return float(np.mean(scores)), float(np.std(scores))


def pc_shares(anom, n=5):
    lam = np.linalg.eigvalsh(anom @ anom.T)[::-1]
    lam = lam[lam > 1e-10]
    return [round(float(v), 4) for v in (lam / lam.sum())[:n]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--store", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--regions", default="IGAD_EA,EQ_INDIAN,CONGO")
    ap.add_argument("--out", default=None)
    ap.add_argument("--seed", type=int, default=20260806)
    args = ap.parse_args()

    reg = load_registry()
    repo, ds = open_cycle(args.store, args.tag)
    rng = np.random.default_rng(args.seed)
    keys = [k.strip() for k in args.regions.split(",")]
    results = {"meta": {"tag": args.tag, "seed": args.seed,
                        "created": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
                        "question": "is k=4 over-partitioning a rank-~2 ensemble spread?"},
               "cases": {}}

    for wname, w in reg["lead_windows"].items():
        steps = window_steps(ds, w["hours"])
        for key in keys:
            box = reg["regions"][key]
            uid, uname, cells = box_unit(ds, box)[0]
            z = np.nanmean(ds["z_500"].isel(time=steps,
                                            values=np.where(cells)[0]).values, axis=1)
            anom = z - z.mean(axis=0, keepdims=True)
            ess, redundancy = ensemble_ess(anom)
            feat = whiten(anom.astype(float))
            shares = pc_shares(anom)

            print(f"\n=== {key} / {wname}  ({int(cells.sum())} cells, {z.shape[0]} members)")
            print(f"    ESS={ess:.2f}  redundancy={redundancy:.3f}  "
                  f"PC variance shares={shares}")
            print(f"    {'k':>2} {'stability(ARI)':>16} {'silhouette':>11} {'var_expl':>9}   occupancy")

            case = {"region": key, "region_name": uname, "lead_window": wname,
                    "n_cells": int(cells.sum()), "n_members": int(z.shape[0]),
                    "ess": round(float(ess), 3), "redundancy": round(float(redundancy), 3),
                    "pc_variance_shares": shares, "k": {}}

            for k in K_RANGE:
                lab = cluster(feat, k, seed=1234)
                mu, sd = stability(feat, k, rng)
                sil = silhouette(feat, lab)
                ve = var_explained(feat, lab)
                occ = np.bincount(lab, minlength=k).tolist()
                verdict = "stable" if mu > 0.75 else ("marginal" if mu > 0.60 else "UNSTABLE")
                case["k"][str(k)] = {"stability_ari_mean": round(mu, 3),
                                     "stability_ari_sd": round(sd, 3),
                                     "silhouette": round(sil, 3),
                                     "var_explained": round(ve, 3),
                                     "occupancy": occ, "verdict": verdict}
                print(f"    {k:>2} {mu:>9.3f}+-{sd:<5.3f} {sil:>11.3f} {ve:>9.3f}   "
                      f"{occ}  {verdict}")

            results["cases"][f"{key}|{wname}"] = case

    # ---- verdict across cases ----
    print("\n" + "=" * 78)
    print("VERDICT — mean stability (ARI) across all cases:")
    summary = {}
    for k in K_RANGE:
        vals = [c["k"][str(k)]["stability_ari_mean"] for c in results["cases"].values()]
        summary[str(k)] = round(float(np.mean(vals)), 3)
        print(f"  k={k}: {np.mean(vals):.3f}   "
              f"({'stable' if np.mean(vals) > 0.75 else 'marginal' if np.mean(vals) > 0.60 else 'UNSTABLE'})")
    best = max(summary, key=lambda k: summary[k])
    results["meta"]["mean_stability_by_k"] = summary
    results["meta"]["most_stable_k"] = int(best)
    print(f"\n  most stable k = {best}")
    print(f"  k=4 (current production setting) stability = {summary['4']}")
    print(f"  k=2 (the proposed fallback)      stability = {summary['2']}")

    if args.out:
        with open(args.out, "w") as f:
            json.dump(results, f, indent=1)
        print(f"\n[out] {args.out}")


if __name__ == "__main__":
    main()

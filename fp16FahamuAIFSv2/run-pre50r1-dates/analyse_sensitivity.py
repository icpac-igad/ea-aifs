#!/usr/bin/env python3
"""Score the pre-50r1 sensitivity runs against the two baselines that make them readable.

Every treatment is compared with the control **at the same seed**. Two reference scales
decide whether a difference means anything:

* ``controlB`` vs ``control``, same seed -- the irreducible FP16/cuDNN non-determinism
  floor. Anything at or below this is unmeasurable, not "small".
* ``control`` seed 0 vs seed 1 -- the model's own stochastic ensemble spread. A treatment
  at or above this has perturbed the forecast as much as drawing a different member.

The verdict rule: a fill is *usable* only if it sits near the floor and well under the
ensemble scale. Sitting between them means the fill changes the forecast in a way you
cannot separate from noise with one member -- which is not the same as harmless.

Usage::

    python analyse_sensitivity.py --dir /tank/projects/aifs-run/wave_sensitivity_20260813
"""

import argparse
import glob
import os

import numpy as np

VARS = ["2t", "tp", "msl"]
# ICPAC domain, on the N320 unstructured axis (lat/lon come from the store).
ICPAC = dict(lat=(-12.0, 18.0), lon=(21.0, 52.0))


def load(d, tag):
    p = os.path.join(d, f"{tag}.npz")
    return np.load(p) if os.path.exists(p) else None


def coords():
    """N320 lat/lon from the reference store (same order the runner yields)."""
    import icechunk
    import zarr
    B = "/tank/projects/aifs-run/20260813_0000/icechunk_v2"
    r = zarr.open_group(icechunk.Repository.open(
        icechunk.local_filesystem_storage(B)).readonly_session(
        tag="cycle-20260813_0000").store, mode="r")
    return np.asarray(r["latitude"][:]), np.asarray(r["longitude"][:])


def stats(a, b, mask):
    """rms difference over all captured steps, globally and inside the mask."""
    d = a - b
    g = float(np.sqrt(np.nanmean(d ** 2)))
    r = float(np.sqrt(np.nanmean(d[:, mask] ** 2))) if mask is not None else float("nan")
    return g, r


def main():
    ap = argparse.ArgumentParser(description="analyse pre-50r1 sensitivity runs")
    ap.add_argument("--dir", required=True)
    ap.add_argument("--seeds", default="0,1")
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]

    tags = sorted({os.path.basename(f)[:-4] for f in glob.glob(f"{args.dir}/*.npz")})
    modes = sorted({t.rsplit("_seed", 1)[0] for t in tags})
    print(f"runs found: {len(tags)}  modes: {modes}  seeds: {seeds}\n")

    try:
        lat, lon = coords()
        lon180 = np.where(lon > 180, lon - 360, lon)
        mask = ((lat > ICPAC["lat"][0]) & (lat < ICPAC["lat"][1]) &
                (lon180 > ICPAC["lon"][0]) & (lon180 < ICPAC["lon"][1]))
        print(f"ICPAC mask: {mask.sum()} of {mask.size} N320 cells\n")
    except Exception as e:
        print(f"(no ICPAC mask: {e})\n")
        mask = None

    # --- reference scale: a different seed = a different ensemble member ---
    ens = {}
    c0, c1 = load(args.dir, f"control_seed{seeds[0]}"), (
        load(args.dir, f"control_seed{seeds[1]}") if len(seeds) > 1 else None)
    if c0 is not None and c1 is not None:
        for v in VARS:
            ens[v] = stats(c0[v], c1[v], mask)

    floor = {}
    cb = load(args.dir, f"controlB_seed{seeds[0]}")
    if c0 is not None and cb is not None:
        for v in VARS:
            floor[v] = stats(c0[v], cb[v], mask)

    hdr = f"{'mode':10s} {'var':4s} {'global rms':>12s} {'ICPAC rms':>11s}"
    if floor:
        hdr += f" {'x floor':>9s}"
    if ens:
        hdr += f" {'% of ens':>9s}"
    print(hdr)
    print("-" * len(hdr))

    for mode in modes:
        if mode == "control":
            continue
        for seed in seeds:
            a = load(args.dir, f"control_seed{seed}")
            b = load(args.dir, f"{mode}_seed{seed}")
            if a is None or b is None:
                continue
            for v in VARS:
                g, r = stats(a[v], b[v], mask)
                line = f"{mode + '/s' + str(seed):10s} {v:4s} {g:12.5g} {r:11.5g}"
                if floor.get(v) and floor[v][0] > 0:
                    line += f" {g / floor[v][0]:9.1f}"
                if ens.get(v) and ens[v][0] > 0:
                    line += f" {100 * g / ens[v][0]:8.1f}%"
                print(line)
        print()

    if floor:
        print("non-determinism floor (controlB vs control, same seed):")
        for v in VARS:
            print(f"   {v:4s} global rms {floor[v][0]:.5g}   ICPAC {floor[v][1]:.5g}")
    if ens:
        print("\nensemble scale (control seed%d vs seed%d):" % (seeds[0], seeds[1]))
        for v in VARS:
            print(f"   {v:4s} global rms {ens[v][0]:.5g}   ICPAC {ens[v][1]:.5g}")


if __name__ == "__main__":
    main()

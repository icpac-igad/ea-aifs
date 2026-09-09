#!/usr/bin/env python3
"""Acceptance check for a tier-B run (an O96 store plus its native N320 sidecar).

Run this after every ``--grid o96`` inference cycle, and after every transcode, BEFORE
deleting anything.

The load-bearing check is #2: the O96 store and the sidecar come from the **same**
rollout, so regridding the sidecar's fields must reproduce the O96 store to storage
precision. That does not depend on the model being reproducible — and it is not:
aifs-ens-2.0 is stochastic, so re-running a member from the same input pkl gives a
different realisation (``2t`` rms ~2.8 K against a field sd of ~14 K). Never "verify" a
store by re-running inference and diffing.

Usage::

    python validate_o96_run.py --o96 .../icechunk_o96 --sidecar .../icechunk_n320_aiwq \
        --tag cycle-20260813_0000 --members 1-50
    # a live single-member smoke run writes no tag:
    python validate_o96_run.py --o96 .../icechunk_o96_gputest \
        --sidecar .../icechunk_n320_gputest --members 1
"""

import argparse
import os
import sys

import numpy as np
import zarr
import icechunk

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from o96_grid import N_VALUES as O96_N_VALUES, load_matrix, regrid_block  # noqa: E402

COORD_ARRAYS = ("latitude", "longitude", "time", "member")


def open_store(path, tag=None):
    """Open read-only at ``tag``, falling back to ``main`` if the tag is gone.

    Compaction (expire + GC) tombstones the original cycle tag permanently, so a
    compacted store legitimately has no ``cycle-<date>_0000``. Its branch tip is the same
    snapshot, so the fallback is exact -- but it is reported, never silent.
    """
    repo = icechunk.Repository.open(icechunk.local_filesystem_storage(path))
    if tag:
        try:
            return repo, zarr.open_group(repo.readonly_session(tag=tag).store, mode="r")
        except Exception:
            print(f"  NOTE tag {tag!r} not found; tags present: "
                  f"{sorted(repo.list_tags()) or 'none'} -> falling back to branch main")
    return repo, zarr.open_group(repo.readonly_session("main").store, mode="r")


def parse_members(spec):
    spec = spec.strip()
    if "-" in spec:
        a, b = map(int, spec.split("-"))
        return list(range(a, b + 1))
    return [int(x) for x in spec.split(",")]


def main():
    ap = argparse.ArgumentParser(description="validate an O96 store against its sidecar")
    ap.add_argument("--o96", required=True)
    ap.add_argument("--sidecar", default=None,
                    help="native N320 sidecar from the same rollout (enables check 2)")
    ap.add_argument("--tag", default=None, help="read both stores at this tag")
    ap.add_argument("--members", default="1", help="'1-50' or '1,2'")
    ap.add_argument("--steps", default=None,
                    help="comma-separated absolute time indices (default: the first and "
                         "last step actually present)")
    ap.add_argument("--sentinel", default="2t")
    args = ap.parse_args()

    o96_repo, o96 = open_store(args.o96, args.tag)
    members = parse_members(args.members)
    var_names = [k for k in o96.array_keys() if k not in COORD_ARRAYS]
    n_members, n_steps, n_values = o96[args.sentinel].shape
    fail = 0

    print(f"O96 store : {len(var_names)} vars, {args.sentinel} {o96[args.sentinel].shape} "
          f"chunks {o96[args.sentinel].chunks}")
    if n_values != O96_N_VALUES:
        print(f"  [FAIL] values axis is {n_values}, expected {O96_N_VALUES}")
        fail += 1
    try:
        print(f"  tags: {o96_repo.list_tags()}")
    except Exception:
        pass

    # --- which steps hold data, and are they the intended window? ---
    probe = np.asarray(o96[args.sentinel][members[0] - 1, :, 0])
    present = np.flatnonzero(np.isfinite(probe))
    if present.size == 0:
        print(f"  [FAIL] member {members[0]} has no data at all")
        raise SystemExit(1)
    print(f"\n1. write window: steps {present[0]}-{present[-1]} "
          f"(h{(present[0]+1)*6}-{(present[-1]+1)*6}), {present.size}/{n_steps} stored")
    steps = ([int(s) for s in args.steps.split(",")] if args.steps
             else [int(present[0]), int(present[-1])])

    # --- every requested member complete at those steps, both stores ---
    print(f"\n2. all {len(members)} member(s) finite at steps {steps}")
    for label, g in [("o96", o96)] + (
            [("sidecar", open_store(args.sidecar, args.tag)[1])] if args.sidecar else []):
        sent = args.sentinel if args.sentinel in list(g.array_keys()) else \
            [k for k in g.array_keys() if k not in COORD_ARRAYS][0]
        missing = [(m, t) for m in members for t in steps
                   if not np.isfinite(np.asarray(g[sent][m - 1, t, :])).all()]
        ok = not missing
        fail += 0 if ok else 1
        print(f"   {label:8s} {'OK' if ok else f'FAIL - incomplete at {missing[:5]}'}")

    # --- the real check: O96 == regrid(sidecar), same rollout ---
    if args.sidecar:
        _, side = open_store(args.sidecar, args.tag)
        shared = [v for v in side.array_keys() if v in var_names]
        print(f"\n3. O96 == interpolate(sidecar, N320, O96) for {shared}")
        mat = load_matrix()
        for m in members[:5]:                       # 5 members is plenty; it is exact
            for v in shared:
                for t in steps:
                    got = np.asarray(o96[v][m - 1, t, :]).astype("float64")
                    red = regrid_block(np.asarray(side[v][m - 1, t, :])[None, :], mat)[0]
                    d = float(np.nanmax(np.abs(got - red)))
                    tol = float(np.nanmax(np.abs(red))) * 1.2e-7 + 1e-30
                    if d > tol:
                        fail += 1
                        print(f"   [FAIL] m{m:03d} t{t:3d} {v}: max|d|={d:.3e} > {tol:.3e}")
        print("   OK (all within f4 rounding)" if not fail else "   FAIL")

    print("\n" + ("PASS" if fail == 0 else f"FAILED ({fail} problem(s))"))
    return 0 if fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())

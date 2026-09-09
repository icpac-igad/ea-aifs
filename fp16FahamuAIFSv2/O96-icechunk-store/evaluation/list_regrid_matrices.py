#!/usr/bin/env python3
"""What can earthkit-regrid actually reach from N320? Settles the archive-grid choice.

The choice of O96 is forced, not preferred. Enumerating earthkit-regrid's matrix
database shows that **from N320 there are exactly 25 targets: 24 regular lat/lon grids
and exactly one Gaussian grid, O96.**

The two candidates that get proposed instead both fail, for *different* reasons:

* **1.0 deg regular lat/lon** — reachable (``N320 -> 1.0deg`` exists) but a dead end:
  there is no ``1.0deg -> 1.5deg`` matrix to reach the AI-WQ grid. The second hop fails.
* **N96** — the second hop is fine (``N96 -> 1.5deg`` exists, and 48 matrices involve
  N96 in general), but ``N320 -> N96`` does not exist and nothing routes to it. The
  first hop fails.

And N96 would not even be smaller: same 96 latitude rows per hemisphere, but the classic
reduced rule keeps 50,662 points against the octahedral ``16 + 4i`` rule's 40,320.

Usage::

    python list_regrid_matrices.py              # from N320 (default)
    python list_regrid_matrices.py --from O96 --to 1.5x1.5deg
"""

import argparse

from earthkit.regrid import db


def name(spec):
    g = spec.get("grid")
    return str(g) if not isinstance(g, list) else f"{g[0]}x{g[1]}deg"


def main():
    ap = argparse.ArgumentParser(description="enumerate earthkit-regrid matrices")
    ap.add_argument("--from", dest="src", default="N320",
                    help="list every target reachable from this grid")
    ap.add_argument("--to", dest="dst", default="1.5x1.5deg",
                    help="list every source that can reach this grid")
    args = ap.parse_args()

    idx = db.SYS_DB.index
    rows = [(name(v["input"]), name(v["output"]),
             v["input"].get("type"), v["output"].get("type"),
             (v["input"].get("shape") or [None])[0],
             (v["output"].get("shape") or [None])[0])
            for v in idx.values()]
    print(f"earthkit-regrid matrix database: {len(rows)} matrices\n")

    print(f"=== every target reachable FROM {args.src} ===")
    outs = sorted({(o, ot, os_) for i, o, it, ot, is_, os_ in rows if i == args.src})
    for o, ot, n in outs:
        print(f"  {args.src} -> {o:14s} {ot:11s} {n:,}" if n else
              f"  {args.src} -> {o:14s} {ot}")
    gauss = [o for o, ot, _ in outs if ot == "reduced_gg"]
    print(f"  -> {len(outs)} targets, of which Gaussian: {gauss or '(none)'}")

    print(f"\n=== every source that can reach {args.dst} ===")
    ins = sorted({i for i, o, it, ot, is_, os_ in rows if o == args.dst})
    print("  " + ", ".join(ins))

    print("\n=== ALL reduced_gg -> reduced_gg matrices (the archive-grid candidates) ===")
    for i, o in sorted({(i, o) for i, o, it, ot, *_ in rows
                        if it == "reduced_gg" and ot == "reduced_gg"}):
        print(f"  {i} -> {o}")

    print("\n=== point counts ===")
    shapes = {}
    for v in idx.values():
        for side in ("input", "output"):
            s = v[side]
            if isinstance(s.get("grid"), str) and s.get("type") == "reduced_gg" \
                    and s.get("shape"):
                shapes[s["grid"]] = s["shape"][0]
    for g in ("N320", "O96", "N96", "O320"):
        if g in shapes:
            store = shapes[g] * 4 * 50 * 120 * 132 * 0.733 / 1e9
            print(f"  {g:6s} {shapes[g]:>9,} points   "
                  f"({store:7.1f} GB for 120 vars x 50 members x 132 steps)")


if __name__ == "__main__":
    main()

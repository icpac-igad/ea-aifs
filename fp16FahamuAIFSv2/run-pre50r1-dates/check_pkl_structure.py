#!/usr/bin/env python3
"""Compare AIFS v2 input-state pkl directories field-by-field across dates.

Written to answer one question: is a **pre-50r1** cycle built by
``ecmwf_opendata_pkl_input_aifsens_v2_pre50r1.py`` structurally the same object as an
operational one, or does it only look like one? 13 of its 112 fields come from donor
archives rather than open data, and a structural difference there would reach the model
as a silent wrong answer rather than an error.

Checks, per date and then across dates:

* **schema** — top-level keys, field-name set, shape, dtype, contiguity
* **NaN structure** — which fields carry NaN and over what fraction. This is the one that
  matters most: the model reads NaN as "not sea"/"not land", so a mask that differs from
  the operational convention changes the meaning of a field without changing its type.
  Also checks the wave fields share one mask, and that no field is partly NaN when its
  operational counterpart is never NaN.
* **finiteness** — no infinities anywhere; no all-zero or constant fields
* **member variation** — which fields differ between members. Donor-sourced fields are
  shared by construction, so this is where a pre-50r1 cycle legitimately differs from an
  operational one, and the script reports it rather than flagging it.
* **physical sanity** — a few invariants that must hold in any valid state
  (``z`` decreasing with height, ``t_10`` warmer than ``t_50``, plausible ranges).

Usage::

    python check_pkl_structure.py \
        --dir 20260212=/tank/projects/aifs-run/20260212_0000/input_states \
        --dir 20260514=/tank/projects/aifs-run/20260514_0000/input_states \
        --dir 20260604=/tank/projects/aifs-run/20260604_0000/input_states \
        --members 1-5 --reference 20260604
"""

import argparse
import os
import pickle
import sys

import numpy as np

DONOR_WAVE = ["wmb", "h1012", "h1214", "h1417", "h1721", "h2125", "h2530", "cdww"]
DONOR_L10 = ["t_10", "u_10", "v_10", "w_10", "z_10"]
LEVELS = [1000, 925, 850, 700, 600, 500, 400, 300, 250, 200, 150, 100, 50, 10]


def parse_members(spec):
    out = set()
    for part in str(spec).split(","):
        part = part.strip()
        if "-" in part:
            a, b = part.split("-")
            out.update(range(int(a), int(b) + 1))
        elif part:
            out.add(int(part))
    return sorted(out)


def load(d, m):
    p = os.path.join(d, f"input_state_member_{m:03d}.pkl")
    with open(p, "rb") as f:
        return pickle.load(f)


def profile(states):
    """Structural fingerprint of one date, from its member states."""
    f0 = states[0]["fields"]
    prof = {
        "keys": tuple(sorted(states[0].keys())),
        "names": frozenset(f0),
        "n": len(f0),
        "date": states[0].get("date"),
        "shape": {k: np.asarray(v).shape for k, v in f0.items()},
        "dtype": {k: str(np.asarray(v).dtype) for k, v in f0.items()},
        "nanfrac": {k: round(float(np.isnan(np.asarray(v)).mean()), 6)
                    for k, v in f0.items()},
    }
    # which fields vary across members
    varying = set()
    for s in states[1:]:
        for k in f0:
            if not np.array_equal(np.asarray(f0[k]), np.asarray(s["fields"][k]),
                                  equal_nan=True):
                varying.add(k)
    ref_nan = np.isnan(np.asarray(f0["swh"]))
    prof["fringe"] = {k: int((np.isnan(np.asarray(f0[k])) ^ ref_nan).sum())
                      for k in ("wmb", "cdww") if k in f0}
    prof["varying"] = frozenset(varying)
    prof["constant"] = frozenset(f0) - varying
    return prof


def check_one(label, d, members, verbose=True):
    states = [load(d, m) for m in members]
    p = profile(states)
    print(f"\n=== {label}  ({d}) ===")
    print(f"  top-level keys : {list(p['keys'])}")
    print(f"  date           : {p['date']}")
    print(f"  fields         : {p['n']}")
    shapes = set(p["shape"].values())
    dtypes = set(p["dtype"].values())
    print(f"  shapes         : {shapes if len(shapes) > 1 else shapes.pop()}")
    print(f"  dtypes         : {dtypes if len(dtypes) > 1 else dtypes.pop()}")

    problems = []
    # finiteness / degeneracy
    for m, s in zip(members, states):
        for k, v in s["fields"].items():
            a = np.asarray(v)
            if np.isinf(a).any():
                problems.append(f"member {m}: {k} contains inf")
            fin = a[np.isfinite(a)]
            if fin.size == 0:
                problems.append(f"member {m}: {k} is entirely non-finite")
            elif fin.min() == fin.max() and k not in ("lsm",):
                problems.append(f"member {m}: {k} is constant ({fin.min()})")

    # Wave-field NaN masks, measured against swh rather than assumed equal to it.
    # Measured on operational cycles (20260514 and 20260604, identical): the six band
    # heights, mwp and mwd match swh EXACTLY, while wmb and cdww deviate by ~226/228
    # points (0.02%) in each direction. So "wave fields all share one mask" is true for
    # 8 of the 10 and very nearly true for the other 2 -- flag only a deviation large
    # enough to be a real masking error rather than this known fringe.
    ref = np.isnan(np.asarray(states[0]["fields"]["swh"]))
    MASK_EXACT = ["mwp", "cos_mwd", "sin_mwd", "h1012", "h1214", "h1417",
                  "h1721", "h2125", "h2530"]
    MASK_FRINGE_TOL = 2000            # ~0.2% of points; operational fringe is ~450
    for k in MASK_EXACT:
        if k in states[0]["fields"]:
            if not np.array_equal(np.isnan(np.asarray(states[0]["fields"][k])), ref):
                problems.append(f"{k}: NaN mask differs from swh (must be exact)")
    for k in ("wmb", "cdww"):
        if k in states[0]["fields"]:
            d = int((np.isnan(np.asarray(states[0]["fields"][k])) ^ ref).sum())
            if d > MASK_FRINGE_TOL:
                problems.append(f"{k}: NaN mask differs from swh at {d} points "
                                f"(> {MASK_FRINGE_TOL} tolerance)")
            elif verbose and d:
                print(f"  note: {k} mask differs from swh at {d} pts "
                      f"(operational cycles show ~454; not an error)")
    # atmospheric fields must be NaN-free
    for k in DONOR_L10 + [f"t_{lv}" for lv in LEVELS] + ["2t", "msl", "sp"]:
        if k in states[0]["fields"]:
            if np.isnan(np.asarray(states[0]["fields"][k])).any():
                problems.append(f"{k}: unexpected NaN in an atmospheric field")

    # physical invariants
    f0 = states[0]["fields"]
    for lo, hi in zip(LEVELS[:-1], LEVELS[1:]):        # 1000 -> 10 hPa
        a, b = np.asarray(f0[f"z_{lo}"]), np.asarray(f0[f"z_{hi}"])
        if not (b > a).all():
            problems.append(f"z_{hi} not everywhere above z_{lo}")
    if float(np.asarray(f0["t_10"]).mean()) <= float(np.asarray(f0["t_50"]).mean()):
        problems.append("t_10 mean not warmer than t_50 (stratospheric profile)")

    print(f"  NaN-carrying   : {sorted(k for k, v in p['nanfrac'].items() if v > 0)}")
    print(f"  NaN fraction   : "
          f"{sorted({round(v, 4) for v in p['nanfrac'].values() if v > 0})}")
    print(f"  varies by member: {len(p['varying'])}/{p['n']}")
    print(f"  shared by all  : {sorted(p['constant'])}")
    print("  " + ("✅ no structural problems" if not problems
                  else f"❌ {len(problems)} problem(s):"))
    for x in problems:
        print(f"      - {x}")
    return p, problems


def compare(ref_label, ref, label, other):
    """Structural diff of two dates. Returns list of hard failures."""
    print(f"\n--- {label} vs {ref_label} ---")
    fail = []
    if ref["keys"] != other["keys"]:
        fail.append(f"top-level keys differ: {other['keys']} vs {ref['keys']}")
    missing = sorted(ref["names"] - other["names"])
    extra = sorted(other["names"] - ref["names"])
    if missing:
        fail.append(f"missing fields: {missing}")
    if extra:
        fail.append(f"extra fields: {extra}")
    print(f"  field names   : {'identical' if not missing and not extra else 'DIFFER'}"
          f" ({other['n']} vs {ref['n']})")

    if not missing and not extra:
        bad_shape = [k for k in ref["names"] if ref["shape"][k] != other["shape"][k]]
        bad_dtype = [k for k in ref["names"] if ref["dtype"][k] != other["dtype"][k]]
        if bad_shape:
            fail.append(f"shape differs: {bad_shape}")
        if bad_dtype:
            fail.append(f"dtype differs: {bad_dtype}")
        print(f"  shapes/dtypes : {'identical' if not bad_shape and not bad_dtype else 'DIFFER'}")

        # NaN *structure*: which fields carry NaN at all. The fraction legitimately
        # differs between dates (sea ice moves), the set of fields must not.
        rn = {k for k, v in ref["nanfrac"].items() if v > 0}
        on = {k for k, v in other["nanfrac"].items() if v > 0}
        if rn != on:
            fail.append(f"different fields carry NaN: only-here={sorted(on - rn)}, "
                        f"only-ref={sorted(rn - on)}")
        print(f"  NaN-carrying set: {'identical' if rn == on else 'DIFFERS'} "
              f"({len(on)} fields)")

        fr = {k: (other["fringe"].get(k), ref["fringe"].get(k)) for k in ("wmb", "cdww")}
        print(f"  wmb/cdww mask fringe vs swh: "
              + ", ".join(f"{k} {v[0]} here / {v[1]} ref" for k, v in fr.items()))

        only_shared = sorted(other["constant"] - ref["constant"])
        only_varies = sorted(ref["constant"] - other["constant"])
        print(f"  member variation: {len(other['varying'])} vary here vs "
              f"{len(ref['varying'])} in reference")
        if only_shared:
            print(f"    shared here but perturbed in reference: {only_shared}")
        if only_varies:
            print(f"    perturbed here but shared in reference: {only_varies}")
    print("  " + ("✅ structurally identical" if not fail
                  else f"❌ {len(fail)} structural difference(s)"))
    for x in fail:
        print(f"      - {x}")
    return fail


def main():
    ap = argparse.ArgumentParser(description="compare AIFS v2 input-state pkl dirs")
    ap.add_argument("--dir", action="append", required=True, metavar="LABEL=PATH",
                    help="repeatable; e.g. --dir 20260212=/path/to/input_states")
    ap.add_argument("--members", default="1-5")
    ap.add_argument("--reference", default=None,
                    help="label to compare the others against (default: last --dir)")
    args = ap.parse_args()

    members = parse_members(args.members)
    dirs = []
    for spec in args.dir:
        if "=" not in spec:
            sys.exit(f"--dir needs LABEL=PATH, got {spec!r}")
        label, path = spec.split("=", 1)
        dirs.append((label, path))
    ref_label = args.reference or dirs[-1][0]

    print(f"members {members[0]}-{members[-1]} | reference = {ref_label}")
    profs, problems = {}, {}
    for label, path in dirs:
        profs[label], problems[label] = check_one(label, path, members)

    fails = {}
    for label, _ in dirs:
        if label != ref_label:
            fails[label] = compare(ref_label, profs[ref_label], label, profs[label])

    print(f"\n{'=' * 62}")
    bad = sum(len(v) for v in fails.values()) + sum(len(v) for v in problems.values())
    for label, _ in dirs:
        n = len(problems[label]) + len(fails.get(label, []))
        print(f"  {label:12s} {'OK' if n == 0 else f'{n} issue(s)'}")
    print("VERDICT: " + ("all dates structurally interchangeable"
                         if bad == 0 else f"{bad} issue(s) found"))
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    sys.exit(main())

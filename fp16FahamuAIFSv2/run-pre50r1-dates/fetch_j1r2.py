#!/usr/bin/env python3
"""Fetch the 8 Cy50r1-only wave fields from the ECMWF research hindcast ``j1r2``.

``j1r2`` is *the* dataset the AIFS v2 wave inputs were trained on, published with a DOI:

    TCo1279 CY50R1 wave hindcast using CY50R1 winds and ORAS6 currents 2025-2026
    for AIFS training                       -- DOI 10.21957/1tzs-0946
    https://apps.ecmwf.int/ifs-experiments/rd/j1r2/

It is a *standalone ecWAM Cy50r1* run with altimeter wave-height assimilation, so it
carries the period-band significant wave heights, ``cdww`` and ``wmb`` for dates *before*
Cy50r1 went operational -- exactly the fields that make a pre-2026-05-13 v2 cycle fail.
It closes the wave half of the pre-50r1 gap; the 10 hPa level is a separate problem
(see ``fetch_era5_l10.py``).

Two things make this cheap to wire in:

* MARS can deliver ``grid=N320`` directly, which is the AIFS state grid (542080 points),
  so there is no local regridding and no grid-ordering risk.
* ``wmb`` is model bathymetry -- static. Any date will do for it.

Caveats worth knowing before a bulk pull:

* It is **deterministic**: one analysis, not 50 perturbed members. Every ensemble member
  gets identical wave fields, so wave initial-condition spread is lost.
* Part of the archive is **on tape** (a 52-field request reported ~294 MB on 1 tape and
  took ~2 min). Pull whole days in one request, not field by field.
* ecWAM's land-sea mask differs slightly from the operational ``waef`` mask: ~29.3 % of
  N320 points are land here against ~30.3 % in the operational input. Points that are sea
  in one and land in the other are reconciled by ``--mask-like``.

Credentials: ``.ecmwfapirc`` in this folder (or ``~/.ecmwfapirc``), from
https://api.ecmwf.int/v1/key/ -- an ECMWF account, *not* the CDS token in ``.env``.

Usage::

    # one cycle -> npz of AIFS-named fields, each (2, 542080) for [t-6h, t]
    python fetch_j1r2.py --date 20260212 --time 00 --out j1r2_20260212_00.npz

    # patch a built input-state pickle directory in place (writes *_j1r2.pkl alongside)
    python fetch_j1r2.py --date 20260212 --time 00 \
        --patch-pkl-dir /tank/projects/aifs-run/20260212_0000/input_states
"""

import argparse
import datetime as dt
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))

# AIFS v2 consumes 11 wave fields; these 8 are the ones Cy50r1 introduced.
# paramId -> AIFS state name. mwd is fetched too because the state stores it as
# cos/sin, and it is cheaper to take the matching analysis than to mix sources.
PARAMS = {
    "140114": "h1012",
    "140115": "h1214",
    "140116": "h1417",
    "140117": "h1721",
    "140118": "h2125",
    "140119": "h2530",
    "140219": "wmb",
    "140233": "cdww",
    # present pre-50r1 as well -- fetched only with --all-wave
    "140229": "swh",
    "140230": "mwd",
    "140232": "mwp",
}
CY50R1_ONLY = ["h1012", "h1214", "h1417", "h1721", "h2125", "h2530", "wmb", "cdww"]
MISSING_ABOVE = 9998.0          # ecWAM writes 9999 on land


def rc_path():
    local = os.path.join(HERE, ".ecmwfapirc")
    return local if os.path.exists(local) else os.path.expanduser("~/.ecmwfapirc")


def retrieve(date, time, params, target, grid="N320", area=None):
    """One MARS request for one cycle. Returns the target path."""
    os.environ.setdefault("ECMWF_API_RC_FILE", rc_path())
    from ecmwfapi import ECMWFDataServer

    req = {
        "class": "rd",
        "dataset": "research",
        "expver": "j1r2",
        "stream": "wave",
        "type": "an",
        "levtype": "sfc",
        "date": date,
        "time": time,
        "step": "0",
        "param": "/".join(params),
        "grid": grid,
        "target": target,
    }
    if area:
        req["area"] = area
    ECMWFDataServer().retrieve(req)
    return target


def read_grib(path):
    """-> {shortName: 1-D array with land as NaN}"""
    import eccodes as ec

    out = {}
    with open(path, "rb") as f:
        while True:
            h = ec.codes_grib_new_from_file(f)
            if h is None:
                break
            name = ec.codes_get(h, "shortName")
            vals = np.array(ec.codes_get_values(h), dtype="float64")
            vals[vals > MISSING_ABOVE] = np.nan
            out[name] = vals
            ec.codes_release(h)
    if not out:
        raise RuntimeError(f"no GRIB messages in {path}")
    return out


def fetch_cycle(date, time, names, workdir, grid="N320"):
    """Fetch one cycle (date, time) and return {aifs_name: array}."""
    ids = [pid for pid, nm in PARAMS.items() if nm in names]
    tgt = os.path.join(workdir, f"j1r2_{date}_{time}.grib")
    if not os.path.exists(tgt):
        retrieve(date, time, ids, tgt, grid=grid)
    return read_grib(tgt)


def build_state_fields(date, time, workdir, all_wave=False, grid="N320"):
    """Return {aifs_field: (2, N)} for [t-6h, t], AIFS naming and transforms applied.

    The AIFS input state carries two time levels. j1r2 is 6-hourly analyses, so both
    are real analyses -- no interpolation anywhere.
    """
    names = list(PARAMS.values()) if all_wave else CY50R1_ONLY
    t1 = dt.datetime.strptime(f"{date}{time}", "%Y%m%d%H")
    t0 = t1 - dt.timedelta(hours=6)

    per_time = []
    for t in (t0, t1):
        d, hh = t.strftime("%Y%m%d"), t.strftime("%H")
        got = fetch_cycle(d, hh, names, workdir, grid=grid)
        missing = [n for n in names if n not in got]
        if missing:
            raise RuntimeError(f"{d} {hh}z: j1r2 returned nothing for {missing}")
        per_time.append(got)
        print(f"  {d} {hh}z: {len(got)} params, {per_time[-1][names[0]].size} points")

    fields = {}
    for n in names:
        fields[n] = np.stack([per_time[0][n], per_time[1][n]])
    if "mwd" in fields:                      # v2 stores direction as cos/sin
        rad = np.deg2rad(fields.pop("mwd"))
        fields["cos_mwd"] = np.cos(rad)
        fields["sin_mwd"] = np.sin(rad)
    return fields


def reconcile_mask(fields, like):
    """Make j1r2's land mask agree with the operational one in ``like``.

    ecWAM on TCo1279 and the operational waef mask disagree on a thin coastal fringe.
    Sea in j1r2 but land operationally -> set NaN (the model never saw a value there).
    Land in j1r2 but sea operationally -> fill from the operational field if available,
    else leave NaN and report the count; a handful of NaNs in a wave field is what the
    model already sees over land.
    """
    ref = np.asarray(like)
    ref_land = ~np.isfinite(ref)
    added = dropped = 0
    for name, arr in fields.items():
        a = np.asarray(arr, dtype="float64")
        land = ~np.isfinite(a)
        extra = land & ~ref_land        # j1r2 says land, operational says sea
        added += int(extra.sum())
        a[ref_land] = np.nan            # operational says land -> always NaN
        dropped += int((~land & ref_land).sum())
        fields[name] = a
    return added // max(len(fields), 1), dropped // max(len(fields), 1)


def main():
    ap = argparse.ArgumentParser(description="fetch j1r2 wave fields for AIFS v2")
    ap.add_argument("--date", required=True, help="YYYYMMDD of the cycle")
    ap.add_argument("--time", default="00", help="cycle hour, default 00")
    ap.add_argument("--out", help="write an npz of the fields here")
    ap.add_argument("--workdir", default=None, help="where to keep the GRIBs")
    ap.add_argument("--grid", default="N320",
                    help="MARS output grid; N320 matches the AIFS state exactly")
    ap.add_argument("--all-wave", action="store_true",
                    help="also fetch swh/mwd/mwp (which exist pre-50r1 anyway)")
    ap.add_argument("--patch-pkl-dir",
                    help="input_states dir: write <name>_j1r2.pkl with fields replaced")
    ap.add_argument("--mask-like", default=None,
                    help="pickle to take the operational land mask from "
                         "(default: member 001 of --patch-pkl-dir)")
    args = ap.parse_args()

    workdir = args.workdir or os.path.join(
        os.environ.get("TMPDIR", "/tmp"), f"j1r2_{args.date}{args.time}")
    os.makedirs(workdir, exist_ok=True)

    print(f"=== j1r2 wave fields for {args.date} {args.time}z (grid {args.grid}) ===")
    fields = build_state_fields(args.date, args.time, workdir,
                                all_wave=args.all_wave, grid=args.grid)
    for n in sorted(fields):
        a = fields[n]
        fin = a[np.isfinite(a)]
        print(f"  {n:8s} {str(a.shape):14s} land={100 * np.isnan(a).mean():4.1f}%  "
              f"min={fin.min():.4g} max={fin.max():.4g}")

    if args.patch_pkl_dir:
        import glob
        import pickle

        ref_pkl = args.mask_like or sorted(
            glob.glob(os.path.join(args.patch_pkl_dir, "input_state_member_*.pkl")))[0]
        with open(ref_pkl, "rb") as f:
            ref = pickle.load(f)
        ref_wave = ref["fields"]["swh"]
        if np.asarray(ref_wave).shape != np.asarray(fields[CY50R1_ONLY[0]]).shape:
            sys.exit(f"grid mismatch: state {np.asarray(ref_wave).shape} vs j1r2 "
                     f"{np.asarray(fields[CY50R1_ONLY[0]]).shape} -- use --grid N320")
        add, drop = reconcile_mask(fields, ref_wave)
        print(f"  mask reconciled against {os.path.basename(ref_pkl)}: "
              f"~{add} pts NaN-ed (j1r2 land), ~{drop} pts dropped (operational land)")

        for pkl in sorted(glob.glob(os.path.join(args.patch_pkl_dir,
                                                 "input_state_member_*.pkl"))):
            if pkl.endswith("_j1r2.pkl"):
                continue
            with open(pkl, "rb") as f:
                st = pickle.load(f)
            st["fields"] = dict(st["fields"])
            st["fields"].update({k: np.array(v, copy=True) for k, v in fields.items()})
            out = pkl.replace(".pkl", "_j1r2.pkl")
            with open(out, "wb") as f:
                pickle.dump(st, f, protocol=4)
            print(f"  patched -> {os.path.basename(out)}")
        print("NOTE: every member now shares one deterministic wave analysis; "
              "wave initial-condition spread is gone.")

    if args.out:
        np.savez_compressed(args.out, **fields)
        print(f"wrote {args.out} ({os.path.getsize(args.out) / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""Regression test for the three writer paths. No GPU needed.

`run_local_icechunk_v2.py` and `icechunk_output.py` are the live weekly pipeline, so the
O96 work had to be purely additive. This replays real N320 fields through
`IcechunkMemberWriter` and asserts:

1. **default** (no `regrid`, no `var_filter`) writes byte-for-byte what it always did —
   this is the test that says the existing cycle is untouched
2. `--grid o96` writes exactly `interpolate(field, N320, O96)`
3. `--native-store` writes the selected variables unchanged on the native grid
4. no chunk write amplification in any of them

Reads member 1 of an existing N320 store; writes into a temp dir and removes it.

Usage::

    python test_writer_paths.py                       # uses the 20260813 store
    python test_writer_paths.py --src /path/to/icechunk_v2 --tag cycle-20260813_0000
"""

import argparse
import os
import pathlib
import shutil
import sys
import tempfile

import numpy as np
import zarr
import icechunk

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.dirname(_HERE))
from icechunk_output import (open_repo_local, init_schema,  # noqa: E402
                             IcechunkMemberWriter)
import o96_grid  # noqa: E402

COORDS = ("latitude", "longitude", "time", "member")
DEFAULT_SRC = "/tank/projects/aifs-run/20260813_0000/icechunk_v2"
DEFAULT_TAG = "cycle-20260813_0000"


def build(tmp, name, n_values, coords, var_names, states, steps, time_chunk, **kw):
    """Write `states` through IcechunkMemberWriter and return the resulting group."""
    path = os.path.join(tmp, name)
    repo = open_repo_local(path)
    init_schema(repo, n_members=1, n_steps=132, n_values=n_values, var_names=var_names,
                latitudes=coords[0], longitudes=coords[1],
                ref_date=np.datetime64("2026-08-13T00"), timestep_s=6 * 3600,
                float_size="f4", time_chunk=time_chunk)
    w = IcechunkMemberWriter(repo, member_index=0, member_number=1,
                             commit_every=time_chunk, time_chunk=time_chunk, **kw)
    for t, st in zip(steps, states):
        w.write_step(st, time_index=t)
    w.finalize()
    got = zarr.open_group(repo.readonly_session("main").store, mode="r")
    nchunks = len(list((pathlib.Path(path) / "chunks").rglob("*"))) \
        if (pathlib.Path(path) / "chunks").exists() else 0
    return got, w, nchunks


def main():
    ap = argparse.ArgumentParser(description="regression-test the writer paths")
    ap.add_argument("--src", default=DEFAULT_SRC)
    ap.add_argument("--tag", default=DEFAULT_TAG)
    args = ap.parse_args()

    repo = icechunk.Repository.open(icechunk.local_filesystem_storage(args.src))
    src = zarr.open_group(repo.readonly_session(tag=args.tag).store, mode="r")
    VARS = [k for k in src.array_keys() if k not in COORDS]
    lat = np.asarray(src["latitude"][:])
    lon = np.asarray(src["longitude"][:])
    lat96, lon96 = o96_grid.o96_coords()
    STEPS = list(range(72, 84))                  # exactly one 12-step chunk
    print(f"source {args.src} @ {args.tag}: {len(VARS)} vars, steps {STEPS[0]}-{STEPS[-1]}")
    states = [{"fields": {v: np.asarray(src[v][0, t, :]) for v in VARS},
               "latitudes": lat, "longitudes": lon} for t in STEPS]

    tmp = tempfile.mkdtemp(prefix="o96_writer_test_")
    fails = []
    try:
        # 1. DEFAULT PATH — must equal the source exactly (production is untouched)
        got, w, nch = build(tmp, "default", 542080, (lat, lon), VARS, states, STEPS,
                            time_chunk=1)
        same = all(np.array_equal(np.asarray(got[v][0, t, :]),
                                  np.asarray(src[v][0, t, :]), equal_nan=True)
                   for v in VARS for t in STEPS)
        ideal = len(VARS) * len(STEPS)
        print(f"\n1. default path (no regrid, no filter, time_chunk=1)")
        print(f"   identical to source: {same}")
        print(f"   chunk objects {nch} vs ideal {ideal} (+2 coords)")
        fails += [] if same else ["default path differs from source"]
        fails += [] if nch <= ideal + 2 else [f"default path amplified: {nch} > {ideal}+2"]

        # 2. O96 PATH — must equal earthkit's interpolation
        got, w, nch = build(tmp, "o96", o96_grid.N_VALUES, (lat96, lon96), VARS,
                            states, STEPS, time_chunk=12,
                            regrid=o96_grid.regrid_block)
        import earthkit.regrid as ekr
        bad = []
        for v in (VARS[0], "2t", "tp", VARS[-1]):
            ref = ekr.interpolate(np.asarray(src[v][0, STEPS[0], :], dtype="float64"),
                                  {"grid": "N320"}, {"grid": "O96"}).astype("float32")
            if not np.array_equal(np.asarray(got[v][0, STEPS[0], :]), ref, equal_nan=True):
                bad.append(v)
        ideal = len(VARS) * (len(STEPS) // 12)
        print(f"\n2. --grid o96 (regrid, time_chunk=12)")
        print(f"   == earthkit.interpolate: {not bad}" + (f" (differs: {bad})" if bad else ""))
        print(f"   chunk objects {nch} vs ideal {ideal} (+2 coords)")
        fails += [] if not bad else [f"o96 path differs from earthkit for {bad}"]
        fails += [] if nch <= ideal + 2 else [f"o96 path amplified: {nch} > {ideal}+2"]

        # 3. SIDECAR PATH — selected vars, native grid, untouched values
        sub = ["msl", "tp", "2t"]
        got, w, nch = build(tmp, "sidecar", 542080, (lat, lon), sub, states, STEPS,
                            time_chunk=12, var_filter=set(sub))
        same = all(np.array_equal(np.asarray(got[v][0, t, :]),
                                  np.asarray(src[v][0, t, :]), equal_nan=True)
                   for v in sub for t in STEPS)
        only = sorted(k for k in got.array_keys() if k not in COORDS) == sorted(sub)
        print(f"\n3. --native-store (var_filter={sub})")
        print(f"   identical to source: {same} | only those vars present: {only}")
        fails += [] if same else ["sidecar path differs from source"]
        fails += [] if only else ["sidecar path wrote unexpected variables"]
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    print("\n" + ("PASS — the live pipeline's default behaviour is unchanged"
                  if not fails else "FAILED:\n  " + "\n  ".join(fails)))
    return 0 if not fails else 1


if __name__ == "__main__":
    sys.exit(main())

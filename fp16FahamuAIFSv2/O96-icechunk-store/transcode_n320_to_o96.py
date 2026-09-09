#!/usr/bin/env python3
"""Transcode an N320 Icechunk store to a much smaller O96 (~1 deg) store.

The inference stores every field on the native N320 reduced-Gaussian grid (542,080
cells). At 120 variables x 50 members that is ~583 GB for the 432-792 h window alone
and ~1.26 TB for the full 0-792 h rollout -- more than the box can keep for more than
two cycles. **O96 is the same kind of grid** (octahedral reduced Gaussian, quasi-equal-
area cells on an unstructured ``values`` axis) at 40,320 cells, so it is a drop-in for
every existing reader -- including ``epistemic-reasoning-risk/s2s_bn_evidence_prep.py``,
which assumes quasi-equal-area cells and takes an unweighted cell mean per spatial unit.

Why O96 and not 1.0 deg regular lat/lon:

* ``earthkit-regrid`` ships an ``O96 -> 1.5 deg`` matrix, so the AI-WQ product is still
  produced by the same earthkit interpolation it uses today. There is **no**
  ``1.0 deg -> 1.5 deg`` matrix, so a regular-latlon archive would force a hand-rolled
  second hop with different numerics.
* Equal-area cells keep the unweighted-cell-mean statistics in the downstream
  diagnostics unbiased; a regular lat/lon grid over-weights high latitudes.

The regrid itself is ``o96_grid.regrid_block`` — the same sparse matrix earthkit uses,
applied field by field, verified bit-identical to ``earthkit.regrid.interpolate``.

Usage::

    python transcode_n320_to_o96.py \
        --src   /tank/projects/aifs-run/20260813_0000/icechunk_v2 \
        --dst   /tank/projects/aifs-run/20260813_0000/icechunk_o96 \
        --src-tag cycle-20260813_0000 --date 20260813_0000

    # validate a couple of members against earthkit, write nothing
    python transcode_n320_to_o96.py --src ... --dst ... --members 1,2 --verify --dry-run

    # tier B also keeps the three AI-WQ variables on the native grid, so the graded
    # submission stays bit-identical (~14.5 GB) -- see O96-icechunk-store/README.md
    python transcode_n320_to_o96.py --src ... --dst .../icechunk_n320_aiwq \
        --native --vars msl,tp,2t --src-tag cycle-20260813_0000 --date 20260813_0000
"""

import argparse
import os
import sys
import time

import numpy as np
import zarr
import icechunk

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.dirname(_HERE))          # icechunk_output.py lives one up
from icechunk_output import open_repo_local  # noqa: E402
# The grid definition and the regrid live in ONE place. They used to be duplicated here,
# which is how this file kept the slow batched form after o96_grid.py was fixed.
from o96_grid import (o96_coords, load_matrix, regrid_block,  # noqa: E402
                      N_VALUES as O96_N_VALUES, N320_N_VALUES)

COORD_ARRAYS = ("latitude", "longitude", "time", "member")


def main():
    ap = argparse.ArgumentParser(description="N320 Icechunk store -> O96 Icechunk store")
    ap.add_argument("--src", required=True, help="source N320 store")
    ap.add_argument("--dst", required=True, help="destination O96 store (created)")
    ap.add_argument("--src-tag", default=None, help="read the source at this tag")
    ap.add_argument("--src-branch", default="main")
    ap.add_argument("--date", required=True, help="cycle date YYYYMMDD_0000 (for the tag)")
    ap.add_argument("--members", default=None, help="'1-50' or '1,2' (default: all)")
    ap.add_argument("--commit-every", type=int, default=12,
                    help="commit every N steps; also the time chunk (12 = 72 h). At O96 a "
                         "1-step chunk is only 161 KB, so 12 gives ~1.9 MB chunks.")
    ap.add_argument("--float-size", default="f4", choices=["f4", "f2"])
    ap.add_argument("--vars", default=None,
                    help="comma-separated subset to copy (default: every variable). Used "
                         "for the tier-B native sidecar, e.g. --vars msl,tp,2t")
    ap.add_argument("--native", action="store_true",
                    help="copy on the native N320 grid instead of regridding to O96. With "
                         "--vars this cuts the small N320 sidecar that keeps the AI-WQ "
                         "product bit-identical (see O96-icechunk-store/README.md tier B).")
    ap.add_argument("--verify", action="store_true",
                    help="check one field per member against earthkit.interpolate")
    ap.add_argument("--dry-run", action="store_true", help="do everything but write")
    ap.add_argument("--no-tag", action="store_true")
    args = ap.parse_args()

    src_repo = icechunk.Repository.open(icechunk.local_filesystem_storage(args.src))
    # fall back to the branch if the tag was tombstoned by a compaction
    try:
        sess = (src_repo.readonly_session(tag=args.src_tag) if args.src_tag
                else src_repo.readonly_session(args.src_branch))
    except Exception:
        print(f"  NOTE tag {args.src_tag!r} not found; tags present: "
              f"{sorted(src_repo.list_tags()) or 'none'} -> using branch {args.src_branch}")
        sess = src_repo.readonly_session(args.src_branch)
    src = zarr.open_group(sess.store, mode="r")

    all_vars = [k for k in src.array_keys() if k not in COORD_ARRAYS]
    if args.vars:
        var_names = [v.strip() for v in args.vars.split(",") if v.strip()]
        missing = [v for v in var_names if v not in all_vars]
        if missing:
            raise SystemExit(f"--vars: not in source store: {missing}")
    else:
        var_names = all_vars
    n_members, n_steps, n_values = src[var_names[0]].shape
    if n_values != N320_N_VALUES:
        raise SystemExit(f"source is not N320 ({n_values} values)")

    members = (list(range(1, n_members + 1)) if not args.members
               else _parse_members(args.members))
    # --native is a straight copy: no matrix, and the destination keeps N320 coords.
    mat = None if args.native else load_matrix()
    dst_n_values = N320_N_VALUES if args.native else O96_N_VALUES
    if args.native:
        dst_lat = np.asarray(src["latitude"][:], "f4")
        dst_lon = np.asarray(src["longitude"][:], "f4")
    else:
        dst_lat, dst_lon = o96_coords()

    print("=" * 70)
    print(f"{'N320 -> N320 (subset copy)' if args.native else 'N320 -> O96 transcode'}"
          f": {args.src}")
    print(f"                    -> {args.dst}")
    print(f"{len(var_names)}/{len(all_vars)} vars x {n_members} members x {n_steps} steps")
    print(f"{N320_N_VALUES} -> {dst_n_values} values "
          f"({N320_N_VALUES/dst_n_values:.1f}x fewer)")
    print(f"commit/time-chunk: {args.commit_every} steps ({args.commit_every*6} h)")
    print("=" * 70)

    if not args.dry_run:
        dst_repo = open_repo_local(args.dst)
        _init_dst(dst_repo, src, var_names, n_members, n_steps,
                  dst_lat, dst_lon, args.float_size, args.commit_every, dst_n_values)

    for member in members:
        t0 = time.time()
        m = member - 1
        # Steps the source actually holds for this member (a --write-hours run leaves
        # the rest as NaN); transcoding those would just burn chunks on NaN.
        probe = np.asarray(src[var_names[0]][m, :, 0])
        present = np.flatnonzero(np.isfinite(probe))
        if present.size == 0:
            print(f"--- member {member:03d}: [SKIP] no data in source")
            continue

        # Write one WHOLE time chunk per commit. Writing step by step into a 12-step
        # chunk makes zarr read-modify-write that chunk 12 times, and Icechunk is
        # copy-on-write, so every one of those touches leaves a chunk object behind —
        # a one-member test that way cost 4.9 GB instead of ~1 GB. Slab writes aligned
        # to the chunk boundary touch each chunk exactly once.
        tc = args.commit_every
        for lo in range(int(present[0]) // tc * tc, int(present[-1]) + 1, tc):
            hi = min(lo + tc, n_steps)
            steps = [int(t) for t in present if lo <= t < hi]
            if not steps:
                continue
            # (n_vars, n_steps_in_chunk, dst_n_values), NaN where the source has no step
            slab = np.full((len(var_names), hi - lo, dst_n_values), np.nan, args.float_size)
            for t in steps:
                block = np.stack([np.asarray(src[v][m, t, :]) for v in var_names])
                out = block if mat is None else regrid_block(block, mat)
                if args.verify and mat is not None and t == steps[0] \
                        and lo == int(present[0]) // tc * tc:
                    _verify(block[0], out[0], var_names[0], member)
                slab[:, t - lo, :] = out
            if args.dry_run:
                continue
            writer_sess = dst_repo.writable_session("main")
            dst = zarr.open_group(writer_sess.store, mode="r+")
            for vi, v in enumerate(var_names):
                dst[v][m, lo:hi, :] = slab[vi]
            writer_sess.commit(f"member {member:03d} steps {lo:03d}-{hi-1:03d}")
        print(f"--- member {member:03d}: {present.size} steps in {time.time()-t0:.1f}s")

    if not args.dry_run and not args.no_tag:
        _tag_if_complete(dst_repo, args.date, var_names[0], members, n_members)
    print("done")


def _parse_members(spec):
    spec = spec.strip()
    if "-" in spec:
        a, b = map(int, spec.split("-"))
        return list(range(a, b + 1))
    return [int(x) for x in spec.split(",")]


def _init_dst(repo, src, var_names, n_members, n_steps, dst_lat, dst_lon,
              float_size, time_chunk, n_values):
    """Same schema as the source store, on the destination values axis."""
    from icechunk_output import schema_exists
    if schema_exists(repo):
        print("[schema] destination already initialised — resuming")
        return
    s = repo.writable_session("main")
    root = zarr.group(s.store, overwrite=True)
    root.create_array("latitude", shape=(n_values,), chunks=(n_values,),
                      dtype="f4", dimension_names=("values",))[:] = dst_lat
    root.create_array("longitude", shape=(n_values,), chunks=(n_values,),
                      dtype="f4", dimension_names=("values",))[:] = dst_lon
    t = root.create_array("time", shape=(n_steps,), chunks=(n_steps,), dtype="i4",
                          dimension_names=("time",))
    t[:] = np.asarray(src["time"][:], dtype="i4")
    t.attrs.update(dict(src["time"].attrs))
    root.create_array("member", shape=(n_members,), chunks=(n_members,), dtype="i4",
                      dimension_names=("member",))[:] = np.asarray(src["member"][:], "i4")
    tc = max(1, min(int(time_chunk), n_steps))
    for v in var_names:
        root.create_array(v, shape=(n_members, n_steps, n_values),
                          chunks=(1, tc, n_values), dtype=float_size,
                          fill_value=float("nan"),
                          dimension_names=("member", "time", "values"))
    s.commit(f"init schema: {n_members}x{n_steps}x{n_values}, "
             f"{len(var_names)} vars ({float_size}), time_chunk={tc}")


def _verify(n320_field, got, var, member):
    import earthkit.regrid as ekr
    ref = ekr.interpolate(np.asarray(n320_field, dtype="float64"),
                          {"grid": "N320"}, {"grid": "O96"})
    d = float(np.nanmax(np.abs(ref - got)))
    print(f"    [VERIFY] member {member:03d} {var}: max|batched - earthkit| = {d:.3e}"
          + ("  OK" if d == 0.0 else "  *** MISMATCH ***"))


def _tag_if_complete(repo, date_str, sentinel, members, n_members):
    """Tag only a complete ensemble — an Icechunk tag is immutable and tombstoned."""
    from icechunk_output import member_written
    root = zarr.open_group(repo.readonly_session("main").store, mode="r")
    probe = np.asarray(root[sentinel][0, :, 0])
    last = int(np.flatnonzero(np.isfinite(probe))[-1])
    complete = [m for m in range(1, n_members + 1)
                if member_written(repo, m - 1, last, sentinel=sentinel)]
    if len(complete) < n_members:
        print(f"[TAG] not tagged: {len(complete)}/{n_members} members present")
        return
    try:
        snap = repo.lookup_branch("main")
        repo.create_tag(f"cycle-{date_str}", snapshot_id=snap)
        print(f"[TAG] cycle-{date_str} -> {snap}")
    except Exception as e:
        print(f"[TAG] skipped ({e})")


if __name__ == "__main__":
    main()

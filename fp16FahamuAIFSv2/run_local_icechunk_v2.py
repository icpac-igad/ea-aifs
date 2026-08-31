#!/usr/bin/env python3
"""
Fully-local AIFS-ENS-2.0 FP16 runner -> local Icechunk store (no GCS, no GRIB).

On-box counterpart of ``fp16_automate_aifs_gpu_pipeline_v2.py``. Where that
orchestrator does download(GCS) -> GRIB inference -> upload(GCS) -> cleanup, this
script collapses the loop to **read local pkl -> infer -> write straight into a local
Icechunk store -> commit** (one ACID snapshot per member). Nothing touches GCS, so no
``coiled-data.json`` is needed. The existing GRIB+GCS pipeline is left untouched as the
fallback.

Output is the ``(member, time, values)`` Icechunk schema from ``ICECHUNK_PATH_A.md``
(native N320 cells, ALL fields kept incl. ``cos_mwd``/``sin_mwd``), written via
``icechunk_output.IcechunkMemberWriter`` to a local-filesystem store, e.g.
``/tank/projects/aifs-run/<date>/icechunk_v2``.

Input pkls are the 112-field member states from
``ecmwf_opendata_pkl_input_aifsens_v2.py`` (run it with ``--no-upload --keep-local``
to produce them locally).

All ensemble members land in **one store**: the schema's ``member`` axis is sized by
``--n-members`` (default 50) and each member writes its own slice, owning disjoint chunks
(member-chunk size 1), so members never conflict. ``--skip-existing`` makes a multi-member
run idempotent — an interrupted 50-member run resumes without redoing finished members.

Usage::

    # single member, short lead
    python run_local_icechunk_v2.py --date 20260625_0000 --members 1 --lead-time 72 \
        --input-dir /tank/projects/aifs-run/20260625_0000/input_states \
        --store     /tank/projects/aifs-run/20260625_0000/icechunk_v2

    # full ensemble, all 50 members into the SAME store, resumable
    python run_local_icechunk_v2.py --date 20260625_0000 --members 1-50 --lead-time 960 \
        --input-dir .../input_states --store .../icechunk_v2 \
        --commit-every 1 --skip-existing
"""

import os
import sys
import time
import argparse
from concurrent.futures import ThreadPoolExecutor

# This script lives in the model subfolder; import the sibling modules directly.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# o96_grid lives in O96-icechunk-store/ (hyphens, so not importable as a package).
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "O96-icechunk-store"))

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
DEFAULT_LEAD_TIME = 960          # hours (40 days); pass --lead-time 72 for a smoke test
TIME_STEP_HOURS = 6              # aifs-ens-2.0 timestep
INFERENCE_PRECISION = "16"       # FP16
INFERENCE_NUM_CHUNKS = 16        # higher -> less VRAM, slower
DEFAULT_N_MEMBERS = 50           # size of the member axis in the store

# Downstream (shared/aifs_n320_grib_1p5defg_nc_cli.py) reads only the 432-792 h
# windows (days 18-33). Steps outside this are computed by the rollout but need
# not be stored; skipped steps allocate no chunks and read back as NaN.
DOWNSTREAM_WINDOW = "432-792"


def parse_member_range(member_str):
    """'1-50' or '1,2,3' or '7' -> list[int]."""
    member_str = member_str.strip()
    if "-" in member_str:
        a, b = map(int, member_str.split("-"))
        return list(range(a, b + 1))
    if "," in member_str:
        return [int(m) for m in member_str.split(",")]
    return [int(member_str)]


def parse_write_hours(spec):
    """'432-792' -> (432, 792); None/'all' -> (None, None) meaning write every step."""
    if not spec or str(spec).lower() == "all":
        return (None, None)
    a, b = str(spec).split("-")
    lo, hi = int(a), int(b)
    if lo > hi:
        raise ValueError(f"--write-hours {spec}: start > end")
    return (lo, hi)


def pkl_path_for(member, input_dir):
    return os.path.join(input_dir, f"input_state_member_{member:03d}.pkl")


# Cells on the ``values`` axis for each grid the runner can write. N320 is the native
# model grid; O96 is the archive grid (see O96-icechunk-store/README.md).
GRID_N_VALUES = {"n320": 542080, "o96": 40320}


def store_values_axis(repo):
    """Length of an existing store's ``values`` axis, or None if not initialised yet."""
    import zarr
    try:
        root = zarr.open_group(repo.readonly_session("main").store, mode="r")
        return int(root["latitude"].shape[0]) if "latitude" in root else None
    except Exception:
        return None


def check_store_grid(repo, path, grid, label):
    """Refuse to write one grid into a store that already holds another.

    Without this the run loads the model and starts a rollout before failing deep in the
    writer on a shape mismatch — and ``--skip-existing`` would probe the wrong grid.
    """
    have = store_values_axis(repo)
    want = GRID_N_VALUES[grid]
    if have is None or have == want:
        return
    other = next((g.upper() for g, n in GRID_N_VALUES.items() if n == have),
                 f"{have}-cell")
    raise SystemExit(
        f"ERROR: the {label} store already holds a {other} grid ({have} values) but "
        f"--grid {grid} writes {want}.\n       {path}\n"
        f"       Point at a different path, or remove that store.")


def fetch_pkl(member, input_dir, bucket, gcs_prefix, service_account_key):
    """Download one member's input pkl from GCS if not already present locally.

    Downloads to ``*.part`` and only renames after the byte count matches the blob,
    so an interrupted transfer can never leave a truncated pkl in place.
    """
    from google.cloud import storage

    path = pkl_path_for(member, input_dir)
    blob_name = f"{gcs_prefix}/input_state_member_{member:03d}.pkl"
    os.makedirs(input_dir, exist_ok=True)

    client = storage.Client.from_service_account_json(service_account_key)
    blob = client.bucket(bucket).blob(blob_name)
    blob.reload()

    if os.path.exists(path) and os.path.getsize(path) == blob.size:
        return path                                   # already complete

    tmp = path + ".part"
    blob.download_to_filename(tmp)
    got = os.path.getsize(tmp)
    if got != blob.size:
        os.remove(tmp)
        raise IOError(f"member {member:03d}: got {got} bytes, expected {blob.size}")
    os.replace(tmp, path)
    return path


def run(date_str, members, input_dir, store_path, lead_time,
        n_members=DEFAULT_N_MEMBERS, precision=INFERENCE_PRECISION,
        num_chunks=INFERENCE_NUM_CHUNKS, float_size="f4", tag=True,
        commit_every=1, skip_existing=False, write_hours=None,
        gcs_fetch=False, bucket=None, gcs_prefix=None, service_account_key=None,
        cleanup_pkl=False, grid="n320", native_store=None, native_vars=None,
        native_write_hours=None):

    # VRAM knobs must be set before the model / CUDA context is created.
    os.environ["ANEMOI_INFERENCE_NUM_CHUNKS"] = str(num_chunks)
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    import fp16_multi_run_AIFS_ENS_v2 as runmod
    from icechunk_output import (open_repo_local, init_schema, schema_exists,
                                 member_written, IcechunkMemberWriter)
    from anemoi.inference.runners.simple import SimpleRunner

    bad = [m for m in members if m < 1 or m > n_members]
    if bad:
        raise ValueError(f"members {bad} outside 1..{n_members} (raise --n-members)")

    n_steps = lead_time // TIME_STEP_HOURS
    h_lo, h_hi = parse_write_hours(write_hours)

    def keep(hour):
        """Store this step? (rollout still computes every step regardless)"""
        return (h_lo is None or hour >= h_lo) and (h_hi is None or hour <= h_hi)

    kept = [s for s in range(1, n_steps + 1) if keep(s * TIME_STEP_HOURS)]
    if not kept:
        raise ValueError(f"--write-hours {write_hours} selects no step within "
                         f"lead-time {lead_time}h")
    last_kept_index = kept[-1] - 1        # absolute 0-based index of final stored step

    # The sidecar may need a DIFFERENT window from the main store: the O96 corpus wants
    # every step (MJO needs day 8 and 15, which a 432-792 h window cannot reach) while the
    # N320 sidecar only needs the downstream window it feeds. Omitted -> inherit the main
    # window, which is the behaviour every existing command line already has.
    if native_write_hours is None:
        nh_lo, nh_hi, native_window_src = h_lo, h_hi, write_hours
    else:
        nh_lo, nh_hi = parse_write_hours(native_write_hours)
        native_window_src = native_write_hours

    def keep_native(hour):
        return (nh_lo is None or hour >= nh_lo) and (nh_hi is None or hour <= nh_hi)

    kept_native = [s for s in range(1, n_steps + 1) if keep_native(s * TIME_STEP_HOURS)]
    if native_store and not kept_native:
        raise ValueError(f"--native-write-hours {native_write_hours} selects no step "
                         f"within lead-time {lead_time}h")
    last_kept_native_index = (kept_native[-1] - 1) if kept_native else None

    print("=" * 70)
    print("AIFS-ENS-2.0 FP16  ->  LOCAL Icechunk store (no GCS, no GRIB)")
    print("=" * 70)
    print(f"Date:        {date_str}")
    print(f"Members:     {members}")
    print(f"Input pkls:  {input_dir}")
    print(f"Store:       {store_path}")
    print(f"Precision:   FP16 | chunks={num_chunks} | dtype={float_size}")
    print(f"Lead time:   {lead_time}h ({lead_time // 24}d) -> {n_steps} steps @ {TIME_STEP_HOURS}h")
    print(f"Commit:      every {commit_every} step(s) = {commit_every * TIME_STEP_HOURS}h "
          f"| time_chunk={commit_every} (aligned, no amplification)")
    if h_lo is None:
        print(f"Write window: ALL {n_steps} steps stored")
    else:
        print(f"Write window: {h_lo}-{h_hi}h -> storing {len(kept)}/{n_steps} steps "
              f"({100*len(kept)/n_steps:.0f}%); others computed but not stored (NaN)")
    print(f"Member axis: 1..{n_members} in ONE store"
          + (" | skip already-written members (resume)" if skip_existing else ""))
    if grid == "o96":
        print(f"Grid:        N320 -> O96 regridded at write time (40,320 cells, "
              f"~13.4x smaller); N320 never hits disk")
    if native_store:
        print(f"Sidecar:     {','.join(native_vars)} kept on native N320 -> {native_store}")
        if nh_lo is None:
            print(f"Sidecar window: ALL {n_steps} steps stored")
        else:
            print(f"Sidecar window: {nh_lo}-{nh_hi}h -> storing {len(kept_native)}/"
                  f"{n_steps} steps ({100*len(kept_native)/n_steps:.0f}%)"
                  + ("" if native_write_hours is None else "  [--native-write-hours]"))
    if gcs_fetch:
        print(f"Input:       fetch from gs://{bucket}/{gcs_prefix}/ (prefetch next member)"
              + (" | delete pkl after each member" if cleanup_pkl else ""))
    print("=" * 70)

    repo = open_repo_local(store_path)
    native_repo = open_repo_local(native_store) if native_store else None

    # Fail before the model loads if a store already holds the other grid.
    check_store_grid(repo, store_path, grid, "--store")
    if native_repo is not None:
        check_store_grid(native_repo, native_store, "n320", "--native-store")

    # The O96 regrid is the same sparse matrix earthkit uses, applied FIELD BY FIELD
    # (batching a whole step is ~9-10x slower — see O96-icechunk-store/README.md §4) and
    # bit-identical to earthkit. ~25 ms per stored step, so the corpus can be written
    # coarse directly off the rollout and N320 never reaches disk.
    if grid == "o96":
        import o96_grid
        regrid_fn = o96_grid.regrid_block
        o96_lats, o96_lons = o96_grid.o96_coords()
    else:
        regrid_fn = None

    print("\nLoading model ecmwf/aifs-ens-2.0 (FP16)...")
    t0 = time.time()
    runner = SimpleRunner({"huggingface": "ecmwf/aifs-ens-2.0"},
                          device="cuda", precision=precision)
    print(f"Model loaded in {time.time() - t0:.1f}s\n")

    ok, failed, skipped, snap = [], [], [], None

    # Resolve the work list up front (before any download) so resumed/complete members
    # never cost a GCS transfer.
    todo = []
    for member in members:
        done = (skip_existing and schema_exists(repo)
                and member_written(repo, member - 1, last_kept_index))
        # A member is only complete if the sidecar has it too, at the sidecar's own final
        # step -- otherwise adding a sidecar to an existing run would silently skip every
        # member and leave it empty.
        if done and native_repo is not None:
            done = (schema_exists(native_repo)
                    and member_written(native_repo, member - 1, last_kept_native_index))
        if done:
            print(f"--- Member {member:03d}: [SKIP] already complete in store")
            skipped.append(member)
        else:
            todo.append(member)
    print(f"\nTo process: {len(todo)} member(s); skipped {len(skipped)}\n")

    # Prefetch the next member's pkl (~1 GB) on a worker thread while the GPU runs the
    # current one, so the download is hidden behind inference.
    pool = ThreadPoolExecutor(max_workers=1) if gcs_fetch else None
    futures = {}
    if pool and todo:
        futures[todo[0]] = pool.submit(fetch_pkl, todo[0], input_dir, bucket,
                                       gcs_prefix, service_account_key)

    for i, member in enumerate(todo):
        m_t0 = time.time()
        print(f"\n--- Member {member:03d} ({i + 1}/{len(todo)}) ---")

        if pool:
            try:
                futures[member].result()             # ensure this member's pkl landed
                print(f"    [FETCH] pkl ready")
            except Exception as e:
                print(f"    [FAILED] download: {e}")
                failed.append(member)
                continue
            if i + 1 < len(todo):                    # kick off the next download now
                nxt = todo[i + 1]
                futures[nxt] = pool.submit(fetch_pkl, nxt, input_dir, bucket,
                                           gcs_prefix, service_account_key)

        try:
            input_state = runmod.load_input_state_from_pickle(member, input_dir)
        except FileNotFoundError as e:
            print(f"    [SKIP] {e}")
            failed.append(member)
            continue

        ref_date = input_state["date"]
        runner.time_step = TIME_STEP_HOURS
        runner.lead_time = lead_time
        runner.reference_date = ref_date

        gen = runner.run(input_state=input_state, lead_time=lead_time)
        try:
            first = next(gen)
        except StopIteration:
            print("    [FAIL] inference yielded no states")
            failed.append(member)
            continue

        # One-time schema init, from the first real state (var names/coords exact).
        var_names = list(first["fields"].keys())
        if grid == "o96":
            lats, lons = o96_lats, o96_lons
        else:
            lats, lons = first["latitudes"], first["longitudes"]
        n_values = len(lats)
        if not schema_exists(repo):
            print(f"    [SCHEMA] init {n_members}x{n_steps}x{n_values}, "
                  f"{len(var_names)} vars ({float_size}), time_chunk={commit_every}")
            init_schema(repo, n_members=n_members, n_steps=n_steps,
                        n_values=n_values, var_names=var_names,
                        latitudes=lats, longitudes=lons,
                        ref_date=ref_date, timestep_s=TIME_STEP_HOURS * 3600,
                        float_size=float_size, time_chunk=commit_every)
        if native_repo is not None and not schema_exists(native_repo):
            print(f"    [SCHEMA] native sidecar {n_members}x{n_steps}x"
                  f"{len(first['latitudes'])}, {len(native_vars)} vars")
            init_schema(native_repo, n_members=n_members, n_steps=n_steps,
                        n_values=len(first["latitudes"]), var_names=list(native_vars),
                        latitudes=first["latitudes"], longitudes=first["longitudes"],
                        ref_date=ref_date, timestep_s=TIME_STEP_HOURS * 3600,
                        float_size=float_size, time_chunk=commit_every)

        writer = IcechunkMemberWriter(repo, member_index=member - 1,
                                      member_number=member, commit_every=commit_every,
                                      regrid=regrid_fn)
        native_writer = (IcechunkMemberWriter(
            native_repo, member_index=member - 1, member_number=member,
            commit_every=commit_every, var_filter=native_vars)
            if native_repo is not None else None)

        # The rollout is autoregressive: every step must be COMPUTED. Steps outside the
        # write window are simply not stored (no chunks allocated -> read back as NaN).
        # Index by the ABSOLUTE step so skipped steps don't shift later ones.
        def consume(state, step):
            hour = step * TIME_STEP_HOURS
            if keep(hour):
                writer.write_step(state, time_index=step - 1)   # keep ALL fields
            if native_writer is not None and keep_native(hour):
                native_writer.write_step(state, time_index=step - 1)
            if step % 20 == 0:
                print(f"    {hour}h / {lead_time}h "
                      f"(stored {writer.n}/{len(kept)}, {len(writer.snapshots)} commits)")

        consume(first, 1)
        for step, state in enumerate(gen, start=2):
            consume(state, step)

        if writer.n != len(kept):
            print(f"    [WARN] stored {writer.n} steps, expected {len(kept)}")
        if native_writer is not None and native_writer.n != len(kept_native):
            print(f"    [WARN] sidecar stored {native_writer.n} steps, "
                  f"expected {len(kept_native)}")
        snap = writer.finalize()                 # flush any uncommitted tail steps
        if native_writer is not None:
            native_writer.finalize()
        ok.append(member)
        print(f"    [OK] member {member:03d}: stored {writer.n}/{n_steps} steps, "
              f"{len(writer.snapshots)} commits (last {snap}) in {time.time() - m_t0:.1f}s")

        # Free the ~1 GB pkl immediately; staging all 50 would cost ~48 GB.
        if cleanup_pkl:
            p = pkl_path_for(member, input_dir)
            if os.path.exists(p):
                os.remove(p)
                print(f"    [CLEANUP] removed {os.path.basename(p)}")

    if pool:
        pool.shutdown(wait=False, cancel_futures=True)

    # Tag ONLY once the whole ensemble is present. Icechunk tags are immutable *and*
    # deleted names are tombstoned (never reusable), so tagging after a partial run
    # would permanently bind `cycle-<date>` to a 1-member snapshot -- a reader would
    # then silently get NaNs for every other member.
    if tag and snap is not None:
        complete = [m for m in range(1, n_members + 1)
                    if member_written(repo, m - 1, last_kept_index)]
        if len(complete) < n_members:
            print(f"\n[TAG] not tagged: {len(complete)}/{n_members} members present. "
                  f"Re-run the remaining members (--skip-existing); the cycle is tagged "
                  f"only when the ensemble is complete.")
        else:
            try:
                repo.create_tag(f"cycle-{date_str}", snapshot_id=snap)
                print(f"\n[TAG] cycle-{date_str} -> {snap} ({n_members}/{n_members} members)")
            except Exception as e:
                print(f"\n[TAG] skipped ({e})")

    print("\n" + "=" * 70)
    print(f"DONE: {len(ok)} written, {len(skipped)} skipped, {len(failed)} failed "
          f"of {len(members)} members -> {store_path}")
    if skipped:
        print(f"Skipped (already in store): {skipped}")
    if failed:
        print(f"Failed: {failed}")
    print("=" * 70)
    return len(failed) == 0


def main():
    ap = argparse.ArgumentParser(
        description="Local AIFS-ENS-2.0 FP16 inference into a local Icechunk store")
    ap.add_argument("--date", required=True, help="cycle date YYYYMMDD_0000")
    ap.add_argument("--members", default="1", help="'1-50' or '1,2,3' or '7'")
    ap.add_argument("--lead-time", type=int, default=DEFAULT_LEAD_TIME,
                    help="forecast lead time (hours), default 960")
    ap.add_argument("--input-dir", required=True,
                    help="dir holding input_state_member_NNN.pkl")
    ap.add_argument("--store", required=True,
                    help="local Icechunk store path (created if missing)")
    ap.add_argument("--n-members", type=int, default=DEFAULT_N_MEMBERS,
                    help="size of the member axis in the store (default 50)")
    ap.add_argument("--precision", default=INFERENCE_PRECISION, choices=["16", "32"])
    ap.add_argument("--chunks", type=int, default=INFERENCE_NUM_CHUNKS,
                    help="ANEMOI_INFERENCE_NUM_CHUNKS (default 16)")
    ap.add_argument("--float-size", default="f4", choices=["f4", "f2"],
                    help="stored dtype (f4 exact / f2 bulk)")
    ap.add_argument("--commit-every", type=int, default=1,
                    help="commit every N model steps (1 = every 6h = per step, "
                         "default; 2 = 12h, 12 = 72h). time_chunk is aligned to this "
                         "so writes never amplify.")
    ap.add_argument("--skip-existing", action="store_true",
                    help="skip members whose final stored step is already present "
                         "(idempotent resume of an interrupted multi-member run)")
    ap.add_argument("--write-hours", default="all",
                    help=f"only STORE steps whose lead hour is in this window, e.g. "
                         f"'{DOWNSTREAM_WINDOW}' (what the downstream regrid reads). "
                         f"'all' (default) stores every step. The rollout always "
                         f"computes every step; skipped steps use no disk (NaN).")
    ap.add_argument("--no-tag", action="store_true", help="do not tag the cycle")
    ap.add_argument("--gcs-fetch", action="store_true",
                    help="download each member's input pkl from GCS on demand "
                         "(next member prefetched during inference)")
    ap.add_argument("--bucket", default="aifs-aiquest-us-20251127")
    ap.add_argument("--service-account", default="coiled-data.json")
    ap.add_argument("--gcs-input-prefix", default=None,
                    help="default '<date>/input_v2'")
    ap.add_argument("--cleanup-pkl", action="store_true",
                    help="delete each member's pkl after it is written (saves ~48 GB "
                         "over a 50-member run)")
    ap.add_argument("--grid", default="n320", choices=["n320", "o96"],
                    help="grid the store is written on. 'n320' (default) is the native "
                         "model grid, ~583 GB/cycle for the 432-792 h window. 'o96' "
                         "regrids each step in-flight to 40,320 cells (~47 GB) so the "
                         "N320 corpus never lands on disk; costs ~14 s/member against "
                         "~285 s of inference. See O96-icechunk-store/README.md.")
    ap.add_argument("--native-store", default=None,
                    help="also write --native-vars to a SECOND store on the native N320 "
                         "grid. With --grid o96 this is tier B: the AI-WQ product stays "
                         "bit-identical for ~13 GB while the 120-var corpus goes coarse.")
    ap.add_argument("--native-vars", default="msl,tp,2t",
                    help="variables for --native-store (default: the three AI-WQ ones)")
    ap.add_argument("--native-write-hours", default=None,
                    help="write window for --native-store, if it should differ from "
                         "--write-hours. The case this exists for: keep the O96 corpus "
                         "complete (no --write-hours, so day 8/15 are present for MJO) "
                         "while the N320 sidecar carries only the downstream window, "
                         f"e.g. --native-write-hours {DOWNSTREAM_WINDOW}. Omitted, the "
                         "sidecar inherits --write-hours exactly as before.")
    args = ap.parse_args()

    gcs_prefix = args.gcs_input_prefix or f"{args.date}/input_v2"
    if args.gcs_fetch and not os.path.exists(args.service_account):
        print(f"ERROR: service account not found: {args.service_account}")
        return 1

    members = parse_member_range(args.members)
    ok = run(date_str=args.date, members=members, input_dir=args.input_dir,
             store_path=args.store, lead_time=args.lead_time,
             n_members=args.n_members, precision=args.precision,
             num_chunks=args.chunks, float_size=args.float_size,
             tag=not args.no_tag, commit_every=args.commit_every,
             skip_existing=args.skip_existing, write_hours=args.write_hours,
             gcs_fetch=args.gcs_fetch, bucket=args.bucket, gcs_prefix=gcs_prefix,
             service_account_key=os.path.abspath(args.service_account)
             if args.gcs_fetch else None,
             cleanup_pkl=args.cleanup_pkl, grid=args.grid,
             native_store=args.native_store,
             native_vars=[v.strip() for v in args.native_vars.split(",") if v.strip()],
             native_write_hours=args.native_write_hours)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "numpy",
#     "obstore",
#     "fsspec",
#     "s3fs",
#     "gribberish",
#     "earthkit-regrid",
#     "google-cloud-storage",
# ]
# ///
"""
S3 GRIB -> AIFS ENS input-state pkl, WITHOUT kerchunk.

Produces the same 92-field, (2, 542080) input state as
ecmwf_opendata_pkl_input_aifsens.py, but sourced entirely from the ECMWF
dissemination GRIB on s3://ecmwf-forecasts/ via the small `.index` sidecars
instead of scanning whole GRIB files with kerchunk.scan_grib.

How it works
------------
Every `.grib2` on s3://ecmwf-forecasts/ ships a tiny (~2 MB) JSON-lines
`.index` next to it. Each line is one GRIB message with `_offset`/`_length`,
so we never read 6.3 GB: we read the index, filter to the ~89 messages a
member needs, byte-range fetch only those, and decode each with `gribberish`
(handles ECMWF's grid_ccsds packing; verified pixel-identical to eccodes).

Two enfo files give surface + pressure + soil for members 1-50:
  t-6h : <prev_date>/<prev_hh>z/.../enfo/<stamp>-0h-enfo-ef.grib2
  t    : <date>/<hh>z/.../enfo/<stamp>-0h-enfo-ef.grib2
The 3 pure constants z/slor/sdor (+ lsm) are not in enfo; they come from the
deterministic `oper` 0h file at the analysis time only (time-invariant, so
fetched once and replicated to both timesteps).

OOM-safe: each param group is fetched/decoded/regridded, written to a partial
pickle, then freed; a final pass merges the partials into the single pkl.
This keeps peak memory to one group instead of all 92 fields, and makes a
member restartable from whatever partials already exist.
"""

import argparse
import datetime
import gc
import json
import os
import pickle
import time
import traceback
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

import earthkit.regrid as ekr
import fsspec
import gribberish
import numpy as np
import obstore as obs
from google.cloud import storage
from obstore.store import from_url


# ---------------------------------------------------------------- configuration
PARAM_SFC = ["10u", "10v", "2d", "2t", "msl", "skt", "sp", "tcw"]
PARAM_SFC_CONST = ["lsm", "z", "slor", "sdor"]   # from oper 0h, replicated to 2 ts
PARAM_SOIL = ["sot"]
PARAM_PL = ["gh", "t", "u", "v", "w", "q"]
LEVELS = [1000, 925, 850, 700, 600, 500, 400, 300, 250, 200, 150, 100, 50]
SOIL_LEVELS = [1, 2]
SOIL_MAP = {"sot_1": "stl1", "sot_2": "stl2"}

N320_NPOINTS = 542080
RAW_SHAPE = (721, 1440)            # ECMWF 0p25, lat 90->-90, lon0=180
# Overridable so a member can be throttled (S3 SlowDown) or run single-threaded
# to warm the earthkit-regrid matrix cache without a concurrent-download race.
MAX_WORKERS = int(os.environ.get("S3_PKL_MAX_WORKERS", "16"))

S3_BUCKET = "ecmwf-forecasts"
S3_REGION = "eu-central-1"

# GCS (mirrors ecmwf_opendata_pkl_input_aifsens.py; off by default for validation)
GCS_BUCKET = "aifs-aiquest-us-20251127"
# key sits next to this script; resolve from __file__ so cwd doesn't matter
GCS_SERVICE_ACCOUNT_KEY = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "coiled-data.json")


# ---------------------------------------------------------------- S3 fetch layer
_obstore_cache = {}


def _get_store():
    if "s" not in _obstore_cache:
        _obstore_cache["s"] = from_url(
            f"s3://{S3_BUCKET}", region=S3_REGION, skip_signature=True)
    return _obstore_cache["s"]


def fetch_range(key, offset, length):
    """Byte range from s3://ecmwf-forecasts/<key> (obstore, fsspec fallback)."""
    try:
        return bytes(obs.get_range(_get_store(), key,
                                   start=offset, end=offset + length))
    except Exception:
        fs = fsspec.filesystem("s3", anon=True)
        return fs.read_block(f"{S3_BUCKET}/{key}", offset, length)


def load_index(key):
    """Read+parse the JSON-lines .index sidecar (small, cached per key)."""
    if key in _obstore_cache:
        return _obstore_cache[key]
    fs = fsspec.filesystem("s3", anon=True)
    raw = fs.cat_file(f"{S3_BUCKET}/{key}").decode()
    recs = [json.loads(ln) for ln in raw.splitlines() if ln.strip()]
    _obstore_cache[key] = recs
    return recs


# ---------------------------------------------------------------- S3 path layout
def _stamp(dt):
    return dt.strftime("%Y%m%d"), dt.strftime("%H"), dt.strftime("%Y%m%d%H%M%S")


def enfo_paths(dt):
    ymd, hh, st = _stamp(dt)
    b = f"{ymd}/{hh}z/ifs/0p25/enfo/{st}-0h-enfo-ef"
    return b + ".grib2", b + ".index"


def oper_paths(dt):
    ymd, hh, st = _stamp(dt)
    b = f"{ymd}/{hh}z/ifs/0p25/oper/{st}-0h-oper-fc"
    return b + ".grib2", b + ".index"


# ---------------------------------------------------------------- decode + grid
def decode_grib_bytes(raw):
    """Decode one GRIB2 message with gribberish -> (721,1440) float32.

    gribberish handles ECMWF's grid_ccsds packing (verified pixel-identical to
    eccodes). The dissemination grid starts at longitude 180; roll by -Ni//2
    so it begins at 0 deg, matching what the open-data pipeline produces and
    what earthkit.regrid / the AIFS checkpoint expect.
    """
    msg = gribberish.parse_grib_message(raw, 0)
    # float64 throughout to match the known-good open-data pkl exactly
    # (earthkit's to_numpy()/regrid produce float64; the AIFS ENS checkpoint
    # has been fed float64 input states).
    arr = np.asarray(msg.data(), dtype=np.float64).reshape(RAW_SHAPE)
    return np.roll(arr, -arr.shape[1] // 2, axis=1)


def to_n320(arr2d):
    """Interpolate a (721,1440) 0.25deg field to the AIFS N320 grid (1-D)."""
    return ekr.interpolate(arr2d, {"grid": (0.25, 0.25)}, {"grid": "N320"})


# ---------------------------------------------------------------- index queries
def _match(rec, param, levtype, levelist=None, number=None):
    if rec.get("param") != param or rec.get("levtype") != levtype:
        return False
    if levelist is not None and str(rec.get("levelist")) != str(levelist):
        return False
    if number is not None and str(rec.get("number")) != str(number):
        return False
    return True


def _find(recs, **kw):
    """Exactly one matching message; raise if missing/ambiguous."""
    hits = [r for r in recs if _match(r, **kw)]
    if len(hits) != 1:
        raise LookupError(f"{len(hits)} msgs for {kw} (expected 1)")
    return hits[0]


def _fetch_decode_regrid(grib_key, rec):
    """Worker: byte-range fetch -> gribberish decode -> N320. Thread-safe."""
    raw = fetch_range(grib_key, rec["_offset"], rec["_length"])
    if raw[:4] != b"GRIB":
        raise ValueError(f"not GRIB at off={rec['_offset']} ({raw[:4]!r})")
    return to_n320(decode_grib_bytes(raw))


def _build_group(jobs):
    """Run a list of (out_name, grib_key, rec, timestep_idx) jobs in parallel.

    Returns {out_name: (2, 542080) float32}, stacking the two timesteps.
    """
    parts = defaultdict(lambda: [None, None])
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futs = {ex.submit(_fetch_decode_regrid, gk, rc): (nm, ti)
                for nm, gk, rc, ti in jobs}
        for fut in as_completed(futs):
            nm, ti = futs[fut]
            parts[nm][ti] = fut.result()
    out = {}
    for nm, (a, b) in parts.items():
        if a is None or b is None:
            raise RuntimeError(f"{nm}: missing a timestep")
        out[nm] = np.stack([a, b])  # (2, 542080) float64
    return out


# ---------------------------------------------------------------- group builders
def group_jobs(idx_prev, idx_t, gk_prev, gk_t, member, group):
    """Yield (out_name, grib_key, rec, timestep_idx) for one param group.

    timestep_idx 0 = t-6h, 1 = t. Surface/soil/pressure come from both enfo
    files; constants are handled separately (see build_constants).
    """
    jobs = []
    if group == "surface":
        for p in PARAM_SFC:
            jobs.append((p, gk_prev, _find(idx_prev, param=p, levtype="sfc",
                                           number=member), 0))
            jobs.append((p, gk_t, _find(idx_t, param=p, levtype="sfc",
                                        number=member), 1))
    elif group == "soil":
        for lv in SOIL_LEVELS:
            nm = f"sot_{lv}"
            jobs.append((nm, gk_prev, _find(idx_prev, param="sot",
                         levtype="sol", levelist=lv, number=member), 0))
            jobs.append((nm, gk_t, _find(idx_t, param="sot",
                         levtype="sol", levelist=lv, number=member), 1))
    elif group.startswith("pl_"):
        p = group[3:]
        for lv in LEVELS:
            nm = f"{p}_{lv}"
            jobs.append((nm, gk_prev, _find(idx_prev, param=p, levtype="pl",
                         levelist=lv, number=member), 0))
            jobs.append((nm, gk_t, _find(idx_t, param=p, levtype="pl",
                         levelist=lv, number=member), 1))
    return jobs


def finalize_group(group, fields):
    """Post-decode renames matching ecmwf_opendata_pkl_input_aifsens.py."""
    if group == "soil":
        return {SOIL_MAP[k]: v for k, v in fields.items()}
    if group == "pl_gh":   # geopotential height -> geopotential
        return {f"z_{k.split('_')[1]}": v * 9.80665 for k, v in fields.items()}
    return fields


def build_constants(idx_oper, gk_oper):
    """z/slor/sdor/lsm from the oper 0h file, replicated to both timesteps.

    The oper 0h file also carries 14 pressure-level `z` (50r1 addition); we
    select levtype='sfc' only, so the constant `z` is the single surface
    orography and the (30, N) z-stacking bug structurally cannot occur.
    """
    out = {}
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futs = {}
        for p in PARAM_SFC_CONST:
            rec = _find(idx_oper, param=p, levtype="sfc")
            futs[ex.submit(_fetch_decode_regrid, gk_oper, rec)] = p
        for fut in as_completed(futs):
            v = fut.result()
            out[futs[fut]] = np.stack([v, v])  # replicate to both timesteps
    return out


# ---------------------------------------------------------------- per member
def process_member(member, date, prev_date, out_dir, const_fields,
                    keep_partials=False):
    """Build one member's input state via partial-write + merge."""
    t0 = time.time()
    gk_t, ik_t = enfo_paths(date)
    gk_prev, ik_prev = enfo_paths(prev_date)
    idx_t, idx_prev = load_index(ik_t), load_index(ik_prev)

    pdir = os.path.join(out_dir, "partials", f"m{member:03d}")
    os.makedirs(pdir, exist_ok=True)

    groups = ["surface", "soil"] + [f"pl_{p}" for p in PARAM_PL]
    for g in groups:
        ppath = os.path.join(pdir, f"{g}.pkl")
        if os.path.exists(ppath):
            print(f"    [{g}] partial exists, skip")
            continue
        gt = time.time()
        fields = _build_group(group_jobs(idx_prev, idx_t, gk_prev, gk_t,
                                         member, g))
        fields = finalize_group(g, fields)
        with open(ppath, "wb") as fh:
            pickle.dump(fields, fh)
        n = len(fields)
        del fields
        gc.collect()
        print(f"    [{g}] {n} fields -> {os.path.basename(ppath)} "
              f"({time.time() - gt:.1f}s)")

    # constants partial (built from the shared in-memory const_fields)
    cpath = os.path.join(pdir, "const.pkl")
    if not os.path.exists(cpath):
        with open(cpath, "wb") as fh:
            pickle.dump(const_fields, fh)

    # merge partials -> single input state
    merged = {}
    for fn in sorted(os.listdir(pdir)):
        with open(os.path.join(pdir, fn), "rb") as fh:
            merged.update(pickle.load(fh))
    input_state = dict(date=date, fields=merged)

    out_pkl = os.path.join(out_dir, f"input_state_member_{member:03d}.pkl")
    with open(out_pkl, "wb") as fh:
        pickle.dump(input_state, fh)

    if not keep_partials:
        for fn in os.listdir(pdir):
            os.remove(os.path.join(pdir, fn))
        os.rmdir(pdir)

    sz = os.path.getsize(out_pkl) / 1e6
    print(f"  member {member}: {len(merged)} fields, {sz:.0f} MB, "
          f"{time.time() - t0:.1f}s -> {out_pkl}")
    return out_pkl


# ---------------------------------------------------------------- verification
def verify(out_pkl):
    """Same field-completeness check as ecmwf_opendata_pkl_input_aifsens.py."""
    with open(out_pkl, "rb") as fh:
        st = pickle.load(fh)
    fields = st["fields"]
    expected = PARAM_SFC + PARAM_SFC_CONST + ["stl1", "stl2"]
    for p in PARAM_PL:
        pn = "z" if p == "gh" else p
        expected += [f"{pn}_{lv}" for lv in LEVELS]
    missing = [f for f in expected if f not in fields]
    extra = [f for f in fields if f not in expected]
    shapes = {v.shape for v in fields.values()}
    dtypes = {str(v.dtype) for v in fields.values()}
    ok = (not missing and not extra and shapes == {(2, N320_NPOINTS)}
          and dtypes == {"float64"})
    print(f"  verify: {len(fields)}/{len(expected)} fields, shapes={shapes}, "
          f"dtypes={dtypes}, missing={missing}, extra={extra} "
          f"-> {'OK' if ok else 'FAIL'}")
    return ok


# ---------------------------------------------------------------- GCS upload
def upload_to_gcs(local_path, blob_name):
    client = storage.Client.from_service_account_json(GCS_SERVICE_ACCOUNT_KEY)
    client.bucket(GCS_BUCKET).blob(blob_name).upload_from_filename(local_path)
    print(f"    uploaded gs://{GCS_BUCKET}/{blob_name}")


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--date", default="20260514_0000",
                    help="analysis time YYYYMMDD_HHMM (default 20260514_0000)")
    ap.add_argument("--members", default="1-50",
                    help="e.g. '1', '1-50', '1,2,5' (default 1-50)")
    ap.add_argument("--out", default="s3_input_states")
    ap.add_argument("--gcs", action="store_true",
                    help="upload each pkl to GCS and remove local")
    ap.add_argument("--keep-partials", action="store_true")
    args = ap.parse_args()

    date = datetime.datetime.strptime(args.date, "%Y%m%d_%H%M")
    prev_date = date - datetime.timedelta(hours=6)

    members = []
    for tok in args.members.split(","):
        if "-" in tok:
            a, b = tok.split("-")
            members += list(range(int(a), int(b) + 1))
        else:
            members.append(int(tok))

    os.makedirs(args.out, exist_ok=True)
    datestr = date.strftime("%Y%m%d_%H%M")
    print(f"S3 GRIB -> pkl | t={date} t-6h={prev_date} | "
          f"{len(members)} members -> {args.out}/")

    # constants: fetched once at the analysis time, shared across all members
    gk_oper, ik_oper = oper_paths(date)
    print("  fetching constants (z/slor/sdor/lsm) from oper 0h ...")
    const_fields = build_constants(load_index(ik_oper), gk_oper)
    print(f"  constants: {sorted(const_fields)} "
          f"({list(const_fields.values())[0].shape})")

    ok, fail = [], []
    for i, m in enumerate(members):
        print(f"\n[{i + 1}/{len(members)}] member {m}")
        try:
            out_pkl = process_member(m, date, prev_date, args.out,
                                     const_fields, args.keep_partials)
            if verify(out_pkl):
                ok.append(m)
            else:
                fail.append(m)
            if args.gcs:
                upload_to_gcs(
                    out_pkl, f"{datestr}/input/input_state_member_{m:03d}.pkl")
                os.remove(out_pkl)
        except Exception as e:
            traceback.print_exc()
            print(f"  member {m} ERROR: {e}")
            fail.append(m)

    print(f"\n{'=' * 60}\nDONE  ok={len(ok)}  fail={len(fail)}")
    if fail:
        print(f"failed: {fail}")


if __name__ == "__main__":
    main()

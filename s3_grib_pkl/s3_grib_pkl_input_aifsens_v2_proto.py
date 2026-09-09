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
#     "eccodes",
# ]
# ///
"""
PROTOTYPE: v2.0 input-state pkl via direct S3 .index byte-range + gribberish,
instead of earthkit-data's ecmwf-open-data source (one-combined-multi-range-
request-per-file, serialized across param groups).

Adapts s3_grib_pkl/s3_grib_pkl_input_aifsens.py (v1, 92 fields) to the v2.0
field set (112 fields): adds sd, vsw->swvl1/2, the wave group (waef/wave
stream, mwd->cos/sin), level 10 hPa (drops q_10), lsm-masked sd/swvl1/swvl2.

Key difference from v1 and from the earthkit-data path: EVERY field/level/
timestep is its own independent single-range byte fetch, run in a
ThreadPoolExecutor (MAX_WORKERS concurrent). A throttled/reset field only
costs that one field, not a whole combined-range file request.
"""

import datetime
import json
import os
import time
import urllib.error
import urllib.request
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

import earthkit.regrid as ekr
import eccodes
import fsspec
import gribberish
import numpy as np
import obstore as obs
from obstore.store import from_url

# ---------------------------------------------------------------- v2.0 param lists
PARAM_SFC = ["10u", "10v", "2d", "2t", "msl", "skt", "sp", "tcw", "sd"]
PARAM_SFC_CONST = ["lsm", "z", "slor", "sdor"]
PARAM_SOIL = ["sot", "vsw"]
PARAM_WAVE = ["wmb", "h1012", "h1214", "h1417", "h1721",
              "h2125", "h2530", "mwd", "cdww", "mwp", "swh"]
PARAM_PL = ["gh", "t", "u", "v", "w", "q"]
LEVELS = [1000, 925, 850, 700, 600, 500, 400, 300, 250, 200, 150, 100, 50, 10]
SOIL_LEVELS = [1, 2]
SOIL_MAP = {"sot_1": "stl1", "sot_2": "stl2", "vsw_1": "swvl1", "vsw_2": "swvl2"}
G = 9.80665

N320_NPOINTS = 542080
RAW_SHAPE = (721, 1440)
MAX_WORKERS = int(os.environ.get("S3_PKL_MAX_WORKERS", "16"))

S3_BUCKET = "ecmwf-forecasts"
S3_REGION = "eu-central-1"
GCS_BASE = "https://storage.googleapis.com/ecmwf-open-data"

# Mirror to fetch from. Both serve byte-identical .grib2/.index objects (verified:
# index bytes, message bytes and decoded values all match), so the offsets from
# one mirror are valid against the other and only the transport differs.
#
# Note the earthkit-data/ecmwf-opendata path CANNOT use gcs: it merges many field
# ranges into one combined "Range: bytes=a-b,c-d,..." header, and GCS rejects that
# with 400 InvalidArgument ("Multiple ranges are not supported"). This script issues
# one single-range request per field, which GCS serves normally (HTTP 206).
SOURCE = "aws"

# ---------------------------------------------------------------- fetch layer
_obstore_cache = {}


def _get_store():
    if "s" not in _obstore_cache:
        _obstore_cache["s"] = from_url(f"s3://{S3_BUCKET}", region=S3_REGION, skip_signature=True)
    return _obstore_cache["s"]


def _http_get(url, offset=None, length=None, tries=8):
    """Single-range (or whole-object) HTTP GET with backoff on 503 SlowDown."""
    last = None
    for i in range(tries):
        try:
            req = urllib.request.Request(url)
            if offset is not None:
                req.add_header("Range", f"bytes={offset}-{offset + length - 1}")
            with urllib.request.urlopen(req, timeout=300) as r:
                return r.read()
        except Exception as e:                      # 503 SlowDown, resets, timeouts
            last = e
            time.sleep(min(2 ** i, 60))
    raise RuntimeError(f"GET failed after {tries} tries: {url} ({last})")


def fetch_range(key, offset, length):
    if SOURCE == "gcs":
        return _http_get(f"{GCS_BASE}/{key}", offset, length)
    try:
        return bytes(obs.get_range(_get_store(), key, start=offset, end=offset + length))
    except Exception:
        fs = fsspec.filesystem("s3", anon=True)
        return fs.read_block(f"{S3_BUCKET}/{key}", offset, length)


def load_index(key):
    if key in _obstore_cache:
        return _obstore_cache[key]
    if SOURCE == "gcs":
        raw = _http_get(f"{GCS_BASE}/{key}").decode()
    else:
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


def waef_paths(dt):
    ymd, hh, st = _stamp(dt)
    b = f"{ymd}/{hh}z/ifs/0p25/waef/{st}-0h-waef-ef"
    return b + ".grib2", b + ".index"


def oper_paths(dt):
    ymd, hh, st = _stamp(dt)
    b = f"{ymd}/{hh}z/ifs/0p25/oper/{st}-0h-oper-fc"
    return b + ".grib2", b + ".index"


# ---------------------------------------------------------------- decode + grid
def decode_grib_bytes(raw):
    """gribberish: fast, but only supports HorizontalAnalysisForecast PDTs."""
    msg = gribberish.parse_grib_message(raw, 0)
    arr = np.asarray(msg.data(), dtype=np.float64).reshape(RAW_SHAPE)
    return np.roll(arr, -arr.shape[1] // 2, axis=1)


def decode_grib_bytes_eccodes(raw):
    """eccodes fallback for PDTs gribberish rejects (e.g. wave-group templates).

    Wave params are ocean-only; land points carry the GRIB missing-value
    sentinel (9999). codes_get_values() returns it raw unless the
    missingValue key is set to NaN first - without this, land points come
    back as the literal value 9999 instead of NaN (silently wrong, not an
    error). Confirmed against a production pkl: without this line, all 12
    wave fields had 9999 at ~30% of points (land) where production had NaN.
    """
    gid = eccodes.codes_new_from_message(raw)
    try:
        eccodes.codes_set(gid, "missingValue", np.nan)
        arr = np.asarray(eccodes.codes_get_values(gid), dtype=np.float64).reshape(RAW_SHAPE)
    finally:
        eccodes.codes_release(gid)
    return np.roll(arr, -arr.shape[1] // 2, axis=1)


def to_n320(arr2d):
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
    hits = [r for r in recs if _match(r, **kw)]
    if len(hits) != 1:
        raise LookupError(f"{len(hits)} msgs for {kw} (expected 1)")
    return hits[0]


def _fetch_decode_regrid(grib_key, rec, decoder=decode_grib_bytes):
    raw = fetch_range(grib_key, rec["_offset"], rec["_length"])
    if raw[:4] != b"GRIB":
        raise ValueError(f"not GRIB at off={rec['_offset']} ({raw[:4]!r})")
    return to_n320(decoder(raw))


def _build_group(jobs, decoder=decode_grib_bytes):
    parts = defaultdict(lambda: [None, None])
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futs = {ex.submit(_fetch_decode_regrid, gk, rc, decoder): (nm, ti) for nm, gk, rc, ti in jobs}
        for fut in as_completed(futs):
            nm, ti = futs[fut]
            parts[nm][ti] = fut.result()
    out = {}
    for nm, (a, b) in parts.items():
        if a is None or b is None:
            raise RuntimeError(f"{nm}: missing a timestep")
        out[nm] = np.stack([a, b])
    return out


# ---------------------------------------------------------------- group builders
def group_jobs(idx_prev, idx_t, gk_prev, gk_t, member, group, wave_type_prev, wave_type_t):
    jobs = []
    if group == "surface":
        for p in PARAM_SFC:
            jobs.append((p, gk_prev, _find(idx_prev, param=p, levtype="sfc", number=member), 0))
            jobs.append((p, gk_t, _find(idx_t, param=p, levtype="sfc", number=member), 1))
    elif group == "soil":
        for p in PARAM_SOIL:
            for lv in SOIL_LEVELS:
                nm = f"{p}_{lv}"
                jobs.append((nm, gk_prev, _find(idx_prev, param=p, levtype="sol", levelist=lv, number=member), 0))
                jobs.append((nm, gk_t, _find(idx_t, param=p, levtype="sol", levelist=lv, number=member), 1))
    elif group == "wave":
        for p in PARAM_WAVE:
            jobs.append((p, gk_prev, _find(idx_prev, param=p, levtype="sfc", number=member), 0))
            jobs.append((p, gk_t, _find(idx_t, param=p, levtype="sfc", number=member), 1))
    elif group.startswith("pl_"):
        p = group[3:]
        for lv in LEVELS:
            if p == "q" and lv == 10:      # v2 drops q_10
                continue
            nm = f"{p}_{lv}"
            jobs.append((nm, gk_prev, _find(idx_prev, param=p, levtype="pl", levelist=lv, number=member), 0))
            jobs.append((nm, gk_t, _find(idx_t, param=p, levtype="pl", levelist=lv, number=member), 1))
    return jobs


def finalize_group(group, fields):
    if group == "soil":
        return {SOIL_MAP[k]: v for k, v in fields.items()}
    if group == "pl_gh":
        return {f"z_{k.split('_')[1]}": v * G for k, v in fields.items()}
    if group == "wave":
        mwd_rad = np.deg2rad(fields.pop("mwd"))
        fields["cos_mwd"] = np.cos(mwd_rad)
        fields["sin_mwd"] = np.sin(mwd_rad)
    return fields


def build_constants(idx_oper, gk_oper):
    out = {}
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futs = {}
        for p in PARAM_SFC_CONST:
            rec = _find(idx_oper, param=p, levtype="sfc")
            futs[ex.submit(_fetch_decode_regrid, gk_oper, rec)] = p
        for fut in as_completed(futs):
            v = fut.result()
            out[futs[fut]] = np.stack([v, v])
    return out


# ---------------------------------------------------------------- per member
def process_member(member, date, prev_date, const_fields=None):
    """Build one member. `const_fields` (z/slor/sdor/lsm) are member-independent -
    pass them in to reuse across members instead of re-fetching per member."""
    t0 = time.time()
    gk_t, ik_t = enfo_paths(date)
    gk_prev, ik_prev = enfo_paths(prev_date)
    idx_t, idx_prev = load_index(ik_t), load_index(ik_prev)

    wgk_t, wik_t = waef_paths(date)
    wgk_prev, wik_prev = waef_paths(prev_date)
    widx_t, widx_prev = load_index(wik_t), load_index(wik_prev)

    if const_fields is None:
        gk_oper, ik_oper = oper_paths(date)
        print("  constants (z/slor/sdor/lsm) from oper 0h ...")
        const_fields = build_constants(load_index(ik_oper), gk_oper)
        print(f"    {sorted(const_fields)} ({time.time()-t0:.1f}s elapsed)")

    # copy: the lsm mask write below must not mutate the shared constants
    merged = {k: v.copy() for k, v in const_fields.items()}
    groups = ["surface", "soil", "wave"] + [f"pl_{p}" for p in PARAM_PL]
    for g in groups:
        gt = time.time()
        if g == "wave":
            fields = _build_group(
                group_jobs(widx_prev, widx_t, wgk_prev, wgk_t, member, g, None, None),
                decoder=decode_grib_bytes_eccodes)
        else:
            fields = _build_group(group_jobs(idx_prev, idx_t, gk_prev, gk_t, member, g, None, None))
        fields = finalize_group(g, fields)
        merged.update(fields)
        print(f"    [{g}] {len(fields)} fields ({time.time()-gt:.1f}s)")

    # land-sea mask: NaN over sea for sd/swvl1/swvl2
    mask = np.equal(merged["lsm"][0], 0)
    for f in ("sd", "swvl1", "swvl2"):
        merged[f][:, mask] = np.nan

    print(f"  TOTAL member {member}: {len(merged)} fields, {time.time()-t0:.1f}s")
    return dict(date=date, fields=merged), time.time() - t0


# ---------------------------------------------------------------- verification
def expected_field_names():
    names = list(PARAM_SFC) + list(PARAM_SFC_CONST) + ["stl1", "stl2", "swvl1", "swvl2"]
    names += [w for w in PARAM_WAVE if w != "mwd"] + ["cos_mwd", "sin_mwd"]
    for p in PARAM_PL:
        out = "z" if p == "gh" else p
        for lev in LEVELS:
            if out == "q" and lev == 10:
                continue
            names.append(f"{out}_{lev}")
    return names


def verify(input_state):
    fields = input_state["fields"]
    expected = expected_field_names()
    missing = [f for f in expected if f not in fields]
    extra = [f for f in fields if f not in expected]
    shapes = {v.shape for v in fields.values()}
    print(f"  verify: {len(fields)}/{len(expected)} fields, shapes={shapes}, "
          f"missing={missing}, extra={extra} -> {'OK' if not missing and not extra else 'FAIL'}")
    return not missing and not extra


def parse_members(spec):
    """'1-50', '1,5,10' or '3' -> sorted list of ints."""
    out = set()
    for part in str(spec).split(","):
        part = part.strip()
        if "-" in part:
            a, b = part.split("-")
            out.update(range(int(a), int(b) + 1))
        elif part:
            out.add(int(part))
    return sorted(out)


if __name__ == "__main__":
    import argparse
    import gc
    import pickle

    ap = argparse.ArgumentParser()
    ap.add_argument("--date", default="20260806_0000")
    ap.add_argument("--member", type=int, help="single member (back-compat)")
    ap.add_argument("--members", help="range/list, e.g. '1-50', '1,5,10'")
    ap.add_argument("--out", default=".")
    ap.add_argument("--source", default="aws", choices=["aws", "gcs"],
                    help="mirror to byte-range fetch from (default aws; gcs "
                         "measured ~10x faster on this box, byte-identical data)")
    ap.add_argument("--suffix", default="",
                    help="appended to the output pkl name, e.g. --suffix _gcs")
    ap.add_argument("--skip-existing", action="store_true",
                    help="skip members whose output pkl already exists")
    args = ap.parse_args()

    SOURCE = args.source

    if args.members:
        members = parse_members(args.members)
    elif args.member is not None:
        members = [args.member]
    else:
        members = [25]

    date = datetime.datetime.strptime(args.date, "%Y%m%d_%H%M")
    prev_date = date - datetime.timedelta(hours=6)

    print(f"PROTOTYPE gribberish v2 | members {members[0]}-{members[-1]} "
          f"({len(members)}) | source={args.source} | t={date} t-6h={prev_date}")

    # indexes + constants are member-independent: fetch once, reuse for all members
    gk_oper, ik_oper = oper_paths(date)
    t_const = time.time()
    print("constants (z/slor/sdor/lsm) from oper 0h, shared across members ...")
    const_fields = build_constants(load_index(ik_oper), gk_oper)
    print(f"  {sorted(const_fields)} ({time.time()-t_const:.1f}s)")

    ok, fail, times = [], [], []
    t_all = time.time()
    for i, m in enumerate(members):
        out_pkl = os.path.join(
            args.out, f"proto_input_state_member_{m:03d}{args.suffix}.pkl")
        if args.skip_existing and os.path.exists(out_pkl):
            print(f"\n[{i+1}/{len(members)}] member {m}: exists, skip")
            ok.append(m)
            continue
        print(f"\n[{i+1}/{len(members)}] member {m}")
        try:
            state, elapsed = process_member(m, date, prev_date, const_fields)
            good = verify(state)
            with open(out_pkl, "wb") as fh:
                pickle.dump(state, fh)
            del state
            gc.collect()
            (ok if good else fail).append(m)
            times.append(elapsed)
            avg = sum(times) / len(times)
            print(f"  {elapsed:.1f}s | avg {avg:.1f}s | "
                  f"ETA {(len(members)-i-1)*avg/60:.1f} min -> {out_pkl}")
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"  member {m} FAILED: {e}")
            fail.append(m)

    dt = time.time() - t_all
    print(f"\n{'='*60}\nDONE source={args.source}: {len(ok)}/{len(members)} ok"
          + (f", failed {fail}" if fail else ""))
    print(f"  total {dt/60:.1f} min ({dt/3600:.2f} h)"
          + (f", avg {sum(times)/len(times):.1f}s/member" if times else ""))

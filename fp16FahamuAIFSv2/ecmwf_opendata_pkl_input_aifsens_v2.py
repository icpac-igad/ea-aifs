#!/usr/bin/env python3
"""
AIFS ENS v2.0 — input-state (pkl) creation from ECMWF Open Data (CPU only).

Builds per-member ``input_state`` pickle files for the **AIFS ENS v2.0** model
(checkpoint ``ecmwf/aifs-ens-2.0``) and uploads them to GCS, mirroring the data
preparation in the official notebook ``run_AIFS_ENS_v2.0.ipynb``.

What is different from the v1 prep (`shared/ecmwf_opendata_pkl_input_aifsens.py`):
  * Surface adds ``sd`` (snow depth).
  * Soil adds ``vsw`` (soil moisture) → ``swvl1/swvl2`` alongside ``stl1/stl2``.
  * A whole new **wave** group (11 params from the ``waef``/``wave`` stream),
    with ``mwd`` decomposed into ``cos_mwd``/``sin_mwd``.
  * Pressure levels gain **level 10 hPa** (14 levels), and ``q_10`` is dropped.
  * A land–sea mask sets ``sd``, ``swvl1``, ``swvl2`` to NaN over sea.
  * ``gh → z`` (×g) is kept, same as v1.

No GPU and no model checkpoint are required — this is data retrieval + transforms
only. The resulting pkl (~112 fields, 2 timesteps each) feeds the v2 GPU runner.

Two retrieval backends, selected with ``--fetch`` (both produce bit-identical
pkls — verified field-by-field on 5 members across 2 dates and both mirrors):

  * ``index`` (default) — read the ``.index`` sidecar, then fetch ONE single
    byte range per field, ``INDEX_MAX_WORKERS`` at a time. Decodes with eccodes.
    Needs no earthkit-data and no ecmwf-opendata. Works against the **google**
    mirror, which is much faster from this box.
    **50 members: 2.9 h (20260806) vs 14.5 h (20260730).**
  * ``earthkit`` — the original ``earthkit.data.from_source("ecmwf-open-data")``
    path, kept as a fallback. ecmwf-opendata merges a param group's byte ranges
    into one combined multi-range request and runs the groups serially, so a
    single 503 costs the whole request plus a 120 s backoff. aws/ecmwf only.
"""

import argparse
import datetime
import json
import os
import pickle
import time
import urllib.request
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

import eccodes
import numpy as np
import earthkit.regrid as ekr
from google.cloud import storage

# ecmwf-opendata mirrors (ecmwf/opendata/urls.py URLS). Only `aws` and `ecmwf`
# actually work through THIS code path:
#   aws    - works; slow here (1-3 Mbps, frequent 503 SlowDown)
#   ecmwf  - works; the direct portal, capped at 500 simultaneous connections
#   google - 400 InvalidArgument "Multiple ranges are not supported". Not a
#            mirror fault: earthkit-data/ecmwf-opendata merges many field byte
#            ranges into ONE combined "Range: bytes=a-b,c-d,..." header, which
#            GCS rejects and S3 accepts. Verified 2026-08-06 that GCS serves the
#            SAME bytes fine (HTTP 206) when asked one range at a time - see
#            s3_grib_pkl/s3_grib_pkl_input_aifsens_v2_proto.py --source gcs,
#            which is ~12x faster than aws on this box.
#   azure  - broken (HTTP 409)
OPENDATA_SOURCES = ["ecmwf", "aws", "google", "azure", "ecmwf-esuites"]

# ---------------------------------------------------------------------------
# v2.0 parameter lists (from run_AIFS_ENS_v2.0.ipynb)
# ---------------------------------------------------------------------------
PARAM_SFC = ["10u", "10v", "2d", "2t", "msl", "skt", "sp", "tcw", "sd"]
PARAM_SFC_FC = ["lsm", "z", "slor", "sdor"]              # time-invariant constants
PARAM_SOIL = ["vsw", "sot"]                               # → swvl1/2, stl1/2
PARAM_WAVE = ["wmb", "h1012", "h1214", "h1417", "h1721",
              "h2125", "h2530", "mwd", "cdww", "mwp", "swh"]
PARAM_PL = ["gh", "t", "u", "v", "w", "q"]
LEVELS = [1000, 925, 850, 700, 600, 500, 400, 300, 250, 200, 150, 100, 50, 10]
SOIL_LEVELS = [1, 2]

SOIL_RENAME = {"sot_1": "stl1", "sot_2": "stl2",
               "vsw_1": "swvl1", "vsw_2": "swvl2"}

# GCS configuration (same bucket/key conventions as the v1 prep)
GCS_BUCKET = "aifs-aiquest-us-20251127"
GCS_SERVICE_ACCOUNT_KEY = "coiled-data.json"
GCS_SUBPATH = "input_v2"          # keep v2 pkls separate from v1 ("input/")
G = 9.80665                       # standard gravity, for gh → z

# ===========================================================================
# fetch="index" backend: one single byte range per field, fetched concurrently
# ===========================================================================
# The `earthkit` backend below asks ecmwf-opendata for a whole param group at
# once; that library merges every field's byte range into ONE combined
#     Range: bytes=a-b,c-d,e-f...
# request per file, and runs the 5 param groups serially. One 503 loses the
# whole combined request and costs a 120 s backoff, which is what made Step 1
# take 14.5 h on 20260730.
#
# This backend reads the same .index sidecar, then issues ONE single-range
# request per field, INDEX_MAX_WORKERS at a time. A throttled field costs only
# itself. It also unlocks the Google mirror: GCS rejects multi-range GETs
# (400 InvalidArgument) but serves single-range ones normally, and is far
# faster from this box. Measured 20260806, 50 members: 2.9 h vs 14.5 h.
#
# Verified bit-identical to the earthkit backend (112/112 fields,
# max_abs_diff=0, NaN masks matched) on 5 members across 2 dates, both mirrors.
INDEX_BASES = {
    "aws": "https://ecmwf-forecasts.s3.eu-central-1.amazonaws.com",
    "google": "https://storage.googleapis.com/ecmwf-open-data",
}
INDEX_MAX_WORKERS = int(os.environ.get("INDEX_MAX_WORKERS", "16"))
RAW_SHAPE = (721, 1440)           # ECMWF 0p25, lat 90->-90, lon0=180
_index_cache = {}


def _http_get(base, key, offset=None, length=None, tries=8):
    """Single-range (or whole-object) GET with backoff. 503 SlowDown is normal."""
    url = f"{base}/{key}"
    last = None
    for i in range(tries):
        try:
            req = urllib.request.Request(url)
            if offset is not None:
                req.add_header("Range", f"bytes={offset}-{offset + length - 1}")
            with urllib.request.urlopen(req, timeout=300) as r:
                return r.read()
        except Exception as e:                  # 503 SlowDown, resets, timeouts
            last = e
            time.sleep(min(2 ** i, 60))
    raise RuntimeError(f"GET failed after {tries} tries: {url} ({last})")


def load_index(base, key):
    """Parse the JSON-lines .index sidecar (small; cached per key)."""
    if key in _index_cache:
        return _index_cache[key]
    raw = _http_get(base, key).decode()
    recs = [json.loads(ln) for ln in raw.splitlines() if ln.strip()]
    _index_cache[key] = recs
    return recs


def _stamp(dt):
    return dt.strftime("%Y%m%d"), dt.strftime("%H"), dt.strftime("%Y%m%d%H%M%S")


def _paths(dt, stream, kind):
    ymd, hh, st = _stamp(dt)
    b = f"{ymd}/{hh}z/ifs/0p25/{stream}/{st}-0h-{stream}-{kind}"
    return b + ".grib2", b + ".index"


def enfo_paths(dt):
    return _paths(dt, "enfo", "ef")


def waef_paths(dt):
    return _paths(dt, "waef", "ef")


def oper_paths(dt):
    return _paths(dt, "oper", "fc")


def decode_grib_message(raw):
    """Decode one GRIB2 message -> (721,1440) float64, rolled to lon0=0.

    eccodes handles every product template in these files, including the wave
    group's PDT 104 (which gribberish rejects), and is bitwise-identical to
    gribberish on the templates both support - while being faster (9.4 vs
    15.1 ms/msg measured), so there is one decode path, not two.

    `missingValue -> NaN` MUST be set before reading values: wave params are
    ocean-only and land points carry the sentinel 9999. Without this they come
    back as the literal 9999 over ~30% of the grid - silently wrong, no error.
    """
    gid = eccodes.codes_new_from_message(raw)
    try:
        eccodes.codes_set(gid, "missingValue", np.nan)
        arr = np.asarray(eccodes.codes_get_values(gid),
                         dtype=np.float64).reshape(RAW_SHAPE)
    finally:
        eccodes.codes_release(gid)
    return np.roll(arr, -arr.shape[1] // 2, axis=1)


def _to_n320(arr2d):
    return ekr.interpolate(arr2d, {"grid": (0.25, 0.25)}, {"grid": "N320"})


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


def _fetch_decode_regrid(base, grib_key, rec):
    raw = _http_get(base, grib_key, rec["_offset"], rec["_length"])
    if raw[:4] != b"GRIB":
        raise ValueError(f"not GRIB at off={rec['_offset']} ({raw[:4]!r})")
    return _to_n320(decode_grib_message(raw))


def _build_group(base, jobs):
    """Run (name, grib_key, rec, timestep_idx) jobs concurrently -> stacked dict."""
    parts = defaultdict(lambda: [None, None])
    with ThreadPoolExecutor(max_workers=INDEX_MAX_WORKERS) as ex:
        futs = {ex.submit(_fetch_decode_regrid, base, gk, rc): (nm, ti)
                for nm, gk, rc, ti in jobs}
        for fut in as_completed(futs):
            nm, ti = futs[fut]
            parts[nm][ti] = fut.result()
    out = {}
    for nm, (a, b) in parts.items():
        if a is None or b is None:
            raise RuntimeError(f"{nm}: missing a timestep")
        out[nm] = np.stack([a, b])
    return out


def _group_jobs(idx_prev, idx_t, gk_prev, gk_t, member, group):
    """timestep_idx 0 = t-6h, 1 = t."""
    jobs = []
    if group == "surface":
        for p in PARAM_SFC:
            jobs.append((p, gk_prev, _find(idx_prev, param=p, levtype="sfc",
                                           number=member), 0))
            jobs.append((p, gk_t, _find(idx_t, param=p, levtype="sfc",
                                        number=member), 1))
    elif group == "soil":
        for p in PARAM_SOIL:
            for lv in SOIL_LEVELS:
                nm = f"{p}_{lv}"
                jobs.append((nm, gk_prev, _find(idx_prev, param=p, levtype="sol",
                                                levelist=lv, number=member), 0))
                jobs.append((nm, gk_t, _find(idx_t, param=p, levtype="sol",
                                             levelist=lv, number=member), 1))
    elif group == "wave":
        for p in PARAM_WAVE:
            jobs.append((p, gk_prev, _find(idx_prev, param=p, levtype="sfc",
                                           number=member), 0))
            jobs.append((p, gk_t, _find(idx_t, param=p, levtype="sfc",
                                        number=member), 1))
    elif group.startswith("pl_"):
        p = group[3:]
        for lv in LEVELS:
            if p == "q" and lv == 10:            # v2 drops q_10
                continue
            nm = f"{p}_{lv}"
            jobs.append((nm, gk_prev, _find(idx_prev, param=p, levtype="pl",
                                            levelist=lv, number=member), 0))
            jobs.append((nm, gk_t, _find(idx_t, param=p, levtype="pl",
                                         levelist=lv, number=member), 1))
    return jobs


def build_constants(base, date):
    """lsm/z/slor/sdor from the oper 0h file, replicated to both timesteps.

    Member-independent: build once and pass to every member.
    """
    gk_oper, ik_oper = oper_paths(date)
    idx_oper = load_index(base, ik_oper)
    out = {}
    with ThreadPoolExecutor(max_workers=INDEX_MAX_WORKERS) as ex:
        futs = {}
        for p in PARAM_SFC_FC:
            rec = _find(idx_oper, param=p, levtype="sfc")
            futs[ex.submit(_fetch_decode_regrid, base, gk_oper, rec)] = p
        for fut in as_completed(futs):
            out[futs[fut]] = np.stack([fut.result()] * 2)
    return out


def create_input_state_index(date, number, base, const_fields):
    """Build one member's v2.0 input_state via the .index byte-range backend.

    Produces exactly the same dict as create_input_state() - verified
    bit-identical field-by-field.
    """
    print(f"\nCreating v2.0 input state for ensemble member {number}")
    t0 = time.time()
    prev_date = date - datetime.timedelta(hours=6)

    gk_t, ik_t = enfo_paths(date)
    gk_prev, ik_prev = enfo_paths(prev_date)
    idx_t, idx_prev = load_index(base, ik_t), load_index(base, ik_prev)

    wgk_t, wik_t = waef_paths(date)
    wgk_prev, wik_prev = waef_paths(prev_date)
    widx_t, widx_prev = load_index(base, wik_t), load_index(base, wik_prev)

    # copy: the lsm mask write below must not mutate the shared constants
    fields = {k: v.copy() for k, v in const_fields.items()}

    for g in ["surface", "soil", "wave"] + [f"pl_{p}" for p in PARAM_PL]:
        gt = time.time()
        if g == "wave":
            got = _build_group(base, _group_jobs(widx_prev, widx_t, wgk_prev,
                                                 wgk_t, number, g))
            # mwd -> cos/sin components
            mwd_rad = np.deg2rad(got.pop("mwd"))
            got["cos_mwd"] = np.cos(mwd_rad)
            got["sin_mwd"] = np.sin(mwd_rad)
        else:
            got = _build_group(base, _group_jobs(idx_prev, idx_t, gk_prev,
                                                 gk_t, number, g))
            if g == "soil":
                got = {SOIL_RENAME[k]: v for k, v in got.items()}
            elif g == "pl_gh":                   # geopotential height -> geopotential
                got = {f"z_{k.split('_')[1]}": v * G for k, v in got.items()}
        fields.update(got)
        print(f"    [{g}] {len(got)} fields ({time.time()-gt:.1f}s)")

    # land-sea mask: NaN over sea for snow depth + soil moisture
    mask = np.equal(fields["lsm"][0], 0)
    for f in ("sd", "swvl1", "swvl2"):
        fields[f][:, mask] = np.nan

    input_state = dict(date=date, fields=fields)
    print(f"  done in {time.time()-t0:.1f}s — {len(fields)} fields, "
          f"shape {next(iter(fields.values())).shape}")
    return input_state


# ===========================================================================
# fetch="earthkit" backend: the original path, kept as a fallback
# ===========================================================================
def get_open_data(date, param, levelist=[], number=None, constant=False,
                  source="aws", **kwargs):
    """Retrieve + N320-interpolate ECMWF Open Data for a (param, level) group.

    Returns a dict ``name -> ndarray`` stacked over [date-6h, date] (2 timesteps).

    ``constant=True``: time-invariant fields (lsm, z, slor, sdor). The deterministic
    stream only runs 00z/12z, so a ``date-6h`` step landing on 06z/18z 404s — fetch
    once at ``date`` and replicate to the 2-timestep shape used by prognostic fields.

    ``number`` selects an ensemble perturbation; ``stream`` (via kwargs) picks the
    archive (``enfo`` for atmosphere, ``waef`` for wave ensemble).
    """
    import earthkit.data as ekd          # only this backend needs earthkit-data

    fields = defaultdict(list)
    dates = [date] if constant else [date - datetime.timedelta(hours=6), date]
    for d in dates:
        if number is None:
            data = ekd.from_source("ecmwf-open-data", date=d, param=param,
                                   levelist=levelist, source=source, **kwargs)
        else:
            kwargs.setdefault("stream", "enfo")
            data = ekd.from_source("ecmwf-open-data", date=d, param=param,
                                   levelist=levelist, number=[number],
                                   source=source, **kwargs)

        for f in data:
            # 50r1 added pressure-level `z` to the deterministic stream; when no
            # level list was requested keep only single-level fields, else the
            # constant `z` (orography) collapses into a (2*Nlevels,...) array.
            if not levelist and f.metadata("levtype") == "pl":
                continue
            # Open data is -180..180; shift to 0..360, then interpolate 0.25 -> N320
            assert f.to_numpy().shape == (721, 1440), \
                f"unexpected grid {f.to_numpy().shape} for {f.metadata('param')}"
            values = np.roll(f.to_numpy(), -f.shape[1] // 2, axis=1)
            values = ekr.interpolate(values, {"grid": (0.25, 0.25)}, {"grid": "N320"})
            name = f"{f.metadata('param')}_{f.metadata('levelist')}" if levelist \
                else f.metadata("param")
            fields[name].append(values)
            if constant:                       # fetched once -> replicate 2nd step
                fields[name].append(values)

    for name, values in fields.items():
        fields[name] = np.stack(values)
    return fields


def create_input_state(date, number, source="aws"):
    """Build the v2.0 input_state dict for one ensemble member."""
    print(f"\nCreating v2.0 input state for ensemble member {number}")
    t0 = time.time()
    fields = {}

    print("  surface fields...")
    fields.update(get_open_data(date, param=PARAM_SFC, number=number,
                                levtype="sfc", source=source))

    print("  constant surface fields...")
    fields.update(get_open_data(date, param=PARAM_SFC_FC, constant=True,
                                levtype="sfc", source=source))

    print("  wave fields...")
    wave_stream = "wave" if not number else "waef"
    fields.update(get_open_data(date, param=PARAM_WAVE, number=number,
                                stream=wave_stream, source=source))

    print("  soil fields...")
    soil = get_open_data(date, param=PARAM_SOIL, levelist=SOIL_LEVELS,
                         number=number, source=source)
    for k, v in soil.items():
        fields[SOIL_RENAME[k]] = v

    print("  pressure-level fields...")
    fields.update(get_open_data(date, param=PARAM_PL, levelist=LEVELS,
                                number=number, source=source))

    # ----- v2.0 data transformations -----
    # mwd -> cos/sin components
    mwd_rad = np.deg2rad(fields.pop("mwd"))
    fields["cos_mwd"] = np.cos(mwd_rad)
    fields["sin_mwd"] = np.sin(mwd_rad)

    # drop unused 10 hPa specific humidity
    fields.pop("q_10", None)

    # land-sea mask: NaN over sea for snow depth + soil moisture.
    # Derive the mask from the (already N320) lsm field rather than a side file.
    mask = np.equal(fields["lsm"][0], 0)
    for f in ("sd", "swvl1", "swvl2"):
        fields[f][:, mask] = np.nan

    # geopotential height -> geopotential
    for level in LEVELS:
        fields[f"z_{level}"] = fields.pop(f"gh_{level}") * G

    input_state = dict(date=date, fields=fields)
    print(f"  done in {time.time()-t0:.1f}s — {len(fields)} fields, "
          f"shape {next(iter(fields.values())).shape}")
    return input_state


def expected_field_names():
    """The full set of v2.0 field names, for verification."""
    names = list(PARAM_SFC) + list(PARAM_SFC_FC) + ["stl1", "stl2", "swvl1", "swvl2"]
    names += [w for w in PARAM_WAVE if w != "mwd"] + ["cos_mwd", "sin_mwd"]
    for p in PARAM_PL:
        out = "z" if p == "gh" else p
        for lev in LEVELS:
            if out == "q" and lev == 10:        # q_10 dropped
                continue
            names.append(f"{out}_{lev}")
    return names


def verify_input_state(input_state, member):
    fields = input_state["fields"]
    expected = expected_field_names()
    missing = [f for f in expected if f not in fields]
    extra = [f for f in fields if f not in expected]
    print(f"  verify member {member}: {len(fields)} fields "
          f"(expected {len(expected)})")
    if missing:
        print(f"    ⚠️  missing: {missing}")
    if extra:
        print(f"    ⚠️  extra:   {extra}")
    ok = not missing and not extra
    print("    ✅ all fields present" if ok else "    ❌ field mismatch")
    return ok


def upload_to_gcs(local_path, bucket, blob_name, key):
    try:
        client = storage.Client.from_service_account_json(key)
        client.bucket(bucket).blob(blob_name).upload_from_filename(local_path)
        print(f"    ✅ uploaded gs://{bucket}/{blob_name}")
        return True
    except Exception as e:
        print(f"    ❌ GCS upload failed: {e}")
        return False


def parse_members(spec):
    """Parse '1-50', '1,5,10' or '3' into a sorted list of ints."""
    out = set()
    for part in str(spec).split(","):
        part = part.strip()
        if "-" in part:
            a, b = part.split("-")
            out.update(range(int(a), int(b) + 1))
        elif part:
            out.add(int(part))
    return sorted(out)


def main():
    ap = argparse.ArgumentParser(description="AIFS ENS v2.0 input-state pkl creator")
    ap.add_argument("--date", help="init date YYYYMMDD or YYYYMMDD_HHMM "
                                    "(default: latest open-data)")
    ap.add_argument("--members", default="1-50",
                    help="ensemble members, e.g. '1-50', '1,5,10', '3' (default 1-50)")
    ap.add_argument("--gcs-subpath", default=GCS_SUBPATH,
                    help=f"GCS subfolder under the date (default {GCS_SUBPATH})")
    ap.add_argument("--fetch", default="index", choices=["index", "earthkit"],
                    help="how fields are retrieved. 'index' (default): one "
                         "single byte range per field, concurrent - ~5x faster "
                         "end-to-end and works against the google mirror. "
                         "'earthkit': the original earthkit-data/ecmwf-opendata "
                         "path (combined multi-range requests, aws/ecmwf only)")
    ap.add_argument("--source", default=None, choices=OPENDATA_SOURCES,
                    help="mirror. Default: google for --fetch index (fastest), "
                         "aws for --fetch earthkit. --fetch index supports "
                         "aws|google only; --fetch earthkit cannot use google "
                         "(multi-range 400) - see OPENDATA_SOURCES")
    ap.add_argument("--no-upload", action="store_true", help="skip GCS upload")
    ap.add_argument("--keep-local", action="store_true",
                    help="keep local pkl files after upload")
    ap.add_argument("--out-dir", default="input_states_v2",
                    help="local output dir (default input_states_v2)")
    args = ap.parse_args()

    if not args.date:
        print("ERROR: --date is required (latest date detection removed)")
        return 1

    source = args.source or ("google" if args.fetch == "index" else "aws")
    if args.fetch == "index" and source not in INDEX_BASES:
        print(f"ERROR: --fetch index supports {sorted(INDEX_BASES)}, got {source!r}")
        return 1
    if args.fetch == "earthkit" and source == "google":
        print("ERROR: --fetch earthkit cannot use the google mirror "
              "(ecmwf-opendata sends combined multi-range GETs; GCS rejects "
              "them with 400 InvalidArgument). Use --fetch index instead.")
        return 1

    d = args.date.replace("_", "")
    DATE = datetime.datetime.strptime(d[:8] + (d[8:12] or "0000"), "%Y%m%d%H%M")
    datestr = DATE.strftime("%Y%m%d_%H%M")
    members = parse_members(args.members)
    upload = not args.no_upload
    if upload and not os.path.exists(GCS_SERVICE_ACCOUNT_KEY):
        print(f"⚠️  {GCS_SERVICE_ACCOUNT_KEY} not found — disabling GCS upload")
        upload = False

    os.makedirs(args.out_dir, exist_ok=True)
    print(f"AIFS ENS v2.0 input prep (fetch={args.fetch}, source={source}) "
          f"| init {DATE} | members {members[0]}-{members[-1]} | upload={upload}")

    const_fields = None
    if args.fetch == "index":
        base = INDEX_BASES[source]
        t_c = time.time()
        print(f"constants (lsm/z/slor/sdor) from oper 0h, shared across members ...")
        const_fields = build_constants(base, DATE)
        print(f"  {sorted(const_fields)} ({time.time()-t_c:.1f}s)")
    else:
        import earthkit.data as ekd
        ekd.config.set({"cache-policy": "user"})

    ok, fail, times = [], [], []
    for i, m in enumerate(members):
        print(f"\n{'='*60}\nMember {m} ({i+1}/{len(members)})\n{'='*60}")
        try:
            t0 = time.time()
            if args.fetch == "index":
                state = create_input_state_index(DATE, m, base, const_fields)
            else:
                state = create_input_state(DATE, m, source=source)
            if not verify_input_state(state, m):
                fail.append(m)
                continue
            fname = os.path.join(args.out_dir, f"input_state_member_{m:03d}.pkl")
            with open(fname, "wb") as fh:
                pickle.dump(state, fh)
            print(f"  saved {fname}")
            if upload:
                blob = f"{datestr}/{args.gcs_subpath}/input_state_member_{m:03d}.pkl"
                if upload_to_gcs(fname, GCS_BUCKET, blob, GCS_SERVICE_ACCOUNT_KEY) \
                        and not args.keep_local:
                    os.remove(fname)
            ok.append(m)
            times.append(time.time() - t0)
            avg = sum(times) / len(times)
            print(f"  {times[-1]:.1f}s | ETA {(len(members)-i-1)*avg/60:.1f} min")
        except Exception as e:
            print(f"  ❌ member {m} failed: {e}")
            fail.append(m)

    print(f"\n{'='*60}\nSUMMARY: {len(ok)}/{len(members)} ok"
          + (f", failed {fail}" if fail else ""))
    if times:
        print(f"  total {sum(times)/60:.1f} min, avg {sum(times)/len(times):.1f}s/member")


if __name__ == "__main__":
    main()

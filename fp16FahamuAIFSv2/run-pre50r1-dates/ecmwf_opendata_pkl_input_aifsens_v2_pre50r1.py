#!/usr/bin/env python3
"""
AIFS ENS v2.0 — input-state (pkl) creation for dates **before IFS Cy50r1**.

A copy of ``../ecmwf_opendata_pkl_input_aifsens_v2.py`` with one change: the 13 input
fields Cy50r1 introduced, which open data does not carry before 2026-05-13, are filled
from the two donor archives established in ``README.md`` §2 instead.

For 2026-02-12, ``check_open_data_inputs.py`` reports exactly what is short::

    AIFS v2: NOT runnable
        sfc 9/9   sol 2/2   pl 6/6
        levels  13/14   missing: [10]
        wave     3/11   missing: ['wmb','h1012','h1214','h1417','h1721',
                                  'h2125','h2530','cdww']

and this script sources those, and only those, elsewhere:

    8 wave fields   ECMWF research hindcast ``class=rd expver=j1r2`` on ``grid=N320``
                    -- the ecWAM run AIFS v2's wave inputs were trained on.
                    Agrees with operational 50r1 at r >= 0.96 (README §2.4).
    10 hPa level    ERA5 on CDS, regridded to N320. 3-1140x closer to the real
                    analysis than carrying 50 hPa up (README §2.4).

Everything else -- all 99 other fields, the member perturbations, the transforms, both
retrieval backends, the GCS upload -- is unchanged from the operational script, so a
pre-50r1 pkl differs from an operational one in 13 fields and nothing else.

!!  READ THIS BEFORE USING THE OUTPUT  !!

    That a pkl builds does not mean the forecast is usable, and this has NOT been
    measured. README §5 specifies the experiment that would settle it (donor modes in
    ``wave_sensitivity.py``, run at 20260604 where the real fields exist to score
    against). Until that runs, treat output from this script as an experiment, not a
    product. The limitation that survives:

      * ERA5 is Cy41r2 at 31 km spliced into an otherwise Cy49r1 operational state, and
        it is DETERMINISTIC -- all 50 members share one 10 hPa level, so the initial
        spread of that level is zero.

    The matching worry about the wave donor turns out not to apply, which is worth
    knowing before anyone else re-derives it. Measured on the waef index directly:

        20260212 (pre-50r1)     max |swh(member n) - swh(member 1)| = 0
        20260604 (operational)  max |swh(member n) - swh(member 1)| = 0.066 m

    over n = 2, 10, 25, 50 at both timesteps -- distinct GRIB messages at distinct byte
    offsets, bit-identical values. Before Cy50r1 the wave ensemble carries **no spread at
    analysis time**; every member already shares one wave state and the spread grows only
    from the perturbed winds. So substituting a single deterministic j1r2 analysis for all
    50 members is exactly what open data itself does at these dates. It costs nothing that
    was not already absent.

    ``j1r2`` covers 2024-05-02 -> ~2026-06-20; ERA5 has ~5 days latency. Outside that
    window this script cannot help, and it says so rather than guessing.

Credentials -- two separate systems, both required (README §2.7):
    ``.ecmwfapirc``  ECMWF Web API key   -> j1r2   (https://api.ecmwf.int/v1/key/)
    ``.env``  ``ak=<uuid>``  CDS token   -> ERA5   (https://cds.climate.copernicus.eu/profile)

Usage::

    PY=/tank/projects/micromamba/envs/aifs-gpu/bin/python
    $PY ecmwf_opendata_pkl_input_aifsens_v2_pre50r1.py \
        --date 20260212 --members 1-50 --out-dir input_states_v2_20260212 --no-upload

    # reuse donors already on disk (they are per-cycle, not per-member)
    $PY ecmwf_opendata_pkl_input_aifsens_v2_pre50r1.py --date 20260212 --members 1 \
        --j1r2-npz j1r2_20260212_00.npz --era5-l10-npz era5_l10_20260212_00.npz

--- original header follows ---

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

# --- the pre-50r1 split ----------------------------------------------------
# Cy50r1 went operational with the 06z run on 2026-05-12, so 2026-05-13 00z is the
# first complete cycle carrying these. Earlier open data has neither.
CY50R1_CUTOVER = datetime.datetime(2026, 5, 13)

# 8 wave fields open data lacks before the cutover -> j1r2.
WAVE_FROM_J1R2 = ["wmb", "h1012", "h1214", "h1417", "h1721",
                  "h2125", "h2530", "cdww"]
# 3 that exist pre-50r1 and are still taken from waef, per member, as normal.
PARAM_WAVE_OPEN = [p for p in PARAM_WAVE if p not in WAVE_FROM_J1R2]

# The 10 hPa level open data lacks before the cutover -> ERA5. q_10 is dropped by v2
# anyway, so only t/u/v/w/z are needed.
L10 = 10
LEVELS_OPEN = [lv for lv in LEVELS if lv != L10]
L10_FROM_ERA5 = ["t_10", "u_10", "v_10", "w_10", "z_10"]

# j1r2 coverage, measured by bisection (README §2.2). Outside it, stop.
J1R2_START = datetime.datetime(2024, 5, 2)
J1R2_END = datetime.datetime(2026, 6, 20)

SOIL_RENAME = {"sot_1": "stl1", "sot_2": "stl2",
               "vsw_1": "swvl1", "vsw_2": "swvl2"}

# GCS configuration (same bucket/key conventions as the v1 prep)
GCS_BUCKET = "aifs-aiquest-us-20251127"
GCS_SERVICE_ACCOUNT_KEY = "coiled-data.json"
GCS_SUBPATH = "input_v2_pre50r1"  # never clobber an operational cycle
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
        # PARAM_WAVE_OPEN, not PARAM_WAVE: the other 8 do not exist in this
        # cycle's waef index and are injected from j1r2 afterwards.
        for p in PARAM_WAVE_OPEN:
            jobs.append((p, gk_prev, _find(idx_prev, param=p, levtype="sfc",
                                           number=member), 0))
            jobs.append((p, gk_t, _find(idx_t, param=p, levtype="sfc",
                                        number=member), 1))
    elif group.startswith("pl_"):
        p = group[3:]
        # LEVELS_OPEN, not LEVELS: 10 hPa is absent pre-50r1 and comes from ERA5.
        for lv in LEVELS_OPEN:
            if p == "q" and lv == 10:            # v2 drops q_10 (unreachable here)
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
    fields.update(get_open_data(date, param=PARAM_WAVE_OPEN, number=number,
                                stream=wave_stream, source=source))

    print("  soil fields...")
    soil = get_open_data(date, param=PARAM_SOIL, levelist=SOIL_LEVELS,
                         number=number, source=source)
    for k, v in soil.items():
        fields[SOIL_RENAME[k]] = v

    print("  pressure-level fields...")
    fields.update(get_open_data(date, param=PARAM_PL, levelist=LEVELS_OPEN,
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

    # geopotential height -> geopotential. LEVELS_OPEN: there is no gh_10 to convert;
    # ERA5 supplies z_10 as geopotential already, in the same m2/s2 units.
    for level in LEVELS_OPEN:
        fields[f"z_{level}"] = fields.pop(f"gh_{level}") * G

    input_state = dict(date=date, fields=fields)
    print(f"  done in {time.time()-t0:.1f}s — {len(fields)} fields, "
          f"shape {next(iter(fields.values())).shape}")
    return input_state


# ===========================================================================
# Donor fields: the 13 Cy50r1-only inputs, from j1r2 (waves) and ERA5 (10 hPa)
# ===========================================================================
# Both donors are per-CYCLE, not per-member -- fetched once and shared across all 50,
# exactly like the lsm/z/slor/sdor constants. That sharing is also the main scientific
# caveat: see the module docstring.


def _check_window(date):
    """Refuse dates the donors cannot cover, rather than failing 40 fields in."""
    if date >= CY50R1_CUTOVER:
        raise SystemExit(
            f"{date:%Y-%m-%d} is on/after the Cy50r1 cutover "
            f"({CY50R1_CUTOVER:%Y-%m-%d}) -- open data carries all 13 fields. "
            f"Use the operational ../ecmwf_opendata_pkl_input_aifsens_v2.py")
    if not (J1R2_START <= date <= J1R2_END):
        raise SystemExit(
            f"{date:%Y-%m-%d} is outside j1r2 coverage "
            f"({J1R2_START:%Y-%m-%d} to {J1R2_END:%Y-%m-%d}, README §2.2). "
            f"The 8 wave fields cannot be sourced for this date.")


def load_donors(date, j1r2_npz=None, era5_npz=None, workdir=None):
    """-> {field: (2, 542080)} covering all 13 Cy50r1-only inputs.

    Reads cached npz files when given, else calls fetch_j1r2 / fetch_era5_l10 directly
    (same folder). Donor arrays are already on N320 in the AIFS state's own point order:
    j1r2 because MARS is asked for ``grid=N320``, ERA5 because fetch_era5_l10 regrids
    there and validates the ordering against ``z_50``.
    """
    ymd, hh = date.strftime("%Y%m%d"), date.strftime("%H")
    workdir = workdir or os.path.join(os.environ.get("TMPDIR", "/tmp"),
                                      f"pre50r1_donors_{ymd}{hh}")
    os.makedirs(workdir, exist_ok=True)
    donors = {}

    if j1r2_npz:
        print(f"  waves  <- {j1r2_npz}")
        d = np.load(j1r2_npz)
        donors.update({k: np.asarray(d[k]) for k in WAVE_FROM_J1R2})
    else:
        print(f"  waves  <- MARS class=rd expver=j1r2 (grid=N320)")
        import fetch_j1r2
        donors.update(fetch_j1r2.build_state_fields(ymd, hh, workdir))

    if era5_npz:
        print(f"  10 hPa <- {era5_npz}")
        d = np.load(era5_npz)
        donors.update({k: np.asarray(d[k]) for k in L10_FROM_ERA5})
    else:
        print(f"  10 hPa <- ERA5 on CDS (regridded to N320)")
        import fetch_era5_l10
        donors.update(fetch_era5_l10.build(ymd, hh, L10, workdir,
                                           fetch_era5_l10.token()))

    missing = [f for f in WAVE_FROM_J1R2 + L10_FROM_ERA5 if f not in donors]
    if missing:
        raise SystemExit(f"donors incomplete: {missing}")
    return donors


_repair_cache = {}


def _n320_xyz():
    """Unit vectors for the 542080 N320 points, for nearest-neighbour hole filling."""
    h = eccodes.codes_grib_new_from_samples("reduced_gg_pl_320_grib2")
    try:
        lat = np.deg2rad(np.asarray(eccodes.codes_get_array(h, "latitudes")))
        lon = np.deg2rad(np.asarray(eccodes.codes_get_array(h, "longitudes")))
    finally:
        eccodes.codes_release(h)
    return np.column_stack([np.cos(lat) * np.cos(lon),
                            np.cos(lat) * np.sin(lon),
                            np.sin(lat)])


def _repair_donor_holes(donors, ref_land):
    """Fill donor points that ecWAM calls land but open data calls sea.

    The two land-sea masks disagree on a thin coastal fringe (README §2.4). Where open
    data says LAND the fix is trivial -- NaN the donor to match, which ``inject_donors``
    does. The other direction leaves a hole: ~230-250 points per field where the member's
    own ``swh`` carries a value but the donor does not.

    For the six period bands that combination genuinely never occurs operationally --
    measured on 20260514 and 20260604, ``h1012``..``h2530``, ``mwp`` and ``mwd`` match
    ``swh``'s mask at every one of the 1084160 points -- so a valid ``swh`` beside a NaN
    band height is an input shape the model has never seen, and is the kind that seeds
    NaN through a rollout. Fill it, from the nearest valid donor point on the sphere.

    ``wmb`` and ``cdww`` are the exception and the claim is weaker for them: operationally
    their mask deviates from ``swh``'s by 454 points (0.04%, symmetric -- 226 one way, 228
    the other), so the combination does occur, rarely. Filling them to ``swh``'s mask
    therefore makes this cycle marginally MORE uniform than an operational one rather than
    matching it exactly. That deviation is 0.04% of points and the alternative -- copying
    an operational cycle's absolute ``wmb``/``cdww`` mask -- would import that cycle's sea
    ice edge into a February state, which is worse. ``check_pkl_structure.py`` reports the
    fringe size for both so the difference stays visible instead of being assumed away.

    Cached on the reference mask: the mask is the same for every member, so 50 members
    pay this once.
    """
    from scipy.spatial import cKDTree

    key = ref_land.tobytes()
    if key in _repair_cache:
        return _repair_cache[key]

    xyz = _n320_xyz()
    out, filled = {}, 0
    for f in WAVE_FROM_J1R2:
        a = np.array(donors[f], dtype=np.float64, copy=True)
        for ti in range(a.shape[0]):
            valid = np.isfinite(a[ti])
            hole = ~valid & ~ref_land[ti]
            n = int(hole.sum())
            if not n:
                continue
            _, idx = cKDTree(xyz[valid]).query(xyz[hole])
            a[ti][hole] = a[ti][valid][idx]
            filled += n
        out[f] = a
    print(f"    mask repair: {filled} donor points filled from nearest valid "
          f"neighbour (ecWAM land / open-data sea)")
    _repair_cache[key] = out
    return out


def inject_donors(fields, donors):
    """Write the 13 donor fields into a member's field dict, in place.

    The wave land mask is reconciled to the operational one in both directions, so the
    8 donor fields end up with a NaN mask byte-identical to the member's own ``swh``.
    The model was trained to expect the operational mask; that is the one that wins.
    Exact for the six band heights; for ``wmb``/``cdww`` this is tighter than operational
    by 454 points -- see ``_repair_donor_holes``.
    """
    ref_land = ~np.isfinite(np.asarray(fields["swh"]))   # operational wave mask
    for f in WAVE_FROM_J1R2 + L10_FROM_ERA5:
        if np.asarray(donors[f]).shape != ref_land.shape:
            raise SystemExit(f"{f}: donor {np.asarray(donors[f]).shape} vs state "
                             f"{ref_land.shape} -- donors must be on N320")

    repaired = _repair_donor_holes(donors, ref_land)
    for f in WAVE_FROM_J1R2:
        a = np.array(repaired[f], dtype=np.float64, copy=True)
        a[ref_land] = np.nan                     # open data says land -> always NaN
        fields[f] = a
    for f in L10_FROM_ERA5:                      # atmospheric: no mask, no NaN
        fields[f] = np.array(donors[f], dtype=np.float64, copy=True)

    bad = [f for f in WAVE_FROM_J1R2
           if not np.array_equal(np.isnan(np.asarray(fields[f])), ref_land)]
    if bad:
        raise SystemExit(f"mask reconciliation failed for {bad}")
    nan10 = [f for f in L10_FROM_ERA5 if not np.isfinite(np.asarray(fields[f])).all()]
    if nan10:
        raise SystemExit(f"ERA5 donor has non-finite values in {nan10}")
    return fields


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
    ap.add_argument("--out-dir", default="input_states_v2_pre50r1",
                    help="local output dir (default input_states_v2_pre50r1)")
    ap.add_argument("--j1r2-npz", default=None,
                    help="cached donor from fetch_j1r2.py; fetched from MARS if omitted")
    ap.add_argument("--era5-l10-npz", default=None,
                    help="cached donor from fetch_era5_l10.py; fetched from CDS if omitted")
    ap.add_argument("--donor-workdir", default=None,
                    help="where donor GRIBs are cached (default under $TMPDIR)")
    ap.add_argument("--save-donors", default=None,
                    help="write the merged 13-field donor set to this npz for reuse")
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

    _check_window(DATE)

    os.makedirs(args.out_dir, exist_ok=True)
    print(f"AIFS ENS v2.0 PRE-50r1 input prep (fetch={args.fetch}, source={source}) "
          f"| init {DATE} | members {members[0]}-{members[-1]} | upload={upload}")
    print(f"  {len(WAVE_FROM_J1R2)} wave fields + the {L10} hPa level are absent from "
          f"open data before {CY50R1_CUTOVER:%Y-%m-%d} and come from donors.")

    # Donors first: they are per-cycle and a credential problem should surface now,
    # not after the first member's 99 open-data fields have been downloaded.
    t_d = time.time()
    print("donor fields (shared across all members) ...")
    donors = load_donors(DATE, args.j1r2_npz, args.era5_l10_npz, args.donor_workdir)
    print(f"  {len(donors)} donor fields ({time.time()-t_d:.1f}s)")
    if args.save_donors:
        np.savez_compressed(args.save_donors, **donors)
        print(f"  saved {args.save_donors}")

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
            inject_donors(state["fields"], donors)
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
    if ok:
        print("  NOTE: 13 of 112 fields are donor-sourced and shared by every member. "
              "For the 8 wave fields that matches open data's own pre-50r1 behaviour "
              "(no wave spread at t=0); for the 10 hPa level it is a real loss of "
              "spread. Whether a forecast from these is usable is UNMEASURED -- "
              "see README §5.")
    if times:
        print(f"  total {sum(times)/60:.1f} min, avg {sum(times)/len(times):.1f}s/member")


if __name__ == "__main__":
    main()

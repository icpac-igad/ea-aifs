#!/usr/bin/env python3
"""Fetch the AIFS v2 10 hPa level (``t/u/v/w/z_10``) from ERA5 on CDS.

The 10 hPa level is the *fatal* half of the pre-50r1 gap: ``wave_sensitivity.py``
measured the only substitute previously available -- 50 hPa carried up (``l10copy``) --
at 5.5x the model's own ensemble spread by day 5. That is unusable.

But "absent" was only ever true of the **0.25 deg open-data dissemination set**, which
gained 10 hPa with Cy50r1. ERA5 has carried 10 hPa since 1940, and it is available on CDS
for the blocked dates -- verified live for 2026-02-12. This fetches it as a donor for the
``l10era5`` mode of ``wave_sensitivity.py``.

Units line up with no conversion. ``fp16_multi_run_AIFS_ENS_v2.get_input_fields`` builds
``z_<level>`` as open-data ``gh`` x g, i.e. geopotential in m2/s2 -- which is exactly what
ERA5 ``z`` already is. ``w`` is Pa/s in both.

What does *not* line up, and why this is a measurement and not a fix:

* ERA5 is **Cy41r2 (2016) at 0.25 deg / 31 km**, spliced into an otherwise Cy49r1
  operational state. Different model, different cycle, different resolution.
* ERA5 is a **single deterministic reanalysis**. All 50 members would share one 10 hPa
  level, so stratospheric initial spread is zero.
* ERA5 is regridded here from regular lat-lon to N320. 10 hPa fields are very smooth, so
  bilinear is ample, but ``--validate`` checks the grid ordering rather than assuming it:
  it pulls ERA5 50 hPa for the same cycle and correlates against the state's own
  ``z_50``. Anything below ~0.99 means the point ordering is wrong -- stop and fix it.

Credentials: ``ak=<uuid>`` in ``.env`` (a CDS personal access token from
https://cds.climate.copernicus.eu/profile). This is *not* the ECMWF key that
``fetch_j1r2.py`` uses; the two APIs are separate systems.

Usage::

    python fetch_era5_l10.py --date 20260813 --time 00 --out era5_l10_20260813_00.npz \
        --validate /tank/projects/aifs-run/20260813_0000/input_states/input_state_member_001.pkl
"""

import argparse
import datetime as dt
import json
import os
import sys
import time
import urllib.request

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CDS = "https://cds.climate.copernicus.eu/api"
DATASET = "reanalysis-era5-pressure-levels"

# CDS variable name -> AIFS state field stem
VARS = {
    "temperature": "t",
    "u_component_of_wind": "u",
    "v_component_of_wind": "v",
    "vertical_velocity": "w",
    "geopotential": "z",
}


def token():
    for path in (os.path.join(HERE, ".env"),
                 os.path.join(os.path.dirname(HERE), ".env")):
        if os.path.exists(path):
            for line in open(path):
                if line.strip().startswith("ak="):
                    return line.strip().split("=", 1)[1]
    tok = os.environ.get("CDSAPI_KEY")
    if tok:
        return tok
    sys.exit("no CDS token: expected 'ak=<uuid>' in .env or $CDSAPI_KEY")


def _req(url, data=None, tok=None):
    r = urllib.request.Request(url, data=data, method="POST" if data else "GET")
    r.add_header("PRIVATE-TOKEN", tok)
    if data:
        r.add_header("Content-Type", "application/json")
    with urllib.request.urlopen(r, timeout=120) as resp:
        return json.loads(resp.read().decode())


def cds_fetch(times, level, target, tok, timeout=1800):
    """Submit one CDS request covering ``times`` (list of datetimes) at ``level`` hPa."""
    days = sorted({t.strftime("%d") for t in times})
    months = sorted({t.strftime("%m") for t in times})
    years = sorted({t.strftime("%Y") for t in times})
    hours = sorted({t.strftime("%H:00") for t in times})
    body = {"inputs": {
        "product_type": ["reanalysis"],
        "variable": list(VARS),
        "pressure_level": [str(level)],
        "year": years, "month": months, "day": days, "time": hours,
        "data_format": "grib", "download_format": "unarchived",
    }}
    job = _req(f"{CDS}/retrieve/v1/processes/{DATASET}/execute",
               json.dumps(body).encode(), tok)
    jid = job["jobID"]
    print(f"  CDS job {jid} ({len(years)}y {len(months)}m {len(days)}d {len(hours)}h)")
    t0 = time.time()
    while True:
        st = _req(f"{CDS}/retrieve/v1/jobs/{jid}", None, tok)["status"]
        if st == "successful":
            break
        if st in ("failed", "dismissed"):
            sys.exit(f"CDS job {jid} {st}")
        if time.time() - t0 > timeout:
            sys.exit(f"CDS job {jid} still {st} after {timeout}s")
        time.sleep(5)
    url = _req(f"{CDS}/retrieve/v1/jobs/{jid}/results", None, tok)["asset"]["value"]["href"]
    urllib.request.urlretrieve(url, target)
    print(f"  downloaded {os.path.getsize(target) / 1e6:.1f} MB -> {target}")
    return target


def read_latlon_grib(path):
    """-> {(shortName, dataDate, dataTime): 2-D array}, lats desc, lons asc."""
    import eccodes as ec

    out, lats, lons = {}, None, None
    with open(path, "rb") as f:
        while True:
            h = ec.codes_grib_new_from_file(f)
            if h is None:
                break
            ni = ec.codes_get(h, "Ni")
            nj = ec.codes_get(h, "Nj")
            if lats is None:
                lats = np.array(ec.codes_get_array(h, "distinctLatitudes"))
                lons = np.array(ec.codes_get_array(h, "distinctLongitudes"))
            key = (ec.codes_get(h, "shortName"), ec.codes_get(h, "dataDate"),
                   ec.codes_get(h, "dataTime"))
            out[key] = np.array(ec.codes_get_values(h)).reshape(nj, ni)
            ec.codes_release(h)
    return out, lats, lons


def n320_latlon():
    """Target grid: the AIFS v2 state grid, 542080 reduced-Gaussian points."""
    import eccodes as ec

    h = ec.codes_grib_new_from_samples("reduced_gg_pl_320_grib2")
    lat = np.array(ec.codes_get_array(h, "latitudes"))
    lon = np.array(ec.codes_get_array(h, "longitudes"))
    ec.codes_release(h)
    return lat, lon


def to_n320(grid2d, lats, lons, tlat, tlon):
    """Bilinear regrid regular lat-lon -> N320 point list, wrapping in longitude."""
    from scipy.interpolate import RegularGridInterpolator

    # RegularGridInterpolator needs ascending axes; ERA5 latitudes run 90 -> -90.
    if lats[0] > lats[-1]:
        lats = lats[::-1]
        grid2d = grid2d[::-1, :]
    # close the longitude seam so points between 359.75 and 360 interpolate correctly
    lons_w = np.concatenate([lons, [lons[-1] + (lons[1] - lons[0])]])
    grid_w = np.concatenate([grid2d, grid2d[:, :1]], axis=1)
    f = RegularGridInterpolator((lats, lons_w), grid_w,
                                method="linear", bounds_error=False, fill_value=None)
    return f(np.column_stack([tlat, tlon % 360.0]))


def build(date, time_, level, workdir, tok):
    """-> {aifs_field_name: (2, 542080)} for [t-6h, t]."""
    t1 = dt.datetime.strptime(f"{date}{time_}", "%Y%m%d%H")
    t0 = t1 - dt.timedelta(hours=6)
    tgt = os.path.join(workdir, f"era5_l{level}_{date}_{time_}.grib")
    if not os.path.exists(tgt):
        cds_fetch([t0, t1], level, tgt, tok)
    data, lats, lons = read_latlon_grib(tgt)
    tlat, tlon = n320_latlon()
    print(f"  ERA5 {lats.size}x{lons.size} lat-lon -> N320 {tlat.size} points")

    fields = {}
    for stem in VARS.values():
        per_t = []
        for t in (t0, t1):
            key = (stem, int(t.strftime("%Y%m%d")), int(t.strftime("%H")) * 100)
            if key not in data:
                sys.exit(f"ERA5 GRIB has no {key}; got {sorted(data)[:6]}...")
            per_t.append(to_n320(data[key], lats, lons, tlat, tlon))
        fields[f"{stem}_{level}"] = np.stack(per_t)
    return fields


def main():
    ap = argparse.ArgumentParser(description="fetch ERA5 10 hPa donor for AIFS v2")
    ap.add_argument("--date", required=True, help="YYYYMMDD of the cycle")
    ap.add_argument("--time", default="00")
    ap.add_argument("--level", type=int, default=10, help="hPa; 50 is the --validate ref")
    ap.add_argument("--workdir", default=None)
    ap.add_argument("--out", help="npz to write")
    ap.add_argument("--validate", metavar="PKL",
                    help="also pull 50 hPa and correlate against this state's z_50; "
                         "this is the grid-ordering check, do not skip it")
    args = ap.parse_args()

    workdir = args.workdir or os.path.join(
        os.environ.get("TMPDIR", "/tmp"), f"era5_l10_{args.date}{args.time}")
    os.makedirs(workdir, exist_ok=True)
    tok = token()

    print(f"=== ERA5 {args.level} hPa for {args.date} {args.time}z ===")
    fields = build(args.date, args.time, args.level, workdir, tok)
    for n in sorted(fields):
        a = fields[n]
        print(f"  {n:6s} {str(a.shape):14s} min={a.min():.6g} max={a.max():.6g}")

    if args.validate:
        import pickle

        print("--- grid-ordering validation at 50 hPa ---")
        ref50 = build(args.date, args.time, 50, workdir, tok)
        with open(args.validate, "rb") as f:
            st = pickle.load(f)
        ok = True
        for stem in ("z", "t", "u"):
            name = f"{stem}_50"
            if name not in st["fields"]:
                continue
            a = np.asarray(ref50[name])[1]
            b = np.asarray(st["fields"][name])[1]
            if a.shape != b.shape:
                sys.exit(f"shape mismatch {name}: ERA5 {a.shape} vs state {b.shape}")
            r = float(np.corrcoef(a, b)[0, 1])
            bias = float(np.mean(a - b))
            rms = float(np.sqrt(np.mean((a - b) ** 2)))
            flag = "OK " if r > 0.99 else "BAD"
            ok &= r > 0.99
            print(f"  {flag} {name}: r={r:.5f}  bias={bias:+.4g}  rms={rms:.4g}")
        if not ok:
            sys.exit("correlation below 0.99 -- N320 point ordering does not match the "
                     "state; do not use this donor")
        print("  grid ordering confirmed; ERA5-vs-operational rms above is the "
              "cycle/resolution difference, which is the thing being measured")

    if args.out:
        np.savez_compressed(args.out, **fields)
        print(f"wrote {args.out} ({os.path.getsize(args.out) / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

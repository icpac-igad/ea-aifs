"""VPM band series and day-of-year climatology from ARCO-ERA5.

Why this exists
---------------
`vpm_index.py` stops after normalisation because it has no climatology, and a
run on three cycles (`MJO_PHASE.md` 6b) showed why that matters: raw chi200 is
~86% wavenumber-1, but that wavenumber 1 is the *stationary* Walker cell, not
the MJO. Without an anomaly there is nothing for any EOF basis to project.

Grid: no regridding is needed and none is done
-----------------------------------------------
ARCO-ERA5 publishes a **240x121 equiangular** analysis-ready store -- exactly
the 1.5 deg grid `vpm_index.py` already solves the Poisson equation on
(`REGRID_DEG`). So the climatology is built on the forecast's own solve grid and
never touches O96. The O96 corpus is where the *forecast* wind lives; the
climatology only has to match the grid the solve happens on, which is 1.5 deg
for both.

The polar rows are kept rather than trimmed, because `vpm_index.py` keeps them:
`cos(phi)` is 6e-17 there, not 0, so the `m^2/cos(phi)` diagonal term becomes
enormous and the solver effectively pins chi=0 at the poles. That is not elegant,
but it is what the forecast path does, and a climatology must be computed the
same way as the field it will be subtracted from. Consistency beats elegance
here; the +-15 deg band is 75 deg away and cannot see either choice.

Two modes
---------
  --mode series : daily band series for a date range   -> validation
  --mode clim   : day-of-year climatology over years   -> production

`--mode series` over a documented MJO event is the test that `MJO_PHASE.md` 6b
should have proposed: if ERA5 shows 4-8 deg/day eastward through this exact
chain and a forecast does not, the chain is sound and that forecast simply has
no active MJO. The 6b test (remove a stationary field, look for drift) cannot
separate those two cases, because an absent MJO and a broken pipeline both leave
an incoherent residual.
"""
from __future__ import annotations

import argparse
import datetime as dt
import sys
import time

import numpy as np

from velocity_potential import divergence_latlon, solve_poisson_sphere
from mjo_index import N_LON_BINS, LAT_BAND

ARCO = ("https://storage.googleapis.com/gcp-public-data-arco-era5/ar/"
        "1959-2022-6h-240x121_equiangular_with_poles_conservative.zarr")
FIELDS = ("chi200", "u850", "u200")


def band_mean(field, lat, lon180, lat_band=LAT_BAND, nbins=N_LON_BINS):
    """(nt, nlat, nlon) -> (nt, 144) cosine-weighted mean over +-lat_band.

    Same reduction as `mjo_index.meridional_band`, but on a regular grid the
    latitude weights are shared by every longitude, so it is a plain tensor
    contraction rather than a bincount.
    """
    m = np.abs(lat) <= lat_band
    w = np.cos(np.deg2rad(lat[m]))
    b = np.einsum("tij,i->tj", field[:, m, :], w) / w.sum()   # (nt, nlon)
    idx = np.clip(np.floor((lon180 + 180.0) / (360.0 / nbins)).astype(int), 0, nbins - 1)
    cnt = np.bincount(idx, minlength=nbins)
    out = np.stack([np.bincount(idx, weights=b[t], minlength=nbins) for t in range(b.shape[0])])
    return out / np.maximum(cnt, 1)[None, :]


def open_era5():
    import xarray as xr
    ds = xr.open_zarr(ARCO, chunks=None, consolidated=True)
    lat = ds.latitude.values.astype(float)            # -90..90 increasing
    lon = ds.longitude.values.astype(float)           # 0..358.5
    lon180 = np.where(lon > 180.0, lon - 360.0, lon)
    return ds, lat, lon, lon180


def bands_for_slice(ds, sl, lat, lon, lon180):
    """Band series for a time slice -> dict of (nt, 144)."""
    sub = ds[["u_component_of_wind", "v_component_of_wind"]].isel(time=sl)
    # Read each variable ONCE and slice levels in memory. The store chunks all 13
    # levels together, so `.sel(level=200)` and `.sel(level=850)` on the same
    # variable fetch the *same* chunks twice over the network -- 50% extra
    # transfer on u, 33% on the job, which on a ~17 h stream is hours.
    lev = list(ds.level.values)
    i200, i850 = lev.index(200), lev.index(850)
    # stored as (time, level, longitude, latitude); the solver wants (t, lat, lon)
    uu = sub.u_component_of_wind.values
    vv = sub.v_component_of_wind.values
    u2 = uu[:, i200].transpose(0, 2, 1)
    u8 = uu[:, i850].transpose(0, 2, 1)
    v2 = vv[:, i200].transpose(0, 2, 1)
    del uu, vv

    d = divergence_latlon(u2, v2, lat, lon)
    chi = solve_poisson_sphere(d, lat, lon)
    return {"chi200": band_mean(chi, lat, lon180),
            "u850": band_mean(u8, lat, lon180),
            "u200": band_mean(u2, lat, lon180)}


def daily(bands, times):
    """6-hourly (nt,144) -> daily (nd,144) plus the dates."""
    days = np.array([np.datetime64(t, "D") for t in times])
    uniq = np.unique(days)
    out = {k: np.stack([v[days == d].mean(axis=0) for d in uniq]) for k, v in bands.items()}
    return out, uniq


def calendar_index(dates):
    """Dates -> 0..364 index into a calendar-day climatology.

    Indexed by (month, day), NOT by days-since-Jan-1: in a leap year every date
    after Feb 28 is one day later in the year, so binning on day-of-year smears
    the seasonal cycle by a day across a quarter of the samples. Feb 29 folds
    onto Feb 28.

    Both the builder and every consumer must use THIS function. A climatology
    subtracted with a different day convention than it was built with is wrong
    by one day in three years out of four, silently.
    """
    cal = {}
    for m in range(1, 13):
        for d in range(1, 32):
            try:
                cal[(m, d)] = dt.date(2001, m, d).timetuple().tm_yday - 1
            except ValueError:
                pass
    cal[(2, 29)] = cal[(2, 28)]
    return np.array([cal[(int(str(d)[5:7]), int(str(d)[8:10]))] for d in dates])


def build_climatology(dates, series, args):
    """Fold a daily band series into a smoothed calendar-day climatology."""
    # Calendar-day climatology, indexed by (month, day) rather than by
    # days-since-Jan-1. Those differ: in a leap year every date after Feb 28 is
    # one day later in the year than the same date in a common year, so binning
    # on day-of-year smears the seasonal cycle by a day across 1/4 of the
    # samples. Feb 29 folds onto Feb 28.
    idx = calendar_index(dates)

    clim, n = {}, np.zeros(365)
    for k in FIELDS:
        acc_ = np.zeros((365, N_LON_BINS)); cnt = np.zeros(365)
        np.add.at(acc_, idx, series[k]); np.add.at(cnt, idx, 1)
        clim[k] = acc_ / np.maximum(cnt, 1)[:, None]
        n = cnt
    if (n == 0).any():
        miss = int((n == 0).sum())
        print(f"  !! {miss} calendar days have no sample; harmonic fit will "
              f"interpolate them, but check the date range")

    # Smooth with the leading annual harmonics, as MJO climatologies
    # conventionally are. The seasonal cycle of a planetary-scale field is a few
    # harmonics wide; keeping all 365 retains sampling noise, which for a 2.5 deg
    # band mean over ~30 samples per day is not small. This also fills any
    # calendar day that happened to draw no sample.
    if args.harmonics:
        t = 2 * np.pi * np.arange(365) / 365.0
        cols = [np.ones(365)]
        for h in range(1, args.harmonics + 1):
            cols += [np.cos(h * t), np.sin(h * t)]
        G = np.stack(cols, axis=1)
        have = n > 0
        if have.sum() < 4 * G.shape[1]:
            sys.exit(f"only {have.sum()} calendar days have samples; a "
                     f"{G.shape[1]}-parameter harmonic fit would extrapolate, not "
                     f"smooth. Widen the date range or pass --harmonics 0.")
        for k in FIELDS:
            coef, *_ = np.linalg.lstsq(G[have], clim[k][have], rcond=None)
            fit = G @ coef
            # A smoother cannot amplify. If it does, the fit is extrapolating
            # through gaps rather than averaging over them -- which is exactly
            # what thin coverage looks like, and it is silent otherwise.
            r0, r1 = clim[k][have].std(), fit.std()
            if r1 > 3.0 * max(r0, 1e-30):
                sys.exit(f"harmonic fit for {k} has std {r1:.3e} against {r0:.3e} "
                         f"in the data it was fitted to -- it is extrapolating. "
                         f"Check calendar-day coverage ({have.sum()}/365 days).")
            clim[k] = fit
        print(f"  smoothed with {args.harmonics} annual harmonics "
              f"({G.shape[1]} parameters per longitude per field), "
              f"{have.sum()}/365 calendar days sampled")

    # record the span actually covered, not the CLI args -- in `combine` mode
    # those are unset, and the inputs are what define the base period.
    np.savez(args.out, **clim, n_per_day=n,
             span=f"{dates.min()}..{dates.max()}", n_days=len(dates),
             harmonics=args.harmonics, grid="240x121_1p5deg",
             # in combine mode --stride is the CLI default, not what built the
             # inputs; the day count is the honest record of sampling density
             days_per_year=len(dates) / max(1, len(set(str(d)[:4] for d in dates))),
             stride=args.stride, fields=np.array(FIELDS))
    print(f"  wrote calendar-day climatology ({n.min():.0f}-{n.max():.0f} samples "
          f"per day) -> {args.out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=("series", "clim", "combine"), required=True)
    ap.add_argument("--inputs", nargs="*", default=[],
                    help="--mode combine: per-year series .npz files to fold into one climatology")
    ap.add_argument("--start", help="YYYY-MM-DD")
    ap.add_argument("--end", help="YYYY-MM-DD (exclusive)")
    ap.add_argument("--block", type=int, default=240, help="timesteps per read (multiple of 8)")
    ap.add_argument("--harmonics", type=int, default=3,
                    help="annual harmonics to smooth the climatology with; 0 keeps raw daily means")
    ap.add_argument("--stride", type=int, default=1,
                    help="keep every Nth block of steps. The link to GCS measured "
                         "1.4 MB/s and does NOT improve with concurrency, so this is "
                         "the only lever on wall clock. Safe for a harmonic-smoothed "
                         "climatology, which fits ~7 parameters per longitude and does "
                         "not need every calendar day sampled in every year.")
    ap.add_argument("--phase", type=int, default=0,
                    help="offset the --stride pattern by this many blocks. Without it every "
                         "year samples the SAME calendar days (blocks are counted from the "
                         "year start), so stride 2 would cover half the calendar 30 times over "
                         "and the other half never. The driver sets phase = year %% stride.")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    if args.mode == "combine":
        # Streaming 30 years in one process is a ~17 h job on a 1.4 MB/s link,
        # and this box has lost three overnight runs to `apt-daily-upgrade`
        # replacing the Python runtime underneath them. So the climatology is
        # built from per-year `--mode series` files instead: each is ~35 min,
        # a killed year costs only itself, and re-running skips what exists.
        dates, series = [], {k: [] for k in FIELDS}
        for f in sorted(args.inputs):
            z = np.load(f)
            dates.append(z["dates"].astype("datetime64[D]"))
            for k in FIELDS:
                series[k].append(z[k])
            print(f"  + {f}  {len(z['dates'])} days")
        if not dates:
            sys.exit("--mode combine needs --inputs")
        dates = np.concatenate(dates)
        series = {k: np.concatenate(v) for k, v in series.items()}
        o = np.argsort(dates)
        dates, series = dates[o], {k: v[o] for k, v in series.items()}
        print(f"  {len(dates)} days total, {dates[0]} .. {dates[-1]}")
        build_climatology(dates, series, args)
        return

    if not (args.start and args.end):
        sys.exit("--mode series/clim need --start and --end")
    ds, lat, lon, lon180 = open_era5()
    t = ds.time.values
    sel = np.where((t >= np.datetime64(args.start)) & (t < np.datetime64(args.end)))[0]
    if sel.size == 0:
        sys.exit(f"no ERA5 timesteps in [{args.start}, {args.end})")
    i0, i1 = int(sel[0]), int(sel[-1]) + 1
    print(f"ERA5 VPM | {args.mode} | {args.start} .. {args.end} | {i1-i0} steps "
          f"({(i1-i0)/4/365.25:.1f} yr) | grid {lat.size}x{lon.size}", flush=True)

    acc, dates_all, t0, nread = [], [], time.time(), 0
    # with --stride N we read `block` steps then skip (N-1)*block, starting
    # --phase blocks in so that successive years cover complementary days
    for a in range(i0 + args.phase * args.block, i1, args.block * args.stride):
        b = min(a + args.block, i1)
        bands = bands_for_slice(ds, slice(a, b), lat, lon, lon180)
        d, dd = daily(bands, t[a:b])
        acc.append(d); dates_all.append(dd)
        nread += b - a
        # `done` is progress through the *calendar*, which is what the ETA must
        # extrapolate on; `nread` is how many steps were actually fetched. With
        # --stride they differ, and reporting only the first reads as if the
        # skipped steps had been downloaded.
        done = (b - i0) / (i1 - i0)
        el = time.time() - t0
        print(f"  {b-i0:7d}/{i1-i0} calendar ({100*done:5.1f}%)  "
              f"{nread:6d} steps read  elapsed {el/60:6.1f} min  "
              f"eta {el*(1-done)/max(done,1e-9)/60:6.1f} min", flush=True)

    dates = np.concatenate(dates_all)
    series = {k: np.concatenate([a[k] for a in acc]) for k in FIELDS}

    if args.mode == "series":
        np.savez(args.out, dates=dates.astype("datetime64[D]").astype(str), **series)
        print(f"  wrote {len(dates)} days -> {args.out}")
        return

    build_climatology(dates, series, args)


if __name__ == "__main__":
    main()

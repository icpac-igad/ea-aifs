"""Tropical-storm-days forecast + tercile probabilities from the Icechunk store.

AI-WQ defines the *observed* diagnostic from IBTrACS
(`training_data.html#tropical-storm-days-data-processing`, implemented in
`AI_WQ_package.TS_processing.count_nstormdays`):

  - a record counts if `usa_wind >= 33.0454` kt (17 m/s) and the storm's
    position at that record lies in the basin box;
  - the diagnostic is the number of unique **`(storm, day)` pairs** in the
    7-day window - so two concurrent storms on one day contribute **two** storm
    days and a week is **not** capped at 7 (the 20-year sample reaches 28);

and evaluates a *forecast* as tercile probabilities per basin, scored by RPSS
(`forecast_evaluation.html#tropical-storm-days-ts`).

IBTrACS is observation-only, so a forecast must detect storms in the model
fields. Because the target counts *storms*, not *days with a storm*, the
detection has to carry storm identity: see `ts_tracks.py`, which reduces the
qualifying points at each step to discrete centres, links those centres into
tracks across the 6-hourly steps, and counts `(track, day)` pairs under the
same TS-strength and in-basin record filters AI-WQ applies.

Centre criteria, all evaluated at the store's 6-hourly steps:

  candidate net (vectorised):
    cyclonic:     zeta850 >= --vorticity                          (1e-4 1/s)
    low:          msl     <= latitude-band median - --msl-drop    (5 hPa)
  then, per centre, against its OWN surroundings:
    closed low:   msl(centre) <= annulus mean - --msl-drop        (5 hPa)
    warm core:    T(200-500 hPa) within --core-radius exceeds the
                  annulus out to --env-radius by --warm-core      (1.0 K)
  and, at counting time:
    TS strength:  max |V10| within --wind-radius >= --wind-threshold (17 m/s)

The wind test is a neighbourhood maximum rather than a co-located value
because IBTrACS `usa_wind` is the storm's maximum sustained wind and the
eyewall is never the same gridpoint as the pressure minimum. The warm core is
measured against the storm's own environment rather than a zonal reference:
that contrast is what distinguishes a tropical cyclone from an extratropical
one, which is cold-core aloft.

`vo` is not an AIFS-ENS-2.0 output, so the vorticity is derived from `u_850`/
`v_850` directly on the reduced Gaussian grid (see `grid_ops.py`) - ~1.5 s per
member, versus ~26 min if every field were regridded to differentiate it.

*** DETECTION CAVEAT ***
The tracker fixes the *quantity* (unbounded `(storm, day)` pairs, matching
AI-WQ) and the *wind semantics*, but the thresholds remain a proxy for
IBTrACS's agency-assigned intensities, and N320 (~28 km) under-resolves TC peak
winds. Compare the resulting counts against the official IBTrACS terciles
(`--aiwq-tercile-dir`) before treating them as submission-grade; if the
detector's climatological mean is offset from the observed one, every forecast
lands in the same category and RPSS is negative by construction.
"""
from __future__ import annotations

import argparse
import datetime as dt
import glob
import os

import numpy as np

import store_io as sio
import ts_tracks as tt
from grid_ops import ReducedGaussianGrid, cyclonic

WIND_THRESHOLD = 17.0     # m/s, AI-WQ / IBTrACS tropical-storm strength
MSL_DROP = 500.0          # Pa below the latitude-band median = "closed low"
WARM_CORE_K = 1.0         # K, 200-500 hPa core minus its own environment
VORTICITY = 1.0e-4        # 1/s, cyclonic 850 hPa relative vorticity at the core
LAT_BIN = 2.0             # deg, for the latitude-banded reference
BELT_LAT = 45.0           # deg, detection belt - the basin boxes span +-40


def counts_for_member(g, m, steps, times, lat, lon180, belt, pidx, grid, args):
    """`({basin: {date: n}}, n_tracks)` for one member - see `ts_tracks`."""
    u = sio.read_member_window(g, "10u", m, steps)
    v = sio.read_member_window(g, "10v", m, steps)
    spd = np.sqrt(u * u + v * v)
    del u, v
    p = sio.read_member_window(g, "msl", m, steps)

    tlayer = None
    if args.warm_core:
        # 200-500 hPa layer mean: the warm core of a tropical cyclone is a deep
        # upper-tropospheric feature, and averaging levels suppresses the
        # single-level noise that a 28 km grid carries at day 18+
        for lev in tt.WARM_CORE_LEVELS:
            t = sio.read_member_window(g, lev, m, steps)
            tlayer = t if tlayer is None else tlayer + t
            del t
        tlayer /= len(tt.WARM_CORE_LEVELS)

    czeta = None
    if args.vorticity and grid is not None:
        u850 = sio.read_member_window(g, "u_850", m, steps)
        v850 = sio.read_member_window(g, "v_850", m, steps)
        # `vo` is not an output variable; derive it on the reduced Gaussian grid
        czeta = cyclonic(grid.relative_vorticity(u850, v850), lat).astype(np.float32)
        del u850, v850

    centres = tt.detect_centres(
        p, czeta, tlayer, spd, lat, lon180, belt, pidx,
        msl_drop=args.msl_drop, vort_thr=args.vorticity,
        warm_core_k=args.warm_core, sep_km=args.centre_separation,
        wind_radius_km=args.wind_radius, core_radius_km=args.core_radius,
        env_radius_km=args.env_radius, band_deg=LAT_BIN)
    tracks = tt.link_tracks(centres, step_hours=6.0,
                            max_speed_ms=args.max_translation)
    return tt.storm_day_counts(tracks, times, sio.BASINS,
                               wind_thr=args.wind_threshold, step_hours=6.0,
                               min_track_hours=args.min_track_hours)


def load_aiwq_terciles(directory, weeks, basins):
    """(nweeks, nbasin, 2) bounds from AI-WQ's per-week IBTrACS tercile files.

    `TS_processing.compute_20yr_TStercile_climatology` writes one file per
    verification week, `TS_20yrCLIM_WEEKLYTSDAYS_terciles_<YYYYMMDD>.nc`, with
    dims `(tercile, basin)` - so the bounds are per-week, not shared, and must
    be matched to each week's start date.
    """
    import xarray as xr

    bounds = np.empty((len(weeks), len(basins), 2))
    used = []
    for wi, wk in enumerate(weeks):
        stamp = wk[0].strftime("%Y%m%d")
        path = os.path.join(directory, f"TS_20yrCLIM_WEEKLYTSDAYS_terciles_{stamp}.nc")
        if not os.path.exists(path):
            have = sorted(os.path.basename(f) for f in
                          glob.glob(os.path.join(directory, "TS_*terciles_*.nc")))
            raise SystemExit(
                f"no IBTrACS tercile file for week starting {stamp}: {path}\n"
                f"  present: {have or '(none)'}\n"
                f"  build with AI_WQ_package.TS_processing."
                f"compute_20yr_TStercile_climatology('{stamp}', ibtracs, '{directory}')")
        ds = xr.open_dataset(path)
        da = ds[[v for v in ds.data_vars][0]]          # (tercile, basin)
        da = da.sel(basin=list(basins))
        bounds[wi] = np.asarray(da.values).T           # -> (basin, 2)
        used.append(os.path.basename(path))
    return bounds, used


def tercile_probs(counts, lower, upper):
    """Ensemble tercile probabilities, binned exactly as AI-WQ bins the observation.

    The convention is not ours to choose: `AI_WQ_package.forecast_evaluation.
    conditional_obs_probs` decides which category the *observation* falls in, and a
    forecast binned any other way is being scored against a different partition.
    Reading that function, for bounds (lower, upper):

        below :  x <  lower
        near  :  lower <= x <  upper     -- and only if lower != upper
        above :  x >= upper

    Both edges differ from the obvious `<=` / `>` reading, and with integer storm-day
    counts against integer-valued terciles ties at a bound are common, so the difference
    is not academic: it moved 0.14 of probability mass in NWP on cycle 20260820.

    When `lower == upper` AI-WQ's `all_equal` mask sets the observation to NaN, i.e. the
    basin-week cannot be scored at all. We mirror that by emitting an empty middle
    category and reporting it, rather than implying a forecast that nothing can grade.

    Returns (probs[3], degenerate).
    """
    x = np.asarray(counts, float)
    degenerate = not (lower != upper)
    near = 0.0 if degenerate else np.mean((x >= lower) & (x < upper))
    return np.array([np.mean(x < lower), near, np.mean(x >= upper)]), degenerate


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--store", required=True)
    ap.add_argument("--tag")
    ap.add_argument("--init", required=True, help="cycle init YYYYMMDD")
    ap.add_argument("--members", type=int, default=None)
    ap.add_argument("--wind-threshold", type=float, default=WIND_THRESHOLD,
                    help="m/s required as the maximum wind near the centre "
                         "(default 17 = IBTrACS 33.0454 kt)")
    ap.add_argument("--wind-radius", type=float, default=tt.WIND_RADIUS_KM,
                    help="km around the centre searched for that maximum "
                         "(default 300; `usa_wind` is a storm maximum, not a "
                         "value at the pressure minimum)")
    ap.add_argument("--msl-drop", type=float, default=MSL_DROP,
                    help="Pa below the latitude-band median msl required "
                         "(default 500 = 5 hPa)")
    ap.add_argument("--vorticity", type=float, default=VORTICITY,
                    help="cyclonic 850 hPa relative vorticity required at the "
                         "core [1/s]; 0 disables (default 1e-4). Derived on the "
                         "reduced Gaussian grid - `vo` is not a model output")
    ap.add_argument("--warm-core", type=float, default=WARM_CORE_K,
                    help="K by which the 200-500 hPa layer mean inside "
                         "--core-radius must exceed the surrounding annulus; "
                         "0 disables the warm-core test (default 1.0)")
    ap.add_argument("--core-radius", type=float, default=tt.CORE_RADIUS_KM,
                    help="km; inner disc averaged for the warm-core and "
                         "pressure-deficit tests (default 200)")
    ap.add_argument("--env-radius", type=float, default=tt.ENV_RADIUS_KM,
                    help="km; outer radius whose annulus is the environment "
                         "the core is compared against (default 800)")
    ap.add_argument("--centre-separation", type=float, default=tt.CENTRE_SEPARATION_KM,
                    help="km; qualifying points within this distance of an "
                         "accepted centre are the same system (default 500)")
    ap.add_argument("--max-translation", type=float, default=tt.MAX_TRANSLATION_MS,
                    help="m/s cap on centre displacement between 6-hourly steps "
                         "when linking tracks (default 25 = ~540 km/step)")
    ap.add_argument("--min-track-hours", type=float, default=tt.MIN_TRACK_HOURS,
                    help="a linked centre must persist this long to count "
                         "(default 24; suppresses single-step blips)")
    ap.add_argument("--aiwq-tercile-dir",
                    help="directory of AI-WQ IBTrACS tercile files "
                         "(TS_20yrCLIM_WEEKLYTSDAYS_terciles_<week>.nc), used "
                         "per verification week - the submission-grade path")
    ap.add_argument("--tercile-clim",
                    help=".npz with `bounds` (basin, 2) lower/upper tercile "
                         "boundaries built with THIS detector; without either "
                         "clim option the ensemble's own terciles are used and "
                         "flagged")
    ap.add_argument("--out", default="ts_days_probs.nc")
    args = ap.parse_args()

    init = dt.datetime.strptime(args.init, "%Y%m%d")
    g = sio.open_store(args.store, args.tag)
    lat, lon180, _ = sio.coords(g)
    steps = sio.written_steps(g)
    times = sio.valid_times(g, init, steps)
    nmem = args.members or sio.n_members(g)
    grid = ReducedGaussianGrid(lat, np.asarray(g["longitude"][:])) if args.vorticity else None
    belt = np.abs(lat) <= BELT_LAT
    pidx = tt.PointIndex(lat, lon180)

    days = sio.group_steps_by_day(times)
    weeks = sio.seven_day_windows(days)
    basins = list(sio.BASINS)

    print(f"TS days | init {init:%Y-%m-%d} | members {nmem} | "
          f"{times[0]:%Y-%m-%d} .. {times[-1]:%Y-%m-%d}")
    print(f"  centres: msl<=band median-{args.msl_drop:.0f} Pa and <=annulus mean-"
          f"{args.msl_drop:.0f} Pa"
          + (f", zeta850>={args.vorticity:.1e} 1/s" if args.vorticity else ", NO vorticity test")
          + (f", warm core>=+{args.warm_core} K (T200-500 within {args.core_radius:.0f} km "
             f"minus annulus to {args.env_radius:.0f} km)" if args.warm_core else ", NO warm-core")
          + f", separation {args.centre_separation:.0f} km")
    print(f"  tracks:  link <= {args.max_translation:.0f} m/s, keep >= "
          f"{args.min_track_hours:.0f} h; count (track, day) pairs with "
          f"max|V10| within {args.wind_radius:.0f} km >= {args.wind_threshold} m/s")
    print(f"  {len(days)} days -> {len(weeks)} full 7-day windows, "
          f"detection belt |lat|<={BELT_LAT:.0f} ({belt.sum()} of {lat.size} points)")

    counts = np.zeros((nmem, len(weeks), len(basins)), dtype=int)
    ntracks = np.zeros(nmem, dtype=int)
    for m in range(nmem):
        per_basin, kept = counts_for_member(g, m, steps, times, lat, lon180,
                                            belt, pidx, grid, args)
        ntracks[m] = kept
        for wi, wk in enumerate(weeks):
            for bi, b in enumerate(basins):
                counts[m, wi, bi] = sum(per_basin[b].get(d, 0) for d in wk)
        if (m + 1) % 10 == 0 or m == nmem - 1:
            print(f"    member {m+1}/{nmem} done ({kept} tracks)")

    # ---- tercile probabilities -------------------------------------------
    if args.aiwq_tercile_dir:
        bounds, used = load_aiwq_terciles(args.aiwq_tercile_dir, weeks, basins)
        clim_src = f"AI-WQ IBTrACS terciles: {', '.join(used)}"
    elif args.tercile_clim:
        b = np.load(args.tercile_clim)["bounds"]              # (basin, 2)
        bounds = np.repeat(np.asarray(b, float)[None], len(weeks), axis=0)
        clim_src = args.tercile_clim
    else:
        b = np.stack([np.nanpercentile(counts[:, :, bi], [100 / 3, 200 / 3])
                      for bi in range(len(basins))])
        bounds = np.repeat(b[None], len(weeks), axis=0)
        clim_src = "ENSEMBLE_SELF (uncalibrated - see caveat)"
        print("\n  !! no --aiwq-tercile-dir / --tercile-clim: using the ensemble's "
              "own terciles, so the probabilities are self-referential and NOT "
              "submission-grade.")

    probs = np.zeros((len(weeks), len(basins), 3))
    degenerate = np.zeros((len(weeks), len(basins)), bool)
    for wi in range(len(weeks)):
        for bi in range(len(basins)):
            probs[wi, bi], degenerate[wi, bi] = tercile_probs(counts[:, wi, bi],
                                                              *bounds[wi, bi])
    if degenerate.any():
        pairs = ", ".join(f"{weeks[wi][0]}/{basins[bi]}"
                          for wi, bi in zip(*np.nonzero(degenerate)))
        print(f"\n  !! degenerate terciles (lower == upper) for {pairs}: under AI-WQ's "
              "partition every count lands in 'above' there, and its scorer masks the "
              "observation to NaN anyway (conditional_obs_probs `all_equal`), so those "
              "basin-weeks are NOT scoreable. The 0/0/1 emitted for them is arithmetic, "
              "not a forecast - it keeps the submission check's sum-to-1 satisfied.")

    import xarray as xr
    ds = xr.Dataset(
        {
            "TS_tercile_probability": (("week", "basin", "category"), probs),
            "storm_days": (("member", "week", "basin"), counts),
            "tercile_bounds": (("week", "basin", "bound"), np.asarray(bounds, float)),
            "n_tracks": (("member",), ntracks),
        },
        coords={
            "week": [str(w[0]) for w in weeks],
            "basin": basins,
            "category": ["below", "near", "above"],
            "member": np.arange(nmem),
            "bound": ["lower", "upper"],
        },
        attrs={
            "detector": ("tracked cyclone centres (relative msl minimum + cyclonic "
                         "850 hPa vorticity + 300 hPa warm core), linked across "
                         "6-hourly steps; counts unique (track, day) pairs"),
            "counts_quantity": ("unique (storm, day) pairs per basin per week - "
                                "matches AI_WQ_package.TS_processing.count_nstormdays; "
                                "NOT capped at 7"),
            "warm_core_k": args.warm_core,
            "vorticity_threshold_s-1": args.vorticity,
            "vorticity_source": "derived from u_850/v_850 on the N320 reduced Gaussian grid",
            "wind_threshold_ms": args.wind_threshold,
            "wind_radius_km": args.wind_radius,
            "msl_drop_pa": args.msl_drop,
            "centre_separation_km": args.centre_separation,
            "max_translation_ms": args.max_translation,
            "min_track_hours": args.min_track_hours,
            "step_hours": 6,
            "ibtracs_native_step_hours": 3,
            "tercile_source": clim_src,
            "source_store": str(args.store),
            "cycle_init": args.init,
        },
    )
    ds.to_netcdf(args.out)
    print(f"\n  wrote {args.out}   ({ntracks.mean():.1f} tracks/member)")
    for wi, wk in enumerate(weeks):
        print(f"  week from {wk[0]}:")
        for bi, b in enumerate(basins):
            c = counts[:, wi, bi]
            lo, hi = bounds[wi, bi]
            print(f"    {b:<5} mean {c.mean():5.1f} d  range {c.min()}-{c.max()}  "
                  f"terciles {lo:g}/{hi:g}  "
                  f"P(below/near/above) = {probs[wi,bi,0]:.2f}/"
                  f"{probs[wi,bi,1]:.2f}/{probs[wi,bi,2]:.2f}")


if __name__ == "__main__":
    main()

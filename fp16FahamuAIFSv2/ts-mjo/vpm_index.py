"""VPM (Velocity Potential MJO) index from the Icechunk store.

Wheeler & Hendon's RMM needs OLR, which AIFS-ENS-2.0 does not output and which
cannot be added by re-running inference (`MJO_PHASE.md`). The **VPM index**
(Ventrice et al. 2013) is an established RMM-like index in which **200 hPa
velocity potential replaces OLR**:

    WH04 RMM :  [ OLR   , U850, U200 ]
    VPM      :  [ chi200, U850, U200 ]

Unlike a truncated WH04 projection - which is not an EOF of the wind-only space
and is refused by `mjo_index.py` - VPM is a real index with its own published
basis, and `chi200` is **diagnosed exactly** from the forecast wind rather than
statistically emulated. That makes it the one MJO index this model can support
on its own output.

Pipeline
--------
    1. D200 = divergence(u200, v200)            native reduced Gaussian grid
    2. regrid D200 -> regular 1.5 deg           one field per step
    3. solve laplacian(chi) = D                 `velocity_potential.py`
    4. cosine-weighted mean over +-15 deg -> 144 longitudes, for each of
       chi200, U850, U200                       (U850/U200 need no regrid)
    5. daily means from the 6-hourly steps
    6. day-of-year climatology removed          [needs --clim]
    7. preceding 120-day rolling mean removed   [needs --lowfreq]
    8. normalise each field by its own std
    9. project onto the VPM combined EOFs       [needs --eofs]
   10. VPM1/VPM2 -> amplitude, phase -> the 9 AI-WQ categories

Only step 1 touches the model. Steps 4-10 are the same machinery `mjo_index.py`
uses and are imported from it rather than reimplemented.

Why divergence is computed natively and only D is regridded
-----------------------------------------------------------
`meridional_band` already works on the flat reduced-Gaussian vector, so U850 and
U200 never need a regular grid. Only the Poisson solve does. Computing D on the
native grid and regridding **one** field per step instead of two halves the
interpolation cost, which dominates the runtime.

*** WHAT THIS DOES NOT SHIP ***
The **VPM EOFs are an external asset**, exactly as WH04's are. AI-WQ distributes
`WH04_combinedEOFs.nc` for RMM but nothing for VPM; the reference basis comes
from NOAA PSL. Without `--eofs` this script **stops after step 8** and writes
the normalised band series (`--dump-bands`) rather than inventing a basis - the
same refusal `mjo_index.py` makes. A self-computed EOF basis would not be VPM:
it would be a new index with no published phase convention, and its phases would
not correspond to VPM's or to RMM's.
"""
from __future__ import annotations

import argparse
import datetime as dt

import numpy as np

import store_io as sio
from grid_ops import ReducedGaussianGrid
from mjo_index import (N_LON_BINS, LAT_BAND, meridional_band, daily_mean,
                       phase_from_pcs, load_eofs)
from velocity_potential import solve_poisson_sphere, divergence_latlon
from era5_vpm_clim import calendar_index

REGRID_DEG = 1.5        # the Poisson grid; MJO is zonal wavenumber 1-3
FIELDS = ("chi200", "u850", "u200")


def chi200_bands(g, m, steps, lat, lon180, grid, regrid_deg=REGRID_DEG,
                 lat_band=LAT_BAND, nbins=N_LON_BINS, div_path="regular"):
    """(nsteps, 144) band-mean chi200 for one member.

    `div_path` decides WHERE the divergence is taken, and it is not a free
    choice when a climatology is involved. Measured on 20260917, one member,
    20 steps, the two paths give:

        chi200 band std ratio      1.60   (native / regular)
        band correlation           0.64
        k=1 amplitude ratio        1.38
        k=2 amplitude ratio        3.41
        k=1 crest offset           +22.8 deg mean, 36.9 deg std

    That last line is the one that matters: the drift diagnostic reads the
    motion of the k=1 crest, and a 37 deg path-dependent scatter is several days
    of MJO propagation at 4-8 deg/day. The paths are not interchangeable.

    "regular" (default) regrids u,v and differentiates on the regular grid --
    the same thing `era5_vpm_clim.py` does to ERA5. **The climatology must be
    subtracted from a field computed the same way it was**, so this is the
    default despite costing two interpolations per step instead of one.

    "native" takes the divergence on the reduced Gaussian grid and regrids only
    D. It is ~2x cheaper and was the original default, chosen on cost alone
    before the difference was measured. It retains more small-scale divergence
    (hence the k=2/k=3 excess), which is arguably the better-resolved answer --
    but it is NOT the quantity the ERA5 climatology represents.
    """
    import earthkit.regrid as ekr

    u = sio.read_member_window(g, "u_200", m, steps)
    v = sio.read_member_window(g, "v_200", m, steps)
    src = {"grid": "O96" if lat.size == 40320 else "N320"}
    tgt = {"grid": [regrid_deg, regrid_deg]}

    def to_reg(x):
        return np.stack([ekr.interpolate(x[k], src, tgt) for k in range(x.shape[0])])

    if div_path == "native":
        d = np.nan_to_num(grid.divergence(u, v), nan=0.0)
        del u, v
        dg = to_reg(d)
    else:
        # Note the polar asymmetry this also removes: the native path sets D=0
        # at the polar rows (nan_to_num), while `divergence_latlon` leaves the
        # 1/cos(phi) value there. Both are harmless -- the Poisson RHS weights
        # by cos(phi), which cancels it -- but they are not the same, and using
        # one path for both sides makes the question moot.
        ug, vg = to_reg(u), to_reg(v)
        del u, v
        nlat = ug.shape[-2]
        alat = np.linspace(-90.0, 90.0, nlat)
        rlon_ = np.arange(ug.shape[-1]) * (360.0 / ug.shape[-1])
        dg = divergence_latlon(ug[:, ::-1, :], vg[:, ::-1, :], alat, rlon_)[:, ::-1, :]

    nlat, nlon = dg.shape[-2], dg.shape[-1]
    rlat = np.linspace(90.0, -90.0, nlat)
    rlon = np.arange(nlon) * (360.0 / nlon)

    chi = solve_poisson_sphere(dg[:, ::-1, :], rlat[::-1], rlon)[:, ::-1, :]

    flat = chi.reshape(chi.shape[0], -1)
    la = np.repeat(rlat, nlon)
    lo = np.tile(np.where(rlon > 180.0, rlon - 360.0, rlon), nlat)
    return meridional_band(flat, la, lo, lat_band, nbins)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--store", required=True)
    ap.add_argument("--tag")
    ap.add_argument("--init", required=True, help="cycle init YYYYMMDD")
    ap.add_argument("--members", type=int, default=None)
    ap.add_argument("--regrid-deg", type=float, default=REGRID_DEG,
                    help="regular grid for the Poisson solve (default 1.5)")
    ap.add_argument("--div-path", choices=("regular", "native"), default="regular",
                    help="where to take the divergence. 'regular' (default) regrids u,v "
                         "first, matching how era5_vpm_clim.py builds the climatology -- "
                         "required for --clim to be meaningful. 'native' is ~2x cheaper but "
                         "puts the k=1 crest ~23 deg elsewhere; see chi200_bands.__doc__")
    ap.add_argument("--eofs", help=".npz with VPM `eof1`,`eof2` (3*144, ordered "
                                  "[chi200, u850, u200]) and optional `sd1`,`sd2`. "
                                  "WITHOUT THIS the script stops after "
                                  "normalisation -- it will not invent a basis")
    ap.add_argument("--clim", help=".npz day-of-year climatology per field")
    ap.add_argument("--lowfreq", help=".npz preceding 120-day means per field")
    ap.add_argument("--dump-bands", help="write the normalised band series here (.npz)")
    ap.add_argument("--out", default="vpm_probs.nc")
    args = ap.parse_args()

    init = dt.datetime.strptime(args.init, "%Y%m%d")
    g = sio.open_store(args.store, args.tag)
    lat, lon180, _ = sio.coords(g)
    steps = sio.written_steps(g)
    times = sio.valid_times(g, init, steps)
    nmem = args.members or sio.n_members(g)
    grid = ReducedGaussianGrid(lat, np.asarray(g["longitude"][:]))

    print(f"VPM | init {init:%Y-%m-%d} | members {nmem} | "
          f"{times[0]:%Y-%m-%d} .. {times[-1]:%Y-%m-%d} | {len(steps)} steps")
    print(f"  chi200: divergence on the {args.div_path} grid -> {args.regrid_deg} deg "
          f"Poisson solve; U850/U200 banded on the native grid")
    if args.clim and args.div_path != "regular":
        print("  !! --clim with --div-path native: the climatology is built on the "
              "regular\n     grid, and the two paths put the k=1 crest ~23 deg apart. "
              "This subtraction\n     mixes two different quantities.")

    per_member = []
    for m in range(nmem):
        bands = {"chi200": chi200_bands(g, m, steps, lat, lon180, grid,
                                    args.regrid_deg, div_path=args.div_path)}
        for name, var in (("u850", "u_850"), ("u200", "u_200")):
            bands[name] = meridional_band(
                sio.read_member_window(g, var, m, steps), lat, lon180)
        daily = {k: daily_mean(v, times) for k, v in bands.items()}
        per_member.append({k: v[0] if isinstance(v, tuple) else v
                           for k, v in daily.items()})
        if (m + 1) % 10 == 0 or m == nmem - 1:
            print(f"    member {m+1}/{nmem} done")

    dates = daily_mean(bands["u850"], times)[1]
    stacked = {k: np.stack([pm[k] for pm in per_member]) for k in FIELDS}

    if args.clim:
        # The climatology is (365, 144) by CALENDAR DAY, so it must be indexed by
        # each forecast day's own date, not subtracted wholesale. The previous
        # form did `z[k][None, :, :]`, which cannot even broadcast against
        # (member, ndays, 144) -- so --clim had never run.
        # `calendar_index` is imported rather than reimplemented: a climatology
        # subtracted under a different day convention than it was built with is
        # silently wrong by a day in three years out of four.
        z = np.load(args.clim)
        ci = calendar_index(dates)
        for k in FIELDS:
            if z[k].shape != (365, N_LON_BINS):
                raise SystemExit(f"--clim {k} is {z[k].shape}, expected (365, {N_LON_BINS})")
            stacked[k] = stacked[k] - z[k][ci][None, :, :]
        print(f"  removed calendar-day climatology {args.clim} "
              f"(days {ci.min()}..{ci.max()})")
    else:
        print("  !! no --clim: anomalies are not referenced to an observed "
              "climatology, so these are NOT VPM-comparable")
    if args.lowfreq:
        # Same shape trap as --clim: this is a per-date series, so it must align
        # with `dates` on its own date axis, not be broadcast.
        z = np.load(args.lowfreq)
        for k in FIELDS:
            if z[k].shape[0] != len(dates):
                raise SystemExit(f"--lowfreq {k} has {z[k].shape[0]} days, "
                                 f"forecast has {len(dates)}; they must align by date")
            stacked[k] = stacked[k] - z[k][None, :, :]
    else:
        print("  !! no --lowfreq: the preceding 120-day mean is not removed")

    norm = {k: float(np.nanstd(stacked[k])) for k in FIELDS}
    for k in FIELDS:
        stacked[k] = stacked[k] / max(norm[k], 1e-30)
    print("  normalisation (std per field): "
          + ", ".join(f"{k}={norm[k]:.3e}" for k in FIELDS))

    if args.dump_bands:
        np.savez(args.dump_bands, dates=np.array([str(d) for d in dates]),
                 **{k: stacked[k] for k in FIELDS}, **{f"sd_{k}": norm[k] for k in FIELDS})
        print(f"  wrote bands -> {args.dump_bands}")

    if not args.eofs:
        print("\n  STOPPING after step 8. The VPM EOFs are an external asset and\n"
              "  none was given (--eofs). Projecting onto a self-computed basis\n"
              "  would produce a different index with no published phase\n"
              "  convention, so it is not done. Bands above are complete and\n"
              "  correct; supply --eofs to finish.")
        return

    e1, e2, sd1, sd2 = load_eofs(args.eofs)
    x = np.concatenate([stacked[k] for k in FIELDS], axis=2)   # (mem, ndays, 3*144)
    if x.shape[2] != e1.size:
        raise SystemExit(f"state vector {x.shape[2]} != EOF length {e1.size}. "
                         f"VPM EOFs must be 3*{N_LON_BINS} ordered [chi200, u850, u200].")
    vpm1, vpm2 = (x @ e1) / sd1, (x @ e2) / sd2
    # `phase_from_pcs` returns (phase, amplitude) and is already vectorised, so
    # it takes the whole (member, day) array at once. Mapping it element-wise and
    # stacking the result silently built a (member, day, 2) array instead.
    ph, amp = phase_from_pcs(vpm1, vpm2)

    import xarray as xr
    probs = np.stack([(ph == c).mean(axis=0) for c in range(9)], axis=-1)
    xr.Dataset(
        {"MJO_phase_probability": (("day", "MJO_phase"), probs),
         "vpm1": (("member", "day"), vpm1), "vpm2": (("member", "day"), vpm2),
         "amplitude": (("member", "day"), amp), "phase": (("member", "day"), ph)},
        coords={"day": [str(d) for d in dates], "MJO_phase": np.arange(9),
                "member": np.arange(nmem)},
        attrs={"index_kind": "vpm", "basis": args.eofs,
               "components": "chi200,u850,u200",
               "chi200_source": f"divergence on the {args.div_path} grid, "
                                f"Poisson solve on a {args.regrid_deg} deg regular grid",
               "climatology_removed": bool(args.clim),
               "lowfreq_removed": bool(args.lowfreq),
               "source_store": str(args.store), "cycle_init": args.init},
    ).to_netcdf(args.out)
    print(f"\n  wrote {args.out}")


if __name__ == "__main__":
    main()

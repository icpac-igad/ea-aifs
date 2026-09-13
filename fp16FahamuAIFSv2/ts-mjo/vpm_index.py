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
from velocity_potential import solve_poisson_sphere

REGRID_DEG = 1.5        # the Poisson grid; MJO is zonal wavenumber 1-3
FIELDS = ("chi200", "u850", "u200")


def chi200_bands(g, m, steps, lat, lon180, grid, regrid_deg=REGRID_DEG,
                 lat_band=LAT_BAND, nbins=N_LON_BINS):
    """(nsteps, 144) band-mean chi200 for one member."""
    import earthkit.regrid as ekr

    u = sio.read_member_window(g, "u_200", m, steps)
    v = sio.read_member_window(g, "v_200", m, steps)
    d = grid.divergence(u, v)
    del u, v
    # `valid` excludes the polar rows, where the centred meridional difference
    # has no neighbour. They are 75 deg from the band and contribute nothing.
    d = np.nan_to_num(d, nan=0.0)

    dg = np.stack([ekr.interpolate(d[k], {"grid": "O96"},
                                   {"grid": [regrid_deg, regrid_deg]})
                   for k in range(d.shape[0])])
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
    print(f"  chi200 via divergence on the native grid -> {args.regrid_deg} deg "
          f"Poisson solve; U850/U200 banded on the native grid")

    per_member = []
    for m in range(nmem):
        bands = {"chi200": chi200_bands(g, m, steps, lat, lon180, grid, args.regrid_deg)}
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
        z = np.load(args.clim)
        for k in FIELDS:
            stacked[k] = stacked[k] - z[k][None, :, :]
    else:
        print("  !! no --clim: anomalies are not referenced to an observed "
              "climatology, so these are NOT VPM-comparable")
    if args.lowfreq:
        z = np.load(args.lowfreq)
        for k in FIELDS:
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
    amp = np.sqrt(vpm1**2 + vpm2**2)
    ph = np.stack([[phase_from_pcs(a, b) for a, b in zip(r1, r2)]
                   for r1, r2 in zip(vpm1, vpm2)])

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
               "chi200_source": "divergence on the native reduced Gaussian grid, "
                                f"Poisson solve on a {args.regrid_deg} deg regular grid",
               "climatology_removed": bool(args.clim),
               "lowfreq_removed": bool(args.lowfreq),
               "source_store": str(args.store), "cycle_init": args.init},
    ).to_netcdf(args.out)
    print(f"\n  wrote {args.out}")


if __name__ == "__main__":
    main()

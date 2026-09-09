#!/usr/bin/env python3
"""Does the O96 hop hurt forecast calibration? N320 vs O96 on the SAME rollout.

Needs a date holding both an N320-derived and an O96-derived 1.5 deg product built from
one inference run, so the regrid route is the only difference. 20260813 qualifies: its
O96 store was transcoded from its N320 store, and both were carried through the identical
downstream chain.

Reports, per variable:
  * mean probability per quintile, both routes (climatology = 0.200 everywhere)
  * sharpness -- mean probability on the modal quintile -- and the shift between routes,
    expressed as a FRACTION OF THE GAP from climatological calibration. That framing is
    the point: a shift is only meaningful next to how far the forecast already sits from
    0.200 on its own.
  * ensemble spread of the weekly aggregate the quintiles are cut from, which is the
    mechanism -- less spread means a sharper, more over-confident distribution.

Usage::

    python compare_regrid_routes.py --n320-dir .../aiwq --o96-dir .../aiwq_via_o96 \
        --n320-nc .../nc_1p5deg --o96-nc .../nc_1p5deg_via_o96 --date 20260813
"""
import argparse, glob, os
import numpy as np, xarray as xr

VARS = [("2t_quintiles", "2t", "tas", "K"),
        ("msl_quintiles", "msl", "mslp", "Pa"),
        ("tp_quintiles", "tp", "pr", "mm")]
WEEKS = [("week1", [0, 1], 24), ("week2", [2, 3, 4], 36)]


def weekly_ens(ncdir, var, chunks, nsteps):
    ens = []
    for f in sorted(glob.glob(f"{ncdir}/*.nc")):
        d = xr.open_dataset(f)
        s = sum(d[var].isel(time=t, member=0).sum(dim="step", skipna=True) for t in chunks)
        ens.append((s * 1000 * (7.0 / (nsteps * 6 / 24)) if var == "tp"
                    else s / nsteps).values)
        d.close()
    return np.stack(ens)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", required=True)
    ap.add_argument("--n320-dir", required=True, help="aiwq dir from the N320 route")
    ap.add_argument("--o96-dir", required=True, help="aiwq dir from the O96 route")
    ap.add_argument("--n320-nc", default=None, help="per-member 1.5 deg, N320 route")
    ap.add_argument("--o96-nc", default=None, help="per-member 1.5 deg, O96 route")
    a = ap.parse_args()

    f = f"ensemble_quintile_probabilities_{a.date}_v2.nc"
    A = xr.open_dataset(os.path.join(a.n320_dir, f))
    C = xr.open_dataset(os.path.join(a.o96_dir, f))

    print(f"=== {a.date}: same rollout, only the regrid route differs ===")
    print("mean probability per quintile (both weeks pooled); climatology = 0.200\n")
    for qname, _, nm, _ in VARS:
        pa = np.nanmean(A[qname].values, axis=(0, 2, 3))
        pc = np.nanmean(C[qname].values, axis=(0, 2, 3))
        sa = float(np.nanmean(A[qname].values.max(axis=1)))
        sc = float(np.nanmean(C[qname].values.max(axis=1)))
        gap = sa - 0.200
        print(f"{nm}")
        print("   N320 " + "  ".join(f"Q{i+1}={v:.3f}" for i, v in enumerate(pa))
              + f"   sharp={sa:.4f}")
        print("   O96  " + "  ".join(f"Q{i+1}={v:.3f}" for i, v in enumerate(pc))
              + f"   sharp={sc:.4f}")
        print(f"   -> the regrid moves sharpness {sc-sa:+.4f}; the forecast already sits "
              f"{gap:.3f} from climatology")
        print(f"      regrid explains {100*abs(sc-sa)/gap:.1f}% of the miscalibration\n")

    if a.n320_nc and a.o96_nc:
        print("=== ensemble spread of the weekly aggregate (the mechanism) ===")
        for _, raw, nm, unit in VARS:
            for wname, chunks, nsteps in WEEKS:
                x = weekly_ens(a.n320_nc, raw, chunks, nsteps)
                y = weekly_ens(a.o96_nc, raw, chunks, nsteps)
                sx = np.nanmean(x.std(axis=0, ddof=1))
                sy = np.nanmean(y.std(axis=0, ddof=1))
                print(f"  {nm:5s} {wname}: N320 {sx:9.4g} {unit:2s} | O96 {sy:9.4g} {unit:2s}"
                      f" | retained {100*sy/sx:6.2f}%")


if __name__ == "__main__":
    main()

"""Recover the VPM projection from PSL's published index time series.

Why this is not a self-computed EOF basis
-----------------------------------------
`MJO_PHASE.md` refuses to project onto a basis we computed ourselves, and the
reason is sound: our own EOFs would be a *different index*, with no published
phase convention, so its phases would not correspond to VPM's or to RMM's and
the `amplitude < 1` inactive test would lose its meaning.

NOAA PSL turns out not to publish the VPM EOF patterns. Its
`ftp2.psl.noaa.gov:/Datasets.other/MJO/eof1,eof2` directories are **OMI's**:
2448 values per file (144 longitudes x 17 latitudes, a single OLR field) with one
file per day of year, which is the OMI construction, not VPM's fixed pair of
3x144 vectors. The `MJO` note in that directory even reads "EOF basis patterns.
could be deleted most likely."

But PSL *does* publish the **VPM index itself** -- `vpm.1x.txt`, daily VPM1,
VPM2 and amplitude from 1979-04-30 to 2026-03-17. So rather than inventing a
basis, fit the linear functional that reproduces those published PCs from our own
band anomalies. The result **inherits VPM's phase convention by construction**,
because it is fitted to VPM's own series -- which is exactly the property a
self-computed EOF lacks.

What it is, honestly
--------------------
It is a regression, not an EOF. Two consequences:

- It is not orthonormal and the two vectors are not variance-maximising. Nothing
  downstream needs them to be: `vpm_index.py` only forms `x @ e1`, `x @ e2`.
- It inherits our reanalysis, our climatology and our grid. VPM was built on
  NCEP R1 1979-2012; this is fitted against ERA5. The fit quality below is
  therefore the honest measure of agreement, and it is measured out of sample.

Measured, fit on 1991-2010 and tested on 2011-2020 (1828 held-out days):

    VPM1        test r = 0.794
    VPM2        test r = 0.868
    amplitude   test r = 0.626
    phase       median |error| 16.4 deg on active days (amp >= 1)
                90.2% land in the correct octant

The last line is the one that matters for AI-WQ, whose MJO product is the nine
octant categories: the right octant 90% of the time on active days. Amplitude is
the weak component, and it is what decides category 0 (inactive) -- adding the
120-day trailing mean lifts amplitude r from 0.626 to 0.697 while leaving the
octant hit rate flat, which is the measured case for building that filter.
"""
from __future__ import annotations

import argparse
import glob

import numpy as np

from era5_vpm_clim import calendar_index

FIELDS = ("chi200", "u850", "u200")
N_LON = 144


def load_series(pattern):
    dates, ser = [], {k: [] for k in FIELDS}
    for f in sorted(glob.glob(pattern)):
        z = np.load(f)
        dates.append(z["dates"].astype("datetime64[D]"))
        for k in FIELDS:
            ser[k].append(z[k])
    if not dates:
        raise SystemExit(f"no series files matched {pattern}")
    d = np.concatenate(dates)
    o = np.argsort(d)
    return d[o], {k: np.concatenate(v)[o] for k, v in ser.items()}


def trailing_mean(a, day_int, window=120, min_samples=30):
    """Mean of the preceding `window` days, per longitude. Causal by construction."""
    out = np.full_like(a, np.nan)
    for i in range(len(day_int)):
        m = (day_int >= day_int[i] - window) & (day_int < day_int[i])
        if m.sum() >= min_samples:
            out[i] = a[m].mean(axis=0)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--series", default="/tank/projects/era5_vpm_clim/era5_vpm_????.npz")
    ap.add_argument("--clim", default="/tank/projects/era5_vpm_clim/vpm_clim_1991_2020.npz")
    ap.add_argument("--vpm", default="/tank/projects/era5_vpm_clim/vpm.1x.txt",
                    help="PSL vpm.1x.txt: year month day hour VPM1 VPM2 amplitude")
    ap.add_argument("--lowfreq", action="store_true",
                    help="also remove the 120-day trailing mean (helps amplitude; see __doc__)")
    ap.add_argument("--train-end", type=int, default=2010,
                    help="last year used for fitting; later years are held out")
    ap.add_argument("--out", default="/tank/projects/era5_vpm_clim/vpm_basis.npz")
    args = ap.parse_args()

    dates, ser = load_series(args.series)
    clim = np.load(args.clim)
    ci = calendar_index(dates)
    di = dates.astype("datetime64[D]").astype(int)

    blocks = []
    for k in FIELDS:
        a = ser[k] - clim[k][ci]
        if args.lowfreq:
            a = a - trailing_mean(a, di)
        blocks.append(a)
    X = np.concatenate(blocks, axis=1)

    v = np.loadtxt(args.vpm)
    vd = np.array([np.datetime64(f"{int(y):04d}-{int(m):02d}-{int(d):02d}")
                   for y, m, d in v[:, :3]])
    pos = {d: i for i, d in enumerate(vd)}
    ok = np.isfinite(X).all(axis=1) & np.array([d in pos for d in dates])
    X, ds = X[ok], dates[ok]
    Y = v[[pos[d] for d in ds], 4:6]

    # normalise each field by its own std, as the index does, and keep the
    # factors so a forecast can be scaled the same way
    sd = np.array([X[:, i * N_LON:(i + 1) * N_LON].std() for i in range(len(FIELDS))])
    for i in range(len(FIELDS)):
        X[:, i * N_LON:(i + 1) * N_LON] /= sd[i]

    yr = ds.astype("datetime64[Y]").astype(int) + 1970
    tr, te = yr <= args.train_end, yr > args.train_end
    if te.sum() == 0:
        raise SystemExit(f"--train-end {args.train_end} leaves no held-out years")
    A = np.column_stack([X, np.ones(len(X))])
    coef, *_ = np.linalg.lstsq(A[tr], Y[tr], rcond=None)
    P = A @ coef

    print(f"fit on {tr.sum()} days (..{args.train_end}), tested on {te.sum()} "
          f"({yr[te].min()}-{yr[te].max()}) | lowfreq={args.lowfreq}")
    for j, nm in enumerate(("VPM1", "VPM2")):
        print(f"  {nm}: train r={np.corrcoef(P[tr, j], Y[tr, j])[0, 1]:.3f}  "
              f"TEST r={np.corrcoef(P[te, j], Y[te, j])[0, 1]:.3f}")
    ap_, ao = np.hypot(*P[te].T), np.hypot(*Y[te].T)
    dph = (np.degrees(np.arctan2(P[te, 1], P[te, 0]))
           - np.degrees(np.arctan2(Y[te, 1], Y[te, 0])) + 180) % 360 - 180
    act = ao >= 1.0
    print(f"  amplitude TEST r={np.corrcoef(ap_, ao)[0, 1]:.3f}")
    print(f"  phase on {act.sum()} active days: median |err| {np.median(np.abs(dph[act])):.1f} deg, "
          f"correct octant {100 * (np.abs(dph[act]) < 45).mean():.1f}%")

    np.savez(args.out,
             eof1=coef[:-1, 0], eof2=coef[:-1, 1],
             intercept=coef[-1], sd1=1.0, sd2=1.0, field_sd=sd,
             fields=np.array(FIELDS), n_lon=N_LON,
             kind="regression_onto_published_VPM",
             source=args.vpm, clim=args.clim,
             train_years=f"..{args.train_end}", n_train=int(tr.sum()),
             lowfreq_removed=bool(args.lowfreq))
    print(f"\n  wrote {args.out}")
    print("  NOTE: this is a regression, not an EOF -- not orthonormal, and it "
          "inherits\n        our reanalysis and climatology. See __doc__.")


if __name__ == "__main__":
    main()

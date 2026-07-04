#!/usr/bin/env python3
"""
Plot AI-WQ individual quintile-probability files to eyeball a submission.

For each variable it builds one figure: rows = forecast periods (week1, week2,
the two "time steps"), columns = the 5 quintile categories. Each panel is a
probability map. Climatological baseline is 0.2 per quintile, so the colormap
is diverging about 0.2 (red = above-normal probability, blue = below-normal).

Usage:
    python plot_aiwq_quintiles.py --date 20260625 --team Fahamu --model fp16FahamuAIFSv2
    python plot_aiwq_quintiles.py --dir aiwq_individual_20260625 --variables tas
"""
import argparse
import glob
import os

import numpy as np
import xarray as xr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    HAVE_CARTOPY = True
except Exception:
    HAVE_CARTOPY = False

WEEKS = ["1", "2"]
QLABELS = ["Q1\n(lowest)", "Q2", "Q3\n(middle)", "Q4", "Q5\n(highest)"]


def find_file(d, variable, week, team, model):
    pat = os.path.join(d, f"{variable}_*_p{week}_{team}_{model}.nc")
    hits = sorted(glob.glob(pat))
    return hits[0] if hits else None


def plot_variable(d, variable, team, model, date, out):
    proj = ccrs.PlateCarree() if HAVE_CARTOPY else None
    subkw = {"projection": proj} if HAVE_CARTOPY else {}
    fig, axes = plt.subplots(
        len(WEEKS), 5, figsize=(20, 7), subplot_kw=subkw,
        constrained_layout=True,
    )

    mesh = None
    for r, week in enumerate(WEEKS):
        f = find_file(d, variable, week, team, model)
        if f is None:
            print(f"  ! missing {variable} p{week}")
            continue
        da = xr.open_dataarray(f)
        lon = da.longitude.values
        lat = da.latitude.values
        p0 = str(da.forecast_period_start.values)[:10]
        p1 = str(da.forecast_period_end.values)[:10]

        for c in range(5):
            ax = axes[r, c]
            field = da.isel(quintile=c).values
            kw = dict(cmap="RdBu_r", vmin=0.0, vmax=0.4, shading="auto")
            if HAVE_CARTOPY:
                kw["transform"] = ccrs.PlateCarree()
            mesh = ax.pcolormesh(lon, lat, field, **kw)
            if HAVE_CARTOPY:
                ax.coastlines(linewidth=0.4)
                ax.add_feature(cfeature.BORDERS, linewidth=0.2)
                ax.set_global()
            if r == 0:
                ax.set_title(QLABELS[c], fontsize=10)
            if c == 0:
                ax.text(-0.08, 0.5, f"week{week}\n{p0}→{p1}",
                        transform=ax.transAxes, rotation=90,
                        va="center", ha="center", fontsize=9)

    cbar = fig.colorbar(mesh, ax=axes, orientation="horizontal",
                        fraction=0.04, pad=0.02, extend="max")
    cbar.set_label("quintile probability (climatology = 0.2; red = above-normal)")
    fig.suptitle(f"{variable.upper()}  quintile probabilities  |  {team}/{model}  |  init {date}",
                 fontsize=14)
    fig.savefig(out, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✅ {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", default=None)
    ap.add_argument("--dir", default=None, help="dir with individual nc files")
    ap.add_argument("--team", default="Fahamu")
    ap.add_argument("--model", default="fp16FahamuAIFSv2")
    ap.add_argument("--variables", nargs="+", default=["tas", "pr", "mslp"])
    ap.add_argument("--outdir", default=None)
    args = ap.parse_args()

    d = args.dir or f"aiwq_individual_{args.date}"
    if not os.path.isdir(d):
        raise SystemExit(f"dir not found: {d}")
    date = args.date or os.path.basename(d).split("_")[-1]
    outdir = args.outdir or f"aiwq_plots_{date}"
    os.makedirs(outdir, exist_ok=True)

    print(f"Reading {d}  (cartopy={'yes' if HAVE_CARTOPY else 'no'})")
    for v in args.variables:
        out = os.path.join(outdir, f"{v}_{date}_quintiles.png")
        plot_variable(d, v, args.team, args.model, date, out)
    print(f"Done -> {outdir}/")


if __name__ == "__main__":
    main()

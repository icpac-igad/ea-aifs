#!/usr/bin/env python3
"""Plot a single AIFS-ENS-2.0 forecast field on a global map for a quick sanity check.

Reads one v2 GRIB chunk (local path, or pulled from GCS by date/member), selects a
parameter at a given forecast hour, regrids the native N320 reduced-Gaussian field to a
regular lat/lon grid and renders it with the file name, member and forecast hour labelled
on the figure.

Examples
--------
# total precipitation, member 2, +72 h, from a local GRIB chunk
python plot_forecast_field.py --grib /scratch/ensemble_outputs/aifs_ens_forecast_20260625_0000_member002_h000-072.grib \
    --param tp --hour 72

# 2 m temperature, member 35, +192 h, pulled straight from GCS for the cycle
python plot_forecast_field.py --date 20260625_0000 --member 35 --param 2t --hour 192

The GCS path mirrors the v2 pipeline output: gs://<bucket>/<date>/fp16_v2_forecasts/.
"""
import argparse
import os

import numpy as np
import earthkit.data as ekd
import earthkit.regrid as ekr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import cartopy.crs as ccrs
import cartopy.feature as cfeature

BUCKET = "aifs-aiquest-us-20251127"
SERVICE_ACCOUNT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "coiled-data.json")

# Per-parameter rendering: (label, unit transform, cmap, norm-builder)
PRECIP_LEVELS = [0, 1, 2, 5, 10, 20, 35, 50, 75, 100, 150, 200, 300]


def _chunk_for_hour(hour: int) -> str:
    """Return the 72 h GRIB-chunk suffix (e.g. h144-216) that contains `hour`."""
    start = ((hour - 1) // 72) * 72 if hour > 0 else 0
    end = start + 72
    return f"h{start:03d}-{end:03d}"


def resolve_grib(args) -> str:
    """Return a local GRIB path, downloading the right chunk from GCS if needed."""
    if args.grib:
        return args.grib
    if not (args.date and args.member):
        raise SystemExit("Provide --grib, or both --date and --member (+ --hour).")
    suffix = _chunk_for_hour(args.hour)
    fname = f"aifs_ens_forecast_{args.date}_member{args.member:03d}_{suffix}.grib"
    local = os.path.join(args.cache_dir, fname)
    if not os.path.exists(local):
        from google.cloud import storage
        os.makedirs(args.cache_dir, exist_ok=True)
        blob = f"{args.date}/{args.gcs_subpath}/{fname}"
        print(f"[GCS] downloading gs://{args.bucket}/{blob}")
        client = storage.Client.from_service_account_json(args.service_account)
        client.bucket(args.bucket).blob(blob).download_to_filename(local)
    return local


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    src = p.add_argument_group("source (use --grib OR --date/--member)")
    src.add_argument("--grib", help="Local GRIB chunk path")
    src.add_argument("--date", help="Cycle date, e.g. 20260625_0000 (for GCS fetch)")
    src.add_argument("--member", type=int, help="Ensemble member number (for GCS fetch)")
    p.add_argument("--param", default="tp", help="GRIB shortName (e.g. tp, 2t, msl, swh)")
    p.add_argument("--hour", type=int, default=72, help="Forecast lead hour to plot")
    p.add_argument("--grid", type=float, default=0.5, help="Target regular grid (deg)")
    p.add_argument("--out", help="Output PNG path (default derived from param/member/hour)")
    p.add_argument("--bucket", default=BUCKET)
    p.add_argument("--gcs-subpath", default="fp16_v2_forecasts")
    p.add_argument("--service-account", default=SERVICE_ACCOUNT)
    p.add_argument("--cache-dir", default="/scratch/plot_src")
    args = p.parse_args()

    grib = resolve_grib(args)
    print(f"[GRIB] {grib} ({os.path.getsize(grib)/1e6:.0f} MB)")

    ds = ekd.from_source("file", grib).sel(param=args.param)
    if len(ds) == 0:
        raise SystemExit(f"No '{args.param}' fields in {os.path.basename(grib)}")
    steps = [f.metadata("endStep") for f in ds]
    matches = [f for f in ds if f.metadata("endStep") == args.hour]
    if not matches:
        raise SystemExit(f"hour {args.hour} not in file; available endSteps: {steps}")
    fld = matches[0]

    vals = fld.to_numpy()  # native N320, 1-D
    grid = ekr.interpolate(vals, {"grid": "N320"}, {"grid": [args.grid, args.grid]})
    nlat, nlon = grid.shape
    lat = np.linspace(90, -90, nlat)
    lon = np.linspace(0, 360, nlon, endpoint=False)

    # member from filename (authoritative), falling back to --member
    base = os.path.basename(grib)
    member = args.member
    if "member" in base:
        try:
            member = int(base.split("member")[1][:3])
        except ValueError:
            pass

    fig = plt.figure(figsize=(14, 7.6))
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.set_global()
    ax.coastlines(linewidth=0.5)
    ax.add_feature(cfeature.BORDERS, linewidth=0.2, alpha=0.3)

    if args.param == "tp":
        field = grid * 1000.0  # m -> mm accumulated
        cmap = plt.get_cmap("turbo")
        norm = mcolors.BoundaryNorm(PRECIP_LEVELS, cmap.N, extend="max")
        pc = ax.pcolormesh(lon, lat, field, transform=ccrs.PlateCarree(),
                           cmap=cmap, norm=norm, shading="auto")
        cb = plt.colorbar(pc, ax=ax, orientation="horizontal", pad=0.05, shrink=0.85,
                          extend="max", ticks=PRECIP_LEVELS)
        cb.set_label(f"Total precipitation (mm, accumulated 0–{args.hour} h)")
        long_name = "total precipitation"
    elif args.param in ("2t", "skt", "2d"):
        field = grid - 273.15  # K -> degC
        pc = ax.pcolormesh(lon, lat, field, transform=ccrs.PlateCarree(),
                           cmap="RdYlBu_r", vmin=-40, vmax=40, shading="auto")
        cb = plt.colorbar(pc, ax=ax, orientation="horizontal", pad=0.05, shrink=0.85, extend="both")
        cb.set_label(f"{args.param} (°C)")
        long_name = {"2t": "2 m temperature", "skt": "skin temperature",
                     "2d": "2 m dewpoint"}[args.param]
    else:
        field = grid
        pc = ax.pcolormesh(lon, lat, field, transform=ccrs.PlateCarree(),
                           cmap="viridis", shading="auto")
        cb = plt.colorbar(pc, ax=ax, orientation="horizontal", pad=0.05, shrink=0.85, extend="both")
        cb.set_label(args.param)
        long_name = args.param

    ax.set_title(f"AIFS-ENS-2.0 FP16  —  {long_name} ({args.param})",
                 fontsize=14, fontweight="bold")
    info = (f"file: {base}\n"
            f"member: {member:03d}    forecast hour: +{args.hour} h")
    ax.text(0.01, -0.16, info, transform=ax.transAxes, fontsize=10, family="monospace",
            va="top", ha="left", bbox=dict(boxstyle="round", fc="white", ec="0.6", alpha=0.9))

    out = args.out or f"{args.param}_member{member:03d}_h{args.hour:03d}.png"
    plt.savefig(out, dpi=110, bbox_inches="tight")
    valid = np.nanmin(field), np.nanmax(field), np.nanmean(field)
    print(f"[SAVE] {out}")
    print(f"[STATS] min={valid[0]:.2f} max={valid[1]:.2f} mean={valid[2]:.2f}")


if __name__ == "__main__":
    main()

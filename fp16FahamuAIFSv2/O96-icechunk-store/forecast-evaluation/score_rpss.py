#!/usr/bin/env python3
"""Score an AI-WQ quintile forecast locally. Nothing is uploaded.

The forecast probabilities are ours; the observations, 20-yr quintile climatology and
land-sea mask come from the AI-WQ archive, and the score is computed by
``AI_WQ_package.forecast_evaluation`` -- the competition's own code, not a
reimplementation. Retrieval is read-only over FTP with ``AIWQ_PASSWORD``.

    RPSS = 1 - RPS_forecast / RPS_climatology

against a climatological forecast (uniform 1/5 per quintile). 0 = no better than
climatology, >0 = skill, <0 = worse than climatology. Lat-weighted global mean;
``tas`` and ``pr`` are land-only (the competition masks them), ``mslp`` is global.

Verification weeks are init+18 and init+25 days, so a cycle is only scoreable once both
weeks have passed. Observations are ERA5T weekly means on the 1.5 deg grid.

``--diagnostics`` adds the two checks that stop a bad number being reported as a finding:

  bias / rmse / pattern-r   is the forecast FIELD sound? A large bias or a units error
                            shows up here, and separates "poor skill" from "broken
                            pipeline". Note pattern-r on absolute values is high by
                            construction (poles cold, tropics warm) -- it rules out gross
                            errors, it is not evidence of skill.
  sharpness / hit-rate      WHY the score came out as it did. sharpness is the mean
                            probability given to the modal quintile; hit-rate is how
                            often the observation actually landed there. Both are 0.20
                            for a climatological forecast. sharpness >> hit-rate means
                            an over-confident (under-dispersive) ensemble, which RPS
                            punishes harder than saying 0.20 everywhere.

Usage::

    python score_rpss.py --fc .../ensemble_quintile_probabilities_20260514_v2.nc \
        --fc-date 20260514 --workdir /tmp/aiwq_eval --diagnostics
    # add --nc-dir to enable the field diagnostics (needs the per-member 1.5 deg files)
"""

import argparse
import datetime as dt
import glob
import json
import os
import sys

import numpy as np
import xarray as xr

# AI-WQ variable -> the name the quintile CLI writes into the submission file
VARMAP = {"tas": "2t_quintiles", "mslp": "msl_quintiles", "pr": "tp_quintiles"}
# and -> the raw field name in the per-member 1.5 deg NetCDFs
RAWVAR = {"tas": "2t", "mslp": "msl", "pr": "tp"}
# the quintile CLI's weekly aggregation: (label, time chunks, n steps)
WEEKS = [("week1", [0, 1], 24), ("week2", [2, 3, 4], 36)]


def verification_dates(fc_date):
    d0 = dt.datetime.strptime(fc_date, "%Y%m%d").date()
    return [(d0 + dt.timedelta(days=18)).strftime("%Y%m%d"),
            (d0 + dt.timedelta(days=25)).strftime("%Y%m%d")]


def lat_weights(lat):
    return np.cos(np.radians(np.asarray(lat)))[:, None]


def field_diagnostics(nc_dir, var, wi, obs):
    """Ensemble-mean weekly aggregate vs observation: bias, rmse, pattern correlation."""
    files = sorted(glob.glob(os.path.join(nc_dir, "*.nc")))
    if not files:
        return None
    _, chunks, nsteps = WEEKS[wi]
    raw = RAWVAR[var]
    ens = []
    for f in files:
        d = xr.open_dataset(f)
        s = sum(d[raw].isel(time=t, member=0).sum(dim="step", skipna=True) for t in chunks)
        ens.append((s * 1000 * (7.0 / (nsteps * 6 / 24)) if raw == "tp"
                    else s / nsteps).values)
        d.close()
    fc = np.mean(np.stack(ens), axis=0)
    o = np.asarray(obs.values)
    d = fc - o
    w = lat_weights(obs.latitude) * np.ones((1, d.shape[1]))
    m = np.isfinite(d)
    return dict(n_members=len(files),
                bias=float(np.sum(d[m] * w[m]) / np.sum(w[m])),
                rmse=float(np.sqrt(np.sum((d[m] ** 2) * w[m]) / np.sum(w[m]))),
                pattern_r=float(np.corrcoef(fc[m], o[m])[0, 1]))


def calibration(fc, obs_pbs):
    """sharpness (confidence in the modal quintile) and its hit rate."""
    good = np.isfinite(obs_pbs).all(axis=0) & np.isfinite(fc).all(axis=0)
    f, o = fc[:, good], obs_pbs[:, good].astype(float)
    modal = f.argmax(axis=0)
    sharp = float(f.max(axis=0).mean())
    hit = float(o[modal, np.arange(o.shape[1])].mean())
    return dict(sharpness=sharp, hit_rate=hit, reliability=hit - sharp)


def main():
    ap = argparse.ArgumentParser(description="local AI-WQ RPSS (no upload)")
    ap.add_argument("--fc", required=True, help="ensemble_quintile_probabilities_*.nc")
    ap.add_argument("--fc-date", required=True, help="init date YYYYMMDD")
    ap.add_argument("--workdir", required=True, help="cache for obs/clim/mask downloads")
    ap.add_argument("--nc-dir", default=None,
                    help="per-member 1.5 deg NetCDFs; enables the field diagnostics")
    ap.add_argument("--variables", default="tas,mslp,pr")
    ap.add_argument("--diagnostics", action="store_true")
    ap.add_argument("--json", default=None, help="also write results here")
    ap.add_argument("--env", default=os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), ".env"))
    args = ap.parse_args()

    from dotenv import load_dotenv
    load_dotenv(args.env)
    pw = os.environ.get("AIWQ_PASSWORD")
    if not pw:
        raise SystemExit(f"AIWQ_PASSWORD not found in {args.env}")
    from AI_WQ_package import retrieve_evaluation_data as R, forecast_evaluation as E

    os.makedirs(args.workdir, exist_ok=True)
    weeks = verification_dates(args.fc_date)
    today = dt.date.today().strftime("%Y%m%d")
    future = [w for w in weeks if w > today]
    if future:
        raise SystemExit(f"verification weeks {future} are in the future -- "
                         f"a cycle is scoreable only after init+25 days ({weeks[-1]})")
    variables = [v.strip() for v in args.variables.split(",") if v.strip()]
    print(f"init {args.fc_date} -> verification weeks {weeks}\n")

    ds = xr.open_dataset(args.fc)
    lsm = R.retrieve_land_sea_mask(pw, local_destination=args.workdir)
    rows = []

    for wi, wdate in enumerate(weeks):
        for var in variables:
            print(f"--- week{wi+1} {wdate}  {var} " + "-" * 28)
            obs = R.retrieve_weekly_obs(wdate, var, pw, local_destination=args.workdir)
            clim = R.retrieve_20yr_quantile_clim(
                wdate, var, pw, local_destination=args.workdir).squeeze(drop=True)
            obs_pbs = E.conditional_obs_probs(obs, clim).transpose(
                "quantile", "latitude", "longitude")

            fc = ds[VARMAP[var]].isel(time_week=wi)
            if "quintile" in fc.dims:
                fc = fc.rename({"quintile": "quantile"})
            fc = fc.transpose("quantile", "latitude", "longitude").assign_coords(
                quantile=np.arange(fc.sizes["quintile" if "quintile" in fc.dims
                                             else "quantile"]),
                latitude=obs.latitude, longitude=obs.longitude)

            rec = dict(week=f"week{wi+1}", valid=wdate, variable=var,
                       rpss=float(E.work_out_RPSS(fc, obs_pbs, var, lsm).values))
            if args.diagnostics:
                rec.update(calibration(fc.values, obs_pbs.values))
                if args.nc_dir:
                    fd = field_diagnostics(args.nc_dir, var, wi, obs)
                    if fd:
                        rec.update(fd)
            rows.append(rec)
            print()

    print("=" * 60)
    print(f"{'week':7s} {'valid':10s} {'variable':9s} {'RPSS':>9s}")
    print("-" * 60)
    for r in rows:
        print(f"{r['week']:7s} {r['valid']:10s} {r['variable']:9s} {r['rpss']:9.4f}")
    print("-" * 60)
    for v in variables:
        vals = [r["rpss"] for r in rows if r["variable"] == v]
        print(f"{'mean':7s} {'':10s} {v:9s} {np.mean(vals):9.4f}")
    print(f"\noverall mean RPSS: {np.mean([r['rpss'] for r in rows]):.4f}")

    if args.diagnostics:
        print(f"\n{'var':5s} {'week':6s} {'sharpness':>10s} {'hit-rate':>9s} "
              f"{'reliability':>12s} {'bias':>9s} {'rmse':>8s} {'pattern-r':>10s}")
        print("-" * 76)
        for r in rows:
            print(f"{r['variable']:5s} {r['week']:6s} {r['sharpness']:10.3f} "
                  f"{r['hit_rate']:9.3f} {r['reliability']:+12.3f} "
                  + (f"{r['bias']:+9.3f} {r['rmse']:8.3f} {r['pattern_r']:10.4f}"
                     if "bias" in r else f"{'-':>9s} {'-':>8s} {'-':>10s}"))
        print("\nclimatological baseline: sharpness = hit-rate = 0.200")
        print("reliability < 0 means over-confident: RPS punishes that below climatology")

    if args.json:
        with open(args.json, "w") as fh:
            json.dump(dict(fc_date=args.fc_date, weeks=weeks, results=rows), fh, indent=2)
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    sys.exit(main())

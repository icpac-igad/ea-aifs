#!/usr/bin/env python3
"""Submit the TS-days and MJO-phase products to AI Weather Quest.

`shared/forecast_submission_cli.py` only knows `tas`/`mslp`/`pr`. TS and MJO have
different shapes, different valid-time conventions and, for TS, a basin mask that
depends on the forecast month, so they go through the package API directly:

    AI_WQ_create_empty_dataarray(variable, date, period, team, model, password)
    -> fill ->
    AI_WQ_forecast_submission(...)

TS  -- (3, 4) = [below, near, above] x [ATL, NWP, SWIO, SEIO], **per forecast
       period** (1 = days 18-24, 2 = days 25-31), so two submissions.
MJO -- (9, 4) = [inactive, phases 1-8] x four valid times at init +7/14/21/28
       days. NOT split by period: one submission covers all four lags.

Tercile bounds come from AI-WQ's own published climatology,
`TS_20yrCLIM_WEEKLYTSDAYS_terciles_<validdate>.nc` on the FTP under
`/climatologies/<year>/` -- not from a self-derived climatology. Using the
official bounds is the only way the probabilities mean what the scoring assumes.

Inactive basins
---------------
`check_fc_submission.check_data_characteristics` builds a month-dependent mask:
months 6-11 activate ATL and NWP only, months 12/1/2 activate SWIO and SEIO only,
and **only the active columns are required to sum to 1**. The official bounds for
an inactive basin are 0/0, which under the binning rule would send all probability
to "above" -- a confident forecast of an out-of-season basin. So inactive columns
are filled with a uniform 1/3 instead: unscored either way, and it asserts nothing.

*** THE MJO PRODUCT IS KNOWN TO BE MISCALIBRATED ***
`ts-mjo/MJO_METHOD.md` section 5: forecast amplitudes run ~2x observed and
P(inactive) is 0.01-0.07 against an observed 0.37, because an *observed*
climatology is removed from a *model* field, leaving the model's mean-state bias
in the anomaly. It will score poorly, and category 0 will be starved. This script
submits what it is given; it does not fix that.
"""
from __future__ import annotations

import argparse
import datetime as dt
import os

import numpy as np
import xarray as xr

MJO_LAGS = (7, 14, 21, 28)
BASINS = ["ATL", "NWP", "SWIO", "SEIO"]


def load_env(path=".env"):
    for ln in open(path):
        if "=" in ln and not ln.strip().startswith("#"):
            k, v = ln.strip().split("=", 1)
            os.environ.setdefault(k, v)


def active_basins(month):
    """AI-WQ's own rule, mirrored from check_fc_submission."""
    return [j for j in range(4)
            if (month in (6, 7, 8, 9, 10, 11) and j <= 1)
            or (month in (12, 1, 2) and j >= 2)]


def tercile_probs(counts, lower, upper):
    """Same binning as ts_days.tercile_probs / AI-WQ's conditional_obs_probs.

        below : x <  lower
        near  : lower <= x < upper     (only when lower != upper)
        above : x >= upper
    """
    n = counts.size
    if lower == upper:
        return np.array([0.0, 0.0, 1.0])       # degenerate; caller overrides
    below = float((counts < lower).sum()) / n
    near = float(((counts >= lower) & (counts < upper)).sum()) / n
    above = float((counts >= upper).sum()) / n
    return np.array([below, near, above])


def build_ts(ts_nc, tercile_dir, period):
    """-> (3,4) for one forecast period, plus a text summary."""
    ds = xr.open_dataset(ts_nc)
    counts = ds["storm_days"].values                    # (member, week, basin)
    weeks = [str(w) for w in ds["week"].values]
    wk = period - 1                                     # period 1 -> first week
    valid = weeks[wk]
    f = os.path.join(tercile_dir,
                     f"TS_20yrCLIM_WEEKLYTSDAYS_terciles_{valid.replace('-','')}.nc")
    if not os.path.exists(f):
        raise SystemExit(f"missing official terciles {f}")
    tb = xr.open_dataset(f)
    bounds = np.asarray(list(tb.data_vars.values())[0].values)   # (2, basin)
    order = [list(tb.basin.values).index(b) for b in BASINS]
    bounds = bounds[:, order]

    act = active_basins(dt.date.fromisoformat(valid).month)
    out = np.full((3, 4), 1.0 / 3.0)
    lines = []
    for j, b in enumerate(BASINS):
        lo, up = float(bounds[0, j]), float(bounds[1, j])
        c = counts[:, wk, j]
        if j in act:
            out[:, j] = tercile_probs(c, lo, up)
            tag = ""
        else:
            tag = "  [inactive -> uniform 1/3]"
        lines.append(f"    {b:5s} bounds {lo:4.0f}/{up:4.0f}  mean {c.mean():5.2f} days  "
                     f"P(below,near,above) = {out[0,j]:.2f},{out[1,j]:.2f},{out[2,j]:.2f}{tag}")
    return out, valid, act, lines


def build_mjo(mjo_nc):
    """-> (9,4) at init +7/14/21/28 days."""
    ds = xr.open_dataset(mjo_nc)
    P = ds["MJO_phase_probability"].values              # (day, 9)
    days = [str(d) for d in ds["day"].values]
    if max(MJO_LAGS) >= P.shape[0]:
        raise SystemExit(f"product has {P.shape[0]} days, needs > {max(MJO_LAGS)}")
    arr = P[list(MJO_LAGS), :].T
    if not np.allclose(arr.sum(axis=0), 1.0, atol=1e-6):
        raise SystemExit(f"columns do not sum to 1: {arr.sum(axis=0)}")
    return arr, [days[i][:10] for i in MJO_LAGS]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--date", required=True, help="init YYYYMMDD")
    ap.add_argument("--ts-nc")
    ap.add_argument("--tercile-dir")
    ap.add_argument("--mjo-nc")
    ap.add_argument("--variables", nargs="+", default=["TS", "MJO"],
                    choices=["TS", "MJO"])
    ap.add_argument("--dry-run", action="store_true",
                    help="build and validate the arrays, print them, submit nothing")
    ap.add_argument("--env-file", default=".env")
    args = ap.parse_args()

    load_env(args.env_file)
    team = os.environ["AIWQ_TEAM_NAME"]
    model = os.environ.get("AIWQ_MODEL_NAME_V2") or os.environ["AIWQ_MODEL_NAME_FP16"]
    # SUBMISSION goes via ECBox and needs the ecbox TOKEN. Retrieval goes over FTP
    # and needs AIWQ_PASSWORD. The package takes ONE `password` argument for both
    # (`ftp_or_ecbox_loading` uses it as the FTP password, then as a Branca token),
    # so passing the wrong one fails with `403 "Not a Branca token"` and
    # "Could not list files from location '/forecast_submissions'" -- an auth error
    # that reads like a path error. Same precedence as shared/forecast_submission_cli.py.
    pw = (os.environ.get("AIWQ_ECBOX_TOKEN") or os.environ.get("ecbox")
          or os.environ["AIWQ_PASSWORD"])
    ftp_pw = os.environ["AIWQ_PASSWORD"]
    from AI_WQ_package import forecast_submission as FS
    print(f"team={team}  model={model}  init={args.date}  "
          f"cred={'ecbox token' if pw != ftp_pw else 'AIWQ_PASSWORD (fallback)'}"
          + ("   [DRY RUN]" if args.dry_run else ""))

    jobs = []
    if "TS" in args.variables:
        if not (args.ts_nc and args.tercile_dir):
            raise SystemExit("TS needs --ts-nc and --tercile-dir")
        for period in (1, 2):
            arr, valid, act, lines = build_ts(args.ts_nc, args.tercile_dir, period)
            print(f"\nTS period {period}  (days {18 if period==1 else 25}-"
                  f"{24 if period==1 else 31}, valid week {valid})")
            print("\n".join(lines))
            s = arr[:, act].sum(axis=0)
            print(f"    active columns {[BASINS[j] for j in act]} sum to {np.round(s,6)}")
            jobs.append(("TS", str(period), arr))
    if "MJO" in args.variables:
        if not args.mjo_nc:
            raise SystemExit("MJO needs --mjo-nc")
        arr, vd = build_mjo(args.mjo_nc)
        print(f"\nMJO  valid {', '.join(vd)}  (init +7/14/21/28)")
        print(f"    modal category per lag: {[int(np.argmax(arr[:,j])) for j in range(4)]}")
        print(f"    P(inactive):            {np.round(arr[0],3)}")
        print(f"    !! miscalibrated -- see MJO_METHOD.md section 5")
        jobs.append(("MJO", "1", arr))

    if args.dry_run:
        print("\nDRY RUN: nothing submitted.")
        return

    ok = 0
    for var, period, arr in jobs:
        try:
            da = FS.AI_WQ_create_empty_dataarray(var, args.date, period, team, model, pw)
            da[:] = arr
            FS.AI_WQ_forecast_submission(da, var, args.date, period, team, model, pw)
            print(f"  submitted {var} period {period}")
            ok += 1
        except Exception as e:
            print(f"  FAILED {var} period {period}: {e}")
    print(f"\n{ok}/{len(jobs)} submitted")


if __name__ == "__main__":
    main()

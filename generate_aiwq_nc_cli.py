#!/usr/bin/env python3
"""
Generate AI Weather Quest submission-ready NetCDF files (no upload).

Reads the ensemble_quintile_probabilities_<DATE>_fp16.nc file, and for each
variable in {tas, mslp, pr} and each week in {1, 2}, builds the empty
DataArray via AI_WQ_package.forecast_submission.AI_WQ_create_empty_dataarray,
populates its values with the quintile probabilities and writes the file as:

    {save_dir}/{var}_{date}_p{week}_{teamname}_{modelname}.nc

Credentials are read from .env: AIWQ_TEAM_NAME, AIWQ_MODEL_NAME_FP16, AIWQ_PASSWORD.

Run with:
    micromamba run -n zarrv3 python generate_aiwq_nc_cli.py \\
        --input /home/roller/Desktop/ensemble_quintile_probabilities_20260514_fp16.nc \\
        --date 20260514 \\
        --save-dir /home/roller/Desktop/aiwq_submissions/
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import xarray as xr
from dotenv import load_dotenv

try:
    from AI_WQ_package import forecast_submission
except ImportError:
    sys.stderr.write(
        "AI_WQ_package not installed. Install with:\n"
        "    micromamba run -n zarrv3 pip install AI_WQ_package\n"
    )
    raise


VAR_MAPPING = {
    "tas": "2t_quintiles",
    "mslp": "msl_quintiles",
    "pr": "tp_quintiles",
}


def get_credentials():
    team = os.getenv("AIWQ_TEAM_NAME")
    model = os.getenv("AIWQ_MODEL_NAME_FP16")
    password = os.getenv("AIWQ_PASSWORD")
    missing = [k for k, v in {
        "AIWQ_TEAM_NAME": team,
        "AIWQ_MODEL_NAME_FP16": model,
        "AIWQ_PASSWORD": password,
    }.items() if not v]
    if missing:
        raise SystemExit(f"Missing env vars in .env: {', '.join(missing)}")
    return team, model, password


def extract_values(ds: xr.Dataset, variable: str, week_period: str) -> np.ndarray:
    var_name = VAR_MAPPING[variable]
    week_name = f"week{week_period}"
    week_mask = ds.week == week_name
    idx = int(np.where(week_mask)[0][0])
    data = ds[var_name].isel(time_week=idx).values
    assert data.shape == (5, 121, 240), f"unexpected shape {data.shape} for {variable} {week_name}"
    assert np.all((data >= 0) & (data <= 1)), f"values outside [0,1] for {variable} {week_name}"
    return data


def build_and_save(ds: xr.Dataset, variable: str, week: str, date_str: str,
                   teamname: str, modelname: str, password: str, save_dir: Path) -> Path:
    values = extract_values(ds, variable, week)

    empty_da = forecast_submission.AI_WQ_create_empty_dataarray(
        variable, date_str, week, teamname, modelname, password
    )
    empty_da.values = values

    out_path = save_dir / f"{variable}_{date_str}_p{week}_{teamname}_{modelname}.nc"
    empty_da.to_netcdf(out_path)
    return out_path


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", required=True,
                        help="Path to ensemble_quintile_probabilities_<DATE>_fp16.nc")
    parser.add_argument("--date", required=True, help="Forecast start date YYYYMMDD")
    parser.add_argument("--save-dir", default="./aiwq_submissions",
                        help="Directory to write submission-ready NC files")
    parser.add_argument("--env-file", default=".env", help="Path to .env file")
    parser.add_argument("--variables", nargs="+",
                        default=["tas", "mslp", "pr"],
                        choices=["tas", "mslp", "pr"])
    parser.add_argument("--weeks", nargs="+", default=["1", "2"], choices=["1", "2"])
    args = parser.parse_args()

    if os.path.exists(args.env_file):
        load_dotenv(args.env_file)

    teamname, modelname, password = get_credentials()
    print(f"Team: {teamname}  Model: {modelname}  Date: {args.date}")

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    ds = xr.open_dataset(args.input)

    written = []
    for variable in args.variables:
        for week in args.weeks:
            print(f"Building {variable} week{week} ...")
            try:
                out = build_and_save(ds, variable, week, args.date,
                                     teamname, modelname, password, save_dir)
                print(f"  -> {out}")
                written.append(out)
            except Exception as e:
                print(f"  FAILED {variable} week{week}: {e}")

    print(f"\nDone. {len(written)} files written to {save_dir}")
    print("All the best with GHACOF.")


if __name__ == "__main__":
    main()

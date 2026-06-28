#!/usr/bin/env python3
"""
Generate AI Weather Quest individual per-variable/week NetCDF files and zip them.

Same prep as the live submission path, but instead of submitting it SAVES each
populated DataArray to an individual file following the AI-WQ structure:

    {variable}_{date}_p{week}_{teamname}_{modelname}.nc

then bundles all of them into a single .zip. Useful for archival or for a
manual / out-of-window upload when the live submission endpoint is closed.

Pattern (per AI_WQ_package instructions):
    from AI_WQ_package import forecast_submission
    fc = forecast_submission.AI_WQ_create_empty_dataarray('tas','20260611','1',team,model,pw)
    fc.values = fc_pbs
    fc.to_netcdf(f"{save_dir}tas_20260611_p1_{team}_{model}.nc")

Usage:
    python aiwq_individual_files_cli.py --date 20260611 --fp16
    python aiwq_individual_files_cli.py --date 20260611 --fp16 --variables tas --weeks 1
"""
import os
import sys
import argparse
import zipfile
from pathlib import Path

from AI_WQ_package import forecast_submission

# Reuse the exact prep + credential logic the submission path uses.
from ensemble_quintile_analysis_cli import prepare_aiwq_submission
from forecast_submission_cli import get_credentials


def main() -> int:
    ap = argparse.ArgumentParser(description="Build + zip AI-WQ individual forecast files")
    ap.add_argument('--date', required=True, help='Forecast start date YYYYMMDD, e.g. 20260611')
    ap.add_argument('--fp16', action='store_true', default=False,
                    help='Use FP16 credentials + default quintile filename')
    ap.add_argument('--quintile-file', default=None,
                    help='Quintile probabilities .nc (default: ./ensemble_quintile_probabilities_<date>[_fp16].nc)')
    ap.add_argument('--variables', nargs='+', default=['tas', 'pr', 'mslp'],
                    choices=['tas', 'pr', 'mslp'])
    ap.add_argument('--weeks', nargs='+', default=['1', '2'])
    ap.add_argument('--save-dir', default=None,
                    help='Dir for the individual .nc files (default ./aiwq_individual_<date>/)')
    ap.add_argument('--zip', dest='zip_path', default=None,
                    help='Output zip path (default ./aiwq_submission_<date>_<team>_<model>.zip)')
    args = ap.parse_args()

    team, model, password = get_credentials(fp16=args.fp16)

    suffix = '_fp16' if args.fp16 else ''
    qfile = args.quintile_file or f"./ensemble_quintile_probabilities_{args.date}{suffix}.nc"
    if not os.path.exists(qfile):
        print(f"ERROR: quintile file not found: {qfile}")
        return 1

    save_dir = Path(args.save_dir or f"./aiwq_individual_{args.date}")
    save_dir.mkdir(parents=True, exist_ok=True)

    print(f"Team: {team} | Model: {model} | Date: {args.date}")
    print(f"Quintile file: {qfile}")
    print(f"Save dir: {save_dir}")
    print("-" * 60)

    created = []
    for variable in args.variables:
        for week in args.weeks:
            arr = prepare_aiwq_submission(qfile, variable, week)
            da = forecast_submission.AI_WQ_create_empty_dataarray(
                variable, args.date, week, team, model, password
            )
            da.values = arr
            fname = save_dir / f"{variable}_{args.date}_p{week}_{team}_{model}.nc"
            da.to_netcdf(fname)
            created.append(fname)
            print(f"  ✅ {fname.name}  shape={arr.shape}")

    if not created:
        print("No files created.")
        return 1

    zip_path = Path(args.zip_path or f"./aiwq_submission_{args.date}_{team}_{model}.zip")
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
        for f in created:
            zf.write(f, arcname=f.name)

    print("-" * 60)
    print(f"📦 Zipped {len(created)} files → {zip_path}  ({zip_path.stat().st_size/1024:.1f} KB)")
    return 0


if __name__ == '__main__':
    sys.exit(main())

#!/usr/bin/env python3
"""
Manual climatology downloader - bypasses AI_WQ_package to download directly via FTP.
Useful if the main script's FTP calls fail.

Usage:
    python download_climatology.py --date 20260625 --output-dir ./
"""

import argparse
import ftplib
import os
from datetime import datetime, timedelta
from pathlib import Path

def valid_dates(forecast_date: str):
    """Calculate valid dates for a given forecast date."""
    date_obj = datetime.strptime(forecast_date, "%Y%m%d")
    fc_valid_date_obj1 = date_obj + timedelta(days=4+(7*2))
    fc_valid_date_obj2 = date_obj + timedelta(days=4+(7*3))
    fc_valid_date1 = fc_valid_date_obj1.strftime("%Y%m%d")
    fc_valid_date2 = fc_valid_date_obj2.strftime("%Y%m%d")
    return fc_valid_date1, fc_valid_date2

def download_climatology_file(date_str, variable, password, output_dir="./"):
    """Download a single climatology file via FTP."""
    if variable == 'tas' or variable == 'mslp':
        weekly_agg_str = 'WEEKLYMEAN'
    elif variable == 'pr':
        weekly_agg_str = 'WEEKLYSUM'
    else:
        raise ValueError(f"Unknown variable: {variable}")

    filename = f'{variable}_20yrCLIM_{weekly_agg_str}_quintiles_{date_str}.nc'
    local_path = os.path.join(output_dir, filename)
    remote_path = f'/climatologies/{date_str[:4]}/{filename}'

    # Skip if file already exists
    if os.path.exists(local_path):
        size = os.path.getsize(local_path) / 1024
        print(f"✓ {filename} already exists ({size:.1f} KB)")
        return True

    try:
        print(f"  Downloading {filename}...", end="", flush=True)
        session = ftplib.FTP('ftp.ecmwf.int', 'ai_weather_quest', password, timeout=30)

        with open(local_path, 'wb') as f:
            session.retrbinary(f"RETR {remote_path}", f.write)

        session.quit()
        size = os.path.getsize(local_path) / 1024
        print(f" ✓ ({size:.1f} KB)")
        return True
    except Exception as e:
        print(f" ✗ Error: {e}")
        if os.path.exists(local_path):
            os.remove(local_path)
        return False

def main():
    parser = argparse.ArgumentParser(
        description="Manually download climatology files for ensemble quintile analysis"
    )
    parser.add_argument('--date', required=True,
                       help='Forecast date (YYYYMMDD format)')
    parser.add_argument('--output-dir', default='./',
                       help='Output directory for climatology files')
    parser.add_argument('--password', default=None,
                       help='ECMWF FTP password (reads from .env if not provided)')

    args = parser.parse_args()

    # Load password from .env if not provided
    if args.password is None:
        if os.path.exists('.env'):
            with open('.env', 'r') as f:
                for line in f:
                    if 'AIWQ_PASSWORD=' in line:
                        args.password = line.split('=')[1].strip()
                        break

    if not args.password:
        print("Error: No password provided and AIWQ_PASSWORD not found in .env")
        return 1

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Calculate valid dates
    fc_valid_date1, fc_valid_date2 = valid_dates(args.date)
    print(f"📅 Forecast date: {args.date}")
    print(f"   Valid dates: {fc_valid_date1}, {fc_valid_date2}\n")

    variables = ['tas', 'mslp', 'pr']
    dates = [fc_valid_date1, fc_valid_date2]

    success_count = 0
    total_files = len(variables) * len(dates)

    print(f"📥 Downloading {total_files} climatology files...\n")

    for variable in variables:
        print(f"  {variable.upper()}:")
        for date_str in dates:
            if download_climatology_file(date_str, variable, args.password, args.output_dir):
                success_count += 1

    print(f"\n✅ Downloaded {success_count}/{total_files} files")

    if success_count == total_files:
        print(f"\n🎉 All climatology files ready!")
        print(f"   Now run: python ensemble_quintile_analysis_cli.py --date {args.date} --v2 --skip-ensemble")
        return 0
    else:
        print(f"\n⚠️  {total_files - success_count} files failed to download")
        return 1

if __name__ == "__main__":
    exit(main())

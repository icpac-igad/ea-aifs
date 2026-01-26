#!/usr/bin/env python3
"""
Ensemble Quintile Analysis - CLI version with configurable date and FP16 support
================================================================================

Usage:
    # FP32 mode (default): reads from 1p5deg_nc/
    python ensemble_quintile_analysis_cli.py --date 20251127

    # FP16 mode: reads from fp16_1p5deg_nc/
    python ensemble_quintile_analysis_cli.py --date 20251127 --fp16

GCS Path Structure:
    FP32: gs://bucket/{date}_0000/1p5deg_nc/
    FP16: gs://bucket/{date}_0000/fp16_1p5deg_nc/
"""

from datetime import datetime, timedelta
import os
import re
import gc
import argparse
import numpy as np
import xarray as xr
import pandas as pd
import tempfile
import shutil
from pathlib import Path
from typing import List, Optional
from google.cloud import storage
from google.oauth2 import service_account

def parse_member_range(member_str: str) -> List[int]:
    """Parse member range string like '1-50' or '1,2,3' into list of integers."""
    members = []
    if '-' in member_str:
        start, end = map(int, member_str.split('-'))
        members = list(range(start, end + 1))
    elif ',' in member_str:
        members = [int(m.strip()) for m in member_str.split(',')]
    else:
        members = [int(member_str)]
    return members

# Try to import AI_WQ_package (may not be available in all environments)
try:
    from AI_WQ_package import retrieve_evaluation_data
    from AI_WQ_package import forecast_submission
    AIWQ_AVAILABLE = True
except ImportError:
    AIWQ_AVAILABLE = False
    print("Warning: AI_WQ_package not available. Submission functions will not work.")

# Try to import icechunk (optional)
try:
    import icechunk
    ICECHUNK_AVAILABLE = True
except ImportError:
    ICECHUNK_AVAILABLE = False


def valid_dates(forecast_date: str):
    """
    Calculate valid dates for a given forecast date.

    Args:
        forecast_date: Forecast date in YYYYMMDD format

    Returns:
        tuple: (fc_valid_date1, fc_valid_date2) for week 1 and week 2
    """
    date_obj = datetime.strptime(forecast_date, "%Y%m%d")
    fc_valid_date_obj1 = date_obj + timedelta(days=4+(7*2))
    fc_valid_date_obj2 = date_obj + timedelta(days=4+(7*3))
    fc_valid_date1 = fc_valid_date_obj1.strftime("%Y%m%d")
    fc_valid_date2 = fc_valid_date_obj2.strftime("%Y%m%d")
    return fc_valid_date1, fc_valid_date2


def get_quintile_clim(forecast_date: str, variable: str, password: Optional[str] = None):
    """Retrieve quintile climatology for a given forecast date and variable."""
    if not AIWQ_AVAILABLE:
        raise RuntimeError("AI_WQ_package is required for climatology retrieval")

    if password is None:
        password = os.getenv('AIWQ_SUBMIT_PWD')

    fc_valid_date1, fc_valid_date2 = valid_dates(forecast_date)

    clim1 = retrieve_evaluation_data.retrieve_20yr_quintile_clim(
        fc_valid_date1, variable, password=password
    )
    clim2 = retrieve_evaluation_data.retrieve_20yr_quintile_clim(
        fc_valid_date2, variable, password=password
    )

    return clim1, clim2


def download_all_quintiles(forecast_date: str, variables: Optional[List[str]] = None,
                          password: Optional[str] = None):
    """Download quintile climatologies for multiple variables."""
    if variables is None:
        variables = ['tas', 'mslp', 'pr']

    if password is None:
        password = os.getenv('AIWQ_SUBMIT_PWD')

    fc_valid_date1, fc_valid_date2 = valid_dates(forecast_date)

    quintile_data = {}

    for variable in variables:
        print(f"Downloading quintile climatology for {variable}...")
        try:
            clim1, clim2 = get_quintile_clim(forecast_date, variable, password)
            quintile_data[variable] = {
                fc_valid_date1: clim1,
                fc_valid_date2: clim2
            }
            print(f"✓ Successfully downloaded {variable} climatology")
        except Exception as e:
            print(f"✗ Error downloading {variable}: {e}")
            quintile_data[variable] = None

    return quintile_data


def download_ensemble_nc_from_gcs(
    forecast_date: str,
    members: Optional[List[int]] = None,
    gcs_bucket: str = "aifs-aiquest-us-20251127",
    gcs_prefix: Optional[str] = None,
    service_account_path: str = "coiled-data.json",
    local_dir: str = "./ensemble_nc_files",
    skip_download_if_exists: bool = True,
    fp16: bool = False
):
    """
    Download ensemble NetCDF files from GCS and combine into a single dataset.

    Args:
        forecast_date: Forecast date in YYYYMMDD format
        members: List of member numbers. If None, downloads all available
        gcs_bucket: GCS bucket name
        gcs_prefix: GCS prefix. If None, auto-generated based on date and fp16 flag
        service_account_path: Path to GCS service account key
        local_dir: Local directory for downloads
        skip_download_if_exists: Skip downloading existing files
        fp16: If True, use fp16_1p5deg_nc/ path instead of 1p5deg_nc/

    Returns:
        xr.Dataset: Combined ensemble dataset
    """
    # Set GCS prefix based on fp16 flag
    if gcs_prefix is None:
        if fp16:
            gcs_prefix = f"{forecast_date}_0000/fp16_1p5deg_nc/"
        else:
            gcs_prefix = f"{forecast_date}_0000/1p5deg_nc/"

    mode_label = "FP16" if fp16 else "FP32"

    print(f"📥 Downloading ensemble NetCDF files from GCS ({mode_label})")
    print(f"   Bucket: gs://{gcs_bucket}/{gcs_prefix}")
    print(f"   Forecast date: {forecast_date}")

    try:
        # Initialize GCS client
        credentials = service_account.Credentials.from_service_account_file(service_account_path)
        client = storage.Client(credentials=credentials)
        bucket = client.bucket(gcs_bucket)

        # Create local directory
        os.makedirs(local_dir, exist_ok=True)

        # List available NetCDF files
        print(f"   🔍 Scanning for available NetCDF files...")
        blobs = bucket.list_blobs(prefix=gcs_prefix)

        available_files = []
        for blob in blobs:
            if blob.name.endswith('.nc'):
                filename = os.path.basename(blob.name)
                match = re.search(r'member(\d+)\.nc', filename)
                if match:
                    member_num = int(match.group(1))
                    if members is None or member_num in members:
                        available_files.append({
                            'blob_name': blob.name,
                            'filename': filename,
                            'member': member_num
                        })

        if not available_files:
            print(f"   ❌ No NetCDF files found in {gcs_prefix}")
            return None

        available_files.sort(key=lambda x: x['member'])
        print(f"   ✅ Found {len(available_files)} NetCDF files")

        # Download files
        member_datasets = []
        all_local_files = []

        for i, file_info in enumerate(available_files):
            local_path = os.path.join(local_dir, file_info['filename'])

            if os.path.exists(local_path) and skip_download_if_exists:
                print(f"   ✓ Already exists: {file_info['filename']}")
            else:
                print(f"   📥 Downloading {file_info['filename']} ({i+1}/{len(available_files)})")
                blob = bucket.blob(file_info['blob_name'])
                blob.download_to_filename(local_path)

            all_local_files.append({
                'local_path': local_path,
                'member': file_info['member'],
                'filename': file_info['filename']
            })

        # Load datasets
        for file_info in all_local_files:
            try:
                ds = xr.open_dataset(file_info['local_path'])
                member_datasets.append(ds)
                print(f"      ✅ Loaded member {file_info['member']:03d}")
            except Exception as e:
                print(f"      ❌ Error loading {file_info['filename']}: {e}")

        if not member_datasets:
            print(f"   ❌ No datasets could be loaded")
            return None

        # Combine datasets
        print(f"   🔗 Combining {len(member_datasets)} members...")
        ensemble_ds = xr.concat(member_datasets, dim='member')

        ensemble_ds.attrs.update({
            'title': f'AIFS Ensemble Forecast ({mode_label})',
            'description': f'Combined ensemble forecast with {len(member_datasets)} members',
            'forecast_date': forecast_date,
            'precision': mode_label,
            'members': f'{all_local_files[0]["member"]}-{all_local_files[-1]["member"]}',
            'processing_date': str(np.datetime64('now'))
        })

        print(f"   ✅ Combined dataset: {len(member_datasets)} members, {list(ensemble_ds.data_vars)}")
        return ensemble_ds

    except Exception as e:
        print(f"❌ Error downloading from GCS: {e}")
        import traceback
        traceback.print_exc()
        return None


def calculate_grid_quintiles(ensemble_data, climatology_quintiles):
    """Calculate quintile probabilities for each grid point (vectorized)."""
    print(f"      🚀 Using VECTORIZED calculation...")

    if not isinstance(climatology_quintiles, xr.DataArray):
        raise TypeError(f"Expected xr.DataArray, got {type(climatology_quintiles)}")

    n_members = ensemble_data.sizes['member']

    if 'time' in climatology_quintiles.dims:
        clim_data = climatology_quintiles.isel(time=0)
    else:
        clim_data = climatology_quintiles

    ensemble_values = ensemble_data.values
    clim_thresholds = clim_data.values

    quintile_probs = np.zeros((5, ensemble_data.sizes['latitude'], ensemble_data.sizes['longitude']))

    # Vectorized quintile calculation
    quintile_probs[0] = np.sum(ensemble_values < clim_thresholds[0], axis=0) / n_members
    quintile_probs[1] = np.sum((ensemble_values >= clim_thresholds[0]) &
                               (ensemble_values < clim_thresholds[1]), axis=0) / n_members
    quintile_probs[2] = np.sum((ensemble_values >= clim_thresholds[1]) &
                               (ensemble_values < clim_thresholds[2]), axis=0) / n_members
    quintile_probs[3] = np.sum((ensemble_values >= clim_thresholds[2]) &
                               (ensemble_values < clim_thresholds[3]), axis=0) / n_members
    quintile_probs[4] = np.sum(ensemble_values >= clim_thresholds[3], axis=0) / n_members

    print(f"      ✅ Vectorized calculation complete!")

    quintile_da = xr.DataArray(
        quintile_probs,
        dims=['quintile', 'latitude', 'longitude'],
        coords={
            'quintile': [0.2, 0.4, 0.6, 0.8, 1.0],
            'latitude': ensemble_data.latitude,
            'longitude': ensemble_data.longitude
        },
        attrs={
            'long_name': 'Quintile probability',
            'units': 'probability'
        }
    )

    return quintile_da


def calculate_ensemble_quintiles(forecast_ds, forecast_date: str,
                                climatology_base_path: str = "./",
                                variable_mapping: Optional[dict] = None):
    """Calculate quintile probabilities for ensemble forecasts."""
    if variable_mapping is None:
        variable_mapping = {
            'msl': 'mslp',
            'tp': 'pr',
            '2t': 'tas'
        }

    fc_valid_date1, fc_valid_date2 = valid_dates(forecast_date)
    print(f"📅 Using climatology for valid dates: {fc_valid_date1}, {fc_valid_date2}")

    climatology_files = {
        'mslp': {
            'week1': f"{climatology_base_path}/mslp_20yrCLIM_WEEKLYMEAN_quintiles_{fc_valid_date1}.nc",
            'week2': f"{climatology_base_path}/mslp_20yrCLIM_WEEKLYMEAN_quintiles_{fc_valid_date2}.nc"
        },
        'pr': {
            'week1': f"{climatology_base_path}/pr_20yrCLIM_WEEKLYSUM_quintiles_{fc_valid_date1}.nc",
            'week2': f"{climatology_base_path}/pr_20yrCLIM_WEEKLYSUM_quintiles_{fc_valid_date2}.nc"
        },
        'tas': {
            'week1': f"{climatology_base_path}/tas_20yrCLIM_WEEKLYMEAN_quintiles_{fc_valid_date1}.nc",
            'week2': f"{climatology_base_path}/tas_20yrCLIM_WEEKLYMEAN_quintiles_{fc_valid_date2}.nc"
        }
    }

    quintile_results = {}
    quintile_bounds = [0.2, 0.4, 0.6, 0.8, 1.0]

    for var_name in forecast_ds.data_vars:
        if var_name not in variable_mapping:
            print(f"Warning: No climatology mapping for {var_name}")
            continue

        clim_var = variable_mapping[var_name]
        print(f"Processing {var_name} -> {clim_var}")

        if clim_var not in climatology_files:
            print(f"Warning: No climatology files for {clim_var}")
            continue

        var_quintiles = []

        week_configs = [
            ("week1", [0, 1]),
            ("week2", [2, 3, 4])
        ]

        for week_name, time_chunks in week_configs:
            print(f"  Processing {week_name} using time chunks {time_chunks}")

            clim_file = climatology_files[clim_var][week_name]

            try:
                if not os.path.exists(clim_file):
                    print(f"    Warning: Climatology file not found: {clim_file}")
                    continue

                clim_ds = xr.open_dataset(clim_file)

                if clim_var not in clim_ds.data_vars:
                    print(f"    Warning: {clim_var} not found in {clim_file}")
                    continue

                clim_quintiles = clim_ds[clim_var]
                print(f"    Loaded climatology: {clim_quintiles.shape}")

                # Aggregate forecast data
                chunk_sums = []
                total_steps = 0
                for t_idx in time_chunks:
                    if t_idx < len(forecast_ds.time):
                        chunk = forecast_ds[var_name].isel(time=t_idx)
                        chunk_sum = chunk.sum(dim='step', skipna=True)
                        chunk_sums.append(chunk_sum)
                        total_steps += 12
                        print(f"      Added time chunk {t_idx}: 12 steps")

                if not chunk_sums:
                    print(f"    Warning: No valid data for {week_name}")
                    continue

                forecast_sum = sum(chunk_sums)
                actual_days = total_steps * 6 / 24

                if var_name == 'tp':
                    scaling_factor = 7.0 / actual_days
                    weekly_forecast = forecast_sum * 1000 * scaling_factor
                    print(f"    Weekly precipitation (scaled to 7 days)")
                else:
                    weekly_forecast = forecast_sum / total_steps
                    print(f"    Weekly mean (sum / {total_steps} steps)")

                quintile_probs = calculate_grid_quintiles(weekly_forecast, clim_quintiles)

                quintile_probs = quintile_probs.assign_coords({
                    'time': 0,
                    'week': week_name,
                    'quintile': quintile_bounds
                })

                var_quintiles.append(quintile_probs)
                print(f"    ✓ Completed {week_name}")

            except Exception as e:
                print(f"    Error processing {week_name}: {e}")
                import traceback
                traceback.print_exc()

        if var_quintiles:
            var_combined = xr.concat(var_quintiles, dim='time_week')
            quintile_results[f"{var_name}_quintiles"] = var_combined
            print(f"✓ Completed {var_name}")

    if quintile_results:
        result_ds = xr.Dataset(quintile_results)
        result_ds.attrs.update({
            'title': 'Ensemble Forecast Quintile Probabilities',
            'processing_date': str(np.datetime64('now'))
        })
        return result_ds
    else:
        print("No quintile data calculated")
        return None


def prepare_aiwq_submission(quintile_file: str, variable: str, week_period: str,
                           time_index: int = 0):
    """Prepare quintile data for AI Weather Quest submission."""
    ds = xr.open_dataset(quintile_file)

    var_mapping = {
        'tas': '2t_quintiles',
        'mslp': 'msl_quintiles',
        'pr': 'tp_quintiles'
    }

    if variable not in var_mapping:
        raise ValueError(f"Variable must be one of: {list(var_mapping.keys())}")

    var_name = var_mapping[variable]
    week_name = f'week{week_period}'
    week_mask = ds.week == week_name

    data = ds[var_name].isel(time_week=np.where(week_mask)[0][time_index])
    submission_data = data.values

    assert submission_data.shape == (5, 121, 240), f"Wrong shape: {submission_data.shape}"
    assert np.all((submission_data >= 0) & (submission_data <= 1)), "Values must be 0-1"

    return submission_data


def submit_forecast(quintile_file: str, variable: str, fc_start_date: str,
                   fc_period: str, teamname: str, modelname: str, password: str,
                   time_index: int = 0):
    """Complete submission workflow."""
    if not AIWQ_AVAILABLE:
        raise RuntimeError("AI_WQ_package required for submission")

    forecast_array = prepare_aiwq_submission(quintile_file, variable, fc_period, time_index)

    empty_da = forecast_submission.AI_WQ_create_empty_dataarray(
        variable, fc_start_date, fc_period, teamname, modelname, password
    )

    empty_da.values = forecast_array

    submitted_da = forecast_submission.AI_WQ_forecast_submission(
        empty_da, variable, fc_start_date, fc_period, teamname, modelname, password
    )

    return submitted_da


def main():
    parser = argparse.ArgumentParser(
        description="Ensemble Quintile Analysis with CLI arguments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
GCS Path Structure:
    FP32 (default): gs://bucket/{date}_0000/1p5deg_nc/
    FP16 (--fp16):  gs://bucket/{date}_0000/fp16_1p5deg_nc/

Examples:
    # Process FP32 forecasts
    python ensemble_quintile_analysis_cli.py --date 20251127

    # Process FP16 forecasts
    python ensemble_quintile_analysis_cli.py --date 20251127 --fp16

    # Skip redownloading existing files
    python ensemble_quintile_analysis_cli.py --date 20251127 --skip-existing
        """
    )

    parser.add_argument('--date', required=True,
                       help='Forecast date (YYYYMMDD format)')
    parser.add_argument('--members', default=None,
                       help='Member range (e.g., 1-50, 1,2,3). If not specified, downloads all available.')
    parser.add_argument('--fp16', action='store_true',
                       help='Use FP16 paths (fp16_1p5deg_nc/)')
    parser.add_argument('--skip-existing', action='store_true', default=True,
                       help='Skip downloading existing files (default: True)')
    parser.add_argument('--output-dir', default='./',
                       help='Output directory for quintile file')
    parser.add_argument('--clim-dir', default='./',
                       help='Directory containing climatology files')
    parser.add_argument('--bucket', default='aifs-aiquest-us-20251127',
                       help='GCS bucket name')
    parser.add_argument('--service-account', default='coiled-data.json',
                       help='Path to GCS service account key')

    args = parser.parse_args()

    # Clean date format (remove _0000 if present)
    forecast_date = args.date.split('_')[0] if '_' in args.date else args.date
    mode_label = "FP16" if args.fp16 else "FP32"

    # Parse members if specified
    members = None
    if args.members:
        try:
            members = parse_member_range(args.members)
            print(f"Processing {len(members)} members: {members[0]}-{members[-1]}")
        except ValueError as e:
            print(f"ERROR: Invalid member range: {e}")
            return 1

    print("=" * 60)
    print(f"AIFS Ensemble Quintile Analysis Pipeline ({mode_label})")
    print("=" * 60)
    print(f"Forecast date: {forecast_date}")
    print(f"Precision mode: {mode_label}")
    print(f"Members: {args.members if args.members else 'all available'}")
    print(f"GCS bucket: {args.bucket}")
    print()

    # Step 1: Download ensemble NetCDF files
    print("Step 1: Loading ensemble forecast from GCS...")
    fds = download_ensemble_nc_from_gcs(
        forecast_date,
        members=members,
        gcs_bucket=args.bucket,
        service_account_path=args.service_account,
        skip_download_if_exists=args.skip_existing,
        fp16=args.fp16
    )

    if fds is None:
        print("❌ Failed to load ensemble data")
        return 1

    print("✅ Loaded ensemble forecast:")
    print(fds)
    print()

    # Step 2: Download climatology
    print("📥 Step 2: Downloading climatology data...")
    fc_valid_date1, fc_valid_date2 = valid_dates(forecast_date)
    print(f"   Valid dates: {fc_valid_date1}, {fc_valid_date2}")

    if AIWQ_AVAILABLE:
        download_all_quintiles(forecast_date)
    else:
        print("   Warning: AI_WQ_package not available, using local climatology files")
    print()

    # Step 3: Calculate quintiles
    print("🔢 Step 3: Calculating quintile probabilities...")
    quintile_ds = calculate_ensemble_quintiles(fds, forecast_date, args.clim_dir)

    if quintile_ds is None:
        print("❌ Failed to calculate quintiles")
        return 1

    print("\nQuintile analysis complete:")
    print(quintile_ds)

    # Step 4: Save results
    print("\n💾 Step 4: Saving results...")
    if args.fp16:
        output_file = os.path.join(args.output_dir,
                                   f'ensemble_quintile_probabilities_{forecast_date}_fp16.nc')
    else:
        output_file = os.path.join(args.output_dir,
                                   f'ensemble_quintile_probabilities_{forecast_date}.nc')

    quintile_ds.to_netcdf(output_file)

    file_size = os.path.getsize(output_file) / (1024**2)
    print(f"✅ Saved to {output_file} ({file_size:.1f} MB)")

    # Step 5: Summary
    print("\n📊 Step 5: Summary")
    print("-" * 40)
    print(f"Mode: {mode_label}")
    print(f"Variables: {list(quintile_ds.data_vars)}")
    print(f"Output: {output_file}")
    print(f"\n📤 Ready for AI Weather Quest submission")
    print(f"   Use submit_forecast() function with output file")

    return 0


if __name__ == "__main__":
    exit(main())

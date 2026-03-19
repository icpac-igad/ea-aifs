#!/usr/bin/env python3
"""
Diagnostic script to identify unit mismatches between forecast and climatology.
Downloads one member's NC from GCS and climatology, then compares value ranges.
"""

import os
import re
import numpy as np
import xarray as xr
from google.cloud import storage
from google.oauth2 import service_account
from dotenv import load_dotenv

load_dotenv()

# Import climatology download from existing pipeline
from era5t_ensemble_quintile_analysis_cli import (
    valid_dates, download_all_quintiles
)

FORECAST_DATE = "20260312"
GCS_BUCKET = "aifs-aiquest-us-20251127"
GCS_PREFIX = "20260312_0000/era5t_fp16_1p5deg_nc/"
SERVICE_ACCOUNT = "coiled-data.json"
LOCAL_DIR = "./diag_nc_files"

def download_one_member():
    """Download member 000 NC file from GCS for diagnosis."""
    os.makedirs(LOCAL_DIR, exist_ok=True)
    credentials = service_account.Credentials.from_service_account_file(SERVICE_ACCOUNT)
    client = storage.Client(credentials=credentials)
    bucket = client.bucket(GCS_BUCKET)

    # List files
    blobs = list(bucket.list_blobs(prefix=GCS_PREFIX))
    nc_blobs = [b for b in blobs if b.name.endswith('.nc')]
    print(f"Found {len(nc_blobs)} NC files in GCS:")
    for b in nc_blobs[:5]:
        print(f"  {b.name}")

    # Download first available member
    if not nc_blobs:
        print("ERROR: No NC files found!")
        return None

    blob = nc_blobs[0]
    filename = os.path.basename(blob.name)
    local_path = os.path.join(LOCAL_DIR, filename)
    if not os.path.exists(local_path):
        print(f"Downloading {filename}...")
        blob.download_to_filename(local_path)
    else:
        print(f"Using cached {filename}")
    return local_path


def diagnose():
    # Step 1: Download one member
    print("=" * 70)
    print("STEP 1: Download forecast member")
    print("=" * 70)
    nc_path = download_one_member()
    if not nc_path:
        return

    # Step 2: Inspect forecast values
    print("\n" + "=" * 70)
    print("STEP 2: Forecast value ranges")
    print("=" * 70)
    ds = xr.open_dataset(nc_path)
    print(f"\nDataset dimensions: {dict(ds.dims)}")
    print(f"Variables: {list(ds.data_vars)}")
    print(f"Coordinates: {list(ds.coords)}")

    # Check step coordinate to understand time structure
    if 'step' in ds.coords:
        steps = ds.step.values
        print(f"\nStep range: {steps[0]} to {steps[-1]}")
        print(f"Number of steps: {len(steps)}")

    for var in ['2t', 'msl', 'tp']:
        if var in ds:
            data = ds[var]
            # Load a small subset to check values
            sample = data.isel(step=slice(0, 5)).values
            print(f"\n  {var}:")
            print(f"    Shape: {data.shape}")
            print(f"    dtype: {data.dtype}")
            print(f"    Min: {np.nanmin(sample):.4f}")
            print(f"    Max: {np.nanmax(sample):.4f}")
            print(f"    Mean: {np.nanmean(sample):.4f}")
            print(f"    Std: {np.nanstd(sample):.4f}")

            # Check specific lat/lon point (e.g., near equator)
            mid_lat = data.sizes.get('latitude', 121) // 2
            mid_lon = data.sizes.get('longitude', 240) // 2
            point_val = data.isel(step=0, latitude=mid_lat, longitude=mid_lon).values
            print(f"    Sample point (mid-grid, step=0): {point_val}")
        else:
            print(f"\n  {var}: NOT FOUND in dataset")

    # Step 3: Simulate weekly mean computation (same as quintile script)
    print("\n" + "=" * 70)
    print("STEP 3: Simulated weekly aggregation")
    print("=" * 70)

    for var in ['2t', 'msl', 'tp']:
        if var not in ds:
            continue

        # Replicate the quintile script logic for week1 (time chunks 0,1)
        if 'time' in ds.dims:
            chunk_sums = []
            total_steps = 0
            for t_idx in [0, 1]:
                if t_idx < ds.sizes.get('time', 0):
                    chunk = ds[var].isel(time=t_idx)
                    chunk_sum = chunk.sum(dim='step', skipna=True)
                    chunk_sums.append(chunk_sum)
                    total_steps += chunk.sizes['step']

            if chunk_sums:
                forecast_sum = sum(chunk_sums)
                actual_days = total_steps * 6 / 24

                if var == 'tp':
                    scaling_factor = 7.0 / actual_days
                    weekly_val = forecast_sum * 1000 * scaling_factor
                    print(f"\n  {var} (weekly precip, m->mm scaled to 7 days):")
                else:
                    weekly_val = forecast_sum / total_steps
                    print(f"\n  {var} (weekly mean = sum/{total_steps}):")

                vals = weekly_val.values
                print(f"    Total steps used: {total_steps}, actual_days: {actual_days}")
                print(f"    Min: {np.nanmin(vals):.4f}")
                print(f"    Max: {np.nanmax(vals):.4f}")
                print(f"    Mean: {np.nanmean(vals):.4f}")
                print(f"    Median: {np.nanmedian(vals):.4f}")
        else:
            # No time dimension - all steps in one block
            all_sum = ds[var].sum(dim='step', skipna=True)
            n_steps = ds.sizes['step']
            if var == 'tp':
                actual_days = n_steps * 6 / 24
                scaling_factor = 7.0 / actual_days
                weekly_val = all_sum * 1000 * scaling_factor
                print(f"\n  {var} (weekly precip):")
            else:
                weekly_val = all_sum / n_steps
                print(f"\n  {var} (weekly mean = sum/{n_steps}):")
            vals = weekly_val.values
            print(f"    Min: {np.nanmin(vals):.4f}")
            print(f"    Max: {np.nanmax(vals):.4f}")
            print(f"    Mean: {np.nanmean(vals):.4f}")

    # Step 4: Download and inspect climatology
    print("\n" + "=" * 70)
    print("STEP 4: Climatology value ranges")
    print("=" * 70)

    fc_valid_date1, fc_valid_date2 = valid_dates(FORECAST_DATE)
    print(f"Valid dates: {fc_valid_date1}, {fc_valid_date2}")

    # Download climatology
    quintile_data = download_all_quintiles(FORECAST_DATE)

    clim_vars = {'tas': 'tas', 'mslp': 'mslp', 'pr': 'pr'}
    for clim_var in clim_vars:
        if quintile_data.get(clim_var) is None:
            print(f"\n  {clim_var}: No climatology data")
            continue

        for date_key, clim in quintile_data[clim_var].items():
            print(f"\n  {clim_var} climatology ({date_key}):")
            print(f"    Type: {type(clim)}")

            if isinstance(clim, xr.DataArray):
                print(f"    Shape: {clim.shape}")
                print(f"    Dims: {clim.dims}")
                vals = clim.values
                print(f"    Min: {np.nanmin(vals):.4f}")
                print(f"    Max: {np.nanmax(vals):.4f}")
                print(f"    Mean: {np.nanmean(vals):.4f}")
                # Show each quintile threshold
                if 'quantile' in clim.dims or 'quintile' in clim.dims:
                    q_dim = 'quantile' if 'quantile' in clim.dims else 'quintile'
                    for qi in range(min(4, clim.sizes[q_dim])):
                        q_vals = clim.isel({q_dim: qi}).values
                        print(f"    Q{qi+1} thresholds: min={np.nanmin(q_vals):.4f}, "
                              f"max={np.nanmax(q_vals):.4f}, mean={np.nanmean(q_vals):.4f}")
            elif isinstance(clim, xr.Dataset):
                print(f"    Variables: {list(clim.data_vars)}")
                for v in clim.data_vars:
                    cdata = clim[v]
                    print(f"    {v}: shape={cdata.shape}")
                    vals = cdata.values
                    print(f"      Min: {np.nanmin(vals):.4f}")
                    print(f"      Max: {np.nanmax(vals):.4f}")
                    print(f"      Mean: {np.nanmean(vals):.4f}")
                    # Show quintile thresholds
                    if len(cdata.dims) >= 3:
                        for qi in range(min(4, cdata.shape[cdata.dims.index('quantile') if 'quantile' in cdata.dims else 0])):
                            q_vals = cdata.isel({cdata.dims[0]: qi}).values if 'quantile' not in cdata.dims else cdata.isel(quantile=qi).values
                            print(f"      Q{qi+1}: min={np.nanmin(q_vals):.4f}, "
                                  f"max={np.nanmax(q_vals):.4f}, mean={np.nanmean(q_vals):.4f}")
            # Only check first date
            break

    # Step 5: Check climatology files on disk
    print("\n" + "=" * 70)
    print("STEP 5: Climatology files on disk")
    print("=" * 70)

    clim_dir = "./climatologies/2026/"
    if os.path.exists(clim_dir):
        for f in sorted(os.listdir(clim_dir)):
            if f.endswith('.nc'):
                fpath = os.path.join(clim_dir, f)
                try:
                    cds = xr.open_dataset(fpath)
                    print(f"\n  {f}:")
                    print(f"    Variables: {list(cds.data_vars)}")
                    print(f"    Dims: {dict(cds.dims)}")
                    for v in cds.data_vars:
                        vals = cds[v].values
                        print(f"    {v}: min={np.nanmin(vals):.4f}, max={np.nanmax(vals):.4f}, mean={np.nanmean(vals):.4f}")
                    cds.close()
                except Exception as e:
                    print(f"  {f}: Error - {e}")
    else:
        # Check current directory
        import glob
        for f in sorted(glob.glob('./*quintiles*.nc')):
            try:
                cds = xr.open_dataset(f)
                print(f"\n  {f}:")
                print(f"    Variables: {list(cds.data_vars)}")
                for v in cds.data_vars:
                    vals = cds[v].values
                    print(f"    {v}: min={np.nanmin(vals):.4f}, max={np.nanmax(vals):.4f}, mean={np.nanmean(vals):.4f}")
                cds.close()
            except Exception as e:
                print(f"  {f}: Error - {e}")

    # Step 6: Unit comparison summary
    print("\n" + "=" * 70)
    print("STEP 6: DIAGNOSIS SUMMARY")
    print("=" * 70)
    print("""
Expected units:
  - 2t (temperature): Kelvin (K) in both forecast and climatology
    If forecast ~280-300 and climatology ~280-300: OK
    If forecast ~280-300 and climatology ~0-30: UNIT MISMATCH (K vs °C)
    If forecast ~0-30 and climatology ~280-300: UNIT MISMATCH (°C vs K)

  - msl (pressure): Pa in forecast, check climatology
    If forecast ~100000 and climatology ~100000: OK (both Pa)
    If forecast ~100000 and climatology ~1000: UNIT MISMATCH (Pa vs hPa)
    If forecast ~1000 and climatology ~100000: UNIT MISMATCH (hPa vs Pa)

  - tp (precipitation): meters in forecast, check climatology
    Forecast is multiplied by 1000 (m->mm) in quintile script
    If climatology is in mm/day or mm/week: need to check scaling

Compare the values printed above to identify the mismatch!
""")

    ds.close()


if __name__ == "__main__":
    diagnose()

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

# Load environment variables from .env file.
# Search the *working directory* first: the docs say "run from this folder so
# coiled-data.json and .env resolve", but dotenv's default find_dotenv() searches
# upward from THIS module's directory (shared/), which never holds the .env — so the
# AIWQ_PASSWORD silently stayed unset and the climatology FTP login failed with
# "530 Login authentication failed". Fall back to the module-relative search.
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv(usecwd=True) or find_dotenv())

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

# Try to import icechunk (optional but recommended for memory efficiency)
try:
    import icechunk
    ICECHUNK_AVAILABLE = True
except ImportError:
    ICECHUNK_AVAILABLE = False
    print("Warning: icechunk not available. Memory-efficient processing disabled.")


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


def get_quintile_clim(forecast_date: str, variable: str, password: Optional[str] = None,
                      max_retries: int = 5, dest: Optional[str] = None):
    """Retrieve quintile climatology with robust retry logic.

    ``dest`` is the directory to download into (passed to AI_WQ's
    ``local_destination``). If None, AI_WQ writes to the CWD -- which, with 3b run
    under ``--clim-dir``/``--work-dir``, put the files where the calc step could not
    find them. Always pass ``dest=clim_dir``.
    """
    if not AIWQ_AVAILABLE:
        raise RuntimeError("AI_WQ_package is required for climatology retrieval")

    if password is None:
        password = os.getenv('AIWQ_PASSWORD') or os.getenv('AIWQ_SUBMIT_PWD')

    fc_valid_date1, fc_valid_date2 = valid_dates(forecast_date)
    if dest:
        os.makedirs(dest, exist_ok=True)

    import time

    def retry_download(date_str, var, attempt=0):
        try:
            # Add initial delay to avoid FTP server rate limiting
            if attempt == 0:
                time.sleep(2)

            # local_destination is a DIRECTORY -> AI_WQ writes {dest}/{var}_...nc
            return retrieve_evaluation_data.retrieve_20yr_quantile_clim(
                date_str, var, password=password,
                local_destination=(dest.rstrip('/') if dest else None)
            )
        except Exception as e:
            if attempt < max_retries - 1:
                # Exponential backoff: 3s, 6s, 12s, 24s, 48s
                wait_time = (2 ** attempt) * 3
                error_msg = str(e).split('\n')[0]
                print(f"      ⚠️  Download attempt {attempt + 1}/{max_retries} failed: {error_msg}")
                print(f"      Waiting {wait_time}s before retry...")
                time.sleep(wait_time)
                return retry_download(date_str, var, attempt + 1)
            else:
                print(f"      ❌ All {max_retries} attempts failed")
                raise

    clim1 = retry_download(fc_valid_date1, variable)
    clim2 = retry_download(fc_valid_date2, variable)

    return clim1, clim2


def check_local_climatology_files(forecast_date: str, clim_dir: str = "./",
                                 variables: Optional[List[str]] = None) -> dict:
    """Check which climatology files exist locally."""
    if variables is None:
        variables = ['tas', 'mslp', 'pr']

    fc_valid_date1, fc_valid_date2 = valid_dates(forecast_date)

    local_files = {
        'tas': {
            'week1': os.path.join(clim_dir, f"tas_20yrCLIM_WEEKLYMEAN_quintiles_{fc_valid_date1}.nc"),
            'week2': os.path.join(clim_dir, f"tas_20yrCLIM_WEEKLYMEAN_quintiles_{fc_valid_date2}.nc")
        },
        'mslp': {
            'week1': os.path.join(clim_dir, f"mslp_20yrCLIM_WEEKLYMEAN_quintiles_{fc_valid_date1}.nc"),
            'week2': os.path.join(clim_dir, f"mslp_20yrCLIM_WEEKLYMEAN_quintiles_{fc_valid_date2}.nc")
        },
        'pr': {
            'week1': os.path.join(clim_dir, f"pr_20yrCLIM_WEEKLYSUM_quintiles_{fc_valid_date1}.nc"),
            'week2': os.path.join(clim_dir, f"pr_20yrCLIM_WEEKLYSUM_quintiles_{fc_valid_date2}.nc")
        }
    }

    available = {}
    for var in variables:
        available[var] = {
            'week1': os.path.exists(local_files[var]['week1']),
            'week2': os.path.exists(local_files[var]['week2'])
        }

    return available, local_files


def download_all_quintiles(forecast_date: str, variables: Optional[List[str]] = None,
                          password: Optional[str] = None, clim_dir: str = "./"):
    """Download quintile climatologies for multiple variables (skip if local files exist)."""
    if variables is None:
        variables = ['tas', 'mslp', 'pr']

    if password is None:
        password = os.getenv('AIWQ_PASSWORD') or os.getenv('AIWQ_SUBMIT_PWD')

    fc_valid_date1, fc_valid_date2 = valid_dates(forecast_date)

    available, local_files = check_local_climatology_files(forecast_date, clim_dir, variables)

    # Check if all files are available locally
    all_available = all(available[v]['week1'] and available[v]['week2'] for v in variables)
    if all_available:
        print(f"   ✅ All climatology files found locally - skipping download")
        return {}

    quintile_data = {}

    for variable in variables:
        print(f"Downloading quintile climatology for {variable}...")
        try:
            clim1, clim2 = get_quintile_clim(forecast_date, variable, password, dest=clim_dir)
            quintile_data[variable] = {
                fc_valid_date1: clim1,
                fc_valid_date2: clim2
            }
            print(f"✓ Successfully downloaded {variable} climatology")
        except Exception as e:
            print(f"✗ Error downloading {variable}: {e}")
            quintile_data[variable] = None

    return quintile_data


def download_ensemble_nc_from_gcs_chunked(
    forecast_date: str,
    members: Optional[List[int]] = None,
    gcs_bucket: str = "aifs-aiquest-us-20251127",
    gcs_prefix: Optional[str] = None,
    service_account_path: str = "coiled-data.json",
    local_dir: str = "./ensemble_nc_files",
    icechunk_store_path: str = "./ensemble_icechunk_store",
    skip_download_if_exists: bool = True,
    fp16: bool = False,
    v2: bool = False,
    local_only: bool = False
):
    """
    Download ensemble NetCDF files from GCS and combine using icechunk for memory efficiency.

    ``local_only=True`` skips GCS entirely: members are discovered by globbing ``local_dir``
    for ``*member<NNN>.nc``. Use it when Step 3a was run with ``--no-upload --output-dir``
    (fully local pipeline, no service account needed).

    This function processes members one at a time and stores them in an icechunk store,
    avoiding the memory issues that occur when loading all members into RAM at once.

    Args:
        forecast_date: Forecast date in YYYYMMDD format
        members: List of member numbers to download. If None, downloads all available
        gcs_bucket: GCS bucket name
        gcs_prefix: GCS prefix where NetCDF files are stored. If None, uses forecast_date
        service_account_path: Path to GCS service account key
        local_dir: Local directory for temporary file downloads
        icechunk_store_path: Path for icechunk store
        skip_download_if_exists: If True, skip downloading files that already exist locally
        fp16: If True, use fp16_1p5deg_nc/ path instead of 1p5deg_nc/

    Returns:
        xr.Dataset: Combined ensemble dataset with all members (lazy-loaded via icechunk)
    """
    if not ICECHUNK_AVAILABLE:
        raise RuntimeError("icechunk is required for memory-efficient processing. Install with: pip install icechunk")

    # Set GCS prefix based on v2 / fp16 flag (v2 takes precedence)
    if gcs_prefix is None:
        if v2:
            gcs_prefix = f"{forecast_date}_0000/fp16_v2_1p5deg_nc/"
        elif fp16:
            gcs_prefix = f"{forecast_date}_0000/fp16_1p5deg_nc/"
        else:
            gcs_prefix = f"{forecast_date}_0000/1p5deg_nc/"

    mode_label = "FP16-v2" if v2 else ("FP16" if fp16 else "FP32")

    print(f"📥 Downloading ensemble NetCDF files from GCS (Memory-Efficient Mode)")
    print(f"   Bucket: gs://{gcs_bucket}/{gcs_prefix}")
    print(f"   Forecast date: {forecast_date}")
    print(f"   Mode: {mode_label}")
    print(f"   Icechunk store: {icechunk_store_path}")

    try:
        bucket = None
        if not local_only:
            # Initialize GCS client
            credentials = service_account.Credentials.from_service_account_file(service_account_path)
            client = storage.Client(credentials=credentials)
            bucket = client.bucket(gcs_bucket)

        # Create local directory and clean up existing icechunk store
        os.makedirs(local_dir, exist_ok=True)
        if os.path.exists(icechunk_store_path):
            shutil.rmtree(icechunk_store_path)

        available_files = []
        if local_only:
            print(f"   🔍 Scanning local directory {local_dir} for NetCDF files...")
            for filename in sorted(os.listdir(local_dir)):
                match = re.search(r'member(\d+)\.nc$', filename)
                if match:
                    member_num = int(match.group(1))
                    if members is None or member_num in members:
                        available_files.append({
                            'blob_name': None,
                            'filename': filename,
                            'member': member_num
                        })
        else:
            # List available NetCDF files in GCS
            print(f"   🔍 Scanning for available NetCDF files...")
            blobs = bucket.list_blobs(prefix=gcs_prefix)

            for blob in blobs:
                if blob.name.endswith('.nc'):
                    # Extract member number from filename
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
            where = local_dir if local_only else gcs_prefix
            print(f"   ❌ No NetCDF files found in {where}")
            return None

        available_files.sort(key=lambda x: x['member'])  # Sort by member number
        print(f"   ✅ Found {len(available_files)} NetCDF files "
              f"{'locally' if local_only else 'in GCS'}")

        # Create local icechunk repository
        local_storage = icechunk.local_filesystem_storage(icechunk_store_path)
        repo = icechunk.Repository.create(local_storage)
        session = repo.writable_session("main")

        zarr_group = "ensemble_forecast"
        processed_count = 0
        # (icechunk_store, icechunk_ref) seen per member. Collected across ALL
        # members rather than read once off the store, because the group attrs
        # only retain the LAST member written -- which would silently present a
        # mixed-source ensemble as single-source. A cycle can hold two forecast
        # stores (N320 and O96, or a tier-B corpus and its sidecar), so members
        # built from different stores is a real failure, not a hypothetical.
        member_provenance = set()

        # Process members one by one for memory efficiency
        for i, file_info in enumerate(available_files):
            member_num = file_info['member']
            print(f"   📥 Processing member {member_num:03d} ({i+1}/{len(available_files)})")

            # Download file if it doesn't exist locally
            local_path = os.path.join(local_dir, file_info['filename'])
            if local_only:
                print(f"      Using local file: {file_info['filename']}")
            elif not os.path.exists(local_path) or not skip_download_if_exists:
                print(f"      Downloading {file_info['filename']}")
                blob = bucket.blob(file_info['blob_name'])
                blob.download_to_filename(local_path)
            else:
                print(f"      Using cached file: {file_info['filename']}")

            # Load and process single member
            try:
                ds = xr.open_dataset(local_path, chunks={'member': 1, 'step': 10, 'latitude': 60, 'longitude': 120})

                member_provenance.add((ds.attrs.get("icechunk_store", ""),
                                       ds.attrs.get("icechunk_ref", "")))

                # Update member coordinate to the correct value
                ds = ds.assign_coords(member=[member_num])

                if i == 0:
                    # First member - create the store structure
                    print(f"      🏗️  Creating icechunk store with member {member_num:03d}")
                    ds.to_zarr(session.store, group=zarr_group, mode='w', consolidated=False)
                    print(f"      ✅ Created store with dimensions: {dict(ds.dims)}")
                else:
                    # Append subsequent members along member dimension
                    print(f"      ➕ Appending member {member_num:03d} to icechunk store")
                    ds.to_zarr(session.store, group=zarr_group, append_dim='member', consolidated=False)

                processed_count += 1

                # Clean up memory immediately
                ds.close()
                del ds
                gc.collect()

            except Exception as e:
                print(f"      ❌ Error processing member {member_num:03d}: {e}")
                import traceback
                traceback.print_exc()
                continue

        if processed_count == 0:
            print(f"   ❌ No members could be processed")
            return None

        # Commit the session
        session.commit(f"Processed {processed_count} ensemble members for {forecast_date}")
        print(f"   ✅ Committed icechunk store with {processed_count} members")

        # Open the final dataset for reading
        print(f"   🔗 Opening final ensemble dataset from icechunk store...")
        read_session = repo.readonly_session(branch="main")
        ensemble_ds = xr.open_zarr(read_session.store, group=zarr_group,
                                 chunks={'member': 1, 'step': 10, 'latitude': 60, 'longitude': 120})

        # Add ensemble metadata
        ensemble_ds.attrs.update({
            'title': f'AIFS Ensemble Forecast - Icechunk Store ({mode_label})',
            'description': f'Memory-efficient ensemble forecast with {processed_count} members',
            'forecast_date': forecast_date,
            'precision': mode_label,
            'members': f'{available_files[0]["member"]}-{available_files[-1]["member"]}',
            'source': 'ECMWF AIFS processed through icechunk for memory efficiency',
            'processing_date': str(np.datetime64('now')),
            'storage_backend': 'icechunk'
        })

        # Provenance of the FORECAST data, kept distinct from this staging store.
        known = {p for p in member_provenance if any(p)}
        if len(known) == 1:
            src_store, src_ref = known.pop()
            ensemble_ds.attrs['source_icechunk_store'] = src_store
            ensemble_ds.attrs['source_icechunk_ref'] = src_ref
        elif len(known) > 1:
            listed = "; ".join(f"{a}@{b}" for a, b in sorted(known))
            ensemble_ds.attrs['source_icechunk_store'] = f"MIXED: {listed}"
            ensemble_ds.attrs['source_icechunk_ref'] = "MIXED"
            print(f"   ⚠️  MIXED PROVENANCE -- members came from more than one "
                  f"forecast store: {listed}")
            print(f"      The quintile product will say so. Do not submit it "
                  f"until you know why.")
        else:
            ensemble_ds.attrs['source_icechunk_store'] = "unknown"
            ensemble_ds.attrs['source_icechunk_ref'] = "unknown"

        print(f"   ✅ Final ensemble dataset:")
        print(f"      Members: {ensemble_ds.sizes['member']}")
        print(f"      Variables: {list(ensemble_ds.data_vars)}")
        print(f"      Dimensions: {dict(ensemble_ds.dims)}")
        print(f"      Storage: icechunk (lazy-loaded, memory-efficient)")

        return ensemble_ds

    except Exception as e:
        print(f"❌ Error in icechunk processing: {e}")
        import traceback
        traceback.print_exc()
        return None


def download_ensemble_nc_from_gcs(
    forecast_date: str,
    members: Optional[List[int]] = None,
    gcs_bucket: str = "aifs-aiquest-us-20251127",
    gcs_prefix: Optional[str] = None,
    service_account_path: str = "coiled-data.json",
    local_dir: str = "./ensemble_nc_files",
    skip_download_if_exists: bool = True,
    fp16: bool = False,
    v2: bool = False
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
    # Set GCS prefix based on v2 / fp16 flag (v2 takes precedence)
    if gcs_prefix is None:
        if v2:
            gcs_prefix = f"{forecast_date}_0000/fp16_v2_1p5deg_nc/"
        elif fp16:
            gcs_prefix = f"{forecast_date}_0000/fp16_1p5deg_nc/"
        else:
            gcs_prefix = f"{forecast_date}_0000/1p5deg_nc/"

    mode_label = "FP16-v2" if v2 else ("FP16" if fp16 else "FP32")

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


def load_existing_icechunk_store(icechunk_store_path: str = "./ensemble_icechunk_store"):
    """Load existing icechunk store without downloading."""
    if not ICECHUNK_AVAILABLE:
        raise RuntimeError("icechunk is required. Install with: pip install icechunk")

    if not os.path.exists(icechunk_store_path):
        raise FileNotFoundError(f"Icechunk store not found at {icechunk_store_path}")

    print(f"📂 Loading existing icechunk store from {icechunk_store_path}...")
    local_storage = icechunk.local_filesystem_storage(icechunk_store_path)
    repo = icechunk.Repository.open(local_storage)
    read_session = repo.readonly_session(branch="main")

    zarr_group = "ensemble_forecast"
    ensemble_ds = xr.open_zarr(read_session.store, group=zarr_group,
                             chunks={'member': 1, 'step': 10, 'latitude': 60, 'longitude': 120})

    print(f"✅ Loaded existing ensemble dataset:")
    print(f"   Members: {ensemble_ds.sizes['member']}")
    print(f"   Variables: {list(ensemble_ds.data_vars)}")
    print(f"   Dimensions: {dict(ensemble_ds.sizes)}")
    print(f"   Storage: icechunk (lazy-loaded, memory-efficient)")

    return ensemble_ds


def load_ensemble_from_gcs(
    forecast_date: str,
    members: Optional[List[int]] = None,
    gcs_bucket: str = "aifs-aiquest-us-20251127",
    service_account_path: str = "coiled-data.json",
    use_icechunk: bool = True,
    skip_download_if_exists: bool = True,
    fp16: bool = False,
    v2: bool = False,
    local_only: bool = False,
    local_dir: str = "./ensemble_nc_files",
    icechunk_store_path: str = "./ensemble_icechunk_store"
):
    """
    Convenience function to download and load ensemble data from GCS.

    Args:
        forecast_date: Forecast date in YYYYMMDD format
        members: Specific members to download, or None for all
        gcs_bucket: GCS bucket name
        service_account_path: Path to GCS service account key
        use_icechunk: If True, use memory-efficient icechunk approach (recommended)
        skip_download_if_exists: If True, skip downloading files that already exist locally
        fp16: If True, use FP16 paths

    Returns:
        xr.Dataset: Combined ensemble dataset
    """
    if use_icechunk:
        if not ICECHUNK_AVAILABLE:
            print("⚠️  icechunk not available, falling back to standard loading")
            print("   Warning: This may cause memory issues with large ensembles!")
            use_icechunk = False

    if use_icechunk:
        return download_ensemble_nc_from_gcs_chunked(
            forecast_date=forecast_date,
            members=members,
            gcs_bucket=gcs_bucket,
            service_account_path=service_account_path,
            local_dir=local_dir,
            icechunk_store_path=icechunk_store_path,
            skip_download_if_exists=skip_download_if_exists,
            fp16=fp16,
            v2=v2,
            local_only=local_only
        )
    else:
        return download_ensemble_nc_from_gcs(
            forecast_date=forecast_date,
            members=members,
            gcs_bucket=gcs_bucket,
            service_account_path=service_account_path,
            skip_download_if_exists=skip_download_if_exists,
            fp16=fp16,
            v2=v2
        )


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
        # Carry the forecast store forward. This file is the submission input and
        # the ONLY per-cycle artifact cleanup_aifs_run.py keeps (it purges
        # nc_1p5deg, where the member-level attrs live), so without this a cleaned
        # cycle can no longer say which store produced what was submitted.
        for k in ('source_icechunk_store', 'source_icechunk_ref',
                  'forecast_date', 'precision'):
            if k in getattr(forecast_ds, 'attrs', {}):
                result_ds.attrs[k] = forecast_ds.attrs[k]
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
    # Process FP32 forecasts (uses icechunk by default for memory efficiency)
    python ensemble_quintile_analysis_cli.py --date 20251127

    # Process FP16 forecasts
    python ensemble_quintile_analysis_cli.py --date 20251127 --fp16

    # Disable icechunk (not recommended for large ensembles - may cause OOM killed)
    python ensemble_quintile_analysis_cli.py --date 20251127 --no-icechunk

    # Skip ensemble download and use existing icechunk store (fast retry for climatology/quintiles)
    python ensemble_quintile_analysis_cli.py --date 20251127 --skip-ensemble
        """
    )

    parser.add_argument('--date', required=True,
                       help='Forecast date (YYYYMMDD format)')
    parser.add_argument('--members', default=None,
                       help='Member range (e.g., 1-50, 1,2,3). If not specified, downloads all available.')
    parser.add_argument('--fp16', action='store_true',
                       help='Use FP16 paths (fp16_1p5deg_nc/)')
    parser.add_argument('--v2', action='store_true',
                       help='Use AIFS-ENS-2.0 paths (fp16_v2_1p5deg_nc/); '
                            'writes ..._{date}_v2.nc. Takes precedence over --fp16.')
    parser.add_argument('--no-icechunk', action='store_true',
                       help='Disable icechunk memory-efficient loading (not recommended for large ensembles)')
    parser.add_argument('--skip-existing', action='store_true', default=True,
                       help='Skip downloading existing files (default: True)')
    parser.add_argument('--skip-ensemble', action='store_true',
                       help='Skip Step 1 (ensemble download) and load existing icechunk store. Useful for retrying climatology/quintile calculation.')
    parser.add_argument('--work-dir', default='./',
                       help='Home directory for ALL per-cycle artifacts (quintile file, '
                            'climatology, temp icechunk store). --output-dir/--clim-dir '
                            'override individually; both default to --work-dir. Default: ./')
    parser.add_argument('--output-dir', default=None,
                       help='Output directory for quintile file (default: --work-dir)')
    parser.add_argument('--clim-dir', default=None,
                       help='Directory for climatology files (default: --work-dir)')
    parser.add_argument('--local-nc-dir', default=None,
                       help='Read the per-member 1.5deg NetCDFs from this local directory '
                            'instead of GCS (pairs with Step 3a --no-upload --output-dir). '
                            'No service account needed for the ensemble load.')
    parser.add_argument('--bucket', default='aifs-aiquest-us-20251127',
                       help='GCS bucket name')
    parser.add_argument('--service-account', default='coiled-data.json',
                       help='Path to GCS service account key')

    args = parser.parse_args()

    # Clean date format (remove _0000 if present)
    forecast_date = args.date.split('_')[0] if '_' in args.date else args.date
    mode_label = "FP16-v2" if args.v2 else ("FP16" if args.fp16 else "FP32")

    # Parse members if specified
    members = None
    if args.members:
        try:
            members = parse_member_range(args.members)
            print(f"Processing {len(members)} members: {members[0]}-{members[-1]}")
        except ValueError as e:
            print(f"ERROR: Invalid member range: {e}")
            return 1

    # Determine if we should use icechunk
    use_icechunk = not args.no_icechunk
    if use_icechunk and not ICECHUNK_AVAILABLE:
        print("⚠️  icechunk not available, falling back to standard loading")
        print("   Warning: This may cause memory issues (OOM killed) with large ensembles!")
        use_icechunk = False

    print("=" * 60)
    print(f"AIFS Ensemble Quintile Analysis Pipeline ({mode_label})")
    print("=" * 60)
    print(f"Forecast date: {forecast_date}")
    print(f"Precision mode: {mode_label}")
    print(f"Members: {args.members if args.members else 'all available'}")
    print(f"GCS bucket: {args.bucket}")
    print(f"Memory-efficient mode (icechunk): {'ENABLED' if use_icechunk else 'DISABLED'}")
    if args.skip_ensemble:
        print("⏭️  Skipping Step 1 (ensemble download) - will load existing icechunk store")
    print()

    # One --work-dir to home all per-cycle artifacts (quintile file, climatology, temp
    # icechunk store). --output-dir / --clim-dir override individually; all default to
    # --work-dir, which itself defaults to "./" (backward compatible).
    work_dir = args.work_dir
    out_dir = args.output_dir if args.output_dir is not None else work_dir
    clim_dir = args.clim_dir if args.clim_dir is not None else work_dir
    icechunk_store_path = os.path.join(work_dir, "ensemble_icechunk_store")
    os.makedirs(work_dir, exist_ok=True)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    if clim_dir:
        os.makedirs(clim_dir, exist_ok=True)

    # Step 1: Download ensemble NetCDF files or load existing store
    if args.skip_ensemble:
        print("📂 Step 1: Loading existing ensemble forecast from icechunk store...")
        try:
            fds = load_existing_icechunk_store(icechunk_store_path)
        except (FileNotFoundError, RuntimeError) as e:
            print(f"❌ Error loading existing icechunk store: {e}")
            return 1
    else:
        if use_icechunk:
            print("📥 Step 1: Loading ensemble forecast from GCS (memory-efficient icechunk mode)...")
            print("   This processes members one at a time to avoid OOM errors")
        else:
            print("📥 Step 1: Loading ensemble forecast from GCS...")
            print("   ⚠️  Warning: Loading all members into RAM - may cause OOM killed!")

        fds = load_ensemble_from_gcs(
            forecast_date,
            members=members,
            gcs_bucket=args.bucket,
            service_account_path=args.service_account,
            use_icechunk=use_icechunk,
            skip_download_if_exists=args.skip_existing,
            fp16=args.fp16,
            v2=args.v2,
            local_only=bool(args.local_nc_dir),
            local_dir=args.local_nc_dir or os.path.join(work_dir, "ensemble_nc_files"),
            icechunk_store_path=icechunk_store_path,
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
        download_all_quintiles(forecast_date, clim_dir=clim_dir)
    else:
        print("   Warning: AI_WQ_package not available, using local climatology files")
    print()

    # Step 3: Calculate quintiles
    print("🔢 Step 3: Calculating quintile probabilities...")
    quintile_ds = calculate_ensemble_quintiles(fds, forecast_date, clim_dir)

    if quintile_ds is None:
        print("❌ Failed to calculate quintiles")
        return 1

    print("\nQuintile analysis complete:")
    print(quintile_ds)

    # Step 4: Save results
    print("\n💾 Step 4: Saving results...")
    if args.v2:
        output_file = os.path.join(out_dir,
                                   f'ensemble_quintile_probabilities_{forecast_date}_v2.nc')
    elif args.fp16:
        output_file = os.path.join(out_dir,
                                   f'ensemble_quintile_probabilities_{forecast_date}_fp16.nc')
    else:
        output_file = os.path.join(out_dir,
                                   f'ensemble_quintile_probabilities_{forecast_date}.nc')

    quintile_ds.to_netcdf(output_file)

    file_size = os.path.getsize(output_file) / (1024**2)
    print(f"✅ Saved to {output_file} ({file_size:.1f} MB)")

    # Step 5: Summary
    print("\n📊 Step 5: Analysis Summary")
    print("-" * 40)
    print(f"Mode: {mode_label}")
    print(f"Processed variables: {list(quintile_ds.data_vars)}")
    print(f"Time periods: {quintile_ds.sizes.get('time_week', 'N/A')}")
    print(f"Grid resolution: {quintile_ds.sizes['latitude']}x{quintile_ds.sizes['longitude']}")

    print("\nQuintile probability statistics:")
    for var in quintile_ds.data_vars:
        print(f"\n  {var}:")
        var_data = quintile_ds[var]
        for q_idx, q_val in enumerate([0.2, 0.4, 0.6, 0.8, 1.0]):
            mean_prob = var_data.isel(quintile=q_idx).mean().values
            print(f"    Q{q_idx+1} ({q_val}): {mean_prob:.3f} mean probability")

    print(f"\n📤 Step 6: Ready for AI Weather Quest submission")
    print(f"   Output file: {output_file}")
    print(f"   Available variables: {list(quintile_ds.data_vars)}")
    print(f"   Use submit_forecast() function to submit results")

    return 0


if __name__ == "__main__":
    exit(main())

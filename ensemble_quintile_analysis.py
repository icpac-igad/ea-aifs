
from datetime import datetime, timedelta
import os
import numpy as np 

import xarray as xr
import pandas as pd
import tempfile
import shutil
from pathlib import Path
from google.cloud import storage
from google.oauth2 import service_account
from AI_WQ_package import retrieve_evaluation_data
import icechunk

def get_quintile_clim(forecast_date, variable, password=None):
    """
    Retrieve the quintile climatology for a given forecast date and variable.

    Args:
        forecast_date (str): The forecast date in the format YYYYMMDD.
        variable (str): The variable to retrieve quintile climatology for.
        password (str): Password for data access. If None, uses environment variable.

    Returns:
        tuple: (clim1, clim2) - The quintile climatologies for both valid dates.
    """
    if password is None:
        password = os.getenv('AIWQ_SUBMIT_PWD')
    
    fc_valid_date1, fc_valid_date2 = valid_dates(forecast_date)

    clim1 = retrieve_evaluation_data.retrieve_20yr_quintile_clim(fc_valid_date1, variable, password=password)
    clim2 = retrieve_evaluation_data.retrieve_20yr_quintile_clim(fc_valid_date2, variable, password=password)

    return clim1, clim2

def valid_dates(forecast_date):
    """
    Retrieve the valid dates for a given forecast date.

    Args:
        forecast_date (str): The forecast date in the format YYYYMMDD.

    Returns:
        tuple: A tuple containing the valid dates for the forecast. 
    """
    date_obj = datetime.strptime(forecast_date, "%Y%m%d")

    # add number of days to date object depending on lead time
    fc_valid_date_obj1 = date_obj + timedelta(days=4+(7*2))  # get to the next Monday then add number of weeks
    fc_valid_date_obj2 = date_obj + timedelta(days=4+(7*3))

    fc_valid_date1 = fc_valid_date_obj1.strftime("%Y%m%d")
    fc_valid_date2 = fc_valid_date_obj2.strftime("%Y%m%d")

    return fc_valid_date1, fc_valid_date2

def download_all_quintiles(forecast_date, variables=None, password=None):
    """
    Download quintile climatologies for multiple variables and both valid dates.
    
    Args:
        forecast_date (str): The forecast date in the format YYYYMMDD.
        variables (list): List of variables to download. If None, uses default set.
        password (str): Password for data access. If None, uses environment variable.
    
    Returns:
        dict: Dictionary with structure {variable: {valid_date1: clim1, valid_date2: clim2}}
    """
    if variables is None:
        variables = ['tas', 'mslp', 'pr']  # Default variables
    
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

def download_ensemble_nc_from_gcs_chunked(
    forecast_date, 
    members=None, 
    gcs_bucket="ea_aifs_w1", 
    gcs_prefix=None,
    service_account_path="coiled-data-e4drr_202505.json",
    local_dir="./ensemble_nc_files",
    icechunk_store_path="./ensemble_icechunk_store"
):
    """
    Download ensemble NetCDF files from GCS and combine using icechunk for memory efficiency.
    
    Args:
        forecast_date (str): Forecast date in YYYYMMDD format
        members (list): List of member numbers to download. If None, downloads all available
        gcs_bucket (str): GCS bucket name
        gcs_prefix (str): GCS prefix where NetCDF files are stored. If None, uses forecast_date
        service_account_path (str): Path to GCS service account key (relative path)
        local_dir (str): Local directory for temporary file downloads
        icechunk_store_path (str): Path for icechunk store
    
    Returns:
        xr.Dataset: Combined ensemble dataset with all members (lazy-loaded via icechunk)
    """
    
    # Set default GCS prefix based on forecast date if not provided
    if gcs_prefix is None:
        gcs_prefix = f"{forecast_date}_0000/1p5deg_nc/"
    
    print(f"📥 Downloading ensemble NetCDF files from GCS (Memory-Efficient)")
    print(f"   Bucket: gs://{gcs_bucket}/{gcs_prefix}")
    print(f"   Forecast date: {forecast_date}")
    print(f"   Icechunk store: {icechunk_store_path}")
    
    try:
        # Initialize GCS client
        credentials = service_account.Credentials.from_service_account_file(service_account_path)
        client = storage.Client(credentials=credentials)
        bucket = client.bucket(gcs_bucket)
        
        # Create local directory and clean up existing icechunk store
        os.makedirs(local_dir, exist_ok=True)
        if os.path.exists(icechunk_store_path):
            import shutil
            shutil.rmtree(icechunk_store_path)
        
        # List available NetCDF files in GCS
        print(f"   🔍 Scanning for available NetCDF files...")
        blobs = bucket.list_blobs(prefix=gcs_prefix)
        
        available_files = []
        for blob in blobs:
            if blob.name.endswith('.nc') and forecast_date in blob.name:
                # Extract member number from filename
                filename = os.path.basename(blob.name)
                import re
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
            print(f"   ❌ No NetCDF files found for forecast date {forecast_date}")
            return None
        
        available_files.sort(key=lambda x: x['member'])  # Sort by member number
        print(f"   ✅ Found {len(available_files)} NetCDF files in GCS")
        
        # Create local icechunk repository
        local_storage = icechunk.local_filesystem_storage(icechunk_store_path)
        repo = icechunk.Repository.create(local_storage)
        session = repo.writable_session("main")
        
        zarr_group = "ensemble_forecast"
        processed_count = 0
        
        # Process members one by one for memory efficiency
        for i, file_info in enumerate(available_files):
            member_num = file_info['member']
            print(f"   📥 Processing member {member_num:03d} ({i+1}/{len(available_files)})")
            
            # Download file if it doesn't exist locally
            local_path = os.path.join(local_dir, file_info['filename'])
            if not os.path.exists(local_path):
                print(f"      Downloading {file_info['filename']}")
                blob = bucket.blob(file_info['blob_name'])
                blob.download_to_filename(local_path)
            else:
                print(f"      Using cached file: {file_info['filename']}")
            
            # Load and process single member
            try:
                ds = xr.open_dataset(local_path, chunks={'member': 1, 'step': 10, 'latitude': 60, 'longitude': 120})
                
                # Update member coordinate to the correct value
                print(f"      Current member coordinate: {ds.member.values}, updating to: {member_num}")
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
                    print(f"      ✅ Appended member {member_num:03d}")
                
                processed_count += 1
                
                # Clean up memory immediately
                ds.close()
                del ds
                import gc
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
            'title': 'AIFS Ensemble Forecast - Icechunk Store',
            'description': f'Memory-efficient ensemble forecast with {processed_count} members',
            'forecast_date': forecast_date,
            'members': f'{available_files[0]["member"]}-{available_files[-1]["member"]}',
            'source': 'ECMWF AIFS processed through icechunk for memory efficiency',
            'processing_date': str(np.datetime64('now')),
            'storage_backend': 'icechunk'
        })
        
        print(f"   ✅ Final ensemble dataset:")
        print(f"      Members: {ensemble_ds.sizes['member']}")
        print(f"      Variables: {list(ensemble_ds.data_vars)}")
        print(f"      Dimensions: {dict(ensemble_ds.dims)}")
        print(f"      Storage: icechunk (lazy-loaded)")
        
        return ensemble_ds
        
    except Exception as e:
        print(f"❌ Error in icechunk processing: {e}")
        import traceback
        traceback.print_exc()
        return None

def download_ensemble_nc_from_gcs(
    forecast_date, 
    members=None, 
    gcs_bucket="ea_aifs_w1", 
    gcs_prefix=None,
    service_account_path="/home/sparrow/Documents/08-2023/impact_weather_icpac/lab/icpac_gcp/e4drr/gcp-coiled-sa-20250310/coiled-data-e4drr_202505.json",
    local_dir="./ensemble_nc_files"
):
    """
    Download ensemble NetCDF files from GCS and combine into a single dataset.
    
    Args:
        forecast_date (str): Forecast date in YYYYMMDD format
        members (list): List of member numbers to download. If None, downloads all available
        gcs_bucket (str): GCS bucket name
        gcs_prefix (str): GCS prefix where NetCDF files are stored. If None, uses forecast_date
        service_account_path (str): Path to GCS service account key
        local_dir (str): Local directory for temporary file downloads
    
    Returns:
        xr.Dataset: Combined ensemble dataset with all members
    """
    
    # Set default GCS prefix based on forecast date if not provided
    if gcs_prefix is None:
        gcs_prefix = f"{forecast_date}_0000/1p5deg_nc/"
    
    print(f"📥 Downloading ensemble NetCDF files from GCS")
    print(f"   Bucket: gs://{gcs_bucket}/{gcs_prefix}")
    print(f"   Forecast date: {forecast_date}")
    
    try:
        # Initialize GCS client
        credentials = service_account.Credentials.from_service_account_file(service_account_path)
        client = storage.Client(credentials=credentials)
        bucket = client.bucket(gcs_bucket)
        
        # Create local directory
        os.makedirs(local_dir, exist_ok=True)
        
        # List available NetCDF files in GCS
        print(f"   🔍 Scanning for available NetCDF files...")
        blobs = bucket.list_blobs(prefix=gcs_prefix)
        
        available_files = []
        for blob in blobs:
            if blob.name.endswith('.nc') and forecast_date in blob.name:
                # Extract member number from filename
                # Expected format: aifs_ensemble_forecast_1p5deg_memberXXX.nc
                filename = os.path.basename(blob.name)
                import re
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
            print(f"   ❌ No NetCDF files found for forecast date {forecast_date}")
            return None
        
        available_files.sort(key=lambda x: x['member'])  # Sort by member number
        print(f"   ✅ Found {len(available_files)} NetCDF files in GCS")
        
        # Check which files need to be downloaded (only missing files)
        files_to_download = []
        existing_files = []
        
        for file_info in available_files:
            local_path = os.path.join(local_dir, file_info['filename'])
            if os.path.exists(local_path):
                # File already exists locally
                existing_files.append({
                    'local_path': local_path,
                    'member': file_info['member'],
                    'filename': file_info['filename']
                })
                print(f"   ✓ Already exists: {file_info['filename']}")
            else:
                # File needs to be downloaded
                files_to_download.append(file_info)
        
        print(f"   📂 Found {len(existing_files)} files already downloaded")
        print(f"   📥 Need to download {len(files_to_download)} missing files")
        
        # Download only missing files
        member_datasets = []
        all_local_files = []
        
        # Download missing files
        for i, file_info in enumerate(files_to_download):
            print(f"   📥 Downloading {file_info['filename']} ({i+1}/{len(files_to_download)})")
            
            local_path = os.path.join(local_dir, file_info['filename'])
            blob = bucket.blob(file_info['blob_name'])
            blob.download_to_filename(local_path)
            all_local_files.append({
                'local_path': local_path,
                'member': file_info['member'],
                'filename': file_info['filename']
            })
        
        # Combine existing and newly downloaded files
        all_local_files.extend(existing_files)
        all_local_files.sort(key=lambda x: x['member'])  # Sort by member number
        
        # Load all files as xarray datasets
        for file_info in all_local_files:
            try:
                ds = xr.open_dataset(file_info['local_path'])
                member_datasets.append(ds)
                print(f"      ✅ Loaded member {file_info['member']:03d}: {list(ds.data_vars)}")
            except Exception as e:
                print(f"      ❌ Error loading {file_info['filename']}: {e}")
                continue
        
        if not member_datasets:
            print(f"   ❌ No datasets could be loaded")
            return None
        
        # Combine all members into a single ensemble dataset
        print(f"   🔗 Combining {len(member_datasets)} members into ensemble dataset...")
        
        try:
            # Concatenate along member dimension
            ensemble_ds = xr.concat(member_datasets, dim='member')
            
            # Add ensemble metadata
            ensemble_ds.attrs.update({
                'title': 'AIFS Ensemble Forecast - Combined Dataset',
                'description': f'Combined ensemble forecast with {len(member_datasets)} members',
                'forecast_date': forecast_date,
                'members': f'{all_local_files[0]["member"]}-{all_local_files[-1]["member"]}',
                'source': 'ECMWF AIFS processed through grib_to_nc_processor.py',
                'processing_date': str(np.datetime64('now'))
            })
            
            print(f"   ✅ Combined ensemble dataset:")
            print(f"      Members: {len(member_datasets)}")
            print(f"      Variables: {list(ensemble_ds.data_vars)}")
            print(f"      Dimensions: {dict(ensemble_ds.dims)}")
            
            return ensemble_ds
            
        except Exception as e:
            print(f"   ❌ Error combining datasets: {e}")
            return None
            
    except Exception as e:
        print(f"❌ Error downloading from GCS: {e}")
        return None
        
    finally:
        # Note: Files are kept in local_dir for future use to avoid re-downloading
        # Clean up only if explicitly requested
        pass

def load_ensemble_from_gcs(forecast_date, members=None, use_icechunk=True):
    """
    Convenience function to download and load ensemble data from GCS.
    
    Args:
        forecast_date (str): Forecast date in YYYYMMDD format  
        members (list): Specific members to download, or None for all
        use_icechunk (bool): If True, use memory-efficient icechunk approach
        
    Returns:
        xr.Dataset: Combined ensemble dataset
    """
    if use_icechunk:
        return download_ensemble_nc_from_gcs_chunked(
            forecast_date=forecast_date,
            members=members
        )
    else:
        return download_ensemble_nc_from_gcs(
            forecast_date=forecast_date,
            members=members
        ) 

def calculate_ensemble_quintiles(forecast_ds, forecast_date, climatology_base_path="./", variable_mapping=None):
    """
    Calculate quintile probabilities for ensemble forecasts against climatology.
    
    Args:
        forecast_ds (xr.Dataset): Ensemble forecast dataset with dimensions (time, member, step, latitude, longitude)
        forecast_date (str): Forecast date in YYYYMMDD format (e.g., '20250821')
        climatology_base_path (str): Base path where climatology files are located
        variable_mapping (dict): Optional mapping of forecast variables to climatology variables
    
    Returns:
        xr.Dataset: Quintile probabilities for each variable, time period, and grid point
    """
    
    if variable_mapping is None:
        variable_mapping = {
            'msl': 'mslp',  # Mean sea level pressure
            'tp': 'pr',     # Total precipitation -> precipitation rate
            '2t': 'tas'     # 2-meter temperature -> surface air temperature
        }
    
    # Calculate valid dates for climatology based on forecast date
    fc_valid_date1, fc_valid_date2 = valid_dates(forecast_date)
    
    print(f"📅 Using climatology for valid dates: {fc_valid_date1}, {fc_valid_date2}")
    
    # Define climatology file patterns for each variable and week using correct dates
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
    
    print(f"📁 Expected climatology files:")
    for var, weeks in climatology_files.items():
        for week, filepath in weeks.items():
            filename = os.path.basename(filepath)
            print(f"   {filename}")
    
    quintile_results = {}
    
    # Define quintile boundaries
    quintile_bounds = [0.2, 0.4, 0.6, 0.8, 1.0]
    
    for var_name in forecast_ds.data_vars:
        if var_name not in variable_mapping:
            print(f"Warning: No climatology mapping for variable {var_name}")
            continue
            
        clim_var = variable_mapping[var_name]
        print(f"Processing {var_name} -> {clim_var}")
        
        if clim_var not in climatology_files:
            print(f"Warning: No climatology files defined for {clim_var}")
            continue
        
        var_quintiles = []
        
        for time_idx, time_val in enumerate(forecast_ds.time):
            print(f"  Processing time {time_idx + 1}/{len(forecast_ds.time)}")
            
            # Calculate weekly means/accumulations for different step ranges
            week_configs = [
                ("week1", slice(0, 7)),   # Days 19-25 (steps 0-6 for week 1)
                ("week2", slice(7, 14))   # Days 26-32 (steps 7-13 for week 2)
            ]
            
            for week_name, step_slice in week_configs:
                
                # Get the appropriate climatology file for this variable and week
                clim_file = climatology_files[clim_var][week_name]
                
                try:
                    # Check if file exists
                    if not os.path.exists(clim_file):
                        print(f"    Warning: Climatology file not found: {clim_file}")
                        continue
                    
                    # Load climatology data
                    clim_ds = xr.open_dataset(clim_file)
                    
                    if clim_var not in clim_ds.data_vars:
                        print(f"    Warning: {clim_var} not found in {clim_file}")
                        continue
                    
                    # Get climatology quintiles (assuming they're stored as percentiles)
                    clim_quintiles = clim_ds[clim_var]  # Shape: (time, quantile, latitude, longitude)
                    print(f"    Loaded climatology data with shape: {clim_quintiles.shape}")
                    print(f"    Climatology dimensions: {clim_quintiles.dims}")
                    print(f"    Climatology quantile values: {clim_quintiles.quantile}")
                    
                    # Calculate forecast weekly statistics
                    forecast_data = forecast_ds[var_name].isel(time=time_idx, step=step_slice)
                    
                    if var_name == 'tp':
                        # For precipitation, calculate weekly accumulation (sum)
                        weekly_forecast = forecast_data.sum(dim='step')
                        print(f"    Calculated weekly precipitation accumulation for {week_name}")
                    else:
                        # For other variables, calculate weekly mean
                        weekly_forecast = forecast_data.mean(dim='step')
                        print(f"    Calculated weekly mean for {week_name}")
                    
                    # Calculate quintile probabilities for each grid point
                    quintile_probs = calculate_grid_quintiles(weekly_forecast, clim_quintiles)
                    
                    # Add coordinates
                    quintile_probs = quintile_probs.assign_coords({
                        'time': time_val,
                        'week': week_name,
                        'quintile': quintile_bounds
                    })
                    
                    var_quintiles.append(quintile_probs)
                    print(f"    ✓ Completed quintile calculation for {week_name}")
                    
                except Exception as e:
                    print(f"    Error processing {week_name} for {var_name}: {e}")
                    continue
        
        if var_quintiles:
            # Combine all time periods and weeks
            var_combined = xr.concat(var_quintiles, dim='time_week')
            quintile_results[f"{var_name}_quintiles"] = var_combined
            print(f"✓ Completed processing for {var_name}")
        else:
            print(f"✗ No quintile data calculated for {var_name}")
    
    # Combine all variables into a single dataset
    if quintile_results:
        result_ds = xr.Dataset(quintile_results)
        
        # Add metadata
        result_ds.attrs.update({
            'title': 'Ensemble Forecast Quintile Probabilities',
            'description': 'Quintile probabilities calculated from ensemble forecasts against 20-year climatology',
            'quintile_bounds': '0.2, 0.4, 0.6, 0.8, 1.0',
            'quintile_interpretation': '0.2: below normal, 0.4: below-to-near normal, 0.6: near normal, 0.8: above-normal, 1.0: above normal',
            'processing_date': str(np.datetime64('now')),
            'climatology_files': str(climatology_files)
        })
        
        return result_ds
    else:
        print("No quintile data could be calculated")
        return None

def calculate_grid_quintiles(ensemble_data, climatology_quintiles):
    """
    Calculate quintile probabilities for each grid point - OPTIMIZED VERSION.
    
    Args:
        ensemble_data (xr.DataArray): Ensemble forecast data (member, latitude, longitude)
        climatology_quintiles (xr.DataArray): Climatology quintiles (time, quantile, latitude, longitude) or (quantile, latitude, longitude)
    
    Returns:
        xr.DataArray: Quintile probabilities (quintile, latitude, longitude)
    """
    
    print(f"      🚀 Using VECTORIZED calculation for speed...")
    
    # Check if climatology_quintiles is actually a DataArray
    if not isinstance(climatology_quintiles, xr.DataArray):
        raise TypeError(f"Expected xr.DataArray, got {type(climatology_quintiles)}")
    
    n_members = ensemble_data.sizes['member']
    
    # Handle different climatology data structures
    if 'time' in climatology_quintiles.dims:
        clim_data = climatology_quintiles.isel(time=0)
    else:
        clim_data = climatology_quintiles
    
    # Convert to numpy arrays for vectorized operations
    ensemble_values = ensemble_data.values  # Shape: (n_members, n_lat, n_lon)
    clim_thresholds = clim_data.values      # Shape: (4, n_lat, n_lon)
    
    # Initialize output array
    quintile_probs = np.zeros((5, ensemble_data.sizes['latitude'], ensemble_data.sizes['longitude']))
    
    # Vectorized quintile calculation - process all grid points at once!
    # Q1: values < 20th percentile
    quintile_probs[0] = np.sum(ensemble_values < clim_thresholds[0], axis=0) / n_members
    
    # Q2: 20th <= values < 40th
    quintile_probs[1] = np.sum((ensemble_values >= clim_thresholds[0]) & 
                               (ensemble_values < clim_thresholds[1]), axis=0) / n_members
    
    # Q3: 40th <= values < 60th  
    quintile_probs[2] = np.sum((ensemble_values >= clim_thresholds[1]) & 
                               (ensemble_values < clim_thresholds[2]), axis=0) / n_members
    
    # Q4: 60th <= values < 80th
    quintile_probs[3] = np.sum((ensemble_values >= clim_thresholds[2]) & 
                               (ensemble_values < clim_thresholds[3]), axis=0) / n_members
    
    # Q5: values >= 80th
    quintile_probs[4] = np.sum(ensemble_values >= clim_thresholds[3], axis=0) / n_members
    
    print(f"      ✅ Vectorized calculation complete!")
    
    # Create xarray DataArray
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
            'units': 'probability',
            'description': 'Probability of forecast falling within each quintile category',
            'quintile_interpretation': '0.2: <20th percentile, 0.4: 20-40th, 0.6: 40-60th, 0.8: 60-80th, 1.0: >80th percentile'
        }
    )
    
    return quintile_da

def prepare_weekly_forecasts(forecast_ds):
    """
    Prepare weekly mean/accumulation forecasts from daily step data.
    
    Args:
        forecast_ds (xr.Dataset): Daily ensemble forecast dataset
    
    Returns:
        xr.Dataset: Weekly aggregated forecasts
    """
    
    weekly_data = {}
    
    # Define step ranges for weeks (assuming daily steps)
    week_ranges = {
        'week1': slice(18, 25),  # Days 19-25
        'week2': slice(25, 32)   # Days 26-32
    }
    
    for var_name in forecast_ds.data_vars:
        weekly_data[var_name] = []
        
        for week_name, step_range in week_ranges.items():
            
            # Extract data for this week
            week_data = forecast_ds[var_name].isel(step=step_range)
            
            if var_name == 'tp':
                # Precipitation: weekly accumulation
                weekly_agg = week_data.sum(dim='step')
            else:
                # Other variables: weekly mean
                weekly_agg = week_data.mean(dim='step')
            
            # Add week coordinate
            weekly_agg = weekly_agg.expand_dims('week')
            weekly_agg = weekly_agg.assign_coords(week=[week_name])
            
            weekly_data[var_name].append(weekly_agg)
        
        # Combine weeks
        weekly_data[var_name] = xr.concat(weekly_data[var_name], dim='week')
    
    return xr.Dataset(weekly_data)


def prepare_aiwq_submission(quintile_file, variable, week_period, time_index=0):
    """
    Prepare quintile data for AI Weather Quest submission.
    
    Args:
      quintile_file: Path to ensemble_quintile_probabilities.nc
      variable: 'tas', 'mslp', or 'pr'
      week_period: '1' or '2'
      time_index: Which forecast time to use (default: 0)
    
    Returns:
      numpy.ndarray: Shape (5, 121, 240) ready for submission
    """
    
    # Load the quintile dataset
    ds = xr.open_dataset(quintile_file)
    
    # Map variable names
    var_mapping = {
      'tas': '2t_quintiles',
      'mslp': 'msl_quintiles',
      'pr': 'tp_quintiles'
    }
    
    if variable not in var_mapping:
      raise ValueError(f"Variable must be one of: {list(var_mapping.keys())}")
    
    var_name = var_mapping[variable]
    
    # Select the appropriate data slice
    # Filter by week period
    week_name = f'week{week_period}'
    week_mask = ds.week == week_name
    
    # Get data for specific time and week
    data = ds[var_name].isel(time_week=np.where(week_mask)[0][time_index])
    
    # The data is already in shape (quintile, latitude, longitude) = (5, 121, 240)
    # Just need to extract the values
    submission_data = data.values
    
    # Verify shape and constraints
    assert submission_data.shape == (5, 121, 240), f"Wrong shape: {submission_data.shape}"
    assert np.all((submission_data >= 0) & (submission_data <= 1)), "Values must be between 0 and 1"
    
    # Check that probabilities sum to 1 at each grid point
    prob_sums = np.sum(submission_data, axis=0)
    valid_points = ~np.isnan(prob_sums)
    if np.any(valid_points):
      sum_check = np.abs(prob_sums[valid_points] - 1.0) < 1e-10
      if not np.all(sum_check):
          print(f"Warning: {np.sum(~sum_check)} grid points don't sum to 1.0")
    
    return submission_data

# Usage for AI Weather Quest submission:
def submit_forecast(quintile_file, variable, fc_start_date, fc_period, teamname, modelname, password, time_index=0):
    """Complete submission workflow"""
    
    from AI_WQ_package import forecast_submission
    
    # Prepare the data
    forecast_array = prepare_aiwq_submission(quintile_file, variable, fc_period, time_index)
    
    # Create empty DataArray
    empty_da = forecast_submission.AI_WQ_create_empty_dataarray(
      variable, fc_start_date, fc_period, teamname, modelname, password
    )
    
    # Fill with your data
    empty_da.values = forecast_array
    
    # Submit
    submitted_da = forecast_submission.AI_WQ_forecast_submission(
      empty_da, variable, fc_start_date, fc_period, teamname, modelname, password
    )
    
    return submitted_da


# Example usage
if __name__ == "__main__":
    
    forecast_date = '20250821'  # Updated to match your processing date
    
    print("=" * 60)
    print("AIFS Ensemble Quintile Analysis Pipeline")
    print("=" * 60)
    print(f"Forecast date: {forecast_date}")
    print()
    
    # Step 1: Download ensemble NetCDF files from GCS (using icechunk for memory efficiency)
    print("📥 Step 1: Loading ensemble forecast from GCS...")
    print("   Using memory-efficient icechunk approach to avoid RAM issues")
    fds = load_ensemble_from_gcs(forecast_date, use_icechunk=True)
    
    if fds is None:
        print("❌ Failed to load ensemble data from GCS")
        print("   Make sure the NetCDF files have been created and uploaded by grib_to_nc_processor.py")
        exit(1)
    
    print("✅ Loaded ensemble forecast dataset:")
    print(fds)
    print()
    
    # Step 2: Download climatology data
    print("📥 Step 2: Downloading climatology data...")
    
    # Calculate valid dates and show what will be downloaded
    fc_valid_date1, fc_valid_date2 = valid_dates(forecast_date)
    print(f"   Will download climatology for valid dates: {fc_valid_date1}, {fc_valid_date2}")
    
    # Download all default variables for aiquest 
    all_quintiles = download_all_quintiles(forecast_date)
    print()
    
    # Step 3: Calculate quintile probabilities
    print("🔢 Step 3: Calculating quintile probabilities...")
    
    # Calculate correct valid dates for this forecast
    fc_valid_date1, fc_valid_date2 = valid_dates(forecast_date)
    print(f"Expected climatology files for valid dates {fc_valid_date1}, {fc_valid_date2}:")
    print(f"  mslp_20yrCLIM_WEEKLYMEAN_quintiles_{fc_valid_date1}.nc")
    print(f"  mslp_20yrCLIM_WEEKLYMEAN_quintiles_{fc_valid_date2}.nc")
    print(f"  pr_20yrCLIM_WEEKLYSUM_quintiles_{fc_valid_date1}.nc")
    print(f"  pr_20yrCLIM_WEEKLYSUM_quintiles_{fc_valid_date2}.nc")
    print(f"  tas_20yrCLIM_WEEKLYMEAN_quintiles_{fc_valid_date1}.nc")
    print(f"  tas_20yrCLIM_WEEKLYMEAN_quintiles_{fc_valid_date2}.nc")
    print()
    
    # Use current directory as base path (change if files are elsewhere)
    climatology_base_path = "./"
    quintile_ds = calculate_ensemble_quintiles(fds, forecast_date, climatology_base_path)
    
    if quintile_ds is not None:
        print("\nQuintile analysis complete:")
        print(quintile_ds)
        
        # Step 4: Save results
        print("💾 Step 4: Saving results...")
        output_file = f'ensemble_quintile_probabilities_{forecast_date}.nc'
        quintile_ds.to_netcdf(output_file)
        
        # Print file size
        import os
        file_size = os.path.getsize(output_file) / (1024**2)  # MB
        print(f"✅ Saved quintile probabilities to {output_file} ({file_size:.1f} MB)")
        print()
        
        # Step 5: Analysis Summary
        print("📊 Step 5: Analysis Summary")
        print("-" * 40)
        print(f"Processed variables: {list(quintile_ds.data_vars)}")
        print(f"Time periods: {quintile_ds.sizes.get('time_week', 'N/A')}")
        print(f"Grid resolution: {quintile_ds.sizes['latitude']}x{quintile_ds.sizes['longitude']}")
        
        # Display summary statistics
        print("\nQuintile probability statistics:")
        for var in quintile_ds.data_vars:
            print(f"\n  {var}:")
            var_data = quintile_ds[var]
            for q_idx, q_val in enumerate([0.2, 0.4, 0.6, 0.8, 1.0]):
                mean_prob = var_data.isel(quintile=q_idx).mean().values
                print(f"    Q{q_idx+1} ({q_val}): {mean_prob:.3f} mean probability")
        
        # Step 6: Ready for submission
        print(f"\n📤 Step 6: Ready for AI Weather Quest submission")
        print(f"   Output file: {output_file}")
        print(f"   Available variables: {list(quintile_ds.data_vars)}")
        print(f"   Use submit_forecast() function to submit results")
        
    else:
        print("❌ Failed to calculate quintile probabilities")
        print("   Please check that all climatology files are present in the current directory")
    
    # Optional: Submit the forecast (uncomment and configure as needed)
    # result = submit_forecast(
    #     output_file,
    #     'mslp',  # or 'pr', 'tas' depending on available variables
    #     forecast_date,
    #     '1',     # week period
    #     team_name,
    #     model_name,
    #     password)   





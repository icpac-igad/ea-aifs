
from datetime import datetime, timedelta
import os
import numpy as np 

import xarray as xr
import pandas as pd
from AI_WQ_package import retrieve_evaluation_data

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

def calculate_ensemble_quintiles(forecast_ds, climatology_base_path="./", variable_mapping=None):
    """
    Calculate quintile probabilities for ensemble forecasts against climatology.
    
    Args:
        forecast_ds (xr.Dataset): Ensemble forecast dataset with dimensions (time, member, step, latitude, longitude)
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
    
    # Define climatology file patterns for each variable and week
    climatology_files = {
        'mslp': {
            'week1': f"{climatology_base_path}/mslp_20yrCLIM_WEEKLYMEAN_quintiles_20250901.nc",
            'week2': f"{climatology_base_path}/mslp_20yrCLIM_WEEKLYMEAN_quintiles_20250908.nc"
        },
        'pr': {
            'week1': f"{climatology_base_path}/pr_20yrCLIM_WEEKLYSUM_quintiles_20250901.nc",
            'week2': f"{climatology_base_path}/pr_20yrCLIM_WEEKLYSUM_quintiles_20250908.nc"
        },
        'tas': {
            'week1': f"{climatology_base_path}/tas_20yrCLIM_WEEKLYMEAN_quintiles_20250901.nc",
            'week2': f"{climatology_base_path}/tas_20yrCLIM_WEEKLYMEAN_quintiles_20250908.nc"
        }
    }
    
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
                    import os
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
    Calculate quintile probabilities for each grid point.
    
    Args:
        ensemble_data (xr.DataArray): Ensemble forecast data (member, latitude, longitude)
        climatology_quintiles (xr.DataArray): Climatology quintiles (time, quantile, latitude, longitude) or (quantile, latitude, longitude)
    
    Returns:
        xr.DataArray: Quintile probabilities (quintile, latitude, longitude)
    """
    
    print(f"DEBUG: climatology_quintiles type: {type(climatology_quintiles)}")
    print(f"DEBUG: ensemble_data type: {type(ensemble_data)}")
    
    # Check if climatology_quintiles is actually a DataArray
    if not isinstance(climatology_quintiles, xr.DataArray):
        raise TypeError(f"Expected xr.DataArray, got {type(climatology_quintiles)}")
    
    n_members = ensemble_data.sizes['member']
    quintile_probs = np.zeros((5, ensemble_data.sizes['latitude'], ensemble_data.sizes['longitude']))
    
    # Handle different climatology data structures
    if 'time' in climatology_quintiles.dims:
        # Take the first (and usually only) time slice
        clim_data = climatology_quintiles.isel(time=0)
    else:
        clim_data = climatology_quintiles
    
    print(f"DEBUG: clim_data shape: {clim_data.shape}")
    print(f"DEBUG: clim_data dims: {clim_data.dims}")
    
    # Get the actual quantile values from the climatology data
    clim_quantile_values = clim_data.quantile  # Should be [0.2, 0.4, 0.6, 0.8]
    print(f"DEBUG: clim_quantile_values: {clim_quantile_values}")
    
    # For each grid point
    for lat_idx in range(ensemble_data.sizes['latitude']):
        for lon_idx in range(ensemble_data.sizes['longitude']):
            
            # Get ensemble values at this grid point
            ensemble_values = ensemble_data.isel(latitude=lat_idx, longitude=lon_idx).values
            
            # Get climatology quintile thresholds at this grid point
            clim_thresholds = clim_data.isel(latitude=lat_idx, longitude=lon_idx).values  # Shape: (4,) for [0.2, 0.4, 0.6, 0.8]
            
            # Calculate probability for each quintile
            # Q1 (0-20%): values < 20th percentile threshold
            prob_q1 = np.sum(ensemble_values < clim_thresholds[0]) / n_members
            
            # Q2 (20-40%): values >= 20th percentile and < 40th percentile
            prob_q2 = np.sum((ensemble_values >= clim_thresholds[0]) & (ensemble_values < clim_thresholds[1])) / n_members
            
            # Q3 (40-60%): values >= 40th percentile and < 60th percentile  
            prob_q3 = np.sum((ensemble_values >= clim_thresholds[1]) & (ensemble_values < clim_thresholds[2])) / n_members
            
            # Q4 (60-80%): values >= 60th percentile and < 80th percentile
            prob_q4 = np.sum((ensemble_values >= clim_thresholds[2]) & (ensemble_values < clim_thresholds[3])) / n_members
            
            # Q5 (80-100%): values >= 80th percentile
            prob_q5 = np.sum(ensemble_values >= clim_thresholds[3]) / n_members
            
            quintile_probs[0, lat_idx, lon_idx] = prob_q1
            quintile_probs[1, lat_idx, lon_idx] = prob_q2
            quintile_probs[2, lat_idx, lon_idx] = prob_q3
            quintile_probs[3, lat_idx, lon_idx] = prob_q4
            quintile_probs[4, lat_idx, lon_idx] = prob_q5
    
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
    
    forecast_date = '20250814'
    # Download all default variables for aiquest 
    all_quintiles = download_all_quintiles(forecast_date)

    # Load the ensemble forecast
    output_filename = 'aifs_ensemble_forecast_1p5deg_members001-004.nc'
    fds = xr.open_dataset(output_filename)
    
    print("Loaded ensemble forecast dataset:")
    print(fds)
    
    # Calculate quintile probabilities (climatology files auto-detected)
    print("\nCalculating quintile probabilities...")
    print("Expected climatology files:")
    print("  mslp_20yrCLIM_WEEKLYMEAN_quintiles_20250901.nc")
    print("  mslp_20yrCLIM_WEEKLYMEAN_quintiles_20250908.nc")
    print("  pr_20yrCLIM_WEEKLYSUM_quintiles_20250901.nc")
    print("  pr_20yrCLIM_WEEKLYSUM_quintiles_20250908.nc")
    print("  tas_20yrCLIM_WEEKLYMEAN_quintiles_20250901.nc")
    print("  tas_20yrCLIM_WEEKLYMEAN_quintiles_20250908.nc")
    print()
    
    # Use current directory as base path (change if files are elsewhere)
    climatology_base_path = "./"
    quintile_ds = calculate_ensemble_quintiles(fds, climatology_base_path)
    
    if quintile_ds is not None:
        print("\nQuintile analysis complete:")
        print(quintile_ds)
        
        # Save results
        output_file = 'ensemble_quintile_probabilities.nc'
        quintile_ds.to_netcdf(output_file)
        print(f"\nSaved quintile probabilities to {output_file}")
        
        # Display summary statistics
        print("\nSummary of quintile probabilities:")
        for var in quintile_ds.data_vars:
            print(f"\n{var}:")
            var_data = quintile_ds[var]
            for q_idx, q_val in enumerate([0.2, 0.4, 0.6, 0.8, 1.0]):
                mean_prob = var_data.isel(quintile=q_idx).mean().values
                print(f"  Quintile {q_val}: {mean_prob:.3f} mean probability")
                
        # Display processing summary
        print(f"\nProcessed variables: {list(quintile_ds.data_vars)}")
        print(f"Time periods: {quintile_ds.sizes.get('time_week', 'N/A')}")
        print(f"Grid resolution: {quintile_ds.sizes['latitude']}x{quintile_ds.sizes['longitude']}")
        
    else:
        print("Failed to calculate quintile probabilities")
        print("Please check that all climatology files are present in the current directory")
    #submit the forecast 
    result = submit_forecast(
      'ensemble_quintile_probabilities.nc',
      'tas',
      '20250814',
      '1',
       team_name,
       model_name,
       password)   





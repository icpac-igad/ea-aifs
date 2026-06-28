import xarray as xr
import earthkit.data as ekd
import earthkit.regrid as ekr
import numpy as np

# Process members 001-004
members = ['001', '002', '003', '004']
time_ranges = ['h432-504', 'h504-576', 'h576-648', 'h648-720', 'h720-792']

# Variable mapping
var_mapping = {
    'mslp': 'msl',    # Mean sea level pressure
    'pr': 'tp',       # Total precipitation  
    'tas': 't2m'      # 2-meter temperature (if available, otherwise use surface temp)
}

print("Processing GRIB files for members 001-004...")

all_datasets = []

for member in members:
    print(f"\nProcessing member {member}:")
    
    member_datasets = []
    
    for time_range in time_ranges:
        filename = f"downloaded_grib_files/aifs_ens_forecast_20250817_0600_member{member}_{time_range}.grib"
        print(f"  Processing {filename}")
        
        try:
            # 1) Open GRIB
            fl = ekd.from_source("file", filename)
            
            # 2) N320 -> 1.5° regular lon/lat
            fl_ll_1p5 = ekr.interpolate(fl, in_grid={"grid": "N320"}, out_grid={"grid": [1.5, 1.5]})
            
            # 3) Convert to xarray (now on 1.5° grid with latitude/longitude coords)
            ds = fl_ll_1p5.to_xarray()
            
            # Check available variables
            available_vars = list(ds.data_vars)
            print(f"    Available variables: {available_vars}")
            
            # Extract target variables if they exist
            target_vars = []
            for target, alt_name in var_mapping.items():
                if alt_name in available_vars:
                    target_vars.append(alt_name)
                    print(f"    ✓ Found {alt_name} for {target}")
                else:
                    print(f"    ✗ {alt_name} not found for {target}")
            
            if target_vars:
                # Extract only target variables
                extracted_ds = ds[target_vars]
                
                # Add member dimension
                extracted_ds = extracted_ds.expand_dims('member')
                extracted_ds = extracted_ds.assign_coords(member=[member])
                
                member_datasets.append(extracted_ds)
                print(f"    Extracted variables: {target_vars}")
            
        except Exception as e:
            print(f"    Error processing {filename}: {e}")
    
    # Concatenate time ranges for this member
    if member_datasets:
        member_combined = xr.concat(member_datasets, dim='time')
        all_datasets.append(member_combined)

# Combine all members
if all_datasets:
    print(f"\nCombining data from {len(all_datasets)} members...")
    final_ds = xr.concat(all_datasets, dim='member')
    
    print('\nFinal dataset info:')
    print(final_ds)
    
    # Check coordinates
    if 'latitude' in final_ds.coords and 'longitude' in final_ds.coords:
        print('\n✓ Dataset has proper latitude/longitude coordinates at 1.5° resolution')
        print(f'Latitude range: {final_ds.latitude.min().values:.2f} to {final_ds.latitude.max().values:.2f}')
        print(f'Longitude range: {final_ds.longitude.min().values:.2f} to {final_ds.longitude.max().values:.2f}')
        print(f'Grid shape: {final_ds.latitude.shape} x {final_ds.longitude.shape}')
    
    print(f'\nMembers: {final_ds.member.values}')
    print(f'Variables: {list(final_ds.data_vars)}')
    
    # Save to NetCDF file
    output_filename = 'aifs_ensemble_forecast_1p5deg_members001-004.nc'
    print(f'\nSaving dataset to {output_filename}...')
    
    # Add global attributes
    final_ds.attrs.update({
        'title': 'AIFS Ensemble Forecast Data',
        'description': 'Regridded to 1.5 degree resolution, members 001-004',
        'source': 'ECMWF AIFS ensemble forecast',
        'grid_resolution': '1.5 degrees',
        'forecast_date': '2025-08-17 06:00',
        'members': 'member001 to member004',
        'variables': 'msl (mean sea level pressure), tp (total precipitation)',
        'processing_date': str(np.datetime64('now'))
    })
    
    # Clean attributes to avoid NetCDF serialization issues
    def clean_attrs(obj):
        """Remove problematic attributes that can't be serialized to NetCDF"""
        if hasattr(obj, 'attrs'):
            attrs_to_remove = []
            for key, value in obj.attrs.items():
                # Remove earthkit-specific attributes that contain binary data
                if key.startswith('_earthkit') or isinstance(value, dict):
                    attrs_to_remove.append(key)
            for key in attrs_to_remove:
                del obj.attrs[key]
        return obj
    
    # Clean the dataset and all variables
    final_ds = clean_attrs(final_ds)
    for var_name in final_ds.data_vars:
        final_ds[var_name] = clean_attrs(final_ds[var_name])
    for coord_name in final_ds.coords:
        final_ds[coord_name] = clean_attrs(final_ds[coord_name])
    
    # Save to NetCDF
    final_ds.to_netcdf(output_filename, engine='netcdf4')
    print(f'✓ Successfully saved to {output_filename}')
    
    # Print file size
    import os
    file_size = os.path.getsize(output_filename) / (1024**2)  # MB
    print(f'File size: {file_size:.1f} MB')
    
else:
    print('\nNo data could be processed from any files')

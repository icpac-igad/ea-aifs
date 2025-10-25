#!/usr/bin/env python3
"""
Diagnose how cfgrib reads the GRIB2 file and what filter_by_keys are being used.
"""

import xarray as xr
import cfgrib

grib_file = "20251020000000-0h-enfo-ef.grib2"

print("="*80)
print("DIAGNOSING CFGRIB LEVEL EXTRACTION")
print("="*80)

# Try to open with cfgrib to see what datasets are available
print("\n1. Checking available datasets (backends):")
print("-"*60)

try:
    # Open without filter to see what cfgrib finds
    datasets = cfgrib.open_datasets(grib_file)
    print(f"Found {len(datasets)} datasets:")
    for i, ds in enumerate(datasets):
        print(f"\nDataset {i}:")
        print(f"  Variables: {list(ds.data_vars)[:5]}")
        print(f"  Coordinates: {list(ds.coords)}")
        if 'isobaricInhPa' in ds.coords:
            print(f"  Pressure levels: {ds.coords['isobaricInhPa'].values}")
        if 'depthBelowLandLayer' in ds.coords:
            print(f"  Soil levels: {ds.coords['depthBelowLandLayer'].values}")
except Exception as e:
    print(f"Error opening datasets: {e}")

# Try opening with specific filters
print("\n\n2. Testing specific filters for pressure levels:")
print("-"*60)

test_filters = [
    {},  # No filter
    {'typeOfLevel': 'isobaricInhPa'},
    {'typeOfLevel': 'isobaricInhPa', 'number': 1},
]

for filter_dict in test_filters:
    print(f"\nFilter: {filter_dict if filter_dict else 'None'}")
    try:
        if filter_dict:
            ds = xr.open_dataset(grib_file, engine='cfgrib', backend_kwargs={'filter_by_keys': filter_dict})
        else:
            ds = xr.open_dataset(grib_file, engine='cfgrib')

        if 'isobaricInhPa' in ds.coords:
            levels = ds.coords['isobaricInhPa'].values
            print(f"  ✅ Pressure levels found: {levels}")
        else:
            print(f"  ⚠️ No isobaricInhPa coordinate")

        print(f"  Variables: {list(ds.data_vars)[:5]}")

    except Exception as e:
        print(f"  ❌ Error: {e}")

# Check if scan_grib is filtering
print("\n\n3. Checking kerchunk scan_grib behavior:")
print("-"*60)

try:
    from kerchunk.grib2 import scan_grib

    groups = scan_grib(grib_file, storage_options={})
    print(f"scan_grib found {len(groups)} groups/messages")

    # Count how many have pressure level info
    pl_groups = 0
    unique_levels = set()

    for i, group in enumerate(groups[:100]):  # Check first 100
        refs = group.get('refs', {})

        # Look for isobaricInhPa coordinate
        for key in refs.keys():
            if 'isobaricInhPa/isobaricInhPa' in key and not key.endswith(('.zarray', '.zattrs')):
                # This is a level value
                pl_groups += 1
                # Try to extract level value
                val = refs[key]
                if isinstance(val, str) and len(val) == 8:
                    import struct
                    try:
                        level = struct.unpack('<d', val.encode('latin1'))[0]
                        unique_levels.add(int(level))
                    except:
                        pass
                break

    print(f"Groups with pressure level data: {pl_groups}")
    print(f"Unique pressure levels found: {sorted(unique_levels)}")

except Exception as e:
    print(f"Error with scan_grib: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*80)
print("CONCLUSION:")
print("="*80)
print("If scan_grib creates ONE group per (variable, level, ensemble) combination,")
print("then the parquet creation script needs to MERGE groups by level dimension!")

#!/usr/bin/env python3
"""
Create a test parquet file with the fixed code to verify all levels are included.
This creates parquet for just ensemble member 1 as a test.
"""

import fsspec
import pandas as pd
import copy
import json
import os
import time
from pathlib import Path

from test_run_ecmwf_step1_scangrib import (
    ecmwf_filter_scan_grib,
    fixed_ensemble_grib_tree,
    log_checkpoint
)
from kerchunk._grib_idx import strip_datavar_chunks

print("="*80)
print("TESTING PARQUET CREATION WITH ALL PRESSURE LEVELS")
print("="*80)

# Configuration
grib_file = "s3://ecmwf-forecasts/20251020/00z/ifs/0p25/enfo/20251020000000-0h-enfo-ef.grib2"
output_dir = Path("test_fixed_parquet")
output_dir.mkdir(exist_ok=True)

start_time = time.time()

print(f"\nProcessing: {grib_file}")
print(f"Output directory: {output_dir}/")

# Step 1: Scan GRIB file
scan_start = log_checkpoint("\n1. Scanning GRIB file")
groups, idx_mapping = ecmwf_filter_scan_grib(grib_file)
log_checkpoint(f"Scan complete - found {len(groups)} groups", scan_start)

# Step 2: Build ensemble tree
tree_start = log_checkpoint("\n2. Building ensemble tree")
remote_options = {"anon": True}
ensemble_tree = fixed_ensemble_grib_tree(
    groups,
    remote_options=remote_options,
    debug_output=True  # Enable debug to see what's happening
)
log_checkpoint(f"Tree built with {len(ensemble_tree['refs'])} references", tree_start)

# Step 3: Create deflated store
deflate_start = log_checkpoint("\n3. Creating deflated store")
deflated_tree = copy.deepcopy(ensemble_tree)
strip_datavar_chunks(deflated_tree)
log_checkpoint("Deflated store created", deflate_start)

# Step 4: Save parquet
parquet_start = log_checkpoint("\n4. Saving parquet file")

# Create parquet file
def create_parquet_file(zstore: dict, output_file: str):
    """Save zarr store dictionary as parquet file."""
    data = []

    for key, value in zstore.items():
        if isinstance(value, str):
            encoded_value = value.encode('utf-8')
        elif isinstance(value, (list, dict)):
            encoded_value = json.dumps(value).encode('utf-8')
        else:
            encoded_value = str(value).encode('utf-8')

        data.append((key, encoded_value))

    df = pd.DataFrame(data, columns=['key', 'value'])
    df.to_parquet(output_file)
    print(f"  Saved {output_file} ({len(df)} rows)")

# Save comprehensive parquet
comprehensive_parquet = output_dir / "comprehensive_all_levels.parquet"
create_parquet_file(deflated_tree['refs'], str(comprehensive_parquet))
log_checkpoint(f"Parquet saved: {comprehensive_parquet}", parquet_start)

# Step 5: Validate with datatree
validate_start = log_checkpoint("\n5. Validating with xarray datatree")
try:
    import xarray as xr

    fs = fsspec.filesystem("reference", fo=ensemble_tree, remote_protocol='s3',
                          remote_options={'anon': True})
    mapper = fs.get_mapper("")

    dt = xr.open_datatree(mapper, engine="zarr", consolidated=False)
    log_checkpoint("DataTree validation successful", validate_start)

    # Check variables and levels
    print("\nAvailable variables:")
    for var_path in sorted(dt.keys())[:10]:
        print(f"  {var_path}")

    # Check if pressure level variables have all levels
    for var_name in ['t', 'u', 'v', 'q', 'gh', 'w']:
        var_path = f"{var_name}/instant/isobaricInhPa"
        if var_path in dt:
            node = dt[var_path]
            if hasattr(node, 'ds') and node.ds is not None:
                ds = node.ds
                if 'isobaricInhPa' in ds.coords:
                    levels = ds.coords['isobaricInhPa'].values
                    print(f"\n  ✅ {var_name}: {len(levels)} levels - {sorted(levels)}")
                else:
                    print(f"\n  ⚠️ {var_name}: No isobaricInhPa coordinate")
        else:
            print(f"\n  ❌ {var_name}: Not found in datatree")

except Exception as e:
    log_checkpoint(f"DataTree validation failed: {e}", validate_start)
    import traceback
    traceback.print_exc()

total_time = time.time() - start_time
print("\n" + "="*80)
print(f"✅ TEST COMPLETE! Total time: {total_time:.2f} seconds")
print("="*80)

# Show file size
file_size = comprehensive_parquet.stat().st_size / (1024 * 1024)
print(f"\nParquet file size: {file_size:.2f} MB")
print(f"Parquet file location: {comprehensive_parquet}")

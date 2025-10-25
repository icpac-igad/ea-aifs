#!/usr/bin/env python3
"""Quick script to list all variables in the parquet file."""

import pandas as pd
import json
from pathlib import Path

parquet_path = "ecmwf_20251020_00_efficient/members/ens_01/ens_01.parquet"

if not Path(parquet_path).exists():
    print(f"File not found: {parquet_path}")
    exit(1)

print(f"Reading: {parquet_path}\n")
df = pd.read_parquet(parquet_path)

zstore = {}
for _, row in df.iterrows():
    key = row['key']
    value = row['value']

    if isinstance(value, bytes):
        value = value.decode('utf-8')

    if isinstance(value, str) and value.startswith('[') and value.endswith(']'):
        try:
            value = json.loads(value)
        except:
            pass

    zstore[key] = value

print("="*80)
print("ALL VARIABLES WITH DATA")
print("="*80)

# Find all variables that have actual data chunks (not just metadata)
variables_with_data = set()

for key in zstore.keys():
    # Look for data chunk keys (contain numbers like /0 or /0.0.0)
    if '/' in key and not key.endswith(('.zarray', '.zattrs', '.zgroup')):
        # Extract the variable path (everything before the chunk index)
        parts = key.split('/')
        # Find where the chunk indices start (all numeric after this point)
        for i in range(len(parts)-1, -1, -1):
            if not all(c.isdigit() or c == '.' for c in parts[i]):
                var_path = '/'.join(parts[:i+1])
                # Check if this variable has a .zarray (actual data, not just coordinates)
                zarray_key = f"{var_path}/.zarray"
                if zarray_key in zstore:
                    variables_with_data.add(var_path)
                break

# Group by parameter type
params = {}
for var_path in sorted(variables_with_data):
    param_name = var_path.split('/')[0]
    if param_name not in params:
        params[param_name] = []
    params[param_name].append(var_path)

print(f"\nFound {len(variables_with_data)} variables with data:\n")

for param_name, var_list in sorted(params.items()):
    print(f"\n{param_name}:")
    for var_path in var_list:
        # Get shape info
        zarray_key = f"{var_path}/.zarray"
        if zarray_key in zstore:
            try:
                metadata = json.loads(zstore[zarray_key]) if isinstance(zstore[zarray_key], str) else zstore[zarray_key]
                shape = metadata.get('shape', [])
                # Only show variables with spatial data (typically 4D: time, step, lat, lon)
                if len(shape) >= 3:
                    print(f"  {var_path}")
                    print(f"    Shape: {shape}")
            except:
                pass

print("\n" + "="*80)
print("MAPPING FOR aifs-etl.py")
print("="*80)

# Generate the correct mapping based on what we found
print("\nCorrected variable path mapping:\n")
print("return {")

# Define what we need
needed_params = {
    '10u': 'u10m',
    '10v': 'v10m',
    '2d': 'd2m',
    '2t': 't2m',
    'msl': 'msl',
    'sp': 'sp',
    'skt': 'skt',
    'tcw': 'tcw',
    'lsm': 'lsm',
    'z': 'z',
    'slor': 'slor',
    'sdor': 'sdor',
    'sot': 'stl',
    'gh': 'gh',
    't': 't',
    'u': 'u',
    'v': 'v',
    'w': 'w',
    'q': 'q',
}

for param_key, param_name in needed_params.items():
    # Find matching variable path
    matching = [v for v in variables_with_data if v.startswith(param_name + '/')]
    if matching:
        # Pick the one with the most likely data (4D shape)
        best_match = None
        for m in matching:
            zarray_key = f"{m}/.zarray"
            if zarray_key in zstore:
                try:
                    metadata = json.loads(zstore[zarray_key]) if isinstance(zstore[zarray_key], str) else zstore[zarray_key]
                    shape = metadata.get('shape', [])
                    if len(shape) == 4 and shape[2] >= 100:  # Has lat/lon dimensions
                        best_match = m
                        break
                except:
                    pass

        if best_match:
            print(f"    '{param_key}': '{best_match}',")
        else:
            print(f"    # '{param_key}': NOT FOUND (searched for {param_name})")
    else:
        print(f"    # '{param_key}': NOT FOUND (searched for {param_name})")

print("}")

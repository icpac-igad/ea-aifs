#!/usr/bin/env python3
"""Check how pressure levels are stored in the parquet file."""

import pandas as pd
import json
import numpy as np
from pathlib import Path

parquet_path = "ecmwf_20251020_00_efficient/members/ens_01/ens_01.parquet"

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

# Check a pressure level variable
var_path = 'gh/instant/isobaricInhPa/gh'
zarray_key = f"{var_path}/.zarray"

if zarray_key in zstore:
    metadata = json.loads(zstore[zarray_key]) if isinstance(zstore[zarray_key], str) else zstore[zarray_key]
    print(f"Variable: {var_path}")
    print(f"  Shape: {metadata['shape']}")
    print(f"  Chunks: {metadata['chunks']}")
    print(f"  Dtype: {metadata['dtype']}")

# Check for isobaricInhPa coordinate
coord_path = 'gh/instant/isobaricInhPa/isobaricInhPa'
zarray_key2 = f"{coord_path}/.zarray"

if zarray_key2 in zstore:
    print(f"\nCoordinate: {coord_path}")
    metadata2 = json.loads(zstore[zarray_key2]) if isinstance(zstore[zarray_key2], str) else zstore[zarray_key2]
    print(f"  Shape: {metadata2['shape']}")
    print(f"  Dtype: {metadata2['dtype']}")

    # Try to extract the coordinate values
    for key in sorted(zstore.keys()):
        if key.startswith(coord_path + "/") and not key.endswith(('.zarray', '.zattrs')):
            chunk_ref = zstore[key]
            if isinstance(chunk_ref, str) and chunk_ref.startswith('base64:'):
                import base64
                base64_str = chunk_ref[7:]
                decoded = base64.b64decode(base64_str)

                # Try decompression
                try:
                    import blosc
                    decoded = blosc.decompress(decoded)
                except:
                    pass

                levels = np.frombuffer(decoded, dtype=np.dtype(metadata2['dtype']))
                print(f"\nAvailable pressure levels:")
                print(f"  {levels}")
                break
else:
    print(f"\nNo coordinate found at: {coord_path}")

# Check chunk structure for the data variable
print(f"\n\nChunk structure for {var_path}:")
chunk_keys = [k for k in sorted(zstore.keys()) if k.startswith(var_path + "/") and not k.endswith(('.zarray', '.zattrs'))]
print(f"  Found {len(chunk_keys)} chunks")
if chunk_keys:
    print(f"  First few chunks:")
    for ck in chunk_keys[:5]:
        print(f"    {ck}")

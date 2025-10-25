#!/usr/bin/env python3
"""Check what pressure level value is in the coordinate."""

import pandas as pd
import json
import numpy as np
import base64

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

# Check the isobaricInhPa coordinate value
coord_path = 'gh/instant/isobaricInhPa/isobaricInhPa'

# Check for scalar coordinate value
for key in sorted(zstore.keys()):
    if key.startswith(coord_path) and not key.endswith(('.zarray', '.zattrs', '.zgroup')):
        print(f"Found coordinate chunk: {key}")
        chunk_ref = zstore[key]
        print(f"  Type: {type(chunk_ref)}")
        print(f"  Value: {chunk_ref}")

        if isinstance(chunk_ref, str) and chunk_ref.startswith('base64:'):
            base64_str = chunk_ref[7:]
            decoded = base64.b64decode(base64_str)
            print(f"  Decoded bytes: {len(decoded)} bytes")

            # Try decompression
            try:
                import blosc
                decoded = blosc.decompress(decoded)
                print(f"  After decompression: {len(decoded)} bytes")
            except Exception as e:
                print(f"  No compression or failed: {e}")

            # Parse as float64 (scalar)
            value = np.frombuffer(decoded, dtype='<f8')
            print(f"  Pressure level: {value}")

print("\n" + "="*60)
print("Checking all pressure level variables:")
print("="*60)

pl_vars = ['gh', 't', 'u', 'v', 'w', 'q']
for pvar in pl_vars:
    coord_path = f'{pvar}/instant/isobaricInhPa/isobaricInhPa'
    for key in sorted(zstore.keys()):
        if key == coord_path or key.startswith(coord_path + '/0'):
            chunk_ref = zstore.get(key)
            if isinstance(chunk_ref, str) and chunk_ref.startswith('base64:'):
                base64_str = chunk_ref[7:]
                decoded = base64.b64decode(base64_str)
                try:
                    import blosc
                    decoded = blosc.decompress(decoded)
                except:
                    pass
                value = np.frombuffer(decoded, dtype='<f8')
                print(f"{pvar}: level = {value[0] if len(value) > 0 else 'N/A'}")
                break

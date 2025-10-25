#!/usr/bin/env python3
"""Decode the pressure level value."""

import pandas as pd
import json
import struct

parquet_path = "ecmwf_20251020_00_efficient/members/ens_01/ens_01.parquet"

df = pd.read_parquet(parquet_path)

zstore = {}
for _, row in df.iterrows():
    key = row['key']
    value = row['value']

    if isinstance(value, bytes):
        try:
            value = value.decode('utf-8')
        except:
            pass

    if isinstance(value, str) and value.startswith('[') and value.endswith(']'):
        try:
            value = json.loads(value)
        except:
            pass

    zstore[key] = value

# Check the coordinate
coord_key = 'gh/instant/isobaricInhPa/isobaricInhPa/0'
if coord_key in zstore:
    val = zstore[coord_key]
    print(f"Key: {coord_key}")
    print(f"Type: {type(val)}")
    print(f"Raw value: {repr(val)}")

    if isinstance(val, str):
        # Try to interpret as raw bytes
        print(f"Length: {len(val)} characters")
        print(f"Bytes: {[hex(ord(c)) for c in val]}")

        # Try as double (8 bytes)
        if len(val) >= 8:
            try:
                # Interpret first 8 bytes as double
                double_val = struct.unpack('<d', val[:8].encode('latin1'))[0]
                print(f"As double: {double_val}")
            except Exception as e:
                print(f"Failed to decode as double: {e}")

print("\n" + "="*60)
print("Checking if parquet has MULTIPLE files for different levels")
print("="*60)
print("\nThis file might only contain ONE pressure level (1000 hPa)")
print("Other levels might be in different parquet files or directories")

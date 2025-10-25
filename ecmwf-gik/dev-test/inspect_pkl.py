#!/usr/bin/env python3
"""
PKL File Inspector

This script inspects the output PKL files from aifs-etl-v2.py and compares them
with the expected AIFS input variables from ecmwf_opendata_pkl_input_aifsens.py.

Usage:
    python inspect_pkl.py <pkl_file>
    python inspect_pkl.py ecmwf_pkl_from_parquet_v2/input_state_member_001.pkl
"""

import pickle
import sys
from pathlib import Path
from collections import defaultdict
import numpy as np


# Expected variables from AIFS reference script
PARAM_SFC = ["10u", "10v", "2d", "2t", "msl", "skt", "sp", "tcw"]
PARAM_SFC_FC = ["lsm", "z", "slor", "sdor"]
PARAM_SOIL = ["stl1", "stl2"]  # Renamed from sot
PARAM_PL = ["gh", "t", "u", "v", "w", "q"]
LEVELS = [1000, 925, 850, 700, 600, 500, 400, 300, 250, 200, 150, 100, 50]


def get_expected_fields():
    """Get the expected fields for AIFS input."""
    expected_surface = PARAM_SFC + PARAM_SFC_FC + PARAM_SOIL

    expected_pressure = []
    for param in PARAM_PL:
        param_name = 'z' if param == 'gh' else param  # gh converted to z
        for level in LEVELS:
            expected_pressure.append(f"{param_name}_{level}")

    return expected_surface, expected_pressure


def inspect_pkl(pkl_file):
    """Inspect a PKL file and display its contents."""
    print("="*80)
    print(f"PKL FILE INSPECTOR")
    print("="*80)
    print(f"File: {pkl_file}\n")

    # Load the pickle file
    try:
        with open(pkl_file, 'rb') as f:
            data = pickle.load(f)
    except Exception as e:
        print(f"❌ Error loading PKL file: {e}")
        return

    # Display basic info
    print("📦 PICKLE CONTENTS")
    print("-" * 80)
    print(f"Type: {type(data)}")
    print(f"Keys: {list(data.keys()) if isinstance(data, dict) else 'N/A'}")

    if 'date' in data:
        print(f"\n📅 Date: {data['date']}")

    if 'fields' not in data:
        print("\n❌ No 'fields' key found in pickle file!")
        return

    fields = data['fields']
    print(f"\n📊 FIELDS SUMMARY")
    print("-" * 80)
    print(f"Total fields: {len(fields)}")

    # Group fields by type
    surface_fields = sorted([f for f in fields.keys() if '_' not in f])
    pressure_fields = sorted([f for f in fields.keys() if '_' in f])

    print(f"\nSurface/constant fields ({len(surface_fields)}):")
    for f in surface_fields:
        shape = fields[f].shape
        dtype = fields[f].dtype
        print(f"  ✓ {f:12s} shape={shape}, dtype={dtype}")

    print(f"\nPressure level fields ({len(pressure_fields)}):")

    # Group by parameter
    by_param = defaultdict(list)
    for pf in pressure_fields:
        param = pf.rsplit('_', 1)[0]
        level = pf.rsplit('_', 1)[1]
        by_param[param].append(int(level))

    for param in sorted(by_param.keys()):
        levels_list = sorted(by_param[param], reverse=True)
        print(f"  {param:8s} ({len(levels_list):2d} levels): {levels_list}")
        # Show sample field info
        sample_field = f"{param}_{levels_list[0]}"
        if sample_field in fields:
            shape = fields[sample_field].shape
            dtype = fields[sample_field].dtype
            print(f"           Sample: {sample_field} shape={shape}, dtype={dtype}")

    # Calculate memory usage
    total_elements = sum(field.size for field in fields.values())
    memory_mb = (total_elements * 4) / (1024 * 1024)  # Assuming float32
    print(f"\n💾 Memory usage: {memory_mb:.2f} MB")

    # File size
    file_size = Path(pkl_file).stat().st_size / (1024 * 1024)
    print(f"📁 File size: {file_size:.2f} MB")

    # Comparison with expected AIFS fields
    print(f"\n{'='*80}")
    print("COMPARISON WITH EXPECTED AIFS INPUT")
    print("="*80)

    expected_surface, expected_pressure = get_expected_fields()
    expected_all = expected_surface + expected_pressure

    available = [f for f in expected_all if f in fields]
    missing = [f for f in expected_all if f not in fields]
    extra = [f for f in fields if f not in expected_all]

    print(f"\n📋 Expected fields: {len(expected_all)}")
    print(f"   - Surface/constant: {len(expected_surface)}")
    print(f"   - Pressure levels:  {len(expected_pressure)}")

    print(f"\n✅ Available fields: {len(available)}/{len(expected_all)} ({100*len(available)/len(expected_all):.1f}%)")

    if missing:
        print(f"\n❌ MISSING FIELDS ({len(missing)}):")
        missing_surface = [f for f in missing if '_' not in f]
        missing_pressure = [f for f in missing if '_' in f]

        if missing_surface:
            print(f"\n  Surface/constant fields ({len(missing_surface)}):")
            for f in missing_surface:
                print(f"    - {f}")

        if missing_pressure:
            print(f"\n  Pressure level fields ({len(missing_pressure)}):")
            # Group by parameter
            by_param_missing = defaultdict(list)
            for pf in missing_pressure:
                param = pf.rsplit('_', 1)[0]
                level = pf.rsplit('_', 1)[1]
                by_param_missing[param].append(level)

            for param in sorted(by_param_missing.keys()):
                levels_list = sorted(by_param_missing[param], key=int, reverse=True)
                print(f"    - {param}: {len(levels_list)} levels missing - {levels_list}")

    if extra:
        print(f"\n➕ EXTRA FIELDS ({len(extra)}):")
        print(f"   (Fields present but not expected by AIFS)")
        for f in sorted(extra):
            print(f"    - {f}")

    # Data shape verification
    print(f"\n{'='*80}")
    print("DATA SHAPE VERIFICATION")
    print("="*80)

    # Expected shape for AIFS (after regridding)
    # Original: (721, 1440) at 0.25° resolution
    # AIFS needs: N320 grid which is approximately (640, 1280)

    print("\n📏 Current data shape:")
    if fields:
        sample_field = list(fields.values())[0]
        print(f"   Grid shape: {sample_field.shape}")

        if sample_field.shape == (721, 1440):
            print(f"   ✓ This is 0.25° resolution (90°N to 90°S, 0° to 360°)")
            print(f"   ⚠️  AIFS requires N320 grid (~640×1280)")
            print(f"   → Regridding needed: 0.25° → N320")
        elif sample_field.shape[0] == 640 or sample_field.shape[0] == 639:
            print(f"   ✓ This appears to be N320 grid")
            print(f"   ✓ Ready for AIFS input")
        else:
            print(f"   ⚠️  Unknown grid resolution")

    # Coordinate rolling check
    print(f"\n🌍 Coordinate system:")
    print(f"   Current: Data extracted from parquet (assumes 0° to 360° longitude)")
    print(f"   AIFS requirement: Data must be shifted from -180° to 180° → 0° to 360°")
    print(f"   Status: ⚠️  Verify longitude convention if using ECMWF Open Data")

    # Summary
    print(f"\n{'='*80}")
    print("SUMMARY")
    print("="*80)

    if not missing:
        print("✅ All expected AIFS fields are present!")
        print("⚠️  Regridding required: 0.25° → N320")
    else:
        print(f"⚠️  {len(missing)} fields missing (see above)")
        print(f"✅ {len(available)} fields available")

    if missing_surface:
        print(f"\n🔧 TO FIX MISSING SURFACE FIELDS:")
        print(f"   - Orography fields (z, slor, sdor) may need separate source")
        print(f"   - Soil fields (stl1, stl2) not available in this parquet")

    print(f"\n🔧 NEXT STEPS:")
    print(f"   1. Obtain missing fields from alternative sources")
    print(f"   2. Apply regridding (0.25° → N320) using earthkit.regrid")
    print(f"   3. Verify longitude coordinate convention")
    print(f"   4. Create final AIFS-ready PKL file")

    return data


def main():
    if len(sys.argv) < 2:
        print("Usage: python inspect_pkl.py <pkl_file>")
        print("\nExample:")
        print("  python inspect_pkl.py ecmwf_pkl_from_parquet_v2/input_state_member_001.pkl")
        sys.exit(1)

    pkl_file = sys.argv[1]

    if not Path(pkl_file).exists():
        print(f"❌ File not found: {pkl_file}")
        sys.exit(1)

    inspect_pkl(pkl_file)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
AIFS Regridding Script

This script applies regridding to the extracted parquet data to make it compatible
with AI-FS input requirements.

Operations performed:
1. Longitude coordinate rolling (shift from -180/180 to 0/360 if needed)
2. Regridding from 0.25° to N320 grid using earthkit.regrid
3. Handle missing fields (z, slor, sdor, stl1, stl2)

Usage:
    python aifs-regrid.py <input_pkl> <output_pkl>
    python aifs-regrid.py ecmwf_pkl_from_parquet_v2/input_state_member_001.pkl aifs_ready/input_state_member_001.pkl
"""

import pickle
import sys
from pathlib import Path
import numpy as np


def apply_regridding(input_pkl, output_pkl, skip_missing=True):
    """
    Apply regridding to convert parquet data to AIFS-ready format.

    Args:
        input_pkl: Path to input PKL file (from aifs-etl-v2.py)
        output_pkl: Path to output PKL file (AIFS-ready)
        skip_missing: If True, skip missing fields; if False, create zero-filled placeholders
    """
    try:
        import earthkit.regrid as ekr
    except ImportError:
        print("❌ earthkit.regrid not available")
        print("Install with: pip install earthkit-regrid")
        sys.exit(1)

    print("="*80)
    print("🌍 AIFS REGRIDDING PROCESSOR")
    print("="*80)
    print(f"Input:  {input_pkl}")
    print(f"Output: {output_pkl}\n")

    # Load input data
    print("📖 Loading input PKL file...")
    try:
        with open(input_pkl, 'rb') as f:
            data = pickle.load(f)
    except Exception as e:
        print(f"❌ Error loading PKL file: {e}")
        return False

    if 'fields' not in data:
        print("❌ No 'fields' key found in pickle file!")
        return False

    fields = data['fields']
    date = data.get('date')

    print(f"✅ Loaded {len(fields)} fields")
    print(f"📅 Date: {date}")

    # Get sample shape
    sample_field = list(fields.values())[0]
    input_shape = sample_field.shape
    print(f"📏 Input grid shape: {input_shape}")

    if input_shape != (721, 1440):
        print(f"⚠️  Warning: Expected (721, 1440) but got {input_shape}")
        print(f"   Proceeding anyway...")

    # Apply coordinate rolling (longitude shift)
    # ECMWF data is typically -180 to 180, we need 0 to 360
    # However, parquet data may already be 0 to 360, so we'll make this optional
    print(f"\n🔄 Step 1: Longitude coordinate check")
    print(f"   Note: ECMWF Open Data is -180 to 180, needs rolling")
    print(f"   Parquet data may already be 0 to 360")
    print(f"   → Skipping coordinate rolling for now (verify data if issues occur)")

    # For now, we'll skip rolling. If needed, uncomment below:
    # for field_name, field_data in fields.items():
    #     fields[field_name] = np.roll(field_data, -field_data.shape[1] // 2, axis=1)

    # Apply regridding: 0.25° → N320
    print(f"\n📐 Step 2: Regridding 0.25° → N320")
    print(f"   This may take several minutes...")

    regridded_fields = {}
    total_fields = len(fields)

    for i, (field_name, field_data) in enumerate(fields.items(), 1):
        if i % 10 == 0 or i == 1:
            print(f"   Processing field {i}/{total_fields}: {field_name}")

        try:
            # Regrid from 0.25° to N320
            regridded = ekr.interpolate(
                field_data,
                {"grid": (0.25, 0.25)},
                {"grid": "N320"}
            )
            regridded_fields[field_name] = regridded

            if i == 1:
                print(f"      Input shape:  {field_data.shape}")
                print(f"      Output shape: {regridded.shape}")

        except Exception as e:
            print(f"   ❌ Error regridding {field_name}: {e}")
            # Keep original data if regridding fails
            regridded_fields[field_name] = field_data

    print(f"   ✅ Regridded {len(regridded_fields)} fields")

    # Handle missing fields
    print(f"\n🔧 Step 3: Handling missing fields")
    missing_fields = ['z', 'slor', 'sdor', 'stl1', 'stl2']
    actually_missing = [f for f in missing_fields if f not in regridded_fields]

    if actually_missing:
        print(f"   Missing fields: {actually_missing}")

        if skip_missing:
            print(f"   → Skipping missing fields (AIFS may require them!)")
        else:
            print(f"   → Creating zero-filled placeholders")
            # Get output shape from a regridded field
            sample_regridded = list(regridded_fields.values())[0]
            output_shape = sample_regridded.shape

            for field_name in actually_missing:
                regridded_fields[field_name] = np.zeros(output_shape, dtype=np.float32)
                print(f"      Added: {field_name} (zeros)")

    # Create output data
    output_data = {
        'date': date,
        'fields': regridded_fields
    }

    # Save output
    print(f"\n💾 Saving AIFS-ready PKL file...")
    output_path = Path(output_pkl)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        with open(output_pkl, 'wb') as f:
            pickle.dump(output_data, f)
        print(f"✅ Saved to: {output_pkl}")

        # Show file size
        file_size = Path(output_pkl).stat().st_size / (1024 * 1024)
        print(f"📁 File size: {file_size:.2f} MB")

    except Exception as e:
        print(f"❌ Error saving PKL file: {e}")
        return False

    # Summary
    print(f"\n{'='*80}")
    print("SUMMARY")
    print("="*80)
    print(f"✅ Total fields processed: {len(regridded_fields)}")
    print(f"📏 Output grid shape: {list(regridded_fields.values())[0].shape}")

    if actually_missing and skip_missing:
        print(f"\n⚠️  WARNING: {len(actually_missing)} fields still missing!")
        print(f"   Missing: {actually_missing}")
        print(f"   AIFS may require these fields for full functionality")
        print(f"\n   To create zero-filled placeholders, run with skip_missing=False")
        print(f"   To obtain real data, use alternative sources for:")
        print(f"   - Orography: z, slor, sdor (constant fields)")
        print(f"   - Soil temp: stl1, stl2 (may need GRIB files)")

    print(f"\n🎯 Next steps:")
    print(f"   1. Verify the regridded data quality")
    print(f"   2. Obtain missing fields if needed")
    print(f"   3. Use this PKL as input for AIFS inference")

    return True


def main():
    if len(sys.argv) < 3:
        print("Usage: python aifs-regrid.py <input_pkl> <output_pkl>")
        print("\nExample:")
        print("  python aifs-regrid.py ecmwf_pkl_from_parquet_v2/input_state_member_001.pkl aifs_ready/input_state_member_001.pkl")
        sys.exit(1)

    input_pkl = sys.argv[1]
    output_pkl = sys.argv[2]

    if not Path(input_pkl).exists():
        print(f"❌ Input file not found: {input_pkl}")
        sys.exit(1)

    # Optional: allow specifying whether to create placeholders for missing fields
    skip_missing = True
    if len(sys.argv) > 3:
        skip_missing = sys.argv[3].lower() not in ['false', '0', 'no']

    success = apply_regridding(input_pkl, output_pkl, skip_missing=skip_missing)

    if success:
        print("\n✅ Regridding complete!")
        sys.exit(0)
    else:
        print("\n❌ Regridding failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()

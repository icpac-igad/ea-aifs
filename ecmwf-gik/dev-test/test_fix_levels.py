#!/usr/bin/env python3
"""
Test the fix for extracting all pressure levels.
Quick validation before running the full parquet creation.
"""

from test_run_ecmwf_step1_scangrib import ecmwf_idx_df_create_with_keys

grib_file = "s3://ecmwf-forecasts/20251020/00z/ifs/0p25/enfo/20251020000000-0h-enfo-ef.grib2"

print("="*80)
print("TESTING FIX FOR ALL PRESSURE LEVELS")
print("="*80)

print(f"\nProcessing: {grib_file}")
print("This will check if ALL 13 pressure levels are now being extracted...\n")

try:
    idx_mapping, combined_dict = ecmwf_idx_df_create_with_keys(grib_file)

    print(f"\n✅ Index mapping created with {len(idx_mapping)} entries")
    print(f"   Combined dict has {len(combined_dict)} unique parameter combinations")

    # The combined_dict contains the parameter combinations
    # Check for pressure level parameters
    from collections import defaultdict
    params_by_var = defaultdict(set)

    for key, filter_dict in combined_dict.items():
        if filter_dict['levtype'] == 'pl':
            param_name = filter_dict['param']
            level = filter_dict['levelist']
            params_by_var[param_name].add(level)

    print("\nPressure levels found for each parameter:")
    for param, levels in sorted(params_by_var.items()):
        sorted_levels = sorted([int(l) for l in levels])
        print(f"  {param}: {len(sorted_levels)} levels - {sorted_levels}")

    # Check if we have all 13 levels for key parameters
    expected_levels = {50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000}
    key_params = ['t', 'u', 'v', 'q', 'gh', 'w']

    print("\nValidation:")
    all_good = True
    for param in key_params:
        if param in params_by_var:
            found_levels = {int(l) for l in params_by_var[param]}
            missing = expected_levels - found_levels
            if missing:
                print(f"  ❌ {param}: Missing levels {sorted(missing)}")
                all_good = False
            else:
                print(f"  ✅ {param}: All 13 levels present!")
        else:
            print(f"  ⚠️ {param}: Not found in index")
            all_good = False

    # Check for soil parameters
    sol_params = {}
    for key, filter_dict in combined_dict.items():
        if filter_dict['levtype'] == 'sol':
            param = filter_dict['param']
            level = filter_dict['levelist']
            if param not in sol_params:
                sol_params[param] = set()
            sol_params[param].add(level)

    print(f"\n📊 Soil parameters found: {len(sol_params)} unique params")
    if sol_params:
        print("Soil parameters with levels:")
        for param, levels in sorted(sol_params.items()):
            print(f"  {param}: levels {sorted([int(l) for l in levels])}")

    # Check for surface parameters
    sfc_params = set()
    for key, filter_dict in combined_dict.items():
        if filter_dict['levtype'] == 'sfc':
            sfc_params.add(filter_dict['param'])

    print(f"\n📊 Surface parameters found: {len(sfc_params)} unique params")
    if sfc_params:
        print("Sample surface parameters:")
        for param in sorted(sfc_params)[:10]:
            print(f"  {param}")

    print("\n" + "="*80)
    if all_good:
        print("✅ SUCCESS! All 13 pressure levels are now being extracted!")
    else:
        print("⚠️ Some levels are still missing - check output above")
    print("="*80)

except Exception as e:
    print(f"\n❌ Error: {e}")
    import traceback
    traceback.print_exc()

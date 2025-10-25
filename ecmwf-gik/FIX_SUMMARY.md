# Fix Summary: Missing Pressure Levels in ECMWF Parquet Files

## Problem Identified

The parquet files created by `ecmwf_ensemble_par_creator_efficient.py` only contained **ONE pressure level (50 hPa)** instead of all 13 required levels, plus missing soil temperature data.

## Root Cause

Located in `/scratch/notebook/test_run_ecmwf_step1_scangrib.py` at **lines 135-136**:

```python
# OLD CODE (BROKEN):
combined_params = edf[((edf['levtype'] == 'pl') &
                       (edf['levelist'] == '50')) |  # ← HARD-CODED TO LEVEL 50 ONLY!
                      (edf['levtype'] == 'sfc')]
```

The function `ecmwf_idx_unique_dict()` was **hard-coded** to only extract pressure level 50, filtering out all other levels (100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000 hPa) before they could be processed.

## Solution Applied

**File Modified:** `test_run_ecmwf_step1_scangrib.py`
**Function:** `ecmwf_idx_unique_dict()`
**Lines:** 131-173

### Changes Made:

1. **Added all 13 pressure levels:**
   ```python
   pressure_levels = ['50', '100', '150', '200', '250', '300', '400', '500',
                     '600', '700', '850', '925', '1000']
   ```

2. **Added all 3 soil levels:**
   ```python
   soil_levels = ['1', '2', '4']
   ```

3. **Updated filter logic** to include ALL levels:
   ```python
   combined_params = edf[
       ((edf['levtype'] == 'pl') & (edf['levelist'].isin(pressure_levels))) |
       ((edf['levtype'] == 'sol') & (edf['levelist'].isin(soil_levels))) |
       (edf['levtype'] == 'sfc')
   ]
   ```

4. **Updated key generation** to include level in the key:
   ```python
   if row['levtype'] in ['pl', 'sol']:
       key = f"{row['param']}_{row['levtype']}_{row['levelist']}"
   else:
       key = f"{row['param']}_{row['levtype']}"
   ```

## Verification

Ran test script (`test_fix_levels.py`) which confirmed:

### ✅ Pressure Level Variables (9 parameters × 13 levels each):
- `d` (dewpoint depression)
- `gh` (geopotential height)
- `q` (specific humidity)
- `r` (relative humidity)
- `t` (temperature)
- `u` (u-wind component)
- `v` (v-wind component)
- `vo` (vorticity)
- `w` (vertical velocity)

**All 13 levels present:** 50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000 hPa

### ✅ Soil Variables (2 parameters × 3 levels each):
- `sot` (soil temperature): levels 1, 2, 4
- `vsw` (volumetric soil water): levels 1, 2, 4

### ✅ Surface Variables (32 parameters):
Including: `10u`, `10v`, `2d`, `2t`, `lsm`, `msl`, `sp`, `skt`, `tcw`, etc.

## What Was NOT Found (Expected)

The following variables are **NOT in the GRIB2 file** and need to be obtained separately:
- `z` - geopotential at surface (orography)
- `slor` - slope of sub-gridscale orography
- `sdor` - standard deviation of orography

These are static fields and may be in a different file or need to be downloaded from a different source.

## Impact on aifs-etl.py

After regenerating the parquet files with the fix, `aifs-etl.py` should now be able to extract:

### Before Fix:
- 9 surface fields
- 6 pressure level fields **at ONLY 50 hPa** = 6 fields
- **Total: 15 fields** ❌

### After Fix:
- 9 surface fields
- 6 pressure level fields **× 13 levels** = 78 fields
- 2 soil fields × 2 levels (if needed) = 4 fields
- **Total: 87-91 fields** ✅

## Next Steps

1. ✅ **DONE:** Fix applied to `test_run_ecmwf_step1_scangrib.py`
2. ✅ **DONE:** Verified fix extracts all levels
3. 🔄 **IN PROGRESS:** Test parquet creation with fixed code
4. ⏳ **TODO:** Re-run `ecmwf_ensemble_par_creator_efficient.py` to create new parquet files
5. ⏳ **TODO:** Update `aifs-etl.py` to properly handle multi-level variables
6. ⏳ **TODO:** Obtain orography fields (z, slor, sdor) from separate source

## Files Modified

1. `/scratch/notebook/test_run_ecmwf_step1_scangrib.py`
   - Function: `ecmwf_idx_unique_dict()` (lines 131-173)

## Test Scripts Created

1. `/scratch/notebook/test_fix_levels.py` - Validates all levels are extracted
2. `/scratch/notebook/create_test_parquet.py` - Creates test parquet file
3. `/scratch/notebook/diagnose_cfgrib_levels.py` - Diagnoses cfgrib behavior
4. `/scratch/notebook/list_all_variables.py` - Lists all variables in parquet
5. `/scratch/notebook/check_levels.py` - Checks level structure
6. `/scratch/notebook/decode_level.py` - Decodes level values

## Summary

The fix was a **one-line change** (hard-coded level 50 → all 13 levels) that will unlock extraction of **78 pressure level fields** instead of just 6, providing complete data for AI-FS training/inference.

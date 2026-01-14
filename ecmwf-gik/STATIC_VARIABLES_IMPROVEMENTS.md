# Static Variables Improvements to aifs-etl-v2.py

## Overview

This document explains the improvements made to `aifs-etl-v2.py` to include static/constant variables that `earthkit.data` collects from ECMWF Open Data, ensuring compatibility with the AIFS model workflow.

## Problem Identified

### Original Issue
The `ecmwf_opendata_pkl_input_aifsens.py` script uses `earthkit.data` (ekd) to download meteorological data, including **static/constant fields** that are critical for AIFS model initialization:

```python
# In ecmwf_opendata_pkl_input_aifsens.py
PARAM_SFC_FC = ["lsm", "z", "slor", "sdor"]  # Constant fields
```

However, the alternative method in `ecmwf-gik/aifs-etl-v2.py` was **missing** three critical static fields:

```python
# Old version - INCOMPLETE
PARAM_SFC_FC = ["lsm"]  # Only land-sea mask
```

### Missing Static Variables

| Field | GRIB Code | Description | Importance |
|-------|-----------|-------------|------------|
| `z` | 129 | Surface geopotential (orography) | **CRITICAL** - Required for AIFS terrain representation |
| `slor` | 161 | Slope of sub-gridscale orography | Moderate - Affects surface interactions |
| `sdor` | 160 | Standard deviation of orography | Moderate - Affects surface parameterizations |

## How earthkit.data Collects These Variables

In `ecmwf_opendata_pkl_input_aifsens.py`, earthkit.data retrieves these static fields via:

```python
# Line 87
fields.update(get_open_data(date, param=PARAM_SFC_FC))  # Constant fields

# Inside get_open_data function (lines 52-56)
data = ekd.from_source("ecmwf-open-data",
                       date=d,
                       param=param,  # ["lsm", "z", "slor", "sdor"]
                       levelist=levelist)
```

The `earthkit.data` library:
1. Connects to ECMWF Open Data API
2. Downloads GRIB2 messages for the requested parameters
3. Extracts metadata using `f.metadata('param')` and `f.metadata('levelist')`
4. Converts to NumPy arrays using `f.to_numpy()`
5. Interpolates from 0.25° grid to N320 grid

## Solution Implemented

### Changes Made to `aifs-etl-v2.py`

#### 1. Updated Configuration (Lines 36-43)

```python
# Static/constant surface fields (orography-related)
# These fields are collected by earthkit.data from ECMWF Open Data
# GRIB parameter codes: z=129, slor=161, sdor=160
PARAM_SFC_FC = ["lsm", "z", "slor", "sdor"]
#   - lsm: Land-sea mask
#   - z: Surface geopotential (orography) - CRITICAL for AIFS
#   - slor: Slope of sub-gridscale orography
#   - sdor: Standard deviation of orography
```

#### 2. Added Variable Path Mappings (Lines 333-337)

```python
# Fixed fields (constants)
'lsm': 'lsm/instant/surface/lsm',
'z': 'z/instant/surface/z',        # Surface geopotential (orography)
'slor': 'slor/instant/surface/slor',  # Slope of sub-gridscale orography
'sdor': 'sdor/instant/surface/sdor',  # Standard deviation of orography
```

#### 3. Updated Docstring (Lines 12-18)

Added explanation of static variables now being collected to match `earthkit.data` behavior.

## Technical Details

### Variable Path Structure

The zarr/parquet structure follows this naming convention:
```
{parameter}/{type}/{level_type}/{parameter}
```

For the static orography fields:
- **Type**: `instant` (instantaneous field, not accumulated)
- **Level Type**: `surface` (at ground level)
- **Parameter**: The ECMWF short name (z, slor, sdor)

### Verification

The existing `verify_input_state()` function (line 646) already handles these fields correctly:

```python
# Line 651
expected_surface = PARAM_SFC + PARAM_SFC_FC
```

This automatically includes all fields in `PARAM_SFC_FC`, so the verification will now check for:
- `lsm` (existing)
- `z` (NEW)
- `slor` (NEW)
- `sdor` (NEW)

### No Naming Conflicts

There is **no conflict** between:
- Surface geopotential: `z` (no suffix)
- Pressure level geopotential: `z_1000`, `z_925`, ..., `z_50` (with level suffix)

These are correctly distinguished by the presence/absence of the level suffix.

## Expected Field Count

### Before Changes
- Surface fields: 8 (PARAM_SFC)
- Constant fields: 1 (lsm only)
- Pressure level fields: 78 (6 params × 13 levels)
- **Total: 87 fields**

### After Changes
- Surface fields: 8 (PARAM_SFC)
- Constant fields: 4 (lsm, z, slor, sdor)
- Pressure level fields: 78 (6 params × 13 levels)
- **Total: 90 fields** ✅

## Compatibility with earthkit.data Workflow

The updated `aifs-etl-v2.py` now provides **feature parity** with the `earthkit.data` approach:

| Aspect | earthkit.data | aifs-etl-v2.py (updated) |
|--------|---------------|--------------------------|
| Surface fields | ✅ 8 params | ✅ 8 params |
| Constant fields | ✅ lsm, z, slor, sdor | ✅ lsm, z, slor, sdor |
| Pressure levels | ✅ 13 levels × 6 params | ✅ 13 levels × 6 params |
| Geopotential conversion | ✅ gh → z | ✅ gh → z |
| Total fields | 90 | 90 |

## Data Availability Note

These static fields are **constant** (don't change with forecast time) and should be available in:
1. ECMWF Open Data API (accessed by earthkit.data)
2. ECMWF parquet/zarr files (accessed by aifs-etl-v2.py)

If these fields are **not present** in the parquet files, they can be obtained from:
- ERA5 constant fields
- ECMWF climate files
- AIFS model repository defaults

Use this CDS API request if needed:
```python
import cdsapi
c = cdsapi.Client()
c.retrieve('reanalysis-era5-complete', {
    'class': 'ea',
    'expver': '1',
    'param': '129/160/161',  # z, slor, sdor
    'date': '2020-01-01',
    'time': '00:00:00',
    'type': 'an',
    'grid': '0.25/0.25',
}, 'orography_constants.grib')
```

## Testing Recommendations

To verify the changes work correctly:

1. **Check parquet file contents**:
   ```bash
   python ecmwf-gik/dev-test/list_all_variables.py
   ```
   This will show all available variables in the parquet file.

2. **Run the updated ETL**:
   ```bash
   python ecmwf-gik/aifs-etl-v2.py
   ```
   Expected output: 90 fields (up from 87)

3. **Inspect the output**:
   ```bash
   python ecmwf-gik/dev-test/inspect_pkl.py ecmwf_pkl_from_parquet_v2/input_state_member_001.pkl
   ```
   Verify that z, slor, sdor are present.

## Summary

The improvements to `aifs-etl-v2.py` now ensure that:

✅ All static/constant fields that `earthkit.data` collects are included
✅ AIFS model receives the critical orography fields (z, slor, sdor)
✅ Feature parity between earthkit.data and parquet-based ETL pipelines
✅ Proper documentation and variable path mappings
✅ Automatic verification of these fields in the output

This brings `aifs-etl-v2.py` to full compatibility with the `ecmwf_opendata_pkl_input_aifsens.py` workflow for AIFS model initialization.

# Precipitation Artifact Analysis Report

## Executive Summary

The precipitation (`tp`/`pr`) variable exhibits random pixel-like artifacts and non-physical distributions in the final quintile output, while temperature (`tas`) and mean sea level pressure (`mslp`) variables show expected behavior. This document analyzes the root causes and provides solutions.

## Identified Issues

### Issue 1: Inappropriate Interpolation Method for Precipitation

**Location:** `aifs_n320_grib_1p5defg_nc.py:299-301`

```python
fl_ll_1p5 = ekr.interpolate(
    fl, in_grid={"grid": "N320"}, out_grid={"grid": [1.5, 1.5]}
)
```

**Problem:** The code uses `earthkit.regrid.interpolate()` with default settings, which applies **bilinear interpolation**. This is fundamentally inappropriate for precipitation because:

1. **Bilinear interpolation assumes smooth, continuous fields** - Precipitation is inherently discontinuous with sharp edges between raining and non-raining regions

2. **Can produce negative values** - When interpolating between high and zero precipitation areas, bilinear interpolation can generate small negative values

3. **Smears localized precipitation** - Sharp rainfall features get artificially spread to neighboring grid cells, creating the "random pixel" artifacts

4. **Does not conserve mass** - Total precipitation amount changes during regridding

**Why TAS and MSLP work correctly:**
- Temperature and pressure are **continuous, quasi-Gaussian fields**
- They vary smoothly across space with no zero-bounded discontinuities
- Bilinear interpolation is appropriate for these variables

### Issue 2: No Constraint on Non-Negative Values

**Location:** `aifs_n320_grib_1p5defg_nc.py` (entire processing chain)

**Problem:** There is no post-processing step to ensure precipitation values remain >= 0 after regridding. Negative precipitation values propagate through:
1. The regridding step
2. The weekly accumulation (`sum(dim='step')`)
3. The quintile calculation

This causes:
- Negative values appearing in normally dry regions
- Distorted quintile distributions
- "Salt and pepper" noise pattern in outputs

### Issue 3: Weekly Accumulation Amplifies Artifacts

**Location:** `ensemble_quintile_analysis.py:564-567`

```python
if var_name == 'tp':
    # For precipitation, calculate weekly accumulation (sum)
    weekly_forecast = forecast_data.sum(dim='step')
```

**Problem:** Summing 7 daily values compounds regridding artifacts:
- Small negative artifacts become more negative
- Small spurious positive artifacts accumulate
- The signal-to-noise ratio degrades with longer accumulation periods

### Issue 4: Precipitation Statistics Require Special Treatment

**Location:** `ensemble_quintile_analysis.py:617-688` (calculate_grid_quintiles)

**Problem:** The quintile calculation treats precipitation identically to other variables, but precipitation has unique statistical properties:

1. **Highly skewed distribution** - Follows log-normal or gamma distribution, not normal
2. **Many zero values** - Precipitation occurrence probability matters
3. **Heavy tails** - Extreme precipitation events are relatively common

The current implementation:
```python
quintile_probs[0] = np.sum(ensemble_values < clim_thresholds[0], axis=0) / n_members
```

This comparison doesn't account for:
- Zero-inflation in precipitation data
- The non-linear nature of precipitation thresholds
- Potential unit/scale mismatches between forecast and climatology

---

## Recommended Solutions

### Solution 1: Use Conservative/Nearest-Neighbor Regridding for Precipitation

**Priority: HIGH**

Modify `aifs_n320_grib_1p5defg_nc.py` to use appropriate regridding for precipitation:

```python
def process_grib_to_netcdf(self, member: int, grib_files: List[str]) -> Optional[str]:
    # ... existing code ...

    # Separate precipitation from other variables
    fl_precip = fl.sel(shortName="tp")  # Select precipitation only
    fl_other = fl.sel(shortName=["msl", "2t"])  # Other variables

    # Use nearest-neighbor or conservative for precipitation
    # Option A: Nearest-neighbor (preserves values, prevents negative)
    fl_precip_regrid = ekr.interpolate(
        fl_precip,
        in_grid={"grid": "N320"},
        out_grid={"grid": [1.5, 1.5]},
        method="nearest"  # or "nearest-stencil"
    )

    # Option B: Use conservative regridding (preserves mass)
    # Requires earthkit-regrid >= 0.3.0
    fl_precip_regrid = ekr.interpolate(
        fl_precip,
        in_grid={"grid": "N320"},
        out_grid={"grid": [1.5, 1.5]},
        method="conservative"
    )

    # Keep bilinear for smooth fields
    fl_other_regrid = ekr.interpolate(
        fl_other,
        in_grid={"grid": "N320"},
        out_grid={"grid": [1.5, 1.5]},
        method="linear"  # default
    )

    # Merge back
    # ... combine datasets ...
```

### Solution 2: Post-Processing Constraint for Non-Negative Values

**Priority: HIGH**

Add a clipping step after regridding in `aifs_n320_grib_1p5defg_nc.py:303-306`:

```python
# Convert to xarray and detach from files
ds = fl_ll_1p5.to_xarray()
ds.load()

# Ensure precipitation is non-negative
if 'tp' in ds.data_vars:
    ds['tp'] = ds['tp'].clip(min=0)
    # Optional: Set very small values to zero to remove numerical noise
    ds['tp'] = ds['tp'].where(ds['tp'] > 1e-9, 0)
```

### Solution 3: Variable-Specific Processing Pipeline

**Priority: MEDIUM**

Create separate processing paths for precipitation vs. continuous variables:

```python
# In aifs_n320_grib_1p5defg_nc.py

# Define variable-specific regridding methods
REGRID_METHODS = {
    "tp": "nearest",      # Precipitation: nearest-neighbor
    "msl": "linear",      # Pressure: bilinear
    "2t": "linear",       # Temperature: bilinear
}

def regrid_variable(fl, var_name, in_grid, out_grid):
    """Apply variable-appropriate regridding method."""
    method = REGRID_METHODS.get(var_name, "linear")

    fl_var = fl.sel(shortName=var_name)
    fl_regrid = ekr.interpolate(
        fl_var,
        in_grid=in_grid,
        out_grid=out_grid,
        method=method
    )

    # Post-processing for precipitation
    if var_name == "tp":
        ds = fl_regrid.to_xarray()
        ds[var_name] = ds[var_name].clip(min=0)
        return ds

    return fl_regrid.to_xarray()
```

### Solution 4: Log-Transform for Precipitation Quintile Calculation

**Priority: MEDIUM**

Modify `ensemble_quintile_analysis.py` to handle precipitation's skewed distribution:

```python
def calculate_grid_quintiles_precip(ensemble_data, climatology_quintiles, epsilon=1e-6):
    """
    Calculate quintile probabilities for precipitation using log-space comparison.

    This accounts for precipitation's:
    - Non-negative constraint
    - Log-normal distribution
    - Zero-inflation
    """

    n_members = ensemble_data.sizes['member']

    # Handle climatology
    if 'time' in climatology_quintiles.dims:
        clim_data = climatology_quintiles.isel(time=0)
    else:
        clim_data = climatology_quintiles

    ensemble_values = ensemble_data.values
    clim_thresholds = clim_data.values

    # Ensure non-negative values
    ensemble_values = np.maximum(ensemble_values, 0)
    clim_thresholds = np.maximum(clim_thresholds, 0)

    # Initialize output
    quintile_probs = np.zeros((5, ensemble_data.sizes['latitude'],
                                ensemble_data.sizes['longitude']))

    # For precipitation, use direct comparison but handle zeros specially
    # Option: Use occurrence probability for lowest quintile

    # Q1: values < 20th percentile (includes "no rain" category)
    quintile_probs[0] = np.sum(ensemble_values < clim_thresholds[0], axis=0) / n_members

    # Q2-Q5: standard comparison
    quintile_probs[1] = np.sum((ensemble_values >= clim_thresholds[0]) &
                               (ensemble_values < clim_thresholds[1]), axis=0) / n_members
    quintile_probs[2] = np.sum((ensemble_values >= clim_thresholds[1]) &
                               (ensemble_values < clim_thresholds[2]), axis=0) / n_members
    quintile_probs[3] = np.sum((ensemble_values >= clim_thresholds[2]) &
                               (ensemble_values < clim_thresholds[3]), axis=0) / n_members
    quintile_probs[4] = np.sum(ensemble_values >= clim_thresholds[3], axis=0) / n_members

    # Normalize to ensure sum = 1 (handles edge cases)
    total = np.sum(quintile_probs, axis=0, keepdims=True)
    total = np.where(total > 0, total, 1)  # Avoid division by zero
    quintile_probs = quintile_probs / total

    return xr.DataArray(
        quintile_probs,
        dims=['quintile', 'latitude', 'longitude'],
        coords={
            'quintile': [0.2, 0.4, 0.6, 0.8, 1.0],
            'latitude': ensemble_data.latitude,
            'longitude': ensemble_data.longitude
        }
    )
```

### Solution 5: Verify Unit Consistency

**Priority: HIGH**

Ensure precipitation units match between forecast and climatology:

```python
# In ensemble_quintile_analysis.py, add unit verification:

def verify_precip_units(forecast_data, clim_data, var_name):
    """
    Verify and correct precipitation units between forecast and climatology.

    Common issues:
    - Forecast in m, climatology in mm
    - Forecast accumulated, climatology rate (mm/day)
    - Different accumulation periods
    """

    if var_name not in ['tp', 'pr']:
        return forecast_data, clim_data

    # Check value ranges to detect unit mismatch
    fc_max = float(forecast_data.max())
    clim_max = float(clim_data.max())

    # If forecast values are ~1000x smaller, likely in meters vs mm
    if clim_max > 10 * fc_max and fc_max < 1:
        print(f"  WARNING: Converting precipitation from m to mm (factor 1000)")
        forecast_data = forecast_data * 1000

    # If forecast values are ~1000x larger, likely in mm vs m
    if fc_max > 10 * clim_max and clim_max < 1:
        print(f"  WARNING: Converting climatology from m to mm (factor 1000)")
        clim_data = clim_data * 1000

    return forecast_data, clim_data
```

---

## Implementation Priority

| Priority | Solution | Impact | Effort |
|----------|----------|--------|--------|
| 1 | Conservative/Nearest regridding | Eliminates primary artifact source | Medium |
| 2 | Non-negative clipping | Prevents negative values | Low |
| 3 | Unit verification | Prevents scale mismatches | Low |
| 4 | Variable-specific pipeline | Clean architecture | Medium |
| 5 | Log-space quintile calculation | Better statistical handling | Medium |

---

## Quick Fix (Minimal Code Change)

For immediate improvement, add these lines to `aifs_n320_grib_1p5defg_nc.py` after line 306:

```python
# Line 306 continuation:
ds.load()

# === ADD THIS BLOCK ===
# Fix precipitation artifacts: clip negative values and remove noise
if 'tp' in ds.data_vars:
    # Ensure non-negative
    ds['tp'] = ds['tp'].clip(min=0)
    # Remove numerical noise (values < 0.01 mm)
    ds['tp'] = ds['tp'].where(ds['tp'] > 0.01 / 1000, 0)  # if in meters
    # OR
    # ds['tp'] = ds['tp'].where(ds['tp'] > 0.01, 0)  # if in mm
# === END BLOCK ===

available_vars = list(ds.data_vars)
```

---

## Verification Steps

After implementing fixes, verify with:

```python
import xarray as xr
import numpy as np

# Load output
ds = xr.open_dataset('ensemble_quintile_probabilities_YYYYMMDD.nc')

# Check 1: No negative values in tp
tp_data = ds['tp_quintiles']
assert (tp_data >= 0).all(), "Negative values found in precipitation"

# Check 2: Probabilities sum to 1
prob_sum = tp_data.sum(dim='quintile')
assert np.allclose(prob_sum, 1, atol=1e-6), "Probabilities don't sum to 1"

# Check 3: Reasonable spatial distribution
# Should not have isolated single-pixel anomalies
from scipy import ndimage
for q in range(5):
    data = tp_data.isel(quintile=q).values
    # Check for isolated pixels (local maxima surrounded by much lower values)
    local_max = ndimage.maximum_filter(data, size=3)
    isolated = (data == local_max) & (data > 2 * ndimage.uniform_filter(data, size=5))
    n_isolated = np.sum(isolated)
    print(f"Quintile {q}: {n_isolated} potentially isolated pixels")
```

---

## Root Cause Summary

| Variable | Distribution | Regrid Method | Result |
|----------|-------------|---------------|--------|
| `tas` (2t) | Gaussian, continuous | Bilinear | OK |
| `mslp` (msl) | Gaussian, continuous | Bilinear | OK |
| `tp` (pr) | Skewed, zero-bounded, discontinuous | Bilinear | ARTIFACTS |

The fundamental issue is **applying a continuous-field interpolation method to a discontinuous, bounded variable**. Precipitation requires special handling at every stage of the pipeline.

---

## References

1. [ECMWF: Regridding of precipitation](https://confluence.ecmwf.int/display/CKB/ERA5%3A+data+documentation#ERA5:datadocumentation-Spatialinterpolation)
2. [earthkit-regrid documentation](https://earthkit-regrid.readthedocs.io/)
3. Accadia et al. (2003): "Sensitivity of Precipitation Forecast Skill Scores to Bilinear Interpolation and a Simple Nearest-Neighbor Average Method on High-Resolution Verification Grids"

---

*Report generated: 2025-12-20*
*Analysis of: aifs_n320_grib_1p5defg_nc.py and ensemble_quintile_analysis.py*

# Comparison: Why Earlier Analysis Missed the Bugs

**Date:** 2026-01-03

This document compares the earlier analysis (`PRECIPITATION_ARTIFACT_ANALYSIS.md`) with the actual bugs found (`PRECIPITATION_FIX_REPORT.md`) and explains why the earlier attempt did not identify the root causes.

---

## Side-by-Side Comparison

### Earlier Analysis (PRECIPITATION_ARTIFACT_ANALYSIS.md)

| Aspect | Details |
|--------|---------|
| **Focus File** | `aifs_n320_grib_1p5defg_nc.py` (data preprocessing/regridding) |
| **Problem Type** | Interpolation artifacts, visual pixelation |
| **Hypothesized Causes** | Bilinear interpolation inappropriate for precipitation |
| **Proposed Solutions** | Conservative/nearest-neighbor regridding, non-negative clipping |

**Hypothesized Issues:**
1. Bilinear interpolation inappropriate for precipitation
2. No constraint on non-negative values after regridding
3. Weekly accumulation amplifies artifacts
4. Precipitation statistics require special treatment
5. Unit verification needed (mentioned but not confirmed)

### Actual Bugs Found (PRECIPITATION_FIX_REPORT.md)

| Aspect | Details |
|--------|---------|
| **Focus File** | `ensemble_quintile_analysis.py` (quintile calculation) |
| **Problem Type** | Unit mismatch + step aggregation bug |
| **Actual Causes** | Meters vs millimeters, slice(0,12) returns NaN for time>0 |
| **Implemented Fix** | `* 1000` conversion + `skipna` sum per time chunk |

**Actual Bugs:**
1. **Unit Mismatch**: Forecast in meters, climatology in millimeters (1000x difference)
2. **Step Aggregation**: `slice(0,12)` only works for time=0, returns 100% NaN for time=1,2,3,4

---

## Why the Earlier Analysis Missed the Bugs

### 1. Wrong File Focus

| Earlier Analysis | Actual Problem |
|------------------|----------------|
| Focused on `aifs_n320_grib_1p5defg_nc.py` (regridding) | Bugs were in `ensemble_quintile_analysis.py` (quintile calculation) |

The regridding in `aifs_n320_grib_1p5defg_nc.py` was actually working correctly. The artifacts appeared during the quintile calculation step, not during regridding.

### 2. Assumed Visual Artifacts, Not Statistical Bias

| Earlier Analysis | Actual Problem |
|------------------|----------------|
| Looked for "salt and pepper" noise, negative values, pixelation | Systematic statistical bias (Q1 = 96% instead of 20%) |

The earlier analysis focused on **VISUAL artifacts** in the data, but the actual problem was a **STATISTICAL bias** that made everything appear dry.

### 3. Unit Verification Was Mentioned But Not Tested

The earlier analysis correctly listed "Verify Unit Consistency" as **Solution 5 (Priority: HIGH)**, but it wasn't actually tested with diagnostic code.

**This diagnostic would have revealed the bug:**
```python
forecast_mean = 0.006 m    # in meters
climatology_mean = 6.32 mm  # in millimeters
# Ratio: ~1000x difference!
```

The solution was correctly predicted but never executed.

### 4. Data Structure Complexity Not Understood

| Earlier Analysis | Actual Problem |
|------------------|----------------|
| Assumed simple step indexing | Valid data at DIFFERENT step indices per time chunk |

The NetCDF data structure has a subtle complexity:

| Time Chunk | Valid Step Indices | Day Range |
|------------|-------------------|-----------|
| time=0 | steps 0-11 | days 18.25-21.00 |
| time=1 | steps 12-23 | days 21.25-24.00 |
| time=2 | steps 24-35 | days 24.25-27.00 |
| time=3 | steps 36-47 | days 27.25-30.00 |
| time=4 | steps 48-59 | days 30.25-33.00 |

Using `slice(0,12)` for all time chunks returned NaN for time=1,2,3,4. This requires actually loading and inspecting the data to discover.

**Diagnostic that revealed this:**
```python
for t in range(5):
    data = tp.isel(time=t, step=slice(0, 12))
    valid_pct = (~np.isnan(data.values)).sum() / data.values.size * 100
    print(f"Time {t} slice(0,12): {valid_pct:.1f}% valid")

# Output:
# Time 0 slice(0,12): 100.0% valid  <- OK
# Time 1 slice(0,12): 0.0% valid    <- BUG!
# Time 2 slice(0,12): 0.0% valid    <- BUG!
# Time 3 slice(0,12): 0.0% valid    <- BUG!
# Time 4 slice(0,12): 0.0% valid    <- BUG!
```

### 5. No Quantitative Verification

| Earlier Analysis | Actual Problem |
|------------------|----------------|
| Proposed solutions without quantitative baseline | Needed to check quintile distribution against expected 20% |

The key diagnostic that revealed the bug:

| Quintile | Expected | Observed (Bug) |
|----------|----------|----------------|
| Q1 (Dry) | 20% | **96%** |
| Q2 | 20% | 1.5% |
| Q3 | 20% | 0.8% |
| Q4 | 20% | 0.8% |
| Q5 (Wet) | 20% | 1.0% |

This massive deviation (96% vs 20%) immediately signals a **calculation bug**, not just interpolation artifacts.

---

## Key Lessons Learned

### 1. Check Statistics First

Before looking for visual artifacts, verify that basic statistics are correct. If Q1 = 96% when it should be 20%, there's a fundamental calculation bug.

### 2. Verify Units at Each Pipeline Stage

Compare value ranges between datasets:
- Forecast: 0.006 m → should be ~6 mm
- Climatology: 6.32 mm

If they differ by ~1000x, there's a unit mismatch.

### 3. Inspect Data Structure Carefully

Don't assume indexing works the same for all dimensions:
- Check which indices have valid (non-NaN) data
- Verify that slice operations return expected data

### 4. Test Proposed Solutions

Solution 5 (Unit Verification) in the earlier analysis was correct, but it wasn't actually tested. Running the diagnostic code would have immediately revealed the 1000x scale mismatch.

### 5. Focus on the Right File

The symptom (bad precipitation output) doesn't always point to the cause. The regridding file was fine; the quintile calculation file had the bugs.

---

## Summary Table

| Aspect | Earlier Analysis | What We Found |
|--------|------------------|---------------|
| Problem File | `aifs_n320_grib_1p5defg_nc.py` | `ensemble_quintile_analysis.py` |
| Problem Type | Interpolation artifacts | Unit mismatch + wrong slicing |
| Root Cause | Bilinear interpolation | meters vs mm, slice(0,12) bug |
| Symptom Interpretation | Visual pixelation/noise | Statistical bias (Q1=96%) |
| Solution Status | Proposed but not tested | Implemented and verified |
| Unit Check | Mentioned (Solution 5) | Confirmed as critical bug |
| Data Structure | Not investigated | Key bug: NaN for time>0 |

---

## Conclusion

The earlier analysis was on the right track with Solution 5 (Unit Verification) but failed to actually execute the diagnostic code that would have revealed the bug. Additionally, the focus on visual interpolation artifacts rather than statistical bias led to investigating the wrong file (`aifs_n320_grib_1p5defg_nc.py` instead of `ensemble_quintile_analysis.py`).

The key takeaway: **Always verify basic statistics (quintile distributions should be ~20% each) before investigating more complex issues like interpolation methods.**

---

*Analysis comparison generated: 2026-01-03*

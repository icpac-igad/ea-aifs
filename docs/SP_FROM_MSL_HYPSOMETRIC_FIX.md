# Fix: Surface Pressure Derivation from MSL via Hypsometric Equation

## Issue

The ERA5T field adaptation (`adapt_era5t_fields()` in
`era5t_fp16_multi_run_AIFS_ENS_v1.py`) was directly copying Mean Sea Level
Pressure (MSL) to Surface Pressure (SP):

```python
fields["sp"] = fields["msl"].copy()
```

This is physically incorrect. MSL is altitude-corrected: at elevated grid
points MSL can be **100–450 hPa higher** than actual surface pressure. Feeding
MSL as SP made the model think the atmosphere was far denser than reality,
corrupting its thermodynamics over the 40-day forecast horizon.

## Impact (20260312 run)

The model output at forecast days 18–40 was catastrophically biased:

| Variable | Forecast Global Mean | Climatology Mean | Error |
|----------|---------------------|-----------------|-------|
| 2t       | 244 K (−29 °C)      | 277 K (4 °C)    | −33 K |
| msl      | 107 805 Pa (1078 hPa) | 101 012 Pa (1010 hPa) | +68 hPa |

Quintile probabilities were dominated by extreme bins:

| Variable | Q1 | Q2 | Q3 | Q4 | Q5 |
|----------|-----|-----|-----|-----|-----|
| 2t       | 0.768 | 0.009 | 0.008 | 0.008 | 0.207 |
| msl      | 0.348 | 0.003 | 0.003 | 0.003 | 0.643 |

A well-calibrated forecast would show ~0.20 in each quintile.

## Diagnosis Method

A diagnostic script (`diagnose_quintile_bias.py`) downloaded one member's
NetCDF output from GCS and the AI Weather Quest climatology, then compared
value ranges:

- Forecast 2t weekly mean: ~33 K — far below climatology Q1 threshold (~275 K)
- Forecast msl weekly mean: ~14 800 Pa — far below climatology (when incorrectly
  averaged), but raw values ~107 800 Pa are far above Q4 (~101 450 Pa)
- Units were confirmed consistent (both K for temperature, both Pa for pressure)
- The bias is in the model output itself, not in post-processing

## Root Cause

`SP = MSL` ignores the barometric height correction. Examples of the error at
different elevations:

| Elevation | Surface Geopotential (m²/s²) | MSL (Pa) | True SP (Pa) | Error |
|-----------|------------------------------|----------|-------------|-------|
| Sea level | 0 | 101 325 | 101 325 | 0 hPa |
| 1 000 m | 9 807 | 101 325 | 90 023 | +113 hPa |
| 2 000 m | 19 614 | 101 325 | 79 981 | +213 hPa |
| 5 000 m (Tibet) | 49 035 | 101 325 | 56 092 | +452 hPa |

The model's dynamics depend heavily on SP for pressure-gradient forces,
temperature tendencies, and moisture transport. Systematic high-pressure bias
in SP caused the forecast to drift to unrealistic states within days.

## Fix

Replace the direct copy with the **hypsometric equation**, using the surface
geopotential (`z`) and 2-meter temperature (`2t`) already present in the pkl
fields:

```python
SP = MSL × exp(−z_sfc / (R_d × T_v))
```

where:
- `z_sfc` = surface geopotential (constant field `z` in pkl), m²/s²
- `R_d` = 287.058 J/(kg·K), specific gas constant for dry air
- `T_v` = virtual temperature ≈ `2t × 1.003` (small moisture correction)

Updated code in `adapt_era5t_fields()`:

```python
if "sp" not in fields and "msl" in fields:
    if "z" in fields:
        Rd = 287.058
        z_sfc = fields["z"]
        t2m = fields["2t"]
        T_mean = np.maximum(t2m * 1.003, 200.0)
        fields["sp"] = fields["msl"] * np.exp(-z_sfc / (Rd * T_mean))
    else:
        fields["sp"] = fields["msl"].copy()  # fallback
```

## Other Field Adaptation Notes

The remaining approximations in `adapt_era5t_fields()` are less impactful but
still introduce some bias:

| Derived Field | Method | Risk |
|---------------|--------|------|
| tcw ← tcwv | Rename | Low (vapour dominates total column water) |
| 2d ← 2t − 2 K | Fixed dewpoint depression | Medium (too dry in tropics, too moist in polar) |
| skt ← 2t | Copy | Low–Medium (OK over ocean, wrong over land at night) |
| stl1, stl2 ← 2t | Copy | Medium (soil temps lag air temps) |
| w ← 0 at all levels | Zeros | Medium (removes vertical motion, suppresses convection) |

## Files Changed

- `era5t_fp16_multi_run_AIFS_ENS_v1.py` — `adapt_era5t_fields()` SP derivation
- `era5tFp16FahamuAIFSv1.md` — field adaptation table updated

## Re-run Required

After this fix, Steps 2–5 of the pipeline must be re-executed for any affected
forecast date:

1. GPU inference (Step 2) — re-run with corrected field adaptation
2. GRIB → NetCDF (Step 3)
3. Quintile analysis (Step 4)
4. Submission (Step 5)

Existing pkl files (Step 1) do **not** need to be regenerated — the fix is
applied at inference time.

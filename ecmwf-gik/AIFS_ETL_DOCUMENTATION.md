# AIFS ETL Pipeline Documentation

## Overview

This documentation describes the complete ETL (Extract, Transform, Load) pipeline for processing ECMWF parquet files into AIFS-ready input states.

## Pipeline Stages

### Stage 1: Data Extraction (`aifs-etl-v2.py`)

**Purpose:** Extract all pressure levels and surface fields from ECMWF parquet files.

**Key Features:**
- Properly extracts all 13 pressure levels from 5D arrays
- Uses obstore for fast S3 access
- Handles both base64-encoded and S3-referenced chunks
- Converts geopotential height (gh) to geopotential (z)

**Usage:**
```bash
python aifs-etl-v2.py
```

**Output:**
- PKL file: `ecmwf_pkl_from_parquet_v2/input_state_member_001.pkl`
- Grid resolution: 0.25° (721 × 1440)
- Total fields: 87
- File size: ~345 MB

**Extracted Fields:**

**Surface Fields (9):**
- `10u` - 10m u-component of wind
- `10v` - 10m v-component of wind
- `2d` - 2m dewpoint temperature
- `2t` - 2m temperature
- `lsm` - Land-sea mask
- `msl` - Mean sea level pressure
- `skt` - Skin temperature
- `sp` - Surface pressure
- `tcw` - Total column water

**Pressure Level Fields (78):**
For each parameter at 13 levels [1000, 925, 850, 700, 600, 500, 400, 300, 250, 200, 150, 100, 50 hPa]:
- `q_{level}` - Specific humidity (13 fields)
- `t_{level}` - Temperature (13 fields)
- `u_{level}` - U-component of wind (13 fields)
- `v_{level}` - V-component of wind (13 fields)
- `w_{level}` - Vertical velocity (13 fields)
- `z_{level}` - Geopotential (13 fields, converted from geopotential height)

### Stage 2: Data Inspection (`inspect_pkl.py`)

**Purpose:** Inspect PKL files and compare with expected AIFS input variables.

**Usage:**
```bash
python inspect_pkl.py ecmwf_pkl_from_parquet_v2/input_state_member_001.pkl
```

**Output:**
- Field inventory
- Shape and dtype information
- Comparison with AIFS requirements
- Missing field analysis

### Stage 3: Regridding (`aifs-regrid.py`)

**Purpose:** Convert 0.25° data to N320 grid required by AIFS.

**Operations:**
1. Longitude coordinate verification (0° to 360°)
2. Regridding: 0.25° → N320 (~640 × 1280)
3. Missing field handling (optional zero-filling)

**Usage:**
```bash
python aifs-regrid.py ecmwf_pkl_from_parquet_v2/input_state_member_001.pkl aifs_ready/input_state_member_001.pkl
```

**Requirements:**
```bash
pip install earthkit-regrid
```

**Output:**
- PKL file: `aifs_ready/input_state_member_001.pkl`
- Grid resolution: N320
- Total fields: 87 (or 92 if zero-filling enabled)

## Field Status Matrix

### ✅ Available Fields (87 total)

| Category | Count | Fields |
|----------|-------|--------|
| Surface | 8 | 10u, 10v, 2d, 2t, msl, skt, sp, tcw |
| Constants | 1 | lsm |
| Pressure (q) | 13 | q_1000 through q_50 |
| Pressure (t) | 13 | t_1000 through t_50 |
| Pressure (u) | 13 | u_1000 through u_50 |
| Pressure (v) | 13 | v_1000 through v_50 |
| Pressure (w) | 13 | w_1000 through w_50 |
| Pressure (z) | 13 | z_1000 through z_50 |

### ❌ Missing Fields (5 total)

| Field | Type | Description | Impact |
|-------|------|-------------|--------|
| `z` | Constant | Surface geopotential (orography) | **Critical** - AIFS needs orography |
| `slor` | Constant | Slope of sub-gridscale orography | Moderate - affects surface interactions |
| `sdor` | Constant | Std dev of orography | Moderate - affects surface interactions |
| `stl1` | Soil | Soil temperature layer 1 | Moderate - affects surface fluxes |
| `stl2` | Soil | Soil temperature layer 2 | Moderate - affects surface fluxes |

### Missing Field Sources

**Orography Fields (z, slor, sdor):**
- These are constant fields that don't change with forecast
- Can be obtained from:
  - ECMWF climate files
  - ERA5 constant fields
  - AIFS model repository (may have defaults)
  - URL: https://confluence.ecmwf.int/display/UDOC/L137+model+level+definitions

**Soil Temperature Fields (stl1, stl2):**
- Not available in the current parquet structure
- Can be obtained from:
  - ECMWF Open Data API (if available for ensemble forecasts)
  - GRIB files from ECMWF MARS archive
  - Alternative: Use surface temperature (skt) as proxy

## Data Flow Diagram

```
┌─────────────────────────────────────────┐
│  ECMWF Parquet Files                    │
│  (S3: ecmwf-forecasts bucket)           │
│  Format: Zarr references in parquet     │
│  Grid: 0.25° (721 × 1440)              │
└──────────────┬──────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────┐
│  Stage 1: aifs-etl-v2.py                │
│  - Extract all 13 pressure levels       │
│  - Process surface/constant fields      │
│  - Convert gh → z (geopotential)        │
│  - Use obstore for S3 access            │
└──────────────┬──────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────┐
│  PKL File (v2)                          │
│  87 fields @ 0.25° resolution           │
│  Size: ~345 MB per member               │
└──────────────┬──────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────┐
│  Stage 2: inspect_pkl.py (optional)     │
│  - Verify field inventory               │
│  - Compare with AIFS requirements       │
│  - Identify missing fields              │
└──────────────┬──────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────┐
│  Stage 3: aifs-regrid.py                │
│  - Regrid: 0.25° → N320                │
│  - Verify longitude coordinates         │
│  - Optional: zero-fill missing fields   │
└──────────────┬──────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────┐
│  AIFS-Ready PKL File                    │
│  87-92 fields @ N320 resolution         │
│  Ready for inference                    │
└─────────────────────────────────────────┘
```

## Complete Workflow

### Quick Start (Single Member)

```bash
# 1. Extract data from parquet
python aifs-etl-v2.py

# 2. Inspect results (optional)
python inspect_pkl.py ecmwf_pkl_from_parquet_v2/input_state_member_001.pkl

# 3. Regrid for AIFS
python aifs-regrid.py \
    ecmwf_pkl_from_parquet_v2/input_state_member_001.pkl \
    aifs_ready/input_state_member_001.pkl
```

### Processing Multiple Ensemble Members

To process multiple members, modify `aifs-etl-v2.py`:

```python
# In main() function, change:
parquet_file = "ecmwf_20250728_18_efficient/members/ens_01/ens_01.parquet"

# To loop over members:
for member in range(1, 51):  # Members 1-50
    parquet_file = f"ecmwf_20250728_18_efficient/members/ens_{member:02d}/ens_{member:02d}.parquet"
    # ... process each member
```

## Performance Metrics

| Stage | Time (single member) | Memory | Output Size |
|-------|---------------------|--------|-------------|
| Extract (v2) | ~560 seconds | ~350 MB | 345 MB |
| Inspect | ~2 seconds | ~350 MB | - |
| Regrid | ~300 seconds | ~500 MB | ~280 MB |
| **Total** | **~15 minutes** | **~500 MB peak** | **280 MB** |

## Comparison: V1 vs V2

| Aspect | V1 (aifs-etl.py) | V2 (aifs-etl-v2.py) |
|--------|------------------|---------------------|
| Pressure levels extracted | 1 (only level 0) | 13 (all levels) |
| Total fields | 15 | 87 |
| File size | 59 MB | 345 MB |
| AIFS compatibility | ❌ Incomplete | ✅ Nearly complete |
| Code logic | Buggy fallback | Proper multi-level detection |

**Key V2 Improvements:**
1. Correctly identifies 5D arrays with shape `(1, 2, 13, 721, 1440)`
2. Extracts all 13 pressure levels from dimension index 2
3. Removes faulty "single level" fallback logic
4. Properly maps requested levels to array indices

## Grid Specifications

### Input Grid (Parquet)
- **Type:** Regular latitude-longitude
- **Resolution:** 0.25° × 0.25°
- **Dimensions:** 721 × 1440
- **Coverage:** 90°N to 90°S, 0° to 360°E
- **Points:** 1,038,240 per field

### Output Grid (AIFS)
- **Type:** N320 (Gaussian reduced grid)
- **Approximate resolution:** ~0.28° (varies by latitude)
- **Dimensions:** ~640 × 1280
- **Coverage:** Global
- **Points:** ~819,200 per field

## Troubleshooting

### Issue: "Only 1 level extracted instead of 13"
**Solution:** Use `aifs-etl-v2.py` instead of `aifs-etl.py`

### Issue: "earthkit.regrid not found"
**Solution:**
```bash
pip install earthkit-regrid
```

### Issue: "Missing orography fields (z, slor, sdor)"
**Solution:** Download constant fields from ERA5 or ECMWF:
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
}, 'constants.grib')
```

### Issue: "Longitude coordinates incorrect"
**Check:** ECMWF Open Data is -180 to 180, parquet may be 0 to 360
**Solution:** Verify in `aifs-regrid.py` and enable rolling if needed:
```python
# Uncomment in aifs-regrid.py:
for field_name, field_data in fields.items():
    fields[field_name] = np.roll(field_data, -field_data.shape[1] // 2, axis=1)
```

## Expected AIFS Input Format

Based on `ecmwf_opendata_pkl_input_aifsens.py`:

```python
input_state = {
    'date': datetime.datetime object,
    'fields': {
        # Surface (8 fields)
        '10u': np.array (N320),
        '10v': np.array (N320),
        # ... etc

        # Constants (4 fields)
        'lsm': np.array (N320),
        'z': np.array (N320),      # ⚠️ Missing
        'slor': np.array (N320),   # ⚠️ Missing
        'sdor': np.array (N320),   # ⚠️ Missing

        # Soil (2 fields)
        'stl1': np.array (N320),   # ⚠️ Missing
        'stl2': np.array (N320),   # ⚠️ Missing

        # Pressure levels (78 fields)
        'q_1000': np.array (N320),
        # ... all 13 levels for q, t, u, v, w, z
    }
}
```

## Dependencies

```txt
numpy>=1.20.0
pandas>=1.3.0
earthkit-regrid>=0.3.0
obstore>=0.2.0  # Optional, for faster S3 access
fsspec>=2021.0.0  # Fallback for S3 access
s3fs>=2021.0.0  # For fsspec S3 backend
```

Install all dependencies:
```bash
pip install numpy pandas earthkit-regrid obstore fsspec s3fs
```

## References

- ECMWF Open Data: https://www.ecmwf.int/en/forecasts/datasets/open-data
- AIFS Documentation: https://www.ecmwf.int/en/forecasts/documentation-and-support/aifs
- Earthkit Regrid: https://earthkit-regrid.readthedocs.io/
- ECMWF Grid Specifications: https://confluence.ecmwf.int/display/UDOC/L137+model+level+definitions

## GIK Pipeline Integration (grib-index-kerchunk)

The [grib-index-kerchunk](https://github.com/icpac-igad/grib-index-kerchunk) repository provides an alternative parquet-based approach for creating AIFS input states. Instead of downloading full GRIB files, GIK creates lightweight parquet reference files containing `[url, byte_offset, byte_length]` triplets that point into remote GRIB files on S3, enabling on-demand byte-range reads.

### Two ETL Approaches Compared

| Aspect | ECMWF Open Data API (`ecmwf_opendata_pkl_input_aifsens.py`) | GIK Parquet Method (`aifs-etl-v2.py`) |
|--------|-----------------------------------------------------------|---------------------------------------|
| Data Source | ECMWF Open Data API via `earthkit.data` | S3 byte-range reads via parquet references |
| Parquet Creation | Not needed | Required (3-stage pipeline) |
| Grid Output | N320 (regridded inline) | 0.25° (requires separate regridding) |
| Time per Member | ~1-2 minutes | ~15 minutes (extract + regrid) |
| Dependencies | `earthkit-data`, `earthkit-regrid`, `ecmwf-opendata` | `kerchunk`, `fsspec`, `obstore` |
| Missing Fields | None (z, slor, sdor, stl1, stl2 all available) | 5 missing (z, slor, sdor, stl1, stl2) |
| Best For | Production ensemble runs | Research / offline batch processing |

### GIK Three-Stage Pipeline

The GIK method uses a three-stage pipeline to create parquet reference files:

**Stage 1: Build Zarr Metadata (Template-Based)**
- Loads zarr structure from pre-built HuggingFace template (`gik-fmrc-v2ecmwf_fmrc.tar.gz`)
- Takes ~5 seconds (vs ~73 minutes with legacy `scan_grib`)
- Template reference date: 2024-05-29

**Stage 2: Index-Based Reference Building**
- Reads `.index` files (JSON-lines) from `s3://ecmwf-forecasts/`
- Each line: `{"_offset": N, "_length": M, "param": "tp", "number": 1, ...}`
- Processes all 85 timestep index files per member
- I/O-bound, parallelized via `ThreadPoolExecutor`
- Time: ~5-15 minutes

**Stage 3: Final Parquet Creation**
- Merges zarr metadata + byte-range references → one parquet per member
- Output: `{date}{run}z-{member}.parquet`

### GIK Data Streaming Performance

| Metric | Sequential | Parallel (8 threads) |
|--------|-----------|---------------------|
| 51 members, 12 vars, 9 timesteps | ~4 hours | ~24 minutes |
| Decoder: gribberish (Rust) | ~25 ms/chunk | ~25 ms/chunk |
| Decoder: cfgrib (Python) | ~2000 ms/chunk | ~2000 ms/chunk |
| **Speedup (gribberish vs cfgrib)** | **~80x** | **~80x** |

### GIK Parquet Creator (`ecmwf_ensemble_par_creator_efficient.py`)

Creates parquet reference files for all 51 ensemble members:

```bash
# From grib-index-kerchunk/ecmwf/
python ecmwf_ensemble_par_creator_efficient.py
```

- Scans each GRIB file once (not per-member)
- Efficiency gain: 51x fewer file scans vs naive approach
- Output: `ecmwf_{date}_{run}_efficient/members/ens_{NN}/ens_{NN}.parquet`

### Recommended Workflow

For **production ensemble runs** (current approach):
```bash
# Uses ECMWF Open Data API — all fields available, N320 output
python ecmwf_opendata_pkl_input_aifsens.py
```

For **research/batch processing** with GIK parquets:
```bash
# Step 1: Create parquet references (once per date)
python ecmwf_ensemble_par_creator_efficient.py

# Step 2: Extract PKL from parquets
python aifs-etl-v2.py

# Step 3: Regrid to N320
python aifs-regrid.py input.pkl output.pkl
```

### ECMWF S3 Data Structure

- **Bucket**: `s3://ecmwf-forecasts/`
- **Path**: `{YYYYMMDD}/{HH}z/ifs/0p25/enfo/{YYYYMMDDHH}0000-{H}h-enfo-ef.grib2`
- **Index**: `.index` (JSON-lines with `_offset`, `_length`, `param`, `number`, `step`)
- **Members**: 51 (1 control + 50 perturbed)
- **Resolution**: 0.25° global
- **File size**: ~3-4 GB per timestep (all 51 members in one file)
- **Access**: Anonymous (`anon=True`)

## Future Improvements

1. **Automated missing field retrieval**
   - Download z, slor, sdor from ERA5 constants
   - Fetch stl1, stl2 from ECMWF MARS or alternative sources

2. **Parallel processing**
   - Process multiple ensemble members in parallel
   - Use multiprocessing for regridding

3. **Validation**
   - Add data quality checks
   - Verify physical consistency
   - Compare with reference datasets

4. **Optimization**
   - Cache constant fields (lsm, z, slor, sdor)
   - Optimize memory usage for large ensembles
   - Implement chunked processing for very large datasets

5. **GIK Pipeline Migration**
   - Evaluate template-based Stage 1 for faster parquet creation
   - Integrate gribberish decoder for 80x faster data streaming
   - Consider Lithops cloud deployment for multi-date parallel processing

---

**Last Updated:** 2026-03-22
**Version:** 2.1

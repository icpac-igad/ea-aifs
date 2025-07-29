# EarthKit Data Usage in AIFS Ensemble Processing

## Overview

This document explains in detail how EarthKit Data (earthkit.data) is used in the AIFS ensemble processing pipeline to retrieve, process, and prepare meteorological data from ECMWF's Open Data service.

## Key Components

### 1. EarthKit Data (ekd)

EarthKit Data is ECMWF's Python package for handling meteorological data. In our pipeline, it serves as the primary interface for:
- Retrieving data from ECMWF Open Data
- Handling GRIB format data
- Providing metadata access
- Converting data to NumPy arrays

### 2. EarthKit Regrid (ekr)

EarthKit Regrid handles the spatial interpolation of data from one grid to another. In our case:
- **Source grid**: 0.25° x 0.25° regular latitude-longitude grid (721x1440 points)
- **Target grid**: N320 reduced Gaussian grid (542,080 points globally)

## Detailed Data Flow

### Step 1: Data Retrieval

```python
import earthkit.data as ekd

# For control forecast (no ensemble perturbation)
data = ekd.from_source("ecmwf-open-data", 
                       date=date, 
                       param=param, 
                       levelist=levelist)

# For ensemble members (1-50)
data = ekd.from_source("ecmwf-open-data", 
                       date=date, 
                       param=param, 
                       levelist=levelist,
                       number=[number],  # Ensemble member number
                       stream='enfo')    # Ensemble forecast stream
```

**Key points:**
- `from_source()` automatically handles the API connection to ECMWF Open Data
- The `number` parameter selects specific ensemble members
- `stream='enfo'` specifies the ensemble forecast data stream
- Returns a FieldList object containing GRIB messages

### Step 2: Data Processing

```python
for f in data:
    # 1. Convert GRIB field to NumPy array
    values = f.to_numpy()  # Shape: (721, 1440)
    
    # 2. Shift longitude coordinates from [-180, 180] to [0, 360]
    values = np.roll(values, -f.shape[1] // 2, axis=1)
    
    # 3. Interpolate to model grid
    values = ekr.interpolate(values, 
                           {"grid": (0.25, 0.25)},  # Source grid
                           {"grid": "N320"})         # Target grid
```

**Processing details:**

1. **Grid Shift**: The `np.roll()` operation shifts the data by half the width (720 points) to convert from:
   - Source: Longitude range [-180°, 180°] with 0° at center
   - Target: Longitude range [0°, 360°] with 0° at left edge

2. **Grid Interpolation**: The N320 grid is an icosahedral-hexagonal grid with:
   - 320 latitude circles between poles
   - Variable number of points per latitude
   - Total of 542,080 points globally
   - More uniform area per grid cell compared to lat-lon grids

### Step 3: Field Organization

```python
# Access metadata for field identification
name = f.metadata('param')           # Parameter short name (e.g., 't', 'u', 'v')
level = f.metadata('levelist')       # Pressure level if applicable

# Create unique field identifier
if levelist:
    field_name = f"{param}_{level}"  # e.g., "t_850" for temperature at 850 hPa
else:
    field_name = param                # e.g., "2t" for 2-meter temperature
```

## Data Structure

### Parameters Retrieved

1. **Surface Fields** (`PARAM_SFC`):
   - `10u`, `10v`: 10-meter wind components
   - `2d`, `2t`: 2-meter dewpoint and temperature
   - `msl`: Mean sea level pressure
   - `skt`: Skin temperature
   - `sp`: Surface pressure
   - `tcw`: Total column water

2. **Constant Surface Fields** (`PARAM_SFC_FC`):
   - `lsm`: Land-sea mask
   - `z`: Surface geopotential
   - `slor`, `sdor`: Orography parameters

3. **Soil Fields** (`PARAM_SOIL`):
   - `sot`: Soil temperature at levels 1 and 2
   - Renamed to `stl1`, `stl2` for model compatibility

4. **Pressure Level Fields** (`PARAM_PL`):
   - `gh`: Geopotential height (converted to `z`)
   - `t`: Temperature
   - `u`, `v`, `w`: Wind components
   - `q`: Specific humidity
   - At 13 pressure levels: 1000, 925, 850, 700, 600, 500, 400, 300, 250, 200, 150, 100, 50 hPa

### Time Handling

The pipeline retrieves data for two time steps:
```python
for date in [DATE - datetime.timedelta(hours=6), DATE]:
    # Retrieve data for current and previous time
```

This provides:
- Previous time (T-6h): Used for model initialization
- Current time (T): Starting point for forecast

### Data Stacking

```python
# Stack time steps for each field
for param, values in fields.items():
    fields[param] = np.stack(values)  # Shape: (2, 542080)
```

Creates a 3D array where:
- First dimension: Time steps (2)
- Second dimension: Grid points (542,080)

## Memory Considerations

Each field requires approximately:
- Single time step: 542,080 points × 4 bytes (float32) ≈ 2.1 MB
- Both time steps: 2 × 2.1 MB ≈ 4.2 MB
- Total for ~100 fields: ~420 MB per ensemble member

## Error Handling

The code includes several validation steps:

1. **Grid dimension check**:
   ```python
   assert f.to_numpy().shape == (721, 1440)
   ```
   Ensures data is on expected 0.25° grid

2. **Field verification**: The test script verifies all expected fields are present

3. **Metadata consistency**: Uses metadata to properly identify and name fields

## Performance Optimizations

1. **Batch retrieval**: Gets all parameters at once when possible
2. **Memory efficiency**: Processes fields one at a time during interpolation
3. **Reuse of constant fields**: Surface constants (terrain, land-sea mask) retrieved only once

## Special Handling

### Geopotential Conversion
```python
# Convert geopotential height (m) to geopotential (m²/s²)
gh = fields.pop(f"gh_{level}")
fields[f"z_{level}"] = gh * 9.80665
```

### Soil Parameter Renaming
The Open Data soil parameters have different names than expected by the model:
- `sot_1` → `stl1` (Soil temperature level 1)
- `sot_2` → `stl2` (Soil temperature level 2)

## Usage in Ensemble Context

For ensemble forecasts:
- **Control run**: `number=None` - Uses unperturbed analysis
- **Perturbed members**: `number=1` to `50` - Each uses different initial perturbations
- Each member requires the same data retrieval and processing
- Total data volume: ~420 MB × 51 members ≈ 21 GB

This explains why the multi-run script processes members sequentially to manage memory usage effectively.
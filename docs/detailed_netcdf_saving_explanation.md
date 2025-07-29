# Detailed NetCDF Saving Process for AIFS ENS v1 Forecasts

## Overview

This document explains how forecast states from the AIFS ENS v1 model are saved to NetCDF files, including the background of `print_state` and the detailed saving process.

## The Forecast Loop Architecture

### Basic Forecast Loop
```python
from anemoi.inference.runners.simple import SimpleRunner
from anemoi.inference.outputs.printer import print_state

# Initialize the runner
checkpoint = {"huggingface": "ecmwf/aifs-ens-1.0"}
runner = SimpleRunner(checkpoint, device="cuda")

# Run forecast with just printing
for state in runner.run(input_state=input_state, lead_time=12):
    print_state(state)
```

### What `print_state` Does

The `print_state` function (`anemoi.inference.outputs.printer.py:35`) is a utility that:

1. **Displays State Metadata**: Shows date, coordinates, and field count
2. **Samples Field Statistics**: Displays min/max values for selected fields
3. **Provides Progress Feedback**: Visual indication of forecast progress

**Example Output:**
```
😀 date=2025-06-25T12:00:00 latitudes=(542080,) longitudes=(542080,) fields=100

    q_50   shape=(542080,) min=9.71238e-07    max=3.36415e-06   
    t_1000 shape=(542080,) min=234.135        max=323.46        
    v_925  shape=(542080,) min=-39.0201       max=38.1443       
    z_850  shape=(542080,) min=8126.93        max=16131.2       
    ro     shape=(542080,) min=0              max=0.0095754     
```

## State Structure

### What is a "State"?

A `State` is a dictionary containing:
```python
state = {
    "date": datetime.datetime,          # Forecast valid time
    "latitudes": np.ndarray,           # Latitude coordinates (542080,)
    "longitudes": np.ndarray,          # Longitude coordinates (542080,)
    "fields": {                        # Meteorological fields
        "t_1000": np.ndarray,          # Temperature at 1000 hPa
        "u_925": np.ndarray,           # U-wind at 925 hPa
        "v_925": np.ndarray,           # V-wind at 925 hPa
        "q_500": np.ndarray,           # Specific humidity at 500 hPa
        "z_850": np.ndarray,           # Geopotential at 850 hPa
        "2t": np.ndarray,              # 2-meter temperature
        "10u": np.ndarray,             # 10-meter U-wind
        "10v": np.ndarray,             # 10-meter V-wind
        "msl": np.ndarray,             # Mean sea level pressure
        # ... ~100 total fields
    }
}
```

### Grid Information
- **Grid Type**: Icosahedral N320 (reduced Gaussian grid)
- **Total Points**: 542,080 grid points globally
- **Coordinate System**: Latitude/longitude pairs for each grid point
- **Data Type**: 32-bit floating point values

## NetCDF Saving Process

### 1. Enhanced Forecast Loop with NetCDF Output

```python
from anemoi.inference.runners.simple import SimpleRunner
from anemoi.inference.outputs.printer import print_state
from anemoi.inference.outputs.netcdf import NetCDFOutput
from anemoi.inference.context import Context
import os

# Setup
checkpoint = {"huggingface": "ecmwf/aifs-ens-1.0"}
runner = SimpleRunner(checkpoint, device="cuda")

# Create output directory
os.makedirs("forecast_outputs", exist_ok=True)
netcdf_file = "forecast_outputs/forecast.nc"

# Configure NetCDF output
context = Context()
context.time_step = 6        # 6-hour steps
context.lead_time = 12       # 12 hours total
context.reference_date = input_state["date"]

netcdf_output = NetCDFOutput(context, path=netcdf_file)

# Enhanced forecast loop
forecast_states = []
outputs_initialized = False

for i, state in enumerate(runner.run(input_state=input_state, lead_time=12)):
    # Visual progress (original functionality)
    print_state(state)
    
    # Initialize NetCDF file on first iteration
    if not outputs_initialized:
        netcdf_output.open(state)
        outputs_initialized = True
    
    # Save state to NetCDF
    netcdf_output.write_step(state)
    
    # Store for later use
    forecast_states.append(state)

# Close the file
netcdf_output.close()
```

### 2. NetCDF File Structure Creation

When `netcdf_output.open(state)` is called, the NetCDF file is created with:

#### Dimensions
```python
dimensions = {
    "values": 542080,    # Spatial dimension (grid points)
    "time": 3            # Time dimension (lead_time/time_step + 1)
}
```

#### Coordinate Variables
```python
coordinates = {
    "latitude": {
        "dimensions": ("values",),
        "data": state["latitudes"],
        "attributes": {
            "units": "degrees_north",
            "long_name": "latitude"
        }
    },
    "longitude": {
        "dimensions": ("values",),
        "data": state["longitudes"],
        "attributes": {
            "units": "degrees_east",
            "long_name": "longitude"
        }
    },
    "time": {
        "dimensions": ("time",),
        "attributes": {
            "units": "seconds since 2025-06-25T06:00:00",
            "long_name": "time",
            "calendar": "gregorian"
        }
    }
}
```

#### Data Variables
Each field becomes a NetCDF variable:
```python
variables = {
    "t_1000": {
        "dimensions": ("time", "values"),
        "data": field_data,
        "attributes": {
            "fill_value": np.nan,
            "missing_value": np.nan
        }
    },
    # ... for all ~100 fields
}
```

### 3. Step-by-Step Saving Process

For each forecast step, `netcdf_output.write_step(state)` performs:

1. **Ensure Variables Exist**: Creates any new variables found in the state
2. **Calculate Time Offset**: Computes seconds since reference date
3. **Write Time Data**: Updates the time coordinate
4. **Write Field Data**: Saves all meteorological fields

```python
def write_step(self, state: State) -> None:
    # Update time coordinate
    step = state["date"] - self.reference_date
    self.time_var[self.n] = step.total_seconds()
    
    # Write all fields
    for name, value in state["fields"].items():
        if not self.skip_variable(name):
            self.vars[name][self.n] = value
    
    self.n += 1  # Increment time index
```

### 4. Data Flow Visualization

```
Input State (from model)
├── date: 2025-06-25T12:00:00
├── latitudes: [90.0, 89.9, ..., -90.0]
├── longitudes: [0.0, 0.1, ..., 359.9]
└── fields: {
    ├── t_1000: [285.3, 284.1, ...]    # Temperature at 1000 hPa
    ├── u_925: [12.4, -8.7, ...]       # U-wind at 925 hPa
    └── ... 98 more fields
}
                    ↓
NetCDF File Structure
├── dimensions: {values: 542080, time: 3}
├── coordinates: {
│   ├── latitude(values): [90.0, 89.9, ..., -90.0]
│   ├── longitude(values): [0.0, 0.1, ..., 359.9]
│   └── time(time): [0, 21600, 43200]  # seconds
│   }
└── variables: {
    ├── t_1000(time, values): [[285.3, 284.1, ...], ...]
    ├── u_925(time, values): [[12.4, -8.7, ...], ...]
    └── ... 98 more variables
}
```

## Key Differences: print_state vs NetCDF Saving

| Aspect | print_state | NetCDF Saving |
|--------|-------------|---------------|
| **Purpose** | Visual progress feedback | Persistent data storage |
| **Data** | Statistical summary (min/max) | Complete field data |
| **Storage** | Temporary (console output) | Permanent (file) |
| **Format** | Human-readable text | Binary, self-describing |
| **Fields** | Sample of fields (4 by default) | All fields (~100) |
| **Accessibility** | Visual only | Programmatic access |

## Memory and Performance Considerations

### Memory Efficiency
- **Incremental Writing**: Data is written step-by-step, not accumulated
- **Chunking**: NetCDF uses chunked storage for efficient I/O
- **Compression**: Optional compression can reduce file size

### Performance Optimization
```python
# Chunking strategy (automatic)
chunksizes = (1, values)  # One time step, all spatial points
while np.prod(chunksizes) > 1000000:
    chunksizes = tuple(int(np.ceil(x / 2)) for x in chunksizes)
```

### Threading Safety
```python
# Thread-safe writing (important for HDF5 backend)
LOCK = threading.RLock()

with LOCK:
    self.vars[name][self.n] = value
```

## File Output Verification

### Reading Back the NetCDF File
```python
import xarray as xr

# Open the saved file
ds = xr.open_dataset("forecast_outputs/forecast.nc")

print(f"Dimensions: {dict(ds.dims)}")
print(f"Variables: {list(ds.data_vars.keys())}")
print(f"Time steps: {ds.time.values}")

# Access specific field
temperature = ds.t_1000.values  # Shape: (time, values)
print(f"Temperature shape: {temperature.shape}")
print(f"Temperature at first time step: {temperature[0, :5]}")
```

### File Size Expectations
- **Uncompressed**: ~400-800 MB for 72-hour forecast
- **With compression**: ~100-200 MB
- **Size factors**: Number of fields, forecast length, grid resolution

## Advanced Configuration Options

### Custom Variable Selection
```python
# Save only specific variables
selected_vars = ["t_1000", "u_925", "v_925", "2t", "10u", "10v"]
netcdf_output = NetCDFOutput(context, path=netcdf_file, variables=selected_vars)
```

### Compression Settings
```python
# Enable compression (modify in NetCDFOutput source)
compression = {"zlib": True, "complevel": 4}
```

### Float Precision
```python
# Use 64-bit floats for higher precision
netcdf_output = NetCDFOutput(context, path=netcdf_file, float_size="f8")
```

## Error Handling and Troubleshooting

### Common Issues
1. **Permission Errors**: Ensure write access to output directory
2. **Disk Space**: Monitor available space for large files
3. **Memory Issues**: Reduce chunks or use compression
4. **Threading**: HDF5 thread safety can cause issues

### Debug Output
```python
import logging
logging.basicConfig(level=logging.DEBUG)
LOG = logging.getLogger("anemoi.inference.outputs.netcdf")
```

## Integration with Analysis Tools

### With Xarray
```python
import xarray as xr
import matplotlib.pyplot as plt

ds = xr.open_dataset("forecast_outputs/forecast.nc")
temperature = ds.t_1000.isel(time=0)  # First time step
temperature.plot()
plt.show()
```

### With Pandas
```python
import pandas as pd

# Convert to DataFrame for analysis
df = ds.to_dataframe()
df.head()
```

## Summary

The NetCDF saving process transforms the ephemeral forecast states from the AIFS ENS v1 model into persistent, self-describing files. While `print_state` provides visual feedback during execution, the NetCDF output preserves complete meteorological data for subsequent analysis, visualization, and scientific workflows.

The key advantage is that instead of just seeing summary statistics on screen, you get complete access to all forecast data in a standardized format that integrates seamlessly with the scientific Python ecosystem.
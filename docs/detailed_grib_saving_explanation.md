# Detailed GRIB Saving Process for AIFS ENS v1 Forecasts

## Overview

This document explains how forecast states from the AIFS ENS v1 model are saved to GRIB files, the meteorological standard for weather data exchange. This complements the NetCDF saving process and provides insight into the Context object's role.

## The Context Object - Why It's Needed

### What is Context?

The `Context` object (`anemoi.inference.context.py:34`) is a configuration container that provides metadata and settings needed by output classes. It's **not directly used in the forecast loop** but is **essential for output initialization**.

```python
from anemoi.inference.context import Context

# Context acts as a metadata container
context = Context()
context.time_step = 6        # Hours between forecast steps
context.lead_time = 12       # Total forecast duration
context.reference_date = input_state["date"]  # Base date for time calculations
```

### Why Context is Created Before runner.run()

The Context object is created **before** the forecast loop because:

1. **Output Initialization**: Output classes need metadata to set up file structures
2. **Time Calculations**: Determines how many time steps to expect
3. **Reference Date**: Establishes time coordinate baseline
4. **File Preparation**: Pre-allocates space and dimensions

```python
# Context is used INSIDE the output classes, not in the forecast loop
netcdf_output = NetCDFOutput(context, path=netcdf_file)  # Context used here
grib_output = GribFileOutput(context, path=grib_file)    # Context used here

# The forecast loop itself doesn't directly use context
for state in runner.run(input_state=input_state, lead_time=12):
    # state contains the actual data
    # context was already used during output initialization
```

### Context Properties Used by Outputs

```python
class Context(ABC):
    # Key properties used by output classes
    reference_date = None      # Base date for time calculations
    time_step = None          # Hours between forecast steps  
    lead_time = None          # Total forecast duration
    output_frequency = None   # How often to write outputs
    write_initial_state = True # Whether to include initial conditions
```

## GRIB Saving Process

### 1. Enhanced Forecast Loop with GRIB Output

```python
from anemoi.inference.runners.simple import SimpleRunner
from anemoi.inference.outputs.printer import print_state
from anemoi.inference.outputs.gribfile import GribFileOutput
from anemoi.inference.context import Context
import os

# Setup
checkpoint = {"huggingface": "ecmwf/aifs-ens-1.0"}
runner = SimpleRunner(checkpoint, device="cuda")

# Create output directory
os.makedirs("forecast_outputs", exist_ok=True)
grib_file = "forecast_outputs/forecast.grib"

# Configure GRIB output
context = Context()
context.time_step = 6        # 6-hour steps
context.lead_time = 12       # 12 hours total
context.reference_date = input_state["date"]

grib_output = GribFileOutput(context, path=grib_file)

# Enhanced forecast loop
forecast_states = []
outputs_initialized = False

for i, state in enumerate(runner.run(input_state=input_state, lead_time=12)):
    # Visual progress (original functionality)
    print_state(state)
    
    # Initialize GRIB file on first iteration
    if not outputs_initialized:
        grib_output.open(state)
        outputs_initialized = True
    
    # Save state to GRIB
    grib_output.write_step(state)
    
    # Store for later use
    forecast_states.append(state)

# Close the file
grib_output.close()
```

### 2. GRIB File Structure Creation

When `grib_output.open(state)` is called, the GRIB file structure is initialized:

#### GRIB Message Structure
```
forecast.grib
├── Message 1: t_1000 at time=0h
│   ├── Grid Definition Section: N320 icosahedral grid
│   ├── Product Definition Section: Temperature, 1000 hPa level
│   ├── Data Representation Section: Packing method, precision
│   └── Data Section: 542,080 grid point values
├── Message 2: u_925 at time=0h
├── Message 3: v_925 at time=0h
├── ...
├── Message N: t_1000 at time=6h
├── Message N+1: u_925 at time=6h
└── ... (100 variables × 3 time steps = 300 messages)
```

#### GRIB Encoding Parameters
```python
# Default encoding for each field
grib_encoding = {
    "paramId": variable_parameter_id,     # ECMWF parameter ID
    "levelType": "pl",                    # Pressure level
    "level": 1000,                        # Level value
    "step": 0,                            # Forecast step (hours)
    "dataDate": 20250625,                 # Reference date
    "dataTime": 600,                      # Reference time
    "grid": "N320",                       # Grid specification
    "packing": "simple",                  # Data packing method
}
```

### 3. Step-by-Step GRIB Saving Process

For each forecast step, `grib_output.write_step(state)` performs:

1. **Calculate Time Step**: Determines forecast hour from reference date
2. **Iterate Through Fields**: Processes each meteorological variable
3. **Create GRIB Message**: Encodes field data with metadata
4. **Write to File**: Appends message to GRIB file

```python
def write_step(self, state: State) -> None:
    step_hours = (state["date"] - self.reference_date).total_seconds() / 3600
    
    for name, field_data in state["fields"].items():
        if not self.skip_variable(name):
            # Create GRIB message
            message = self.create_grib_message(
                data=field_data,
                variable=name,
                step=step_hours,
                date=self.reference_date
            )
            
            # Write to file
            self.grib_file.write(message)
```

### 4. GRIB vs NetCDF Comparison

| Aspect | GRIB | NetCDF |
|--------|------|---------|
| **Format** | Binary messages | Hierarchical arrays |
| **Structure** | Sequential messages | Multidimensional arrays |
| **Metadata** | Embedded in each message | Separate attributes |
| **Time Storage** | Separate message per time/variable | Single array with time dimension |
| **File Size** | Smaller (compressed) | Larger (uncompressed) |
| **Usage** | Meteorological exchange | Scientific analysis |
| **Tools** | eccodes, cfgrib | xarray, netCDF4 |

### 5. Data Flow Visualization for GRIB

```
Input State (from model)
├── date: 2025-06-25T12:00:00
├── latitudes: [90.0, 89.9, ..., -90.0]
├── longitudes: [0.0, 0.1, ..., 359.9]
└── fields: {
    ├── t_1000: [285.3, 284.1, ...]
    ├── u_925: [12.4, -8.7, ...]
    └── ... 98 more fields
}
                    ↓
GRIB Message Creation
├── Message 1: t_1000 @ 12:00Z
│   ├── Grid: N320 icosahedral
│   ├── Parameter: Temperature (130)
│   ├── Level: 1000 hPa
│   ├── Step: 6 hours
│   └── Data: [285.3, 284.1, ...]
├── Message 2: u_925 @ 12:00Z
│   ├── Grid: N320 icosahedral
│   ├── Parameter: U-wind (131)
│   ├── Level: 925 hPa
│   └── Data: [12.4, -8.7, ...]
└── ... (100 messages per time step)
```

## Why outputs_initialized is Used

### The Initialization Pattern

```python
outputs_initialized = False

for i, state in enumerate(runner.run(input_state=input_state, lead_time=12)):
    if not outputs_initialized:
        netcdf_output.open(state)  # Uses context + first state
        outputs_initialized = True
    
    netcdf_output.write_step(state)    # Uses current state
```

### Why This Pattern Exists

1. **Context Provides Metadata**: Time steps, reference date, total duration
2. **State Provides Data**: Actual grid coordinates, field names, array shapes
3. **Both Are Needed**: Context for file structure, state for data dimensions

```python
# Context tells us HOW MANY time steps to expect
context.lead_time = 12
context.time_step = 6
# Therefore: expect 3 time steps (12/6 + initial = 3)

# State tells us WHAT the data looks like
state["latitudes"].shape  # (542080,) - grid size
state["fields"].keys()    # ['t_1000', 'u_925', ...] - variable names
```

### Why Not Initialize Outside the Loop?

The output classes need **both** context and state information:

```python
# This wouldn't work - no state data available yet
netcdf_output.open(???)  # Needs coordinate arrays from state

# This works - we have both context and state
for state in runner.run(...):
    if not outputs_initialized:
        netcdf_output.open(state)  # Now we have coordinates!
```

## GRIB-Specific Features

### 1. Parameter ID Mapping

```python
# GRIB uses standard parameter IDs
parameter_mapping = {
    "t_1000": {"paramId": 130, "levelType": "pl", "level": 1000},
    "u_925": {"paramId": 131, "levelType": "pl", "level": 925},
    "v_925": {"paramId": 132, "levelType": "pl", "level": 925},
    "2t": {"paramId": 167, "levelType": "sfc"},
    "10u": {"paramId": 165, "levelType": "sfc"},
    "10v": {"paramId": 166, "levelType": "sfc"},
}
```

### 2. Grid Specification

```python
# GRIB grid definition for N320
grid_definition = {
    "gridType": "reduced_gg",
    "N": 320,                    # Gaussian number
    "numberOfPoints": 542080,    # Total grid points
    "coordinateValues": [...],   # Lat/lon for each point
}
```

### 3. Data Packing

```python
# GRIB supports various packing methods
packing_options = {
    "simple": "Direct binary representation",
    "jpeg": "JPEG 2000 compression",
    "ccsds": "CCSDS compression",
    "grid_simple": "Simple grid point packing",
}
```

## File Output Verification

### Reading GRIB Files

```python
import earthkit.data as ekd

# Read GRIB file
grib_data = ekd.from_source("file", "forecast.grib")

print(f"Number of messages: {len(grib_data)}")
print(f"First message: {grib_data[0].metadata()}")

# Filter by parameter
temperature_msgs = grib_data.sel(param="t", level=1000)
print(f"Temperature messages: {len(temperature_msgs)}")
```

### Using cfgrib (GRIB to xarray)

```python
import xarray as xr
import cfgrib

# Convert GRIB to xarray Dataset
ds = xr.open_dataset("forecast.grib", engine="cfgrib")
print(f"Variables: {list(ds.data_vars.keys())}")
print(f"Dimensions: {dict(ds.dims)}")
```

## Advanced GRIB Configuration

### 1. Custom Encoding

```python
# Custom GRIB encoding
custom_encoding = {
    "t_1000": {
        "paramId": 130,
        "levelType": "pl", 
        "level": 1000,
        "packing": "jpeg",
        "accuracy": 0.01
    }
}

grib_output = GribFileOutput(context, path=grib_file, encoding=custom_encoding)
```

### 2. Template Usage

```python
# Use GRIB templates for consistent encoding
grib_output = GribFileOutput(
    context, 
    path=grib_file,
    templates=["ecmwf_deterministic_forecast.yaml"]
)
```

### 3. Split Output Files

```python
# Create separate files for each time step
grib_output = GribFileOutput(
    context, 
    path="forecast_{step}.grib",
    split_output=True
)
```

## Memory and Performance Considerations

### File Size Comparison
- **GRIB**: 50-150 MB (compressed)
- **NetCDF**: 100-400 MB (uncompressed)
- **Compression ratio**: GRIB ~2-3x smaller

### Writing Performance
- **GRIB**: Sequential message writing
- **NetCDF**: Random access array writing
- **GRIB**: Better for streaming workflows
- **NetCDF**: Better for analysis workflows

## Error Handling

### Common GRIB Issues
1. **Missing Parameter IDs**: Variables without standard GRIB codes
2. **Grid Specification**: Complex icosahedral grids
3. **Coordinate Systems**: Longitude wrapping, pole handling
4. **File Corruption**: Incomplete messages from interrupted writes

### Debug Output
```python
import logging
logging.basicConfig(level=logging.DEBUG)
LOG = logging.getLogger("anemoi.inference.outputs.gribfile")
```

## Summary

The GRIB saving process complements NetCDF by providing:
- **Standard meteorological format** for operational use
- **Compressed storage** for efficient distribution
- **Message-based structure** for streaming applications
- **ECMWF compatibility** for weather center integration

The Context object serves as a configuration bridge, providing metadata that output classes need to properly structure files before any forecast data is available. The `outputs_initialized` pattern ensures that both Context metadata and State data are available when setting up the file structure.
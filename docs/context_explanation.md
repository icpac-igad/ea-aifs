# Understanding the Context Object in AIFS ENS v1 Forecasts

## Why Context is Created Before runner.run()

The Context object might seem unnecessary since it's created before the forecast loop but not directly used in it. However, it serves a crucial role in the output initialization process.

## The Context Object Explained

### What is Context?

```python
from anemoi.inference.context import Context

context = Context()
context.time_step = 6        # Hours between forecast steps
context.lead_time = 12       # Total forecast duration (hours)
context.reference_date = input_state["date"]  # Base date for time calculations
```

The Context object is a **metadata container** that provides configuration information needed by output classes to properly set up file structures.

### Key Properties

```python
class Context(ABC):
    # These properties are used by output classes
    reference_date = None      # Base date for time coordinate calculations
    time_step = None          # Hours between forecast steps
    lead_time = None          # Total forecast duration
    output_frequency = None   # How often to write outputs (optional)
    write_initial_state = True # Whether to include initial conditions
```

## The Initialization Flow

### 1. Context Creation (Before forecast loop)

```python
context = Context()
context.time_step = 6
context.lead_time = 12
context.reference_date = input_state["date"]
```

**Purpose**: Provides metadata about the forecast configuration

### 2. Output Object Creation (Before forecast loop)

```python
netcdf_output = NetCDFOutput(context, path=netcdf_file)
grib_output = GribFileOutput(context, path=grib_file)
```

**Purpose**: Output classes store the context but don't use it yet

### 3. File Initialization (First iteration of forecast loop)

```python
for i, state in enumerate(runner.run(input_state=input_state, lead_time=12)):
    if not outputs_initialized:
        netcdf_output.open(state)  # Uses BOTH context AND state
        outputs_initialized = True
```

**Purpose**: Now both metadata (context) and data (state) are available

## Why outputs_initialized = False Pattern?

### The Problem

Output classes need **two types of information**:

1. **Context metadata**: How many time steps? What's the reference date?
2. **State data**: What's the grid size? What variables exist?

### The Solution

```python
outputs_initialized = False

for i, state in enumerate(runner.run(input_state=input_state, lead_time=12)):
    if not outputs_initialized:
        # NOW we have both context AND state
        netcdf_output.open(state)  # Uses context + state together
        outputs_initialized = True
    
    netcdf_output.write_step(state)  # Uses current state data
```

### What Happens in output.open(state)?

#### NetCDF Example:
```python
def open(self, state: State) -> None:
    # FROM CONTEXT: Calculate time dimension
    time_steps = self.context.lead_time // self.context.time_step + 1
    
    # FROM STATE: Get spatial dimensions
    values = len(state["latitudes"])
    
    # Create NetCDF file structure
    self.time_dim = self.ncfile.createDimension("time", time_steps)
    self.values_dim = self.ncfile.createDimension("values", values)
    
    # FROM CONTEXT: Set up time coordinate
    self.time_var.units = f"seconds since {self.context.reference_date}"
    
    # FROM STATE: Set up spatial coordinates
    self.latitude_var[:] = state["latitudes"]
    self.longitude_var[:] = state["longitudes"]
```

## Alternative Approaches (and why they don't work)

### ❌ Approach 1: Initialize outside the loop

```python
# This wouldn't work
netcdf_output.open()  # ERROR: No state data available yet!

for state in runner.run(...):
    netcdf_output.write_step(state)
```

**Problem**: No access to actual grid coordinates or variable names

### ❌ Approach 2: Pass context to runner

```python
# This doesn't exist in the API
for state in runner.run(input_state=input_state, lead_time=12, context=context):
    pass
```

**Problem**: The runner doesn't need context - it only generates states

### ✅ Approach 3: Current pattern (correct)

```python
# Context provides metadata
context = Context()
context.time_step = 6
context.lead_time = 12

# Output stores context for later use
netcdf_output = NetCDFOutput(context, path=netcdf_file)

# On first iteration, both context and state are available
for state in runner.run(...):
    if not outputs_initialized:
        netcdf_output.open(state)  # Uses context + state
        outputs_initialized = True
    
    netcdf_output.write_step(state)
```

## Context Usage in Different Output Classes

### NetCDF Output
```python
# Context used for:
time_steps = context.lead_time // context.time_step + 1
reference_date = context.reference_date
time_var.units = f"seconds since {reference_date}"
```

### GRIB Output
```python
# Context used for:
step_hours = (state["date"] - context.reference_date).total_seconds() / 3600
reference_date = context.reference_date
grib_message["dataDate"] = reference_date.strftime("%Y%m%d")
```

## Visual Flow Diagram

```
Step 1: Create Context
context = Context()
context.time_step = 6
context.lead_time = 12
context.reference_date = DATE
                    ↓
Step 2: Create Output Objects
netcdf_output = NetCDFOutput(context, path="file.nc")
# Output objects store context but don't use it yet
                    ↓
Step 3: Start Forecast Loop
for state in runner.run(...):
    ↓
Step 4: First Iteration Only
if not outputs_initialized:
    netcdf_output.open(state)  # Uses context + state together
    outputs_initialized = True
    ↓
Step 5: Every Iteration
netcdf_output.write_step(state)  # Uses current state
```

## The Key Insight

The Context object **is** used in the forecast loop, but **indirectly**:

1. **Context is stored** in the output objects when they're created
2. **Context is accessed** when `output.open(state)` is called
3. **Context is used** to set up file structures and time coordinates

The `outputs_initialized = False` pattern ensures that the output setup happens exactly once, when both the context metadata and the first state data are available.

## Why This Architecture?

This design separates concerns:
- **Context**: Configuration and metadata
- **State**: Dynamic forecast data
- **Output**: File writing logic

This allows the same output classes to work with different runners, contexts, and data sources while maintaining clean separation of responsibilities.
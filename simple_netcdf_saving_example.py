#!/usr/bin/env python3
"""
Simple example showing how to save AIFS ENS v1 forecast states to NetCDF.

This demonstrates the exact method of extending the basic forecast loop
with NetCDF output capabilities.
"""

import datetime
import os
from collections import defaultdict

import numpy as np
import earthkit.data as ekd
import earthkit.regrid as ekr
from ecmwf.opendata import Client as OpendataClient

# Core anemoi imports
from anemoi.inference.runners.simple import SimpleRunner
from anemoi.inference.outputs.printer import print_state
from anemoi.inference.outputs.netcdf import NetCDFOutput
from anemoi.inference.context import Context


def prepare_minimal_input_state():
    """Prepare a minimal input state for demonstration."""
    # Use the same parameters as the notebook
    PARAM_SFC = ["10u", "10v", "2d", "2t", "msl", "skt", "sp", "tcw"]
    PARAM_SFC_FC = ["lsm", "z", "slor", "sdor"]
    PARAM_SOIL = ["sot"]
    PARAM_PL = ["gh", "t", "u", "v", "w", "q"]
    LEVELS = [1000, 925, 850, 700, 600, 500, 400, 300, 250, 200, 150, 100, 50]
    SOIL_LEVELS = [1, 2]
    
    # Get latest date
    DATE = OpendataClient("ecmwf").latest()
    print(f"Using date: {DATE}")
    
    # Simplified data retrieval function
    def get_data(param, levelist=None, number=None):
        fields = defaultdict(list)
        for date in [DATE - datetime.timedelta(hours=6), DATE]:
            data = ekd.from_source("ecmwf-open-data", date=date, param=param, levelist=levelist)
            for f in data:
                values = np.roll(f.to_numpy(), -f.shape[1] // 2, axis=1)
                values = ekr.interpolate(values, {"grid": (0.25, 0.25)}, {"grid": "N320"})
                name = f"{f.metadata('param')}_{f.metadata('levelist')}" if levelist else f.metadata("param")
                fields[name].append(values)
        
        for param, values in fields.items():
            fields[param] = np.stack(values)
        return fields
    
    # Gather all fields
    fields = {}
    print("Retrieving surface fields...")
    fields.update(get_data(param=PARAM_SFC))
    fields.update(get_data(param=PARAM_SFC_FC))
    
    print("Retrieving soil fields...")
    soil = get_data(param=PARAM_SOIL, levelist=SOIL_LEVELS)
    mapping = {'sot_1': 'stl1', 'sot_2': 'stl2'}
    for k, v in soil.items():
        if k in mapping:
            fields[mapping[k]] = v
    
    print("Retrieving pressure level fields...")
    fields.update(get_data(param=PARAM_PL, levelist=LEVELS))
    
    # Convert geopotential height to geopotential
    for level in LEVELS:
        if f"gh_{level}" in fields:
            gh = fields.pop(f"gh_{level}")
            fields[f"z_{level}"] = gh * 9.80665
    
    return {"date": DATE, "fields": fields}


def method_1_basic_forecast_loop():
    """Method 1: Basic forecast loop with just print_state (original notebook method)."""
    print("=== METHOD 1: Basic Forecast Loop ===")
    
    # Initialize runner
    checkpoint = {"huggingface": "ecmwf/aifs-ens-1.0"}
    runner = SimpleRunner(checkpoint, device="cuda")
    
    # Prepare input
    input_state = prepare_minimal_input_state()
    
    # Basic forecast loop (original method)
    print("Running basic forecast loop...")
    for state in runner.run(input_state=input_state, lead_time=12):
        print_state(state)
    
    print("Basic forecast completed.\n")


def method_2_netcdf_saving():
    """Method 2: Enhanced forecast loop with NetCDF saving."""
    print("=== METHOD 2: Enhanced Forecast Loop with NetCDF Saving ===")
    
    # Initialize runner
    checkpoint = {"huggingface": "ecmwf/aifs-ens-1.0"}
    runner = SimpleRunner(checkpoint, device="cuda")
    
    # Prepare input
    input_state = prepare_minimal_input_state()
    
    # Setup NetCDF output
    output_dir = "forecast_outputs"
    os.makedirs(output_dir, exist_ok=True)
    
    date_str = input_state["date"].strftime("%Y%m%d_%H%M")
    netcdf_file = f"{output_dir}/forecast_{date_str}.nc"
    
    # Configure NetCDF context
    context = Context()
    context.time_step = 6        # 6-hour time steps
    context.lead_time = 12       # 12 hours total forecast
    context.reference_date = input_state["date"]
    
    # Create NetCDF output object
    netcdf_output = NetCDFOutput(context, path=netcdf_file)
    
    # Enhanced forecast loop
    print("Running enhanced forecast loop with NetCDF saving...")
    forecast_states = []
    outputs_initialized = False
    
    for i, state in enumerate(runner.run(input_state=input_state, lead_time=12)):
        # Original functionality: visual progress
        print_state(state)
        
        # NEW: Initialize NetCDF file on first iteration
        if not outputs_initialized:
            print(f"🔧 Initializing NetCDF file: {netcdf_file}")
            netcdf_output.open(state)
            outputs_initialized = True
        
        # NEW: Save state to NetCDF file
        print(f"💾 Saving time step {i+1} to NetCDF...")
        netcdf_output.write_step(state)
        
        # Store state for later use (optional)
        forecast_states.append(state)
    
    # NEW: Close the NetCDF file
    netcdf_output.close()
    
    # Verify the file was created
    if os.path.exists(netcdf_file):
        file_size = os.path.getsize(netcdf_file) / (1024 * 1024)  # MB
        print(f"✅ NetCDF file created successfully: {netcdf_file}")
        print(f"📊 File size: {file_size:.2f} MB")
        print(f"⏱️  Time steps saved: {len(forecast_states)}")
    else:
        print(f"❌ NetCDF file not created: {netcdf_file}")
    
    return forecast_states, netcdf_file


def method_3_verify_netcdf_content(netcdf_file):
    """Method 3: Verify NetCDF file content."""
    print("=== METHOD 3: Verify NetCDF Content ===")
    
    try:
        import xarray as xr
        
        # Open the NetCDF file
        ds = xr.open_dataset(netcdf_file)
        
        print(f"🔍 NetCDF file analysis:")
        print(f"   Dimensions: {dict(ds.dims)}")
        print(f"   Variables: {len(ds.data_vars)} total")
        print(f"   Time steps: {len(ds.time)}")
        print(f"   Grid points: {len(ds.latitude)}")
        
        # Show some key variables
        key_vars = ["t_1000", "u_925", "v_925", "2t", "10u", "10v"]
        available_vars = [v for v in key_vars if v in ds.data_vars]
        print(f"   Key variables available: {available_vars}")
        
        # Show time coordinate
        times = ds.time.values
        print(f"   Time coordinates: {times}")
        
        # Show data for first variable
        if available_vars:
            var_name = available_vars[0]
            var_data = ds[var_name].values
            print(f"   {var_name} data shape: {var_data.shape}")
            print(f"   {var_name} data range: {var_data.min():.3f} to {var_data.max():.3f}")
        
        ds.close()
        
    except ImportError:
        print("❌ xarray not available for verification")
    except Exception as e:
        print(f"❌ Error reading NetCDF file: {e}")


def demonstrate_the_difference():
    """Demonstrate the difference between print_state and NetCDF saving."""
    print("=== DEMONSTRATION: print_state vs NetCDF Saving ===")
    
    # Create a sample state (simulated)
    sample_state = {
        "date": datetime.datetime(2025, 6, 25, 12, 0, 0),
        "latitudes": np.random.uniform(-90, 90, 100),  # Simplified for demo
        "longitudes": np.random.uniform(-180, 180, 100),
        "fields": {
            "t_1000": np.random.uniform(220, 320, 100),  # Temperature
            "u_925": np.random.uniform(-50, 50, 100),    # U-wind
            "v_925": np.random.uniform(-50, 50, 100),    # V-wind
            "2t": np.random.uniform(250, 310, 100),      # 2m temperature
        }
    }
    
    print("📊 What print_state shows:")
    print_state(sample_state)
    
    print("💾 What gets saved to NetCDF:")
    print("   - Complete latitude array: 100 values")
    print("   - Complete longitude array: 100 values") 
    print("   - Complete field data for ALL variables:")
    for name, data in sample_state["fields"].items():
        print(f"     {name}: {data.shape} array with values {data.min():.3f} to {data.max():.3f}")
    
    print("\n🔑 Key Differences:")
    print("   print_state: Shows summary statistics (min/max) of selected fields")
    print("   NetCDF:      Saves complete data arrays for all fields")
    print("   print_state: Temporary console output")
    print("   NetCDF:      Permanent file storage")
    print("   print_state: Human-readable")
    print("   NetCDF:      Machine-readable, analyzable")


if __name__ == "__main__":
    print("🚀 AIFS ENS v1 NetCDF Saving Methods Demonstration")
    print("=" * 60)
    
    # Show the conceptual difference first
    demonstrate_the_difference()
    
    # Method 1: Basic forecast loop (original notebook method)
    # method_1_basic_forecast_loop()
    
    # Method 2: Enhanced forecast loop with NetCDF saving
    forecast_states, netcdf_file = method_2_netcdf_saving()
    
    # Method 3: Verify NetCDF content
    method_3_verify_netcdf_content(netcdf_file)
    
    print("\n✅ Demonstration completed!")
    print(f"📁 Output file: {netcdf_file}")
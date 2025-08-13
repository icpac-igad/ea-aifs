#!/usr/bin/env python3
"""
Multi-Run AIFS ENS v1 - Running 50 Ensemble Members

This script runs ECMWF's aifs-ens-v1 data-driven model for multiple ensemble members (1-50),
using ECMWF's open data dataset and the anemoi-inference package.

Each ensemble member is saved as a separate GRIB file.
"""

import datetime
from collections import defaultdict
import os
import time
import pickle
import argparse

import numpy as np
import earthkit.data as ekd
import earthkit.regrid as ekr

from anemoi.inference.runners.simple import SimpleRunner
from anemoi.inference.outputs.printer import print_state
from anemoi.inference.outputs.gribfile import GribFileOutput

from ecmwf.opendata import Client as OpendataClient


# Configuration
PARAM_SFC = ["10u", "10v", "2d", "2t", "msl", "skt", "sp", "tcw"]
PARAM_SFC_FC = ["lsm", "z", "slor", "sdor"]
PARAM_SOIL = ["sot"]
PARAM_PL = ["gh", "t", "u", "v", "w", "q"]
LEVELS = [1000, 925, 850, 700, 600, 500, 400, 300, 250, 200, 150, 100, 50]
SOIL_LEVELS = [1, 2]

# Multi-run configuration
ENSEMBLE_MEMBERS = list(range(1, 51))  # Members 1-50
LEAD_TIME = 72  # Hours
OUTPUT_DIR = "ensemble_outputs"

# Pickle file configuration
USE_PICKLE_FILES = False  # Set to True to use pre-downloaded pickle files
PICKLE_INPUT_DIR = "./input_states"  # Directory containing downloaded pickle files


def get_open_data(date, param, levelist=[], number=None):
    """Retrieve data from ECMWF Open Data API."""
    fields = defaultdict(list)
    # Get the data for the current date and the previous date
    for d in [date - datetime.timedelta(hours=6), date]:
        if number is None:
            data = ekd.from_source("ecmwf-open-data", date=d, param=param, levelist=levelist)
        else:
            data = ekd.from_source("ecmwf-open-data", date=d, param=param, levelist=levelist, 
                                 number=[number], stream='enfo')
        
        for f in data:
            # Open data is between -180 and 180, we need to shift it to 0-360
            assert f.to_numpy().shape == (721, 1440)
            values = np.roll(f.to_numpy(), -f.shape[1] // 2, axis=1)
            # Interpolate the data from 0.25 to N320
            values = ekr.interpolate(values, {"grid": (0.25, 0.25)}, {"grid": "N320"})
            # Add the values to the list
            name = f"{f.metadata('param')}_{f.metadata('levelist')}" if levelist else f.metadata("param")
            fields[name].append(values)

    # Create a single matrix for each parameter
    for param, values in fields.items():
        fields[param] = np.stack(values)

    return fields


def load_input_state_from_pickle(member, pickle_dir):
    """Load input state from pre-downloaded pickle file."""
    pickle_file = os.path.join(pickle_dir, f"input_state_member_{member:03d}.pkl")
    
    if not os.path.exists(pickle_file):
        raise FileNotFoundError(f"Pickle file not found: {pickle_file}")
    
    try:
        with open(pickle_file, 'rb') as f:
            input_state = pickle.load(f)
        
        print(f"    ✅ Loaded from pickle: {pickle_file}")
        print(f"    Date: {input_state['date']}")
        print(f"    Fields: {len(input_state['fields'])}")
        
        return input_state
    except Exception as e:
        raise RuntimeError(f"Error loading pickle file {pickle_file}: {str(e)}")


def get_input_fields(date, number):
    """Get input fields for a specific ensemble member."""
    fields = {}
    
    # Add single level fields
    fields.update(get_open_data(date, param=PARAM_SFC, number=number))
    fields.update(get_open_data(date, param=PARAM_SFC_FC))  # Constant fields
    
    # Add soil fields
    soil = get_open_data(date, param=PARAM_SOIL, levelist=SOIL_LEVELS, number=number)
    
    # Rename soil parameters
    mapping = {'sot_1': 'stl1', 'sot_2': 'stl2',
               'vsw_1': 'swvl1', 'vsw_2': 'swvl2'}
    for k, v in soil.items():
        fields[mapping[k]] = v
    
    # Add pressure level fields
    fields.update(get_open_data(date, param=PARAM_PL, levelist=LEVELS, number=number))
    
    # Convert geopotential height to geopotential
    for level in LEVELS:
        gh = fields.pop(f"gh_{level}")
        fields[f"z_{level}"] = gh * 9.80665
    
    return fields


def run_ensemble_member(runner, date, member, output_dir):
    """Run forecast for a single ensemble member and save to GRIB."""
    date_str = date.strftime("%Y%m%d_%H%M")
    grib_file = f"{output_dir}/aifs_ens_forecast_{date_str}_member{member:03d}.grib"
    
    # Get input fields (either from pickle or live download)
    if USE_PICKLE_FILES:
        print(f"Loading initial conditions from pickle for member {member}...")
        input_state = load_input_state_from_pickle(member, PICKLE_INPUT_DIR)
        # Use the date from the pickle file
        date = input_state['date']
    else:
        print(f"Retrieving initial conditions for member {member}...")
        fields = get_input_fields(date, member)
        input_state = dict(date=date, fields=fields)
    
    # Set context properties on the runner (which extends Context)
    runner.time_step = 6  # 6-hour time step
    runner.lead_time = LEAD_TIME
    runner.reference_date = date
    
    # Initialize GRIB output using runner as context
    grib_output = GribFileOutput(runner, path=grib_file)
    
    # Run forecast
    print(f"Running forecast for member {member}...")
    outputs_initialized = False
    step_count = 0
    
    for state in runner.run(input_state=input_state, lead_time=LEAD_TIME):
        # Initialize output on first state
        if not outputs_initialized:
            grib_output.open(state)
            outputs_initialized = True
        
        # Write to output
        grib_output.write_step(state)
        step_count += 1
        
        # Print progress every 4 steps (24 hours)
        if step_count % 4 == 0:
            print(f"  Member {member}: {step_count * 6} hours completed")
    
    # Close output
    grib_output.close()
    
    # Verify file
    if os.path.exists(grib_file):
        file_size = os.path.getsize(grib_file) / (1024 * 1024)  # Size in MB
        print(f"✓ Member {member} completed: {grib_file} ({file_size:.2f} MB)")
        return True, file_size
    else:
        print(f"✗ Error: Output file not created for member {member}")
        return False, 0


def main():
    """Main function to run all ensemble members."""
    # Get latest date
    DATE = OpendataClient("ecmwf").latest()
    print("Initial date is", DATE)
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Optional: Set environment variables to reduce memory usage
    # os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True' 
    # os.environ['ANEMOI_INFERENCE_NUM_CHUNKS'] = '16'
    
    # Load model
    print("\nLoading model...")
    checkpoint = {"huggingface": "ecmwf/aifs-ens-1.0"}
    runner = SimpleRunner(checkpoint, device="cuda")
    print("Model loaded successfully!")
    
    # Track processing
    processing_times = []
    successful_members = []
    total_size = 0
    
    # Run all ensemble members
    for member in ENSEMBLE_MEMBERS:
        start_time = time.time()
        print(f"\n{'='*60}")
        print(f"Processing ensemble member {member}/{len(ENSEMBLE_MEMBERS)}")
        print(f"{'='*60}")
        
        try:
            success, file_size = run_ensemble_member(runner, DATE, member, OUTPUT_DIR)
            
            if success:
                successful_members.append(member)
                total_size += file_size
            
            # Track time
            member_time = time.time() - start_time
            processing_times.append(member_time)
            print(f"Member {member} processing time: {member_time:.2f} seconds")
            
            # Estimate remaining time
            if len(processing_times) > 0:
                avg_time = sum(processing_times) / len(processing_times)
                remaining_members = len(ENSEMBLE_MEMBERS) - member
                est_remaining = avg_time * remaining_members
                print(f"Estimated time remaining: {est_remaining/60:.1f} minutes")
                
        except Exception as e:
            print(f"✗ Error processing member {member}: {str(e)}")
            continue
    
    # Summary
    print(f"\n{'='*60}")
    print("Processing complete!")
    print(f"Successfully processed {len(successful_members)}/{len(ENSEMBLE_MEMBERS)} members")
    print(f"Total processing time: {sum(processing_times)/60:.1f} minutes")
    if processing_times:
        print(f"Average time per member: {sum(processing_times)/len(processing_times):.2f} seconds")
    print(f"Total disk space used: {total_size:.2f} MB ({total_size/1024:.2f} GB)")
    
    # List generated files
    date_str = DATE.strftime("%Y%m%d_%H%M")
    grib_files = sorted([f for f in os.listdir(OUTPUT_DIR) 
                        if f.startswith(f"aifs_ens_forecast_{date_str}") and f.endswith('.grib')])
    print(f"\nGenerated {len(grib_files)} GRIB files in {OUTPUT_DIR}/")
    
    # Verify a sample file
    if grib_files:
        sample_file = os.path.join(OUTPUT_DIR, grib_files[0])
        print(f"\nVerifying sample file: {grib_files[0]}")
        try:
            grib_data = ekd.from_source("file", sample_file)
            print(f"File contains {len(grib_data)} fields")
            # Show first few fields
            print("First 5 fields:")
            for i, field in enumerate(grib_data[:5]):
                meta = field.metadata()
                print(f"  {i+1}: {meta.get('param')} at {meta.get('levelist', 'surface')} - "
                      f"step: {meta.get('step')} hours")
        except Exception as e:
            print(f"Error reading file: {e}")


if __name__ == "__main__":
    main()
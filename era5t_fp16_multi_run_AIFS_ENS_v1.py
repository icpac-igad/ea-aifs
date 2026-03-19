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
ENSEMBLE_MEMBERS = list(range(1, 10))  # Members 1-50
LEAD_TIME = 960  # Hours (40 days)
OUTPUT_DIR = "/scratch/ensemble_outputs"

# Pickle file configuration
USE_PICKLE_FILES = True  # Set to True to use pre-downloaded pickle files
PICKLE_INPUT_DIR = "/scratch/input_states"  # Directory containing downloaded pickle files

# GPU Memory Optimization Configuration
# Reference: https://huggingface.co/ecmwf/aifs-ens-1.0/discussions/17
# Full precision requires ~50GB VRAM. These settings reduce memory usage to <24GB.
GPU_MEMORY_OPTIMIZATION = True  # Set to True to enable memory optimizations
INFERENCE_PRECISION = "16"  # "16" for half precision (FP16), "32" for full precision
INFERENCE_NUM_CHUNKS = 16  # Higher values reduce memory but may slow inference (8, 16, 32, 64)


def get_open_data(date, param, levelist=[], number=None):
    """Retrieve data from ECMWF Open Data API."""
    fields = defaultdict(list)
    # Get the data for the current date and the previous date
    for d in [date - datetime.timedelta(hours=6), date]:
        if number is None:
            data = ekd.from_source("ecmwf-open-data",
                                   date=d,
                                   param=param,
                                   levelist=levelist)
        else:
            data = ekd.from_source("ecmwf-open-data",
                                   date=d,
                                   param=param,
                                   levelist=levelist,
                                   number=[number],
                                   stream='enfo')

        for f in data:
            # Open data is between -180 and 180, we need to shift it to 0-360
            assert f.to_numpy().shape == (721, 1440)
            values = np.roll(f.to_numpy(), -f.shape[1] // 2, axis=1)
            # Interpolate the data from 0.25 to N320
            values = ekr.interpolate(values, {"grid": (0.25, 0.25)},
                                     {"grid": "N320"})
            # Add the values to the list
            name = f"{f.metadata('param')}_{f.metadata('levelist')}" if levelist else f.metadata(
                "param")
            fields[name].append(values)

    # Create a single matrix for each parameter
    for param, values in fields.items():
        fields[param] = np.stack(values)

    return fields


def load_input_state_from_pickle(member, pickle_dir):
    """Load input state from pre-downloaded pickle file."""
    pickle_file = os.path.join(pickle_dir,
                               f"input_state_member_{member:03d}.pkl")

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
        raise RuntimeError(
            f"Error loading pickle file {pickle_file}: {str(e)}")


def adapt_era5t_fields(fields):
    """Adapt ERA5T 74-field pkl to the 92 fields expected by aifs-ens-1.0.

    ERA5T (AIFSgaia) has: 10u, 10v, 2t, msl, tcwv + t,q,z,u,v x 13 PL + lsm,z,slor,sdor
    aifs-ens-1.0 expects: 10u, 10v, 2d, 2t, msl, skt, sp, tcw + stl1, stl2
                          + t,q,z,u,v,w x 13 PL + lsm,z,slor,sdor

    Missing fields are approximated from available data.
    """
    added = []

    # tcwv -> tcw (total column water vapour is dominant component of total column water)
    if "tcwv" in fields and "tcw" not in fields:
        fields["tcw"] = fields.pop("tcwv")
        added.append("tcw (from tcwv)")

    # sp: derive from msl using the hypsometric equation
    # SP = MSL * exp(-z_sfc / (R_d * T_virtual))
    # where z_sfc is surface geopotential (m²/s²) and T_virtual ≈ 2t * (1 + 0.61*q)
    # The constant field "z" in the pkl is surface geopotential (orography).
    if "sp" not in fields and "msl" in fields:
        if "z" in fields:
            Rd = 287.058  # J/(kg·K) specific gas constant for dry air
            z_sfc = fields["z"]  # surface geopotential, m²/s²
            t2m = fields["2t"]   # 2-meter temperature, K
            # Use 2t as proxy for mean virtual temperature in the column
            # between surface and sea level. Add small moisture correction.
            T_mean = t2m * 1.003  # ~0.3% virtual temperature correction
            # Avoid division by zero in extreme cold regions
            T_mean = np.maximum(T_mean, 200.0)
            fields["sp"] = fields["msl"] * np.exp(-z_sfc / (Rd * T_mean))
            added.append("sp (from msl + z via hypsometric eq)")
        else:
            # Fallback: direct copy (better than nothing, but loses altitude correction)
            fields["sp"] = fields["msl"].copy()
            added.append("sp (from msl, WARNING: no orography correction)")

    # 2d: dewpoint ≈ 2t - 2K (conservative dewpoint depression)
    if "2d" not in fields and "2t" in fields:
        fields["2d"] = fields["2t"] - 2.0
        added.append("2d (from 2t - 2K)")

    # skt: skin temperature ≈ 2m temperature
    if "skt" not in fields and "2t" in fields:
        fields["skt"] = fields["2t"].copy()
        added.append("skt (from 2t)")

    # stl1, stl2: soil temperature ≈ 2m temperature
    if "stl1" not in fields and "2t" in fields:
        fields["stl1"] = fields["2t"].copy()
        added.append("stl1 (from 2t)")
    if "stl2" not in fields and "2t" in fields:
        fields["stl2"] = fields["2t"].copy()
        added.append("stl2 (from 2t)")

    # w (vertical velocity): set to zero at all pressure levels
    for level in LEVELS:
        key = f"w_{level}"
        if key not in fields:
            # Use shape from any existing PL field
            ref_key = f"t_{level}"
            if ref_key in fields:
                fields[key] = np.zeros_like(fields[ref_key])
                added.append(key)

    if added:
        print(f"    Adapted ERA5T fields: added {len(added)} derived fields")
        print(f"      {', '.join(added[:6])}")
        if len(added) > 6:
            print(f"      ... and {len(added) - 6} more (w at pressure levels)")
        print(f"    Total fields: {len(fields)}")

    return fields


def get_input_fields(date, number):
    """Get input fields for a specific ensemble member."""
    fields = {}

    # Add single level fields
    fields.update(get_open_data(date, param=PARAM_SFC, number=number))
    fields.update(get_open_data(date, param=PARAM_SFC_FC))  # Constant fields

    # Add soil fields
    soil = get_open_data(date,
                         param=PARAM_SOIL,
                         levelist=SOIL_LEVELS,
                         number=number)

    # Rename soil parameters
    mapping = {
        'sot_1': 'stl1',
        'sot_2': 'stl2',
        'vsw_1': 'swvl1',
        'vsw_2': 'swvl2'
    }
    for k, v in soil.items():
        fields[mapping[k]] = v

    # Add pressure level fields
    fields.update(
        get_open_data(date, param=PARAM_PL, levelist=LEVELS, number=number))

    # Convert geopotential height to geopotential
    for level in LEVELS:
        gh = fields.pop(f"gh_{level}")
        fields[f"z_{level}"] = gh * 9.80665

    return fields


def run_ensemble_member(runner, date, member, output_dir):
    """Run forecast for a single ensemble member and save to GRIB files every 72 hours."""
    # Get input fields (either from pickle or live download)
    if USE_PICKLE_FILES:
        print(f"Loading initial conditions from pickle for member {member}...")
        input_state = load_input_state_from_pickle(member, PICKLE_INPUT_DIR)
        # Adapt ERA5T 74-field pkl to 92 fields if needed
        if len(input_state['fields']) < 90:
            input_state['fields'] = adapt_era5t_fields(input_state['fields'])
        # Use the date from the pickle file
        date = input_state['date']
    else:
        print(f"Retrieving initial conditions for member {member}...")
        fields = get_input_fields(date, member)
        input_state = dict(date=date, fields=fields)

    # Create date string AFTER determining the actual forecast date
    # Format: YYYYMMDD_HHMM (e.g., 20251127_0000)
    date_str = date.strftime("%Y%m%d_%H%M")

    # Set context properties on the runner (which extends Context)
    runner.time_step = 6  # 6-hour time step
    runner.lead_time = LEAD_TIME
    runner.reference_date = date

    # Calculate steps per 72-hour period
    hours_per_file = 72
    steps_per_file = hours_per_file // runner.time_step  # 72/6 = 12 steps

    # Track files created and total size
    created_files = []
    total_size = 0

    # Run forecast
    print(f"Running forecast for member {member}...")
    step_count = 0
    current_file_step = 0
    grib_output = None
    outputs_initialized = False

    for state in runner.run(input_state=input_state, lead_time=LEAD_TIME):
        # Check if we need to start a new file
        if current_file_step == 0:
            # Close previous file if exists
            if grib_output is not None:
                grib_output.close()
                outputs_initialized = False

            # Create new file name with time range
            start_hour = step_count * runner.time_step
            end_hour = min(start_hour + hours_per_file, LEAD_TIME)
            grib_file = f"{output_dir}/aifs_ens_forecast_{date_str}_member{member:03d}_h{start_hour:03d}-{end_hour:03d}.grib"

            # Initialize new GRIB output
            grib_output = GribFileOutput(runner, path=grib_file)
            print(f"  Creating file: {os.path.basename(grib_file)}")

        # Initialize output on first state of each file
        if not outputs_initialized:
            grib_output.open(state)
            outputs_initialized = True

        # Write to output
        grib_output.write_step(state)
        step_count += 1
        current_file_step += 1

        # Print progress every 4 steps (24 hours)
        if step_count % 4 == 0:
            print(f"  Member {member}: {step_count * 6} hours completed")

        # Check if we've completed a 72-hour period
        if current_file_step >= steps_per_file or step_count * runner.time_step >= LEAD_TIME:
            # Close current file
            grib_output.close()
            outputs_initialized = False

            # Verify and track the file
            if os.path.exists(grib_file):
                file_size = os.path.getsize(grib_file) / (1024 * 1024
                                                          )  # Size in MB
                created_files.append(os.path.basename(grib_file))
                total_size += file_size
                print(
                    f"  ✓ Completed: {os.path.basename(grib_file)} ({file_size:.2f} MB)"
                )
            else:
                print(
                    f"  ✗ Error: File not created: {os.path.basename(grib_file)}"
                )

            # Reset for next file
            current_file_step = 0

    # Close final file if still open
    if grib_output is not None and outputs_initialized:
        grib_output.close()

    # Summary for this member
    if created_files:
        print(
            f"✓ Member {member} completed: {len(created_files)} files, total {total_size:.2f} MB"
        )
        return True, total_size
    else:
        print(f"✗ Error: No output files created for member {member}")
        return False, 0


def main():
    """Main function to run all ensemble members."""
    # Get latest date
    #DATE = OpendataClient("ecmwf").latest()
    DATE = datetime.datetime(2025, 11, 27, 0, 0)
    print("Initial date is", DATE)

    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # GPU Memory Optimization Settings
    # These settings help reduce VRAM usage from ~50GB to <24GB
    if GPU_MEMORY_OPTIMIZATION:
        print("\n" + "=" * 60)
        print("GPU Memory Optimization: ENABLED")
        print(f"  Precision: {'FP16 (half)' if INFERENCE_PRECISION == '16' else 'FP32 (full)'}")
        print(f"  Inference chunks: {INFERENCE_NUM_CHUNKS}")
        print("=" * 60)

        # Set environment variables for memory optimization
        os.environ['ANEMOI_INFERENCE_NUM_CHUNKS'] = str(INFERENCE_NUM_CHUNKS)
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    else:
        print("\nGPU Memory Optimization: DISABLED (requires ~50GB VRAM)")

    # Load model
    print("\nLoading model...")
    checkpoint = {"huggingface": "ecmwf/aifs-ens-1.0"}

    # Configure runner with precision settings
    if GPU_MEMORY_OPTIMIZATION and INFERENCE_PRECISION == "16":
        runner = SimpleRunner(checkpoint, device="cuda", precision="16")
        print("Model loaded with FP16 precision (reduced memory mode)")
    else:
        runner = SimpleRunner(checkpoint, device="cuda")
        print("Model loaded with FP32 precision (full memory mode)")

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
            success, file_size = run_ensemble_member(runner, DATE, member,
                                                     OUTPUT_DIR)

            if success:
                successful_members.append(member)
                total_size += file_size

            # Track time
            member_time = time.time() - start_time
            processing_times.append(member_time)
            print(
                f"Member {member} processing time: {member_time:.2f} seconds")

            # Estimate remaining time
            if len(processing_times) > 0:
                avg_time = sum(processing_times) / len(processing_times)
                remaining_members = len(ENSEMBLE_MEMBERS) - member
                est_remaining = avg_time * remaining_members
                print(
                    f"Estimated time remaining: {est_remaining/60:.1f} minutes"
                )

        except Exception as e:
            print(f"✗ Error processing member {member}: {str(e)}")
            continue

    # Summary
    print(f"\n{'='*60}")
    print("Processing complete!")
    print(
        f"Successfully processed {len(successful_members)}/{len(ENSEMBLE_MEMBERS)} members"
    )
    print(f"Total processing time: {sum(processing_times)/60:.1f} minutes")
    if processing_times:
        print(
            f"Average time per member: {sum(processing_times)/len(processing_times):.2f} seconds"
        )
    print(
        f"Total disk space used: {total_size:.2f} MB ({total_size/1024:.2f} GB)"
    )

    # List generated files
    date_str = DATE.strftime("%Y%m%d_%H%M")
    grib_files = sorted([
        f for f in os.listdir(OUTPUT_DIR) if
        f.startswith(f"aifs_ens_forecast_{date_str}") and f.endswith('.grib')
    ])
    print(f"\nGenerated {len(grib_files)} GRIB files in {OUTPUT_DIR}/")

    # Count files per member (should be 1 file per 72-hour period for LEAD_TIME=72)
    expected_files_per_member = max(1, LEAD_TIME // 72)
    print(
        f"Expected {expected_files_per_member} file(s) per member for {LEAD_TIME}-hour forecast"
    )

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
                print(
                    f"  {i+1}: {meta.get('param')} at {meta.get('levelist', 'surface')} - "
                    f"step: {meta.get('step')} hours")
        except Exception as e:
            print(f"Error reading file: {e}")


if __name__ == "__main__":
    main()

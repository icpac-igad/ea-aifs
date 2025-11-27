#!/usr/bin/env python3
"""
Test Ensemble Input States Creation - CPU Only

This script tests the creation of input states for ensemble members 1-50
using ECMWF Open Data. It focuses only on data retrieval and preprocessing,
without running the actual model (no GPU required).
"""

import datetime
from collections import defaultdict
import time
import os
import pickle

import numpy as np
import earthkit.data as ekd
import earthkit.regrid as ekr
from google.cloud import storage

from ecmwf.opendata import Client as OpendataClient


# Configuration
PARAM_SFC = ["10u", "10v", "2d", "2t", "msl", "skt", "sp", "tcw"]
PARAM_SFC_FC = ["lsm", "z", "slor", "sdor"]
PARAM_SOIL = ["sot"]
PARAM_PL = ["gh", "t", "u", "v", "w", "q"]
LEVELS = [1000, 925, 850, 700, 600, 500, 400, 300, 250, 200, 150, 100, 50]
SOIL_LEVELS = [1, 2]

# Test configuration
ENSEMBLE_MEMBERS = list(range(1, 14))  # Test all 50 members
OUTPUT_DIR = "test_input_states"
SAVE_STATES = True  # Save input states to disk for verification

# GCS configuration
GCS_BUCKET = "aifs-aiquest-us-20251127"
GCS_SERVICE_ACCOUNT_KEY = "coiled-data.json"
UPLOAD_TO_GCS = True  # Upload pkl files to GCS
CLEANUP_LOCAL_FILES = True  # Remove local files after successful GCS upload


def get_open_data(date, param, levelist=[], number=None):
    """Retrieve data from ECMWF Open Data API."""
    fields = defaultdict(list)
    print(f"    Retrieving {param} data" + (f" at levels {levelist}" if levelist else "") + 
          (f" for member {number}" if number else ""))
    
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


def create_input_state(date, number):
    """Create input state for a specific ensemble member."""
    print(f"\nCreating input state for ensemble member {number}")
    start_time = time.time()
    
    fields = {}
    
    # Add single level fields
    print("  Getting surface fields...")
    fields.update(get_open_data(date, param=PARAM_SFC, number=number))
    
    print("  Getting constant surface fields...")
    fields.update(get_open_data(date, param=PARAM_SFC_FC))  # Constant fields
    
    # Add soil fields
    print("  Getting soil fields...")
    soil = get_open_data(date, param=PARAM_SOIL, levelist=SOIL_LEVELS, number=number)
    
    # Rename soil parameters
    mapping = {'sot_1': 'stl1', 'sot_2': 'stl2',
               'vsw_1': 'swvl1', 'vsw_2': 'swvl2'}
    for k, v in soil.items():
        fields[mapping[k]] = v
    
    # Add pressure level fields
    print("  Getting pressure level fields...")
    fields.update(get_open_data(date, param=PARAM_PL, levelist=LEVELS, number=number))
    
    # Convert geopotential height to geopotential
    print("  Converting geopotential height to geopotential...")
    for level in LEVELS:
        gh = fields.pop(f"gh_{level}")
        fields[f"z_{level}"] = gh * 9.80665
    
    # Create input state
    input_state = dict(date=date, fields=fields)
    
    elapsed_time = time.time() - start_time
    print(f"  Completed in {elapsed_time:.2f} seconds")
    
    # Print summary
    print(f"  Input state created with {len(fields)} fields")
    print(f"  Field shapes: {list(fields.values())[0].shape}")
    
    # Calculate memory usage
    total_elements = sum(field.size for field in fields.values())
    memory_mb = (total_elements * 4) / (1024 * 1024)  # Assuming float32
    print(f"  Approximate memory usage: {memory_mb:.2f} MB")
    
    return input_state


def verify_input_state(input_state, member):
    """Verify the input state has all required fields."""
    fields = input_state['fields']
    
    # Expected fields
    expected_surface = PARAM_SFC + PARAM_SFC_FC + ['stl1', 'stl2']
    expected_pressure = []
    for param in PARAM_PL:
        if param == 'gh':
            param = 'z'  # Converted to geopotential
        for level in LEVELS:
            expected_pressure.append(f"{param}_{level}")
    
    expected_total = expected_surface + expected_pressure
    
    # Check if all fields are present
    missing = [f for f in expected_total if f not in fields]
    extra = [f for f in fields if f not in expected_total]
    
    print(f"\n  Verification for member {member}:")
    print(f"    Expected fields: {len(expected_total)}")
    print(f"    Actual fields: {len(fields)}")
    
    if missing:
        print(f"    ⚠️  Missing fields: {missing}")
    if extra:
        print(f"    ⚠️  Extra fields: {extra}")
    
    if not missing and not extra:
        print(f"    ✅ All fields present and correct!")
        return True
    return False


def upload_to_gcs(local_file_path, gcs_bucket, gcs_blob_name, service_account_key):
    """Upload a file to Google Cloud Storage."""
    try:
        # Initialize GCS client with service account key
        client = storage.Client.from_service_account_json(service_account_key)
        bucket = client.bucket(gcs_bucket)
        blob = bucket.blob(gcs_blob_name)
        
        # Upload the file
        blob.upload_from_filename(local_file_path)
        print(f"    ✅ Uploaded to gs://{gcs_bucket}/{gcs_blob_name}")
        return True
    except Exception as e:
        print(f"    ❌ Failed to upload to GCS: {str(e)}")
        return False


def main():
    """Main function to test ensemble input state creation."""
    # Get latest date
    DATE = datetime.datetime(2025, 11, 27, 0, 0)  # 2025-09-11 00:00 UTC
    #DATE = OpendataClient("ecmwf").latest()
    datestr = DATE.strftime("%Y%m%d_%H%M")
    print(f"Initial date is {DATE}")
    print(f"Date string for GCS: {datestr}")
    print(f"Will test creating input states for {len(ENSEMBLE_MEMBERS)} ensemble members")
    
    # Create output directory if saving states
    if SAVE_STATES:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        print(f"\nInput states will be saved to {OUTPUT_DIR}/")
    
    UPLOAD_TO_GCS=True
    # Check GCS configuration
    if UPLOAD_TO_GCS:
        if not os.path.exists(GCS_SERVICE_ACCOUNT_KEY):
            print(f"❌ GCS service account key not found: {GCS_SERVICE_ACCOUNT_KEY}")
            print("Disabling GCS upload...")
            UPLOAD_TO_GCS = False
        else:
            print(f"✅ GCS upload enabled to bucket: {GCS_BUCKET}")
            print(f"Files will be saved under date folder: {datestr}/")
    
    # Track timing and success
    successful_members = []
    failed_members = []
    processing_times = []
    
    # Test creating input states for all members
    for i, member in enumerate(ENSEMBLE_MEMBERS):
        print(f"\n{'='*60}")
        print(f"Processing member {member} ({i+1}/{len(ENSEMBLE_MEMBERS)})")
        print(f"{'='*60}")
        
        try:
            start_time = time.time()
            
            # Create input state
            input_state = create_input_state(DATE, member)
            
            # Verify the state
            if verify_input_state(input_state, member):
                successful_members.append(member)
                
                # Optionally save the state
                if SAVE_STATES:
                    filename = f"{OUTPUT_DIR}/input_state_member_{member:03d}.pkl"
                    with open(filename, 'wb') as f:
                        pickle.dump(input_state, f)
                    print(f"  Saved to {filename}")
                    
                    # Upload to GCS if enabled
                    if UPLOAD_TO_GCS:
                        gcs_blob_name = f"{datestr}/input/input_state_member_{member:03d}.pkl"
                        upload_success = upload_to_gcs(filename, GCS_BUCKET, gcs_blob_name, GCS_SERVICE_ACCOUNT_KEY)
                        
                        # Clean up local file if upload successful and cleanup enabled
                        if upload_success and CLEANUP_LOCAL_FILES:
                            os.remove(filename)
                            print(f"    🗑️  Removed local file: {filename}")
            else:
                failed_members.append(member)
            
            # Track timing
            elapsed = time.time() - start_time
            processing_times.append(elapsed)
            
            # Estimate remaining time
            avg_time = sum(processing_times) / len(processing_times)
            remaining = (len(ENSEMBLE_MEMBERS) - (i + 1)) * avg_time
            print(f"\n  Time for this member: {elapsed:.2f} seconds")
            print(f"  Estimated time remaining: {remaining/60:.1f} minutes")
            
        except Exception as e:
            print(f"\n  ❌ Error processing member {member}: {str(e)}")
            failed_members.append(member)
            continue
    
    # Final summary
    print(f"\n{'='*60}")
    print("TEST SUMMARY")
    print(f"{'='*60}")
    print(f"Successfully created input states for {len(successful_members)}/{len(ENSEMBLE_MEMBERS)} members")
    
    if failed_members:
        print(f"\nFailed members: {failed_members}")
    
    if processing_times:
        print(f"\nTiming statistics:")
        print(f"  Total time: {sum(processing_times)/60:.1f} minutes")
        print(f"  Average time per member: {sum(processing_times)/len(processing_times):.2f} seconds")
        print(f"  Min time: {min(processing_times):.2f} seconds")
        print(f"  Max time: {max(processing_times):.2f} seconds")
    
    # Show a sample of field names from the last successful state
    if successful_members and 'input_state' in locals():
        print(f"\nSample fields from the input state:")
        field_names = sorted(input_state['fields'].keys())
        print(f"  Surface fields: {[f for f in field_names if '_' not in f]}")
        print(f"  Pressure level fields sample (first 5): {[f for f in field_names if '_' in f][:5]}")
        print(f"  Total number of fields: {len(field_names)}")


if __name__ == "__main__":
    main()


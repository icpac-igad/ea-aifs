#!/usr/bin/env python3
"""
Download GRIB files from Google Cloud Storage - Similar to test_ensemble_input_states.py pattern

This script downloads AIFS ensemble forecast GRIB files from GCS bucket ea_aifs_w1
for the specific time ranges:
- h432-504, h504-576, h576-648, h648-720, h720-792
For ensemble members 001-009.

Based on the pattern from test_ensemble_input_states.py
"""

import os
import time
from pathlib import Path
from google.cloud import storage

# GCS configuration - following test_ensemble_input_states.py pattern
GCS_BUCKET = "ea_aifs_w1"
GCS_PREFIX = "forecasts/20250814_a3/"
GCS_SERVICE_ACCOUNT_KEY = "coiled-data-e4drr_202505.json"

# Forecast configuration
ENSEMBLE_MEMBERS = list(range(1, 10))  # Members 1-9
FORECAST_DATE = "20250817"
FORECAST_TIME = "0600"

# Time periods - exact matches for available files
TIME_RANGES = [
    ("432", "504"),  # Days 18-21
    ("504", "576"),  # Days 21-24  
    ("576", "648"),  # Days 24-27
    ("648", "720"),  # Days 27-30
    ("720", "792")   # Days 30-33
]

# Local configuration
OUTPUT_DIR = "downloaded_grib_files"


def download_grib_from_gcs(bucket, blob_name, local_path, max_retries=3):
    """Download a GRIB file from GCS with retry logic."""
    for attempt in range(max_retries):
        try:
            print(f"    Downloading: {os.path.basename(local_path)}")
            
            blob = bucket.blob(blob_name)
            
            # Check if blob exists
            if not blob.exists():
                raise Exception(f"File not found in bucket: {blob_name}")
            
            # Get file size
            blob.reload()  # Refresh metadata
            total_size = blob.size or 0
            
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            
            # Download file
            start_time = time.time()
            blob.download_to_filename(local_path)
            download_time = time.time() - start_time
            
            # Verify file size
            actual_size = os.path.getsize(local_path)
            if total_size > 0 and actual_size != total_size:
                raise Exception(f"Size mismatch: expected {total_size}, got {actual_size}")
            
            size_mb = actual_size / (1024 * 1024)
            speed_mbps = size_mb / download_time if download_time > 0 else 0
            print(f"    ✅ Downloaded: {os.path.basename(local_path)} ({size_mb:.2f} MB in {download_time:.1f}s, {speed_mbps:.1f} MB/s)")
            return True, actual_size
            
        except Exception as e:
            print(f"    ✗ Attempt {attempt + 1} failed: {str(e)}")
            if attempt < max_retries - 1:
                print(f"    Retrying in 5 seconds...")
                time.sleep(5)
            else:
                print(f"    Failed to download after {max_retries} attempts")
                return False, 0
    
    return False, 0


def create_gcs_client(service_account_key):
    """Create GCS client with service account authentication - following test_ensemble_input_states.py pattern."""
    try:
        if not os.path.exists(service_account_key):
            raise FileNotFoundError(f"Service account file not found: {service_account_key}")
        
        # Initialize GCS client with service account key (same as test_ensemble_input_states.py)
        client = storage.Client.from_service_account_json(service_account_key)
        bucket = client.bucket(GCS_BUCKET)
        
        print(f"✅ Connected to GCS bucket: {GCS_BUCKET}")
        print(f"   Using service account: {service_account_key}")
        return client, bucket
        
    except Exception as e:
        print(f"❌ Failed to initialize GCS client: {str(e)}")
        raise


def generate_file_list():
    """Generate list of GRIB files to download."""
    file_list = []
    
    for member in ENSEMBLE_MEMBERS:
        for start_hour, end_hour in TIME_RANGES:
            filename = f"aifs_ens_forecast_{FORECAST_DATE}_{FORECAST_TIME}_member{member:03d}_h{start_hour}-{end_hour}.grib"
            blob_name = f"{GCS_PREFIX}{filename}"
            
            file_list.append({
                'blob_name': blob_name,
                'filename': filename,
                'member': member,
                'hours': f"{start_hour}-{end_hour}",
                'start_hour': int(start_hour),
                'end_hour': int(end_hour)
            })
    
    return file_list


def main():
    """Main function to download GRIB files - following test_ensemble_input_states.py pattern."""
    print(f"GRIB File Download from GCS")
    print(f"{'='*60}")
    print(f"Bucket: {GCS_BUCKET}")
    print(f"Prefix: {GCS_PREFIX}")
    print(f"Members: {min(ENSEMBLE_MEMBERS)}-{max(ENSEMBLE_MEMBERS)} ({len(ENSEMBLE_MEMBERS)} total)")
    print(f"Time ranges: {len(TIME_RANGES)} periods")
    print(f"Output directory: {OUTPUT_DIR}")
    
    # Check service account key
    if not os.path.exists(GCS_SERVICE_ACCOUNT_KEY):
        print(f"❌ GCS service account key not found: {GCS_SERVICE_ACCOUNT_KEY}")
        return 1
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Generate file list
    file_list = generate_file_list()
    print(f"\nTotal files to download: {len(file_list)}")
    
    # Initialize GCS client
    try:
        client, bucket = create_gcs_client(GCS_SERVICE_ACCOUNT_KEY)
    except Exception as e:
        print(f"Failed to initialize GCS client: {e}")
        return 1
    
    # Track progress
    successful_downloads = []
    failed_downloads = []
    total_size = 0
    download_times = []
    
    # Download files - following test_ensemble_input_states.py progress pattern
    for i, file_info in enumerate(file_list):
        print(f"\n{'='*60}")
        print(f"Processing file {i+1}/{len(file_list)}")
        print(f"Member {file_info['member']:03d}, hours {file_info['hours']}")
        print(f"{'='*60}")
        
        try:
            start_time = time.time()
            
            # Construct local path
            local_path = os.path.join(OUTPUT_DIR, file_info['filename'])
            
            # Skip if file already exists and is not empty
            if os.path.exists(local_path) and os.path.getsize(local_path) > 0:
                existing_size = os.path.getsize(local_path)
                print(f"    ✓ Already exists: {file_info['filename']} ({existing_size // (1024*1024)} MB)")
                successful_downloads.append(file_info)
                total_size += existing_size
                continue
            
            # Download the file
            success, file_size = download_grib_from_gcs(bucket, file_info['blob_name'], local_path)
            
            if success:
                successful_downloads.append(file_info)
                total_size += file_size
                
                # Track timing
                elapsed = time.time() - start_time
                download_times.append(elapsed)
                
                # Estimate remaining time (like test_ensemble_input_states.py)
                if download_times:
                    avg_time = sum(download_times) / len(download_times)
                    remaining = (len(file_list) - (i + 1)) * avg_time
                    print(f"  Time for this file: {elapsed:.2f} seconds")
                    print(f"  Estimated time remaining: {remaining/60:.1f} minutes")
            else:
                failed_downloads.append(file_info)
                
        except Exception as e:
            print(f"\n  ❌ Error downloading {file_info['filename']}: {str(e)}")
            failed_downloads.append(file_info)
            continue
    
    # Final summary - following test_ensemble_input_states.py summary pattern
    print(f"\n{'='*60}")
    print("DOWNLOAD SUMMARY")
    print(f"{'='*60}")
    print(f"Successfully downloaded {len(successful_downloads)}/{len(file_list)} files")
    
    if failed_downloads:
        print(f"\nFailed downloads:")
        for failed in failed_downloads:
            print(f"  {failed['filename']}")
    
    if download_times:
        print(f"\nTiming statistics:")
        print(f"  Total download time: {sum(download_times)/60:.1f} minutes")
        print(f"  Average time per file: {sum(download_times)/len(download_times):.2f} seconds")
        print(f"  Min time: {min(download_times):.2f} seconds")
        print(f"  Max time: {max(download_times):.2f} seconds")
    
    print(f"\nStorage statistics:")
    total_size_gb = total_size / (1024 * 1024 * 1024)
    print(f"  Total size: {total_size // (1024*1024)} MB ({total_size_gb:.2f} GB)")
    print(f"  Files saved to: {OUTPUT_DIR}/")
    
    # Show file breakdown by member
    print(f"\nFiles per member:")
    for member in ENSEMBLE_MEMBERS:
        member_files = [f for f in successful_downloads if f['member'] == member]
        print(f"  Member {member:03d}: {len(member_files)}/{len(TIME_RANGES)} files")
    
    # Show sample of downloaded files
    if successful_downloads:
        print(f"\nSample downloaded files:")
        for file_info in successful_downloads[:5]:
            local_path = os.path.join(OUTPUT_DIR, file_info['filename'])
            if os.path.exists(local_path):
                size_mb = os.path.getsize(local_path) / (1024 * 1024)
                print(f"  {file_info['filename']} ({size_mb:.2f} MB)")
        if len(successful_downloads) > 5:
            print(f"  ... and {len(successful_downloads) - 5} more files")
    
    return 0 if len(failed_downloads) == 0 else 1


if __name__ == "__main__":
    exit_code = main()
    exit(exit_code)
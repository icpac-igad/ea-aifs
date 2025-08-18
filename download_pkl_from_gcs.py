#!/usr/bin/env python3
"""
Download Pickle Files from GCS - GPU Notebook Utility

This script downloads pre-processed input state pickle files from Google Cloud Storage
for use in GPU-based ensemble forecasting. This avoids the time-consuming data
retrieval and preprocessing steps on expensive GPU instances.

Usage:
    python download_pkl_from_gcs.py --date 20250105_0000 --members 1-50 --output-dir ./input_states
"""

import os
import argparse
import pickle
from pathlib import Path
from google.cloud import storage


# GCS configuration
GCS_BUCKET = "hrrr_delete"
GCS_SERVICE_ACCOUNT_KEY = "coiled-data-e4drr_202505.json"


def download_from_gcs(gcs_bucket, gcs_blob_name, local_file_path, service_account_key):
    """Download a file from Google Cloud Storage."""
    try:
        # Initialize GCS client with service account key
        client = storage.Client.from_service_account_json(service_account_key)
        bucket = client.bucket(gcs_bucket)
        blob = bucket.blob(gcs_blob_name)
        
        # Check if blob exists
        if not blob.exists():
            print(f"    ❌ File not found in GCS: gs://{gcs_bucket}/{gcs_blob_name}")
            return False
        
        # Create local directory if it doesn't exist
        os.makedirs(os.path.dirname(local_file_path), exist_ok=True)
        
        # Download the file
        blob.download_to_filename(local_file_path)
        print(f"    ✅ Downloaded gs://{gcs_bucket}/{gcs_blob_name} -> {local_file_path}")
        return True
    except Exception as e:
        print(f"    ❌ Failed to download from GCS: {str(e)}")
        return False


def parse_member_range(member_str):
    """Parse member range string like '1-50' or '1,2,3' into list of integers."""
    members = []
    if '-' in member_str:
        start, end = map(int, member_str.split('-'))
        members = list(range(start, end + 1))
    elif ',' in member_str:
        members = [int(m.strip()) for m in member_str.split(',')]
    else:
        members = [int(member_str)]
    return members


def verify_pickle_file(file_path):
    """Verify that the pickle file contains valid input state data."""
    try:
        with open(file_path, 'rb') as f:
            input_state = pickle.load(f)
        
        # Check structure
        if not isinstance(input_state, dict):
            return False, "Not a dictionary"
        
        if 'date' not in input_state or 'fields' not in input_state:
            return False, "Missing 'date' or 'fields' key"
        
        fields = input_state['fields']
        if not isinstance(fields, dict) or len(fields) == 0:
            return False, "Invalid or empty fields"
        
        # Check field count (should be around 90+ fields)
        field_count = len(fields)
        if field_count < 80:
            return False, f"Too few fields: {field_count}"
        
        return True, f"Valid input state with {field_count} fields"
    
    except Exception as e:
        return False, f"Error reading file: {str(e)}"


def main():
    parser = argparse.ArgumentParser(description='Download pickle files from GCS for ensemble forecasting')
    parser.add_argument('--date', required=True, help='Date string (e.g., 20250105_0000)')
    parser.add_argument('--members', default='1-50', help='Member range (e.g., 1-50, 1,2,3)')
    parser.add_argument('--output-dir', default='./input_states', help='Local output directory')
    parser.add_argument('--bucket', default=GCS_BUCKET, help='GCS bucket name')
    parser.add_argument('--service-account', default=GCS_SERVICE_ACCOUNT_KEY, help='Service account JSON file')
    parser.add_argument('--verify', action='store_true', help='Verify downloaded pickle files')
    
    args = parser.parse_args()
    
    # Parse members
    try:
        members = parse_member_range(args.members)
        print(f"Will download files for {len(members)} ensemble members: {members}")
    except ValueError as e:
        print(f"❌ Error parsing member range: {e}")
        return 1
    
    # Check service account key
    if not os.path.exists(args.service_account):
        print(f"❌ Service account key not found: {args.service_account}")
        return 1
    
    print(f"Using service account: {args.service_account}")
    print(f"Target bucket: {args.bucket}")
    print(f"Date folder: {args.date}")
    print(f"Local output directory: {args.output_dir}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Download files
    successful_downloads = []
    failed_downloads = []
    
    for member in members:
        print(f"\nDownloading ensemble member {member}...")
        
        # Define paths
        gcs_blob_name = f"{args.date}/input_state_member_{member:03d}.pkl"
        local_file_path = os.path.join(args.output_dir, f"input_state_member_{member:03d}.pkl")
        
        # Skip if file already exists
        if os.path.exists(local_file_path):
            print(f"    File already exists: {local_file_path}")
            if args.verify:
                is_valid, msg = verify_pickle_file(local_file_path)
                if is_valid:
                    print(f"    ✅ Verification passed: {msg}")
                    successful_downloads.append(member)
                else:
                    print(f"    ❌ Verification failed: {msg}")
                    failed_downloads.append(member)
                    # Remove corrupted file
                    os.remove(local_file_path)
                    print(f"    Removed corrupted file, will re-download...")
                    # Continue to download
                    success = download_from_gcs(args.bucket, gcs_blob_name, local_file_path, args.service_account)
                    if success:
                        successful_downloads.append(member)
                    else:
                        failed_downloads.append(member)
            else:
                successful_downloads.append(member)
            continue
        
        # Download from GCS
        success = download_from_gcs(args.bucket, gcs_blob_name, local_file_path, args.service_account)
        
        if success:
            # Verify file if requested
            if args.verify:
                is_valid, msg = verify_pickle_file(local_file_path)
                if is_valid:
                    print(f"    ✅ Verification passed: {msg}")
                    successful_downloads.append(member)
                else:
                    print(f"    ❌ Verification failed: {msg}")
                    failed_downloads.append(member)
                    # Remove corrupted file
                    os.remove(local_file_path)
            else:
                successful_downloads.append(member)
        else:
            failed_downloads.append(member)
    
    # Summary
    print(f"\n{'='*60}")
    print("DOWNLOAD SUMMARY")
    print(f"{'='*60}")
    print(f"Successfully downloaded: {len(successful_downloads)}/{len(members)} files")
    
    if failed_downloads:
        print(f"Failed downloads: {failed_downloads}")
    
    if successful_downloads:
        print(f"\nFiles are ready in: {args.output_dir}/")
        print(f"You can now use these files in your GPU ensemble forecasting script.")
        
        # Show disk usage
        total_size = 0
        for member in successful_downloads:
            filepath = os.path.join(args.output_dir, f"input_state_member_{member:03d}.pkl")
            if os.path.exists(filepath):
                total_size += os.path.getsize(filepath)
        
        print(f"Total disk usage: {total_size / (1024*1024):.2f} MB ({total_size / (1024*1024*1024):.2f} GB)")
    
    return 0 if len(failed_downloads) == 0 else 1


if __name__ == "__main__":
    exit(main())
#!/usr/bin/env python3
"""
Download GRIB files for specific forecast periods from Google Cloud Storage bucket.

This script downloads AIFS ensemble forecast GRIB files for:
- Days 19-25 (hours 456-600) 
- Days 26-32 (hours 624-768)
For ensemble members 001-009.

Supports authentication via service account JSON file.
"""

import os
import sys
import argparse
from pathlib import Path
import time

try:
    from google.cloud import storage
    from google.oauth2 import service_account
    GCS_AVAILABLE = True
except ImportError:
    GCS_AVAILABLE = False

# Always import requests for fallback HTTP downloads
import requests
from urllib.parse import urljoin

# GCS bucket configuration
BUCKET_NAME = "ea_aifs_w1"
BUCKET_PREFIX = "forecasts/20250814_a3/"

# Forecast configuration
ENSEMBLE_MEMBERS = [f"{i:03d}" for i in range(1, 10)]  # 001-009
FORECAST_DATE = "20250817"
FORECAST_TIME = "0600"

# Time periods to download - matching actual available files
TIME_PERIODS = {
    "days_19_25": [
        ("432", "504"),  # Days 18-21
        ("504", "576"),  # Days 21-24  
        ("576", "648")   # Days 24-27
    ],
    "days_26_32": [
        ("648", "720"),  # Days 27-30
        ("720", "792")   # Days 30-33
    ]
}


def download_file_gcs(bucket, blob_name, local_path, max_retries=3):
    """Download a file from GCS with retry logic and progress tracking."""
    for attempt in range(max_retries):
        try:
            print(f"  Downloading: {os.path.basename(local_path)}")
            
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
            blob.download_to_filename(local_path)
            
            # Verify file size
            actual_size = os.path.getsize(local_path)
            if total_size > 0 and actual_size != total_size:
                raise Exception(f"Size mismatch: expected {total_size}, got {actual_size}")
            
            print(f"  ✓ Downloaded: {os.path.basename(local_path)} ({actual_size // (1024*1024)} MB)")
            return True
            
        except Exception as e:
            print(f"  ✗ Attempt {attempt + 1} failed: {str(e)}")
            if attempt < max_retries - 1:
                print(f"  Retrying in 5 seconds...")
                time.sleep(5)
            else:
                print(f"  Failed to download after {max_retries} attempts")
                return False
    
    return False


def download_file_http(url, local_path, max_retries=3):
    """Download a file via HTTP with retry logic and progress tracking."""
    for attempt in range(max_retries):
        try:
            print(f"  Downloading: {os.path.basename(local_path)}")
            
            response = requests.get(url, stream=True, timeout=300)
            response.raise_for_status()
            
            # Get file size if available
            total_size = int(response.headers.get('content-length', 0))
            
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            
            # Download with progress
            downloaded = 0
            with open(local_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        
                        # Show progress every 10MB
                        if total_size > 0 and downloaded % (10 * 1024 * 1024) == 0:
                            progress = (downloaded / total_size) * 100
                            print(f"    Progress: {progress:.1f}% ({downloaded // (1024*1024)} MB)")
            
            # Verify file size
            actual_size = os.path.getsize(local_path)
            if total_size > 0 and actual_size != total_size:
                raise Exception(f"Size mismatch: expected {total_size}, got {actual_size}")
            
            print(f"  ✓ Downloaded: {os.path.basename(local_path)} ({actual_size // (1024*1024)} MB)")
            return True
            
        except Exception as e:
            print(f"  ✗ Attempt {attempt + 1} failed: {str(e)}")
            if attempt < max_retries - 1:
                print(f"  Retrying in 5 seconds...")
                time.sleep(5)
            else:
                print(f"  Failed to download after {max_retries} attempts")
                return False
    
    return False


def create_gcs_client(service_account_path=None):
    """Create GCS client with optional service account authentication."""
    if not GCS_AVAILABLE:
        raise RuntimeError("Google Cloud Storage library not available. Install with: pip install google-cloud-storage")
    
    if service_account_path:
        if not os.path.exists(service_account_path):
            raise FileNotFoundError(f"Service account file not found: {service_account_path}")
        
        credentials = service_account.Credentials.from_service_account_file(service_account_path)
        client = storage.Client(credentials=credentials)
        print(f"  Using service account: {service_account_path}")
    else:
        # Use default credentials (environment variables, metadata server, etc.)
        client = storage.Client()
        print(f"  Using default credentials")
    
    return client


def generate_download_list(periods_to_download=None):
    """Generate list of files to download."""
    if periods_to_download is None:
        periods_to_download = ["days_19_25", "days_26_32"]
    
    download_list = []
    
    for period_name in periods_to_download:
        if period_name not in TIME_PERIODS:
            print(f"Warning: Unknown period '{period_name}', skipping")
            continue
            
        time_ranges = TIME_PERIODS[period_name]
        
        for member in ENSEMBLE_MEMBERS:
            for start_hour, end_hour in time_ranges:
                filename = f"aifs_ens_forecast_{FORECAST_DATE}_{FORECAST_TIME}_member{member}_h{start_hour}-{end_hour}.grib"
                
                download_list.append({
                    'blob_name': f"{BUCKET_PREFIX}{filename}",
                    'filename': filename,
                    'member': member,
                    'period': period_name,
                    'hours': f"{start_hour}-{end_hour}"
                })
    
    return download_list


def main():
    parser = argparse.ArgumentParser(description='Download AIFS ensemble forecast GRIB files')
    parser.add_argument('--output-dir', '-o', default='./downloaded_grib_files',
                        help='Output directory for downloaded files (default: ./downloaded_grib_files)')
    parser.add_argument('--periods', nargs='+', choices=['days_19_25', 'days_26_32'],
                        default=['days_19_25', 'days_26_32'],
                        help='Time periods to download (default: both)')
    parser.add_argument('--members', nargs='+', type=int, choices=range(1, 10),
                        help='Specific ensemble members to download (1-9, default: all)')
    parser.add_argument('--dry-run', action='store_true',
                        help='Show what would be downloaded without actually downloading')
    parser.add_argument('--service-account', '-s', type=str,
                        help='Path to service account JSON file for GCS authentication')
    parser.add_argument('--use-http', action='store_true',
                        help='Use HTTP downloads instead of GCS client (fallback mode)')
    
    args = parser.parse_args()
    
    # Filter members if specified
    global ENSEMBLE_MEMBERS
    if args.members:
        ENSEMBLE_MEMBERS = [f"{m:03d}" for m in args.members]
    
    # Generate download list
    download_list = generate_download_list(args.periods)
    
    if not download_list:
        print("No files to download!")
        return 1
    
    # Determine download method
    use_gcs = GCS_AVAILABLE and not args.use_http
    gcs_client = None
    bucket = None
    
    if use_gcs:
        try:
            gcs_client = create_gcs_client(args.service_account)
            bucket = gcs_client.bucket(BUCKET_NAME)
            print(f"  Connected to GCS bucket: {BUCKET_NAME}")
        except Exception as e:
            print(f"  Failed to initialize GCS client: {e}")
            print(f"  Falling back to HTTP downloads...")
            use_gcs = False
    
    # Show summary
    print(f"\nDownload Plan:")
    if use_gcs:
        print(f"  Method: Google Cloud Storage client")
        print(f"  Bucket: {BUCKET_NAME}")
        print(f"  Prefix: {BUCKET_PREFIX}")
    else:
        print(f"  Method: HTTP downloads")
        print(f"  Base URL: https://storage.googleapis.com/{BUCKET_NAME}/{BUCKET_PREFIX}")
    print(f"  Output directory: {args.output_dir}")
    print(f"  Periods: {', '.join(args.periods)}")
    print(f"  Members: {', '.join(ENSEMBLE_MEMBERS)}")
    print(f"  Total files: {len(download_list)}")
    
    if args.dry_run:
        print(f"\nFiles to download (dry run):")
        for item in download_list:
            if use_gcs:
                print(f"  gs://{BUCKET_NAME}/{item['blob_name']} -> {args.output_dir}/{item['period']}/{item['filename']}")
            else:
                url = f"https://storage.googleapis.com/{BUCKET_NAME}/{item['blob_name']}"
                print(f"  {url} -> {args.output_dir}/{item['period']}/{item['filename']}")
        return 0
    
    # Create output directory
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Download files
    print(f"\nStarting downloads...")
    successful_downloads = 0
    failed_downloads = 0
    total_size = 0
    
    for i, item in enumerate(download_list, 1):
        print(f"\n[{i}/{len(download_list)}] Member {item['member']}, {item['hours']} hours:")
        
        # Create period subdirectory
        period_dir = output_path / item['period']
        period_dir.mkdir(exist_ok=True)
        
        local_path = period_dir / item['filename']
        
        # Skip if file already exists and is not empty
        if local_path.exists() and local_path.stat().st_size > 0:
            print(f"  ✓ Already exists: {item['filename']} ({local_path.stat().st_size // (1024*1024)} MB)")
            successful_downloads += 1
            total_size += local_path.stat().st_size
            continue
        
        # Download the file
        if use_gcs:
            success = download_file_gcs(bucket, item['blob_name'], str(local_path))
        else:
            url = f"https://storage.googleapis.com/{BUCKET_NAME}/{item['blob_name']}"
            success = download_file_http(url, str(local_path))
        
        if success:
            successful_downloads += 1
            total_size += local_path.stat().st_size
        else:
            failed_downloads += 1
    
    # Summary
    print(f"\n{'='*60}")
    print(f"Download Summary:")
    print(f"  Successful: {successful_downloads}/{len(download_list)}")
    print(f"  Failed: {failed_downloads}")
    print(f"  Total size: {total_size // (1024*1024)} MB ({total_size / (1024*1024*1024):.2f} GB)")
    print(f"  Files saved to: {args.output_dir}")
    
    # List downloaded files by period
    for period in args.periods:
        period_path = output_path / period
        if period_path.exists():
            files = list(period_path.glob("*.grib"))
            print(f"\n{period.replace('_', ' ').title()}: {len(files)} files")
            for file in sorted(files)[:5]:  # Show first 5 files
                size_mb = file.stat().st_size // (1024*1024)
                print(f"  {file.name} ({size_mb} MB)")
            if len(files) > 5:
                print(f"  ... and {len(files) - 5} more files")
    
    return 0 if failed_downloads == 0 else 1


if __name__ == "__main__":
    sys.exit(main())

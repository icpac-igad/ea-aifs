#!/usr/bin/env python3
"""
GCS Upload for GRIB Files - Upload ensemble GRIB files to Google Cloud Storage

This script uploads generated GRIB files to GCS and optionally removes local files
to manage disk space during long forecasts.
"""

import os
import sys
import time
import argparse
from pathlib import Path
from typing import Optional, List
from google.cloud import storage
from google.cloud.exceptions import GoogleCloudError


class GCSGribUploader:
    """Handles uploading GRIB files to Google Cloud Storage."""
    
    def __init__(self, bucket_name: str, project_id: Optional[str] = None):
        """
        Initialize the GCS uploader.
        
        Args:
            bucket_name: Name of the GCS bucket
            project_id: GCP project ID (optional, uses default from environment)
        """
        self.bucket_name = bucket_name
        self.project_id = project_id
        
        try:
            if project_id:
                self.client = storage.Client(project=project_id)
            else:
                self.client = storage.Client()
            self.bucket = self.client.bucket(bucket_name)
        except GoogleCloudError as e:
            raise RuntimeError(f"Failed to initialize GCS client: {e}")
    
    def upload_file(self, local_path: str, gcs_path: str, remove_local: bool = False) -> bool:
        """
        Upload a single file to GCS.
        
        Args:
            local_path: Path to local file
            gcs_path: Destination path in GCS bucket
            remove_local: Whether to remove local file after successful upload
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            if not os.path.exists(local_path):
                print(f"❌ Local file not found: {local_path}")
                return False
            
            # Get file size for progress reporting
            file_size = os.path.getsize(local_path) / (1024 * 1024)  # MB
            print(f"📤 Uploading {os.path.basename(local_path)} ({file_size:.2f} MB)...")
            
            start_time = time.time()
            
            # Upload the file
            blob = self.bucket.blob(gcs_path)
            blob.upload_from_filename(local_path)
            
            upload_time = time.time() - start_time
            upload_speed = file_size / upload_time if upload_time > 0 else 0
            
            print(f"✅ Uploaded to gs://{self.bucket_name}/{gcs_path}")
            print(f"   Time: {upload_time:.2f}s, Speed: {upload_speed:.2f} MB/s")
            
            # Remove local file if requested
            if remove_local:
                try:
                    os.remove(local_path)
                    print(f"🗑️  Removed local file: {local_path}")
                except OSError as e:
                    print(f"⚠️  Failed to remove local file: {e}")
            
            return True
            
        except GoogleCloudError as e:
            print(f"❌ Upload failed for {local_path}: {e}")
            return False
        except Exception as e:
            print(f"❌ Unexpected error uploading {local_path}: {e}")
            return False
    
    def upload_directory(self, local_dir: str, gcs_prefix: str = "", 
                        pattern: str = "*.grib", remove_local: bool = False) -> tuple:
        """
        Upload all files matching pattern from a directory to GCS.
        
        Args:
            local_dir: Local directory path
            gcs_prefix: Prefix for GCS paths (e.g., 'forecasts/2024/')
            pattern: File pattern to match (default: '*.grib')
            remove_local: Whether to remove local files after upload
            
        Returns:
            tuple: (successful_uploads, failed_uploads, total_size_mb)
        """
        local_path = Path(local_dir)
        
        if not local_path.exists():
            print(f"❌ Directory not found: {local_dir}")
            return 0, 0, 0
        
        # Find matching files
        files = list(local_path.glob(pattern))
        if not files:
            print(f"❌ No files found matching pattern '{pattern}' in {local_dir}")
            return 0, 0, 0
        
        print(f"📂 Found {len(files)} files to upload")
        
        successful = 0
        failed = 0
        total_size = 0
        
        for file_path in files:
            # Calculate GCS path
            if gcs_prefix:
                gcs_path = f"{gcs_prefix.rstrip('/')}/{file_path.name}"
            else:
                gcs_path = file_path.name
            
            # Track file size
            file_size = file_path.stat().st_size / (1024 * 1024)  # MB
            total_size += file_size
            
            # Upload file
            if self.upload_file(str(file_path), gcs_path, remove_local):
                successful += 1
            else:
                failed += 1
        
        print(f"\n📊 Upload Summary:")
        print(f"   ✅ Successful: {successful}/{len(files)}")
        print(f"   ❌ Failed: {failed}/{len(files)}")
        print(f"   📦 Total size: {total_size:.2f} MB ({total_size/1024:.2f} GB)")
        
        return successful, failed, total_size
    
    def list_bucket_contents(self, prefix: str = "") -> List[str]:
        """List contents of the GCS bucket with optional prefix filter."""
        try:
            blobs = self.client.list_blobs(self.bucket, prefix=prefix)
            return [blob.name for blob in blobs]
        except GoogleCloudError as e:
            print(f"❌ Failed to list bucket contents: {e}")
            return []
    
    def check_bucket_exists(self) -> bool:
        """Check if the specified bucket exists and is accessible."""
        try:
            self.bucket.reload()
            return True
        except GoogleCloudError:
            return False


def main():
    """Main function to handle command line usage."""
    parser = argparse.ArgumentParser(
        description="Upload GRIB files to Google Cloud Storage",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Upload all GRIB files from ensemble_outputs/
  python gcs_upload_grib.py --bucket my-forecasts --directory ensemble_outputs/
  
  # Upload with date prefix and remove local files
  python gcs_upload_grib.py --bucket my-forecasts --directory ensemble_outputs/ \\
    --prefix forecasts/20240813/ --remove-local
  
  # Upload single file
  python gcs_upload_grib.py --bucket my-forecasts \\
    --file ensemble_outputs/forecast_001.grib --gcs-path forecasts/latest.grib
        """
    )
    
    parser.add_argument("--bucket", required=True,
                       help="GCS bucket name")
    parser.add_argument("--project", 
                       help="GCP project ID (uses default if not specified)")
    
    # File/directory options
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--file", 
                      help="Upload single file")
    group.add_argument("--directory", 
                      help="Upload all GRIB files from directory")
    
    parser.add_argument("--gcs-path",
                       help="Destination path in GCS (for single file upload)")
    parser.add_argument("--prefix", default="",
                       help="Prefix for GCS paths (for directory upload)")
    parser.add_argument("--pattern", default="*.grib",
                       help="File pattern to match (default: *.grib)")
    parser.add_argument("--remove-local", action="store_true",
                       help="Remove local files after successful upload")
    
    args = parser.parse_args()
    
    try:
        # Initialize uploader
        print(f"🔧 Initializing GCS uploader for bucket: {args.bucket}")
        uploader = GCSGribUploader(args.bucket, args.project)
        
        # Check bucket accessibility
        if not uploader.check_bucket_exists():
            print(f"❌ Bucket '{args.bucket}' not found or not accessible")
            print("   Make sure the bucket exists and you have proper permissions")
            return 1
        
        print("✅ GCS connection established")
        
        if args.file:
            # Single file upload
            gcs_path = args.gcs_path or os.path.basename(args.file)
            success = uploader.upload_file(args.file, gcs_path, args.remove_local)
            return 0 if success else 1
        
        else:
            # Directory upload
            successful, failed, total_size = uploader.upload_directory(
                args.directory, args.prefix, args.pattern, args.remove_local
            )
            
            if failed > 0:
                print(f"\n⚠️  {failed} uploads failed")
                return 1
            else:
                print(f"\n🎉 All {successful} files uploaded successfully!")
                return 0
                
    except Exception as e:
        print(f"❌ Fatal error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())

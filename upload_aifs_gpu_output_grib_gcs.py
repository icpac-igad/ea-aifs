#!/usr/bin/env python3
"""
GCS Upload for GRIB Files - Multi-threaded version for faster uploads
Optimized for a2-ultragpu-1g instances with concurrent uploads

This script uploads generated GRIB files to GCS using multiple threads
to significantly reduce upload time for large files.

python v2upload.py --bucket ea_aifs_w1 --directory /scratch/ensemble_outputs/ \ 
                   --prefix forecasts/20250814_a3/ --threads 15
"""

import os
import sys
import time
import argparse
import threading
from pathlib import Path
from typing import Optional, List, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from queue import Queue
from google.cloud import storage
from google.cloud.exceptions import GoogleCloudError


class GCSGribUploaderMultiThreaded:
    """Handles uploading GRIB files to Google Cloud Storage with multi-threading support."""
    
    def __init__(self, bucket_name: str, project_id: Optional[str] = None, max_workers: int = 10):
        """
        Initialize the GCS uploader with threading support.
        
        Args:
            bucket_name: Name of the GCS bucket
            project_id: GCP project ID (optional, uses default from environment)
            max_workers: Maximum number of concurrent upload threads
        """
        self.bucket_name = bucket_name
        self.project_id = project_id
        self.max_workers = max_workers
        
        try:
            if project_id:
                self.client = storage.Client(project=project_id)
            else:
                self.client = storage.Client()
            self.bucket = self.client.bucket(bucket_name)
        except GoogleCloudError as e:
            raise RuntimeError(f"Failed to initialize GCS client: {e}")
        
        # Thread-safe counters
        self.upload_lock = threading.Lock()
        self.successful_uploads = 0
        self.failed_uploads = 0
        self.total_bytes_uploaded = 0
        self.start_time = None
    
    def _get_thread_safe_client(self) -> storage.Client:
        """Create a new client for thread safety."""
        if self.project_id:
            return storage.Client(project=self.project_id)
        return storage.Client()
    
    def upload_file_thread_safe(self, local_path: str, gcs_path: str, 
                               remove_local: bool = False, thread_id: int = 0) -> Tuple[bool, float]:
        """
        Upload a single file to GCS (thread-safe version).
        
        Args:
            local_path: Path to local file
            gcs_path: Destination path in GCS bucket
            remove_local: Whether to remove local file after successful upload
            thread_id: Thread identifier for logging
            
        Returns:
            tuple: (success: bool, file_size_mb: float)
        """
        try:
            if not os.path.exists(local_path):
                print(f"[Thread {thread_id}] ❌ Local file not found: {local_path}")
                return False, 0
            
            # Get file size
            file_size = os.path.getsize(local_path) / (1024 * 1024)  # MB
            file_name = os.path.basename(local_path)
            
            print(f"[Thread {thread_id}] 📤 Starting upload: {file_name} ({file_size:.2f} MB)")
            
            start_time = time.time()
            
            # Use thread-local client for safety
            client = self._get_thread_safe_client()
            bucket = client.bucket(self.bucket_name)
            blob = bucket.blob(gcs_path)
            
            # Configure for optimal upload performance
            blob.chunk_size = 32 * 1024 * 1024  # 32MB chunks for large files
            
            # Upload the file
            blob.upload_from_filename(local_path)
            
            upload_time = time.time() - start_time
            upload_speed = file_size / upload_time if upload_time > 0 else 0
            
            print(f"[Thread {thread_id}] ✅ Completed: {file_name}")
            print(f"[Thread {thread_id}]    Time: {upload_time:.2f}s, Speed: {upload_speed:.2f} MB/s")
            
            # Update thread-safe counters
            with self.upload_lock:
                self.successful_uploads += 1
                self.total_bytes_uploaded += file_size
            
            # Remove local file if requested
            if remove_local:
                try:
                    os.remove(local_path)
                    print(f"[Thread {thread_id}] 🗑️  Removed local file: {local_path}")
                except OSError as e:
                    print(f"[Thread {thread_id}] ⚠️  Failed to remove local file: {e}")
            
            return True, file_size
            
        except GoogleCloudError as e:
            print(f"[Thread {thread_id}] ❌ Upload failed for {local_path}: {e}")
            with self.upload_lock:
                self.failed_uploads += 1
            return False, 0
        except Exception as e:
            print(f"[Thread {thread_id}] ❌ Unexpected error uploading {local_path}: {e}")
            with self.upload_lock:
                self.failed_uploads += 1
            return False, 0
    
    def upload_files_parallel(self, file_list: List[Tuple[str, str]], 
                            remove_local: bool = False) -> Tuple[int, int, float]:
        """
        Upload multiple files in parallel using thread pool.
        
        Args:
            file_list: List of (local_path, gcs_path) tuples
            remove_local: Whether to remove local files after upload
            
        Returns:
            tuple: (successful_uploads, failed_uploads, total_size_mb)
        """
        if not file_list:
            print("❌ No files to upload")
            return 0, 0, 0
        
        print(f"\n🚀 Starting parallel upload with {self.max_workers} threads")
        print(f"📦 Total files to upload: {len(file_list)}")
        
        # Reset counters
        self.successful_uploads = 0
        self.failed_uploads = 0
        self.total_bytes_uploaded = 0
        self.start_time = time.time()
        
        # Create thread pool and submit all tasks
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all upload tasks
            futures = {}
            for idx, (local_path, gcs_path) in enumerate(file_list):
                future = executor.submit(
                    self.upload_file_thread_safe, 
                    local_path, 
                    gcs_path, 
                    remove_local,
                    idx % self.max_workers  # Assign thread ID
                )
                futures[future] = (local_path, gcs_path)
            
            # Monitor progress
            completed = 0
            for future in as_completed(futures):
                completed += 1
                local_path, gcs_path = futures[future]
                
                # Progress update
                elapsed = time.time() - self.start_time
                if elapsed > 0:
                    avg_speed = self.total_bytes_uploaded / elapsed
                    remaining = len(file_list) - completed
                    eta = (remaining * (elapsed / completed)) if completed > 0 else 0
                    
                    print(f"\n📊 Progress: {completed}/{len(file_list)} files")
                    print(f"   ⏱️  Elapsed: {elapsed:.1f}s, ETA: {eta:.1f}s")
                    print(f"   📈 Avg speed: {avg_speed:.2f} MB/s")
                    print(f"   ✅ Success: {self.successful_uploads}, ❌ Failed: {self.failed_uploads}")
        
        total_time = time.time() - self.start_time
        
        print(f"\n🏁 Upload Complete!")
        print(f"   ⏱️  Total time: {total_time:.2f}s")
        print(f"   ✅ Successful: {self.successful_uploads}/{len(file_list)}")
        print(f"   ❌ Failed: {self.failed_uploads}/{len(file_list)}")
        print(f"   📦 Total uploaded: {self.total_bytes_uploaded:.2f} MB ({self.total_bytes_uploaded/1024:.2f} GB)")
        if total_time > 0:
            print(f"   📈 Overall speed: {self.total_bytes_uploaded/total_time:.2f} MB/s")
        
        return self.successful_uploads, self.failed_uploads, self.total_bytes_uploaded
    
    def upload_directory(self, local_dir: str, gcs_prefix: str = "", 
                        pattern: str = "*.grib", remove_local: bool = False,
                        batch_size: Optional[int] = None) -> Tuple[int, int, float]:
        """
        Upload all files matching pattern from a directory to GCS using parallel uploads.
        
        Args:
            local_dir: Local directory path
            gcs_prefix: Prefix for GCS paths (e.g., 'forecasts/2024/')
            pattern: File pattern to match (default: '*.grib')
            remove_local: Whether to remove local files after upload
            batch_size: Optional batch size for processing (None = all at once)
            
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
        
        # Calculate total size
        total_size = sum(f.stat().st_size for f in files) / (1024 * 1024)  # MB
        print(f"📦 Total size: {total_size:.2f} MB ({total_size/1024:.2f} GB)")
        
        # Prepare file list with GCS paths
        file_list = []
        for file_path in files:
            if gcs_prefix:
                gcs_path = f"{gcs_prefix.rstrip('/')}/{file_path.name}"
            else:
                gcs_path = file_path.name
            file_list.append((str(file_path), gcs_path))
        
        # Upload files in batches if specified
        if batch_size and batch_size < len(file_list):
            total_successful = 0
            total_failed = 0
            total_uploaded = 0
            
            for i in range(0, len(file_list), batch_size):
                batch = file_list[i:i + batch_size]
                print(f"\n📦 Processing batch {i//batch_size + 1}/{(len(file_list) + batch_size - 1)//batch_size}")
                successful, failed, uploaded = self.upload_files_parallel(batch, remove_local)
                total_successful += successful
                total_failed += failed
                total_uploaded += uploaded
            
            return total_successful, total_failed, total_uploaded
        else:
            return self.upload_files_parallel(file_list, remove_local)
    
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


def optimize_thread_count(file_count: int, file_size_mb: float) -> int:
    """
    Determine optimal thread count based on file characteristics.
    
    For a2-ultragpu-1g instance with high bandwidth:
    - Start with 10 threads as baseline
    - Adjust based on file size and count
    """
    # Base thread count
    base_threads = 10
    
    # For very large files (>1GB), use fewer threads to avoid memory issues
    if file_size_mb > 1024:
        return min(5, file_count)
    
    # For medium files (100MB-1GB), use moderate threading
    elif file_size_mb > 100:
        return min(10, file_count)
    
    # For smaller files, can use more threads
    else:
        return min(20, file_count)


def main():
    """Main function to handle command line usage."""
    parser = argparse.ArgumentParser(
        description="Upload GRIB files to Google Cloud Storage with multi-threading support",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Upload all GRIB files with default 10 threads
  python continat_v20250817.py --bucket my-forecasts --directory ensemble_outputs/
  
  # Upload with custom thread count (e.g., 20 threads for many small files)
  python continat_v20250817.py --bucket my-forecasts --directory ensemble_outputs/ --threads 20
  
  # Upload with date prefix and remove local files
  python continat_v20250817.py --bucket my-forecasts --directory ensemble_outputs/ \\
    --prefix forecasts/20240813/ --remove-local --threads 15
  
  # Upload in batches (useful for very large datasets)
  python continat_v20250817.py --bucket my-forecasts --directory ensemble_outputs/ \\
    --batch-size 50 --threads 10
        """
    )
    
    parser.add_argument("--bucket", required=True,
                       help="GCS bucket name")
    parser.add_argument("--project", 
                       help="GCP project ID (uses default if not specified)")
    parser.add_argument("--threads", type=int, default=10,
                       help="Number of parallel upload threads (default: 10, recommended: 5-20)")
    
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
    parser.add_argument("--batch-size", type=int,
                       help="Process files in batches of this size (optional)")
    parser.add_argument("--auto-optimize", action="store_true",
                       help="Automatically optimize thread count based on file characteristics")
    
    args = parser.parse_args()
    
    try:
        # Auto-optimize thread count if requested
        thread_count = args.threads
        if args.auto_optimize and args.directory:
            # Sample first file to estimate size
            local_path = Path(args.directory)
            files = list(local_path.glob(args.pattern))
            if files:
                avg_size = sum(f.stat().st_size for f in files[:5]) / len(files[:5]) / (1024 * 1024)
                thread_count = optimize_thread_count(len(files), avg_size)
                print(f"🎯 Auto-optimized thread count: {thread_count} (based on {len(files)} files, ~{avg_size:.0f}MB avg)")
        
        # Initialize uploader
        print(f"🔧 Initializing GCS uploader for bucket: {args.bucket}")
        print(f"🧵 Using {thread_count} parallel upload threads")
        uploader = GCSGribUploaderMultiThreaded(args.bucket, args.project, thread_count)
        
        # Check bucket accessibility
        if not uploader.check_bucket_exists():
            print(f"❌ Bucket '{args.bucket}' not found or not accessible")
            print("   Make sure the bucket exists and you have proper permissions")
            return 1
        
        print("✅ GCS connection established")
        
        if args.file:
            # Single file upload (still uses threading for consistency)
            gcs_path = args.gcs_path or os.path.basename(args.file)
            file_list = [(args.file, gcs_path)]
            successful, failed, total_size = uploader.upload_files_parallel(
                file_list, args.remove_local
            )
            return 0 if failed == 0 else 1
        
        else:
            # Directory upload with parallel processing
            successful, failed, total_size = uploader.upload_directory(
                args.directory, args.prefix, args.pattern, 
                args.remove_local, args.batch_size
            )
            
            if failed > 0:
                print(f"\n⚠️  {failed} uploads failed")
                return 1
            else:
                print(f"\n🎉 All {successful} files uploaded successfully!")
                return 0
                
    except Exception as e:
        print(f"❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())

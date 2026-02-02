#!/usr/bin/env python3
"""
Automated AIFS GPU Pipeline - Per-Member Processing

CRITICAL DESIGN: Processes ONE ensemble member at a time to avoid storage issues.

For each member:
    1. Download single .pkl file from GCS
    2. Run GPU inference → generate .grib files
    3. Upload .grib files to GCS
    4. Cleanup: delete BOTH .pkl AND .grib files locally
    → Repeat for next member

This ensures minimal local storage usage on GPU machines.

Usage:
    python automate_aifs_gpu_pipeline.py --date 20251127_0000 --members 1-50

GCS Structure:
    gs://aifs-aiquest-us-20251127/
        YYYYMMDD_0000/
            input/          <- pickle files
            forecasts/      <- GRIB output files
"""

import os
import sys
import time
import argparse
from pathlib import Path
from typing import List, Tuple

# Import from existing scripts
from download_pkl_from_gcs import (
    download_from_gcs,
    parse_member_range,
    verify_pickle_file,
    GCS_BUCKET,
    GCS_SERVICE_ACCOUNT_KEY
)

from upload_aifs_gpu_output_grib_gcs import (
    GCSGribUploaderMultiThreaded
)


# =============================================================================
# Configuration
# =============================================================================

LOCAL_INPUT_DIR = "/scratch/input_states"
LOCAL_OUTPUT_DIR = "/scratch/ensemble_outputs"
LEAD_TIME = 792  # Hours


# =============================================================================
# Per-Member Pipeline Functions
# =============================================================================

def download_single_member(member: int, date_str: str, input_dir: str,
                           bucket: str, service_account_key: str) -> Tuple[bool, str]:
    """Download pickle file for a single ensemble member."""
    gcs_blob_name = f"{date_str}/input/input_state_member_{member:03d}.pkl"
    local_path = os.path.join(input_dir, f"input_state_member_{member:03d}.pkl")

    os.makedirs(input_dir, exist_ok=True)

    # Check if valid file exists locally
    if os.path.exists(local_path):
        is_valid, msg = verify_pickle_file(local_path)
        if is_valid:
            print(f"    [DOWNLOAD] Already exists and valid")
            return True, local_path
        else:
            print(f"    [DOWNLOAD] Invalid local file, removing...")
            os.remove(local_path)

    # Download from GCS
    print(f"    [DOWNLOAD] Fetching from GCS...")
    if download_from_gcs(bucket, gcs_blob_name, local_path, service_account_key):
        is_valid, msg = verify_pickle_file(local_path)
        if is_valid:
            print(f"    [DOWNLOAD] Success: {msg}")
            return True, local_path
        else:
            print(f"    [DOWNLOAD] Verification failed: {msg}")
            return False, ""

    return False, ""


def run_single_member(member: int, input_dir: str, output_dir: str,
                      runner) -> Tuple[bool, List[str]]:
    """Run AIFS model for a single ensemble member."""
    import fp32_multi_run_AIFS_ENS_v1
    from fp32_multi_run_AIFS_ENS_v1 import run_ensemble_member

    # Fix Issue 1: Override the hardcoded PICKLE_INPUT_DIR with our input_dir
    fp32_multi_run_AIFS_ENS_v1.PICKLE_INPUT_DIR = input_dir

    os.makedirs(output_dir, exist_ok=True)

    print(f"    [MODEL] Running inference...")
    try:
        # run_ensemble_member returns (success, total_size) - only 2 values
        success, size = run_ensemble_member(runner, None, member, output_dir)

        if success:
            # Collect output files for this member using glob
            files = list(Path(output_dir).glob(f"*_member{member:03d}_*.grib"))
            files = [str(f) for f in files]

            print(f"    [MODEL] Success: {len(files)} files, {size:.1f} MB")
            return True, files
        else:
            print(f"    [MODEL] Failed")
            return False, []

    except Exception as e:
        print(f"    [MODEL] Error: {str(e)}")
        return False, []


def upload_single_member(member: int, output_files: List[str], date_str: str,
                         bucket: str, service_account_key: str,
                         threads: int = 5) -> bool:
    """Upload GRIB files for a single ensemble member."""
    if not output_files:
        print(f"    [UPLOAD] No files to upload")
        return True

    # Prepare file list
    file_list = []
    for local_path in output_files:
        if os.path.exists(local_path):
            gcs_path = f"{date_str}/forecasts/{os.path.basename(local_path)}"
            file_list.append((local_path, gcs_path))

    if not file_list:
        print(f"    [UPLOAD] No valid files found")
        return True

    total_size = sum(os.path.getsize(f[0]) for f in file_list) / (1024**2)
    print(f"    [UPLOAD] Uploading {len(file_list)} files ({total_size:.1f} MB)...")

    # Upload with explicit service account key
    uploader = GCSGribUploaderMultiThreaded(
        bucket, max_workers=threads, service_account_key=service_account_key
    )
    successful, failed, _ = uploader.upload_files_parallel(file_list, remove_local=False)

    if failed == 0:
        print(f"    [UPLOAD] Success: {successful} files uploaded")
        return True
    else:
        print(f"    [UPLOAD] Partial failure: {failed}/{len(file_list)} failed")
        return False


def cleanup_member_files(member: int, pkl_path: str, grib_files: List[str]) -> None:
    """Remove local pkl and grib files for a single member."""
    removed_count = 0
    removed_size = 0

    # Remove pkl file
    if pkl_path and os.path.exists(pkl_path):
        size = os.path.getsize(pkl_path)
        os.remove(pkl_path)
        removed_count += 1
        removed_size += size

    # Remove grib files
    for grib_path in grib_files:
        if os.path.exists(grib_path):
            size = os.path.getsize(grib_path)
            os.remove(grib_path)
            removed_count += 1
            removed_size += size

    print(f"    [CLEANUP] Removed {removed_count} files ({removed_size/(1024**2):.1f} MB)")


def process_single_member(member: int, date_str: str,
                          input_dir: str, output_dir: str,
                          bucket: str, service_account_key: str,
                          runner, upload_threads: int = 5) -> bool:
    """
    Process a single ensemble member through the complete pipeline.

    Steps:
        1. Download .pkl file
        2. Run GPU inference
        3. Upload .grib files
        4. Cleanup local files
    """
    print(f"\n  --- Member {member:03d} ---")
    start_time = time.time()

    pkl_path = ""
    grib_files = []
    success = True

    # Step 1: Download
    download_ok, pkl_path = download_single_member(
        member, date_str, input_dir, bucket, service_account_key
    )
    if not download_ok:
        print(f"    [FAILED] Download failed for member {member}")
        return False

    # Step 2: Run model
    model_ok, grib_files = run_single_member(member, input_dir, output_dir, runner)
    if not model_ok:
        print(f"    [FAILED] Model run failed for member {member}")
        # Still cleanup the pkl file
        cleanup_member_files(member, pkl_path, [])
        return False

    # Step 3: Upload
    upload_ok = upload_single_member(
        member, grib_files, date_str, bucket, service_account_key, upload_threads
    )
    if not upload_ok:
        print(f"    [WARNING] Upload had failures for member {member}")
        success = False

    # Step 4: Cleanup (always cleanup, even if upload had issues)
    cleanup_member_files(member, pkl_path, grib_files)

    elapsed = time.time() - start_time
    status = "SUCCESS" if success else "PARTIAL"
    print(f"    [{status}] Member {member:03d} completed in {elapsed:.1f}s")

    return success


# =============================================================================
# Main Pipeline
# =============================================================================

def run_pipeline(date_str: str, members: List[int],
                 bucket: str = GCS_BUCKET,
                 service_account_key: str = GCS_SERVICE_ACCOUNT_KEY,
                 input_dir: str = LOCAL_INPUT_DIR,
                 output_dir: str = LOCAL_OUTPUT_DIR,
                 upload_threads: int = 5) -> bool:
    """
    Run the complete AIFS GPU pipeline with per-member processing.

    CRITICAL: Each member is fully processed (download → inference → upload → cleanup)
    before moving to the next member. This minimizes local storage usage.
    """

    print("\n" + "=" * 70)
    print("AIFS AUTOMATED GPU PIPELINE (Per-Member Processing)")
    print("=" * 70)
    print(f"Date:       {date_str}")
    print(f"Members:    {members[0]}-{members[-1]} ({len(members)} total)")
    print(f"Bucket:     {bucket}")
    print(f"Input:      {input_dir}")
    print(f"Output:     {output_dir}")
    print("")
    print("Processing mode: ONE MEMBER AT A TIME")
    print("  For each member: Download → Inference → Upload → Cleanup")
    print("  This ensures minimal local storage usage.")
    print("=" * 70)

    # Enable memory optimization
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

    # Load model ONCE (expensive operation)
    print("\nLoading AIFS-ENS model (one-time)...")
    from anemoi.inference.runners.simple import SimpleRunner
    checkpoint = {"huggingface": "ecmwf/aifs-ens-1.0"}
    runner = SimpleRunner(checkpoint, device="cuda")
    print("Model loaded successfully\n")

    start_time = time.time()
    successful_members = []
    failed_members = []
    processing_times = []

    # Process each member through the complete pipeline
    for i, member in enumerate(members):
        print(f"\n{'='*50}")
        print(f"PROCESSING MEMBER {member} ({i+1}/{len(members)})")
        print(f"{'='*50}")

        member_start = time.time()

        success = process_single_member(
            member=member,
            date_str=date_str,
            input_dir=input_dir,
            output_dir=output_dir,
            bucket=bucket,
            service_account_key=service_account_key,
            runner=runner,
            upload_threads=upload_threads
        )

        member_time = time.time() - member_start
        processing_times.append(member_time)

        if success:
            successful_members.append(member)
        else:
            failed_members.append(member)

        # Progress report
        avg_time = sum(processing_times) / len(processing_times)
        remaining = len(members) - (i + 1)
        eta = remaining * avg_time

        print(f"\n  Progress: {i+1}/{len(members)} members")
        print(f"  This member: {member_time:.1f}s")
        print(f"  Average: {avg_time:.1f}s/member")
        print(f"  ETA: {eta/60:.1f} minutes ({eta/3600:.2f} hours)")

    # Final summary
    elapsed = time.time() - start_time

    print("\n" + "=" * 70)
    print("PIPELINE COMPLETE")
    print("=" * 70)
    print(f"Successful: {len(successful_members)}/{len(members)} members")
    if failed_members:
        print(f"Failed:     {failed_members}")
    print(f"Total time: {elapsed/60:.1f} min ({elapsed/3600:.2f} hours)")
    print(f"Avg/member: {elapsed/len(members):.1f}s")
    print(f"Output:     gs://{bucket}/{date_str}/forecasts/")
    print("=" * 70)

    return len(failed_members) == 0


def main():
    parser = argparse.ArgumentParser(
        description="Automated AIFS GPU Pipeline (Per-Member Processing)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
PROCESSING MODE:
  Each ensemble member is fully processed before moving to the next:
    1. Download single .pkl file
    2. Run GPU inference → generate .grib files
    3. Upload .grib files to GCS
    4. Cleanup: delete BOTH .pkl AND .grib files locally

  This ensures minimal local storage usage on GPU machines.

Examples:
  python automate_aifs_gpu_pipeline.py --date 20251127_0000 --members 1-50
  python automate_aifs_gpu_pipeline.py --date 20251127_0000 --members 1,5,10
  python automate_aifs_gpu_pipeline.py --date 20251127_0000 --members 1-10 --upload-threads 10
        """
    )

    parser.add_argument('--date', required=True, help='Date (YYYYMMDD_0000)')
    parser.add_argument('--members', default='1-50', help='Members (1-50 or 1,2,3)')
    parser.add_argument('--bucket', default=GCS_BUCKET)
    parser.add_argument('--service-account', default=GCS_SERVICE_ACCOUNT_KEY)
    parser.add_argument('--input-dir', default=LOCAL_INPUT_DIR)
    parser.add_argument('--output-dir', default=LOCAL_OUTPUT_DIR)
    parser.add_argument('--upload-threads', type=int, default=5,
                       help='Upload threads per member (default: 5)')

    args = parser.parse_args()

    if not os.path.exists(args.service_account):
        print(f"ERROR: Service account not found: {args.service_account}")
        return 1

    try:
        members = parse_member_range(args.members)
    except ValueError as e:
        print(f"ERROR: Invalid members: {e}")
        return 1

    success = run_pipeline(
        date_str=args.date,
        members=members,
        bucket=args.bucket,
        service_account_key=args.service_account,
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        upload_threads=args.upload_threads
    )

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())

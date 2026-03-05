#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "google-cloud-storage",
#     "google-auth",
#     "requests",
#     "python-dotenv",
# ]
# ///
"""
AIFS GPU Ensemble Inference — Local Orchestrator (Cloud Run GPU)
================================================================

Sends HTTP POST requests to a Cloud Run GPU service running AIFS inference.
No Lithops. Each request processes a batch of ensemble members on one L4 GPU.

Architecture:
    run_aifs_ensemble.py (this script, runs locally via uv run)
         |
         v  HTTP POST /run {members: [1,2,3], date: "20260226_0000"}
      /    |    \\
     /     |     \\  (concurrent ThreadPoolExecutor)
    v      v      v
  [GPU 1]  [GPU 2]  ... [GPU N]  (Cloud Run L4 GPU instances)
    |       |            |
    v       v            v
  process_aifs_batch() on each GPU → GCS upload

Usage:
    # Full ensemble (50 members, 17 batches of 3)
    uv run run_aifs_ensemble.py --date 20260301_0000 --members 1-50 --batch-size 3

    # Small test (2 batches)
    uv run run_aifs_ensemble.py --date 20260301_0000 --members 1-6 --batch-size 3

    # Single member test
    uv run run_aifs_ensemble.py --date 20260301_0000 --members 23 --batch-size 1

    # Dry run (show batches without executing)
    uv run run_aifs_ensemble.py --date 20260301_0000 --members 1-50 --dry-run

    # Custom service URL
    uv run run_aifs_ensemble.py --date 20260301_0000 --members 1-6 \\
        --service-url https://aifs-gpu-worker-XXXXX.region.run.app

Author: ICPAC GIK Team
"""

import os
import sys
import time
import argparse
import logging
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Any

import requests
import google.auth
import google.auth.transport.requests
from dotenv import load_dotenv

# Load .env from the script's directory
load_dotenv(Path(__file__).parent / '.env')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ==============================================================================
# CONFIGURATION (loaded from .env, with fallback defaults)
# ==============================================================================

GCS_BUCKET = os.environ.get('GCS_BUCKET', 'aifs-aiquest-us-20251127')
GCS_OUTPUT_SUBPATH = os.environ.get('GCS_OUTPUT_SUBPATH', 'fp16_forecasts')

DEFAULT_SERVICE_URL = os.environ.get('AIFS_SERVICE_URL', '')

MAX_RETRIES = int(os.environ.get('MAX_RETRIES', '3'))
REQUEST_TIMEOUT = int(os.environ.get('REQUEST_TIMEOUT', '3600'))
RETRY_DELAY = 30        # seconds between retries


# ==============================================================================
# AUTH
# ==============================================================================

def get_id_token(audience: str) -> str:
    """
    Get a Google ID token for authenticated Cloud Run invocation.

    Tries in order:
      1. google.oauth2.id_token (works with SA key or metadata server)
      2. gcloud auth print-identity-token (works with user accounts)
    """
    # Try google-auth library first (service accounts, metadata server)
    try:
        from google.oauth2 import id_token as id_token_mod
        from google.auth.transport.requests import Request

        token = id_token_mod.fetch_id_token(Request(), audience)
        return token
    except Exception:
        pass

    # Fall back to gcloud CLI (works for user accounts)
    import subprocess
    result = subprocess.run(
        ["gcloud", "auth", "print-identity-token"],
        capture_output=True, text=True, timeout=15,
    )
    if result.returncode == 0 and result.stdout.strip():
        return result.stdout.strip()

    raise RuntimeError(
        "Cannot obtain ID token. Either set GOOGLE_APPLICATION_CREDENTIALS to a "
        "service account key, or run 'gcloud auth login' with an account that has "
        "roles/run.invoker on the Cloud Run service."
    )


# ==============================================================================
# BATCH HELPERS
# ==============================================================================

def parse_member_range(member_str: str) -> List[int]:
    """Parse member specification: '1-50', '1,2,3', '1-10,15,20-25'."""
    members = []
    for part in member_str.split(','):
        part = part.strip()
        if '-' in part:
            start, end = part.split('-', 1)
            members.extend(range(int(start), int(end) + 1))
        else:
            members.append(int(part))
    return sorted(set(members))


def create_batches(members: List[int], batch_size: int) -> List[List[int]]:
    """Split members into batches of batch_size."""
    return [
        members[i:i + batch_size]
        for i in range(0, len(members), batch_size)
    ]


# ==============================================================================
# GCS VERIFICATION
# ==============================================================================

def verify_gcs_output(date_str: str, member: int) -> bool:
    """Check that at least one GRIB file exists in GCS for this member."""
    try:
        from google.cloud import storage
        client = storage.Client()
        bucket = client.bucket(GCS_BUCKET)
        prefix = f"{date_str}/{GCS_OUTPUT_SUBPATH}/aifs_ens_forecast_{date_str}_member{member:03d}_"

        blobs = list(bucket.list_blobs(prefix=prefix, max_results=1))
        return len(blobs) > 0

    except Exception:
        return False


# ==============================================================================
# HTTP DISPATCH
# ==============================================================================

def check_batch_already_done(batch: List[int], date_str: str) -> List[int]:
    """Return list of members that already have output in GCS (skip these)."""
    return [m for m in batch if verify_gcs_output(date_str, m)]


def wait_for_gcs_output(
    batch: List[int],
    date_str: str,
    poll_interval: int = 60,
    max_wait: int = 2400,  # 40 min — generous for GPU inference
) -> bool:
    """
    Poll GCS until all members in batch have output, or timeout.
    Used after a client-side timeout when the GPU is likely still working.
    """
    elapsed = 0
    while elapsed < max_wait:
        all_done = all(verify_gcs_output(date_str, m) for m in batch)
        if all_done:
            return True
        logger.info(
            f"  GCS poll: waiting for {batch} ({elapsed}s/{max_wait}s)..."
        )
        time.sleep(poll_interval)
        elapsed += poll_interval
    return False


def dispatch_batch(
    service_url: str,
    batch: List[int],
    date_str: str,
    batch_idx: int,
    total_batches: int,
) -> Dict[str, Any]:
    """
    Send a single batch to Cloud Run and return the result.

    Safety: checks GCS before dispatching to avoid duplicate GPU runs.
    On timeout, polls GCS instead of retrying (GPU may still be working).
    """
    endpoint = f"{service_url}/run"

    # Pre-check: skip members that already have output in GCS
    already_done = check_batch_already_done(batch, date_str)
    if already_done:
        remaining = [m for m in batch if m not in already_done]
        if not remaining:
            logger.info(
                f"[{batch_idx+1}/{total_batches}] SKIP batch {batch}: "
                f"all members already in GCS"
            )
            return {
                "batch": batch, "date": date_str, "success": True,
                "completed": batch, "failed": [],
                "message": "All members already in GCS (skipped)",
            }
        logger.info(
            f"[{batch_idx+1}/{total_batches}] Members {already_done} already "
            f"in GCS, only processing {remaining}"
        )
        batch = remaining

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            logger.info(
                f"[{batch_idx+1}/{total_batches}] Attempt {attempt}: "
                f"members {batch} → {endpoint}"
            )

            token = get_id_token(service_url)
            headers = {
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json",
            }
            payload = {"members": batch, "date": date_str}

            resp = requests.post(
                endpoint,
                json=payload,
                headers=headers,
                timeout=REQUEST_TIMEOUT,
            )

            if resp.status_code == 200:
                result = resp.json()
                if result.get("success"):
                    logger.info(f"  OK   batch {batch}: {result.get('message', '')}")
                    return result
                else:
                    logger.warning(
                        f"  FAIL batch {batch}: server returned success=False "
                        f"({result.get('message', 'unknown')})"
                    )
                    # Fall through to retry
            else:
                logger.warning(
                    f"  FAIL batch {batch}: HTTP {resp.status_code} "
                    f"({resp.text[:200]})"
                )

        except requests.exceptions.Timeout:
            # Client timeout does NOT mean the GPU stopped — poll GCS instead
            logger.warning(
                f"  TIMEOUT batch {batch} (attempt {attempt}). "
                f"GPU may still be running — polling GCS..."
            )
            if wait_for_gcs_output(batch, date_str):
                logger.info(f"  GCS-OK batch {batch}: appeared after timeout")
                return {
                    "batch": batch, "date": date_str, "success": True,
                    "completed": batch, "failed": [],
                    "message": "GCS-verified after client timeout",
                }
            # Still not in GCS after long wait — now retry
            logger.warning(f"  GCS poll exhausted for {batch}, will retry")
        except requests.exceptions.ConnectionError as e:
            logger.warning(f"  CONNECTION ERROR batch {batch}: {e}")
        except Exception as e:
            logger.warning(f"  ERROR batch {batch}: {e}")

        # GCS verification rescue: check if files landed despite error
        all_verified = all(verify_gcs_output(date_str, m) for m in batch)
        if all_verified:
            logger.info(f"  GCS-OK batch {batch}: all members verified in GCS")
            return {
                "batch": batch, "date": date_str, "success": True,
                "completed": batch, "failed": [],
                "message": "GCS-verified (HTTP status stale)",
            }

        if attempt < MAX_RETRIES:
            logger.info(f"  Retrying in {RETRY_DELAY}s...")
            time.sleep(RETRY_DELAY)

    # All retries exhausted
    logger.error(f"  GIVE UP batch {batch}: max retries ({MAX_RETRIES}) exceeded")
    return {
        "batch": batch, "date": date_str, "success": False,
        "completed": [], "failed": batch,
        "message": f"Max retries ({MAX_RETRIES}) exceeded",
    }


# ==============================================================================
# PARALLEL EXECUTION
# ==============================================================================

def run_parallel(
    service_url: str,
    date_str: str,
    members: List[int],
    batch_size: int,
    max_concurrent: int = 3,
) -> List[Dict]:
    """
    Dispatch batches to Cloud Run in parallel using ThreadPoolExecutor.

    Cloud Run auto-queues requests beyond available instances, so we limit
    concurrency to max_concurrent to stay within GPU quota.
    """
    batches = create_batches(members, batch_size)
    total = len(batches)

    print(f"\nDispatching {total} batches to {service_url}")
    print(f"Max concurrent: {max_concurrent}  |  Max retries: {MAX_RETRIES}")
    print("=" * 70)

    overall_start = time.time()
    results = [None] * total

    with ThreadPoolExecutor(max_workers=max_concurrent) as pool:
        future_to_idx = {}
        for i, batch in enumerate(batches):
            future = pool.submit(
                dispatch_batch, service_url, batch, date_str, i, total
            )
            future_to_idx[future] = i

        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            try:
                results[idx] = future.result()
            except Exception as e:
                logger.error(f"Batch {idx} raised: {e}")
                results[idx] = {
                    "batch": batches[idx],
                    "date": date_str,
                    "success": False,
                    "completed": [],
                    "failed": batches[idx],
                    "message": str(e),
                }

    total_time = time.time() - overall_start
    success_count = sum(1 for r in results if r and r.get("success"))
    print(
        f"\nCompleted in {total_time:.1f}s ({total_time/60:.1f} min) — "
        f"{success_count}/{total} batches successful"
    )

    return results


# ==============================================================================
# SUMMARY
# ==============================================================================

def print_summary(
    date_str: str,
    batches: List[List[int]],
    results: List[Dict],
    total_time: float
):
    """Print execution summary."""
    all_completed = []
    all_failed = []
    total_files = 0
    total_size = 0.0

    for result in results:
        if not result:
            continue
        all_completed.extend(result.get('completed', []))
        all_failed.extend(result.get('failed', []))
        total_files += result.get('total_files', 0)
        total_size += result.get('total_size_mb', 0.0)

    batch_success = sum(1 for r in results if r and r.get('success'))

    print("\n" + "=" * 70)
    print("EXECUTION SUMMARY — AIFS GPU ENSEMBLE INFERENCE")
    print("=" * 70)

    print(f"\nDate: {date_str}")
    print(f"Batches: {batch_success}/{len(batches)} successful")
    print(f"Members: {len(all_completed)} completed, {len(all_failed)} failed")
    print(f"GRIB files: {total_files}")
    print(f"Total size: {total_size:.1f} MB ({total_size/1024:.2f} GB)")
    print(f"Total time: {total_time:.1f}s ({total_time/60:.1f} min)")

    if all_completed:
        print(f"\nCompleted members: {sorted(all_completed)}")
    if all_failed:
        print(f"Failed members: {sorted(all_failed)}")

    print("\nBatch details:")
    print("-" * 70)
    for i, (batch, result) in enumerate(zip(batches, results)):
        if not result:
            print(f"  ???  Batch {i+1} {batch}: No result")
            continue
        status = "OK" if result.get('success') else "FAIL"
        msg = result.get('message', 'No message')
        print(f"  {status} Batch {i+1} {batch}: {msg}")

        member_times = result.get('member_times', {})
        for member, t in sorted(member_times.items()):
            m_status = "ok" if member in result.get('completed', []) else "FAIL"
            print(f"       Member {int(member):03d}: {t:.1f}s [{m_status}]")

    print(f"\nGCS output: gs://{GCS_BUCKET}/{date_str}/{GCS_OUTPUT_SUBPATH}/")
    print(f"  gsutil ls gs://{GCS_BUCKET}/{date_str}/{GCS_OUTPUT_SUBPATH}/")


# ==============================================================================
# CLI
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='AIFS GPU Ensemble Inference — Cloud Run HTTP Orchestrator',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument('--date', type=str, required=True,
                        help='Forecast date (YYYYMMDD_0000)')
    parser.add_argument('--members', type=str, default='1-50',
                        help='Member specification: 1-50, 1,2,3, or 1-10,15,20-25 (default: 1-50)')
    parser.add_argument('--batch-size', type=int,
                        default=int(os.environ.get('BATCH_SIZE', '3')),
                        help='Members per GPU instance (default: 3)')
    parser.add_argument('--max-concurrent', type=int,
                        default=int(os.environ.get('MAX_CONCURRENT', '3')),
                        help='Max concurrent HTTP requests (default: 3, matches GPU quota)')
    parser.add_argument('--service-url', type=str, default=DEFAULT_SERVICE_URL,
                        help='Cloud Run service URL (default: from .env AIFS_SERVICE_URL)')
    parser.add_argument('--dry-run', action='store_true',
                        help='Show batches without executing')
    parser.add_argument('--yes', '-y', action='store_true',
                        help='Skip interactive confirmation')

    args = parser.parse_args()

    # Parse members
    try:
        members = parse_member_range(args.members)
    except ValueError as e:
        parser.error(f"Invalid members: {e}")

    batches = create_batches(members, args.batch_size)
    n_batches = len(batches)

    # Print configuration
    print("=" * 70)
    print("AIFS GPU ENSEMBLE INFERENCE — CLOUD RUN HTTP")
    print("=" * 70)
    print(f"\nDate:           {args.date}")
    print(f"Members:        {members[0]}-{members[-1]} ({len(members)} total)")
    print(f"Batch size:     {args.batch_size}")
    print(f"Batches:        {n_batches}")
    print(f"Max concurrent: {args.max_concurrent}")
    print(f"Service URL:    {args.service_url}")
    print(f"GCS bucket:     {GCS_BUCKET}")
    print(f"GCS output:     {args.date}/{GCS_OUTPUT_SUBPATH}/")

    print(f"\nBatch layout:")
    for i, batch in enumerate(batches, 1):
        print(f"  Batch {i:2d}: members {batch}")

    if args.dry_run:
        print(f"\nDRY RUN — {n_batches} batches would be dispatched.")
        est_time_min = args.batch_size * 15  # ~15 min per member
        est_cost = n_batches * (est_time_min / 60) * 0.65
        print(f"Estimated wall-clock: ~{est_time_min} min")
        print(f"Estimated GPU cost:   ~${est_cost:.2f}")
        return

    # Confirm before large runs
    if n_batches > 5 and not args.yes:
        response = input(f"\nDispatch {n_batches} GPU batches? (y/N): ")
        if response.lower() != 'y':
            print("Cancelled.")
            return

    # Execute
    start_time = time.time()

    results = run_parallel(
        args.service_url,
        args.date,
        members,
        args.batch_size,
        args.max_concurrent,
    )

    total_time = time.time() - start_time

    # Summary
    print_summary(args.date, batches, results, total_time)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

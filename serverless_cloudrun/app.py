"""
AIFS GPU Ensemble Inference — Cloud Run Flask Worker
=====================================================

Flask app running on Cloud Run GPU (L4). Accepts HTTP POST /run requests
with member batch + date, runs AIFS FP16 inference, uploads GRIB files to GCS.

The AIFS model is loaded once as a global singleton on first request to avoid
reloading per request (Cloud Run instances handle one request at a time via
max_instance_request_concurrency=1).

Endpoints:
    GET  /health  → {"status": "ok"}  (startup/liveness probes)
    POST /run     → accepts {members: [1,2,3], date: "20260226_0000"}

Author: ICPAC GIK Team
"""

import os
import sys
import time
import logging
import pickle
import traceback
from typing import Dict, List, Tuple, Any

from flask import Flask, request, jsonify

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# ==============================================================================
# CONFIGURATION
# ==============================================================================

GCS_BUCKET = os.environ.get('GCS_BUCKET', 'aifs-aiquest-us-20251127')
GCS_OUTPUT_SUBPATH = "fp16_forecasts"

LEAD_TIME = 792          # Hours
INFERENCE_PRECISION = "16"
INFERENCE_NUM_CHUNKS = 16

SCRATCH_INPUT = "/scratch/input_states"
SCRATCH_OUTPUT = "/scratch/ensemble_outputs"

# Global AIFS model runner (loaded once, reused across requests)
_runner = None


# ==============================================================================
# MODEL LOADING
# ==============================================================================

def get_runner():
    """
    Get or initialize the global AIFS SimpleRunner singleton.

    Resets on CUDA errors to avoid stale GPU context issues.
    """
    global _runner
    if _runner is not None:
        return _runner

    import torch
    from anemoi.inference.runners.simple import SimpleRunner

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available")

    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    os.environ['ANEMOI_INFERENCE_NUM_CHUNKS'] = str(INFERENCE_NUM_CHUNKS)

    logger.info("Loading AIFS-ENS model (FP16)...")
    model_start = time.time()
    checkpoint = {"huggingface": "ecmwf/aifs-ens-1.0"}
    _runner = SimpleRunner(checkpoint, device="cuda", precision=INFERENCE_PRECISION)
    logger.info(f"Model loaded in {time.time() - model_start:.1f}s")

    return _runner


def reset_runner():
    """Reset the global runner on CUDA errors."""
    global _runner
    _runner = None
    try:
        import torch
        torch.cuda.empty_cache()
    except Exception:
        pass


# ==============================================================================
# GCS HELPERS
# ==============================================================================

def get_gcs_client():
    """Get authenticated GCS client using Workload Identity."""
    from google.cloud import storage
    return storage.Client()


def download_pkl_from_gcs(member: int, date_str: str, input_dir: str) -> Tuple[bool, str]:
    """Download pickle file for a single ensemble member from GCS."""
    gcs_blob_name = f"{date_str}/input/input_state_member_{member:03d}.pkl"
    local_path = os.path.join(input_dir, f"input_state_member_{member:03d}.pkl")

    os.makedirs(input_dir, exist_ok=True)

    try:
        client = get_gcs_client()
        bucket = client.bucket(GCS_BUCKET)
        blob = bucket.blob(gcs_blob_name)

        if not blob.exists():
            logger.error(f"GCS blob not found: gs://{GCS_BUCKET}/{gcs_blob_name}")
            return False, ""

        blob.download_to_filename(local_path)

        file_size = os.path.getsize(local_path) / (1024 * 1024)
        logger.info(f"Downloaded member {member:03d} pkl: {file_size:.1f} MB")
        return True, local_path

    except Exception as e:
        logger.error(f"Download failed for member {member}: {e}")
        return False, ""


def upload_grib_files_to_gcs(
    member: int,
    output_files: List[str],
    date_str: str
) -> Tuple[int, int]:
    """Upload GRIB files for a single member to GCS. Returns (uploaded, failed)."""
    if not output_files:
        return 0, 0

    uploaded = 0
    failed = 0

    try:
        client = get_gcs_client()
        bucket = client.bucket(GCS_BUCKET)

        for local_path in output_files:
            if not os.path.exists(local_path):
                failed += 1
                continue

            gcs_path = f"{date_str}/{GCS_OUTPUT_SUBPATH}/{os.path.basename(local_path)}"
            blob = bucket.blob(gcs_path)

            try:
                blob.upload_from_filename(local_path)
                uploaded += 1
            except Exception as e:
                logger.error(f"Upload failed {os.path.basename(local_path)}: {e}")
                failed += 1

        logger.info(f"Member {member:03d}: uploaded {uploaded}, failed {failed}")

    except Exception as e:
        logger.error(f"GCS upload error for member {member}: {e}")
        failed = len(output_files) - uploaded

    return uploaded, failed


# ==============================================================================
# GPU INFERENCE
# ==============================================================================

def run_single_member_inference(
    runner,
    member: int,
    input_dir: str,
    output_dir: str,
    date_str: str
) -> Tuple[bool, List[str], float]:
    """Run AIFS FP16 inference for a single member."""
    import numpy as np
    from anemoi.inference.outputs.gribfile import GribFileOutput

    pickle_file = os.path.join(input_dir, f"input_state_member_{member:03d}.pkl")
    if not os.path.exists(pickle_file):
        logger.error(f"Pickle file not found: {pickle_file}")
        return False, [], 0.0

    os.makedirs(output_dir, exist_ok=True)

    try:
        with open(pickle_file, 'rb') as f:
            input_state = pickle.load(f)

        date = input_state['date']
        date_str_formatted = date.strftime("%Y%m%d_%H%M")

        runner.time_step = 6
        runner.lead_time = LEAD_TIME
        runner.reference_date = date

        hours_per_file = 72
        steps_per_file = hours_per_file // runner.time_step  # 12

        created_files = []
        total_size = 0.0
        step_count = 0
        current_file_step = 0
        grib_output = None
        outputs_initialized = False

        for state in runner.run(input_state=input_state, lead_time=LEAD_TIME):
            if current_file_step == 0:
                if grib_output is not None:
                    grib_output.close()
                    outputs_initialized = False

                start_hour = step_count * runner.time_step
                end_hour = min(start_hour + hours_per_file, LEAD_TIME)
                grib_file = os.path.join(
                    output_dir,
                    f"aifs_ens_forecast_{date_str_formatted}_member{member:03d}_h{start_hour:03d}-{end_hour:03d}.grib"
                )
                grib_output = GribFileOutput(runner, path=grib_file)

            if not outputs_initialized:
                grib_output.open(state)
                outputs_initialized = True

            grib_output.write_step(state)
            step_count += 1
            current_file_step += 1

            if step_count % 4 == 0:
                logger.info(f"Member {member:03d}: {step_count * 6}h completed")

            if current_file_step >= steps_per_file or step_count * runner.time_step >= LEAD_TIME:
                grib_output.close()
                outputs_initialized = False

                if os.path.exists(grib_file):
                    file_size = os.path.getsize(grib_file) / (1024 * 1024)
                    created_files.append(grib_file)
                    total_size += file_size

                current_file_step = 0

        if grib_output is not None and outputs_initialized:
            grib_output.close()

        logger.info(f"Member {member:03d}: {len(created_files)} files, {total_size:.1f} MB")
        return len(created_files) > 0, created_files, total_size

    except Exception as e:
        logger.error(f"Inference failed for member {member}: {e}")
        traceback.print_exc()
        return False, [], 0.0


def cleanup_member_files(pkl_path: str, grib_files: List[str]) -> None:
    """Remove local pkl and grib files for a single member."""
    for path in [pkl_path] + grib_files:
        if path and os.path.exists(path):
            try:
                os.remove(path)
            except OSError:
                pass


# ==============================================================================
# BATCH PROCESSING
# ==============================================================================

def process_aifs_batch(member_batch: List[int], date_str: str) -> Dict[str, Any]:
    """
    Process a batch of ensemble members using the global AIFS runner.

    Args:
        member_batch: List of member numbers (e.g., [1, 2, 3])
        date_str: Date string (e.g., "20260301_0000")

    Returns:
        Dictionary with batch results
    """
    import torch

    batch_start = time.time()

    result = {
        "batch": member_batch,
        "date": date_str,
        "completed": [],
        "failed": [],
        "member_times": {},
        "total_files": 0,
        "total_size_mb": 0.0,
        "gpu_device": "none",
        "success": False,
        "message": "",
    }

    if not torch.cuda.is_available():
        result["message"] = "CUDA not available"
        logger.error("CUDA not available on this worker")
        return result

    result["gpu_device"] = torch.cuda.get_device_name(0)
    logger.info(f"GPU: {result['gpu_device']}")
    logger.info(f"Batch: members {member_batch}, date {date_str}")

    # Get or load global runner
    try:
        runner = get_runner()
    except Exception as e:
        result["message"] = f"Model load failed: {e}"
        logger.error(result["message"])
        return result

    # Process each member in the batch
    for member in member_batch:
        member_start = time.time()
        logger.info(f"--- Member {member:03d} ---")

        pkl_path = ""
        grib_files = []

        try:
            # Step 1: Download .pkl from GCS
            download_ok, pkl_path = download_pkl_from_gcs(
                member, date_str, SCRATCH_INPUT
            )
            if not download_ok:
                logger.error(f"Download failed for member {member}")
                result["failed"].append(member)
                result["member_times"][member] = time.time() - member_start
                continue

            # Step 2: Run GPU inference
            inference_ok, grib_files, size_mb = run_single_member_inference(
                runner, member, SCRATCH_INPUT, SCRATCH_OUTPUT, date_str
            )
            if not inference_ok:
                logger.error(f"Inference failed for member {member}")
                result["failed"].append(member)
                cleanup_member_files(pkl_path, [])
                result["member_times"][member] = time.time() - member_start
                continue

            # Step 3: Upload .grib files to GCS
            uploaded, upload_failed = upload_grib_files_to_gcs(
                member, grib_files, date_str
            )

            if upload_failed > 0:
                logger.warning(f"Member {member}: {upload_failed} uploads failed")

            # Step 4: Cleanup local files
            cleanup_member_files(pkl_path, grib_files)

            member_time = time.time() - member_start
            result["completed"].append(member)
            result["member_times"][member] = member_time
            result["total_files"] += uploaded
            result["total_size_mb"] += size_mb

            logger.info(
                f"Member {member:03d} done: {uploaded} files, "
                f"{size_mb:.1f} MB, {member_time:.1f}s"
            )

        except RuntimeError as e:
            if "CUDA" in str(e):
                logger.error(f"CUDA error for member {member}, resetting runner: {e}")
                reset_runner()
            else:
                logger.error(f"Error processing member {member}: {e}")
            traceback.print_exc()
            result["failed"].append(member)
            cleanup_member_files(pkl_path, grib_files)
            result["member_times"][member] = time.time() - member_start
        except Exception as e:
            logger.error(f"Error processing member {member}: {e}")
            traceback.print_exc()
            result["failed"].append(member)
            cleanup_member_files(pkl_path, grib_files)
            result["member_times"][member] = time.time() - member_start

    # Batch summary
    batch_time = time.time() - batch_start
    result["success"] = len(result["failed"]) == 0
    result["total_time_seconds"] = round(batch_time, 1)
    result["message"] = (
        f"Batch {member_batch}: {len(result['completed'])}/{len(member_batch)} "
        f"completed in {batch_time:.1f}s"
    )

    logger.info(result["message"])
    return result


# ==============================================================================
# FLASK ROUTES
# ==============================================================================

@app.route('/health', methods=['GET'])
def health():
    """Health check for startup/liveness probes."""
    return jsonify({"status": "ok"})


@app.route('/run', methods=['POST'])
def run():
    """
    Run AIFS inference for a batch of ensemble members.

    Request JSON:
        {
            "members": [1, 2, 3],
            "date": "20260226_0000"
        }

    Returns:
        Batch result dict with completed/failed members, timing, etc.
    """
    data = request.get_json()
    if not data:
        return jsonify({"error": "Request body must be JSON"}), 400

    members = data.get('members')
    date_str = data.get('date')

    if not members or not isinstance(members, list):
        return jsonify({"error": "'members' must be a non-empty list of integers"}), 400
    if not date_str or not isinstance(date_str, str):
        return jsonify({"error": "'date' must be a string like '20260226_0000'"}), 400

    logger.info(f"Received request: members={members}, date={date_str}")

    try:
        result = process_aifs_batch(members, date_str)
        status_code = 200 if result.get("success") else 500
        return jsonify(result), status_code
    except Exception as e:
        logger.error(f"Unhandled error: {e}")
        traceback.print_exc()
        return jsonify({
            "error": str(e),
            "success": False,
            "batch": members,
            "date": date_str,
        }), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, threaded=False)

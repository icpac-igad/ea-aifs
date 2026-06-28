#!/usr/bin/env python3
"""
PyTorch Memory Profiling for AIFS ENS v1 - FP16 Optimized Version

This script profiles GPU memory usage during AIFS ensemble inference WITH FP16
and chunking optimizations enabled. Use this to verify inference fits within 24GB VRAM.

Target GPUs: A10G, RTX 4090, RTX 3090 (24GB VRAM)

Usage:
    # Quick test (default: 1 member, 72h forecast)
    python pytorch_profile_fp16.py

    # Test with specific chunk settings
    python pytorch_profile_fp16.py --chunks 16

    # More aggressive memory reduction
    python pytorch_profile_fp16.py --chunks 32

Output files:
    - aifs_ens_fp16_memory_snapshot.pickle: Memory snapshot for pytorch.org/memory_viz
    - aifs_ens_fp16_gpu_mem.csv: Per-step memory statistics
    - fp16_profile_summary.txt: Human-readable profiling summary
"""

import datetime
from collections import defaultdict
import os
import time
import pickle
import argparse
import sys

import numpy as np
import earthkit.data as ekd
import earthkit.regrid as ekr
import torch

from anemoi.inference.runners.simple import SimpleRunner
from anemoi.inference.outputs.gribfile import GribFileOutput

# Configuration
PARAM_SFC = ["10u", "10v", "2d", "2t", "msl", "skt", "sp", "tcw"]
PARAM_SFC_FC = ["lsm", "z", "slor", "sdor"]
PARAM_SOIL = ["sot"]
PARAM_PL = ["gh", "t", "u", "v", "w", "q"]
LEVELS = [1000, 925, 850, 700, 600, 500, 400, 300, 250, 200, 150, 100, 50]
SOIL_LEVELS = [1, 2]

# Default profiling configuration
DEFAULT_ENSEMBLE_MEMBERS = [1]
DEFAULT_LEAD_TIME = 72  # Short forecast for quick testing
DEFAULT_OUTPUT_DIR = "/scratch/profile_fp16_outputs"

# Pickle file configuration
USE_PICKLE_FILES = True
PICKLE_INPUT_DIR = "/scratch/input_states"

# FP16 Optimization defaults
DEFAULT_PRECISION = "16"
DEFAULT_NUM_CHUNKS = 16

# Output files
SNAPSHOT_FILE = "aifs_ens_fp16_memory_snapshot.pickle"
MEMORY_LOG_FILE = "aifs_ens_fp16_gpu_mem.csv"
SUMMARY_FILE = "fp16_profile_summary.txt"

# Memory threshold for 24GB GPU (with safety margin)
MEMORY_THRESHOLD_GB = 23.0


def get_open_data(date, param, levelist=[], number=None):
    """Retrieve data from ECMWF Open Data API."""
    fields = defaultdict(list)
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
            assert f.to_numpy().shape == (721, 1440)
            values = np.roll(f.to_numpy(), -f.shape[1] // 2, axis=1)
            values = ekr.interpolate(values, {"grid": (0.25, 0.25)},
                                     {"grid": "N320"})
            name = f"{f.metadata('param')}_{f.metadata('levelist')}" if levelist else f.metadata(
                "param")
            fields[name].append(values)

    for param, values in fields.items():
        fields[param] = np.stack(values)

    return fields


def load_input_state_from_pickle(member, pickle_dir):
    """Load input state from pre-downloaded pickle file."""
    pickle_file = os.path.join(pickle_dir,
                               f"input_state_member_{member:03d}.pkl")

    if not os.path.exists(pickle_file):
        raise FileNotFoundError(f"Pickle file not found: {pickle_file}")

    with open(pickle_file, 'rb') as f:
        input_state = pickle.load(f)

    print(f"    Loaded from pickle: {pickle_file}")
    print(f"    Date: {input_state['date']}")
    print(f"    Fields: {len(input_state['fields'])}")

    return input_state


def get_input_fields(date, number):
    """Get input fields for a specific ensemble member."""
    fields = {}

    fields.update(get_open_data(date, param=PARAM_SFC, number=number))
    fields.update(get_open_data(date, param=PARAM_SFC_FC))

    soil = get_open_data(date,
                         param=PARAM_SOIL,
                         levelist=SOIL_LEVELS,
                         number=number)

    mapping = {
        'sot_1': 'stl1',
        'sot_2': 'stl2',
        'vsw_1': 'swvl1',
        'vsw_2': 'swvl2'
    }
    for k, v in soil.items():
        fields[mapping[k]] = v

    fields.update(
        get_open_data(date, param=PARAM_PL, levelist=LEVELS, number=number))

    for level in LEVELS:
        gh = fields.pop(f"gh_{level}")
        fields[f"z_{level}"] = gh * 9.80665

    return fields


def check_memory_threshold(threshold_gb=MEMORY_THRESHOLD_GB):
    """Check if current memory usage exceeds threshold."""
    allocated = torch.cuda.memory_allocated() / 1024**3
    if allocated > threshold_gb:
        return False, allocated
    return True, allocated


def log_memory_stats(member, output_dir, step=None, threshold_gb=MEMORY_THRESHOLD_GB):
    """Log current GPU memory statistics and check threshold."""
    alloc_gb = torch.cuda.max_memory_allocated() / 1024**3
    reserv_gb = torch.cuda.max_memory_reserved() / 1024**3
    current_gb = torch.cuda.memory_allocated() / 1024**3

    log_file = os.path.join(output_dir, MEMORY_LOG_FILE)

    # Create header if file doesn't exist
    if not os.path.exists(log_file):
        with open(log_file, "w") as f:
            f.write("timestamp,member,step,allocated_GB,reserved_GB,current_GB,within_24GB\n")

    step_str = str(step) if step is not None else "final"
    within_threshold = "YES" if current_gb <= threshold_gb else "NO"

    with open(log_file, "a") as f:
        f.write(f"{datetime.datetime.utcnow().isoformat()},{member},{step_str},"
                f"{alloc_gb:.3f},{reserv_gb:.3f},{current_gb:.3f},{within_threshold}\n")

    return alloc_gb, reserv_gb, current_gb


def run_ensemble_member_fp16(runner, date, member, output_dir, lead_time):
    """Run forecast with FP16 optimizations and detailed memory profiling."""

    # Reset peak memory stats
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()

    # Log initial memory state
    print(f"\n  Initial GPU memory:")
    init_alloc, init_reserv, init_current = log_memory_stats(member, output_dir, step="init")
    print(f"    Allocated: {init_alloc:.2f} GB, Reserved: {init_reserv:.2f} GB")

    # Get input fields
    if USE_PICKLE_FILES:
        print(f"  Loading initial conditions from pickle for member {member}...")
        input_state = load_input_state_from_pickle(member, PICKLE_INPUT_DIR)
        date = input_state['date']
    else:
        print(f"  Retrieving initial conditions for member {member}...")
        fields = get_input_fields(date, member)
        input_state = dict(date=date, fields=fields)

    date_str = date.strftime("%Y%m%d_%H%M")

    # Set context properties
    runner.time_step = 6
    runner.lead_time = lead_time
    runner.reference_date = date

    # Calculate steps
    hours_per_file = 72
    steps_per_file = hours_per_file // runner.time_step

    created_files = []
    total_size = 0
    memory_readings = []
    exceeded_threshold = False
    max_memory_step = 0

    print(f"  Running FP16 forecast for member {member} ({lead_time} hours)...")
    print(f"  Memory threshold: {MEMORY_THRESHOLD_GB} GB (24GB GPU target)")

    step_count = 0
    current_file_step = 0
    grib_output = None
    outputs_initialized = False

    # Run with no_grad for inference
    with torch.no_grad():
        for state in runner.run(input_state=input_state, lead_time=lead_time):
            # File management
            if current_file_step == 0:
                if grib_output is not None:
                    grib_output.close()
                    outputs_initialized = False

                start_hour = step_count * runner.time_step
                end_hour = min(start_hour + hours_per_file, lead_time)
                grib_file = f"{output_dir}/fp16_forecast_{date_str}_member{member:03d}_h{start_hour:03d}-{end_hour:03d}.grib"
                grib_output = GribFileOutput(runner, path=grib_file)

            if not outputs_initialized:
                grib_output.open(state)
                outputs_initialized = True

            grib_output.write_step(state)
            step_count += 1
            current_file_step += 1

            # Log memory every step
            alloc, reserv, current = log_memory_stats(member, output_dir, step=step_count)
            memory_readings.append({
                'step': step_count,
                'hour': step_count * 6,
                'allocated': alloc,
                'reserved': reserv,
                'current': current
            })

            # Track max memory step
            if current > max_memory_step:
                max_memory_step = current

            # Check threshold
            if current > MEMORY_THRESHOLD_GB:
                exceeded_threshold = True
                status = "EXCEEDED"
            else:
                status = "OK"

            # Print progress every step for detailed monitoring
            print(f"    Step {step_count:2d} ({step_count * 6:3d}h): "
                  f"Current={current:.2f}GB, Peak={alloc:.2f}GB [{status}]")

            # File completion
            if current_file_step >= steps_per_file or step_count * runner.time_step >= lead_time:
                grib_output.close()
                outputs_initialized = False

                if os.path.exists(grib_file):
                    file_size = os.path.getsize(grib_file) / (1024 * 1024)
                    created_files.append(os.path.basename(grib_file))
                    total_size += file_size

                current_file_step = 0

    # Close final file
    if grib_output is not None and outputs_initialized:
        grib_output.close()

    # Final memory stats
    torch.cuda.empty_cache()
    final_alloc = torch.cuda.max_memory_allocated() / 1024**3
    final_reserv = torch.cuda.max_memory_reserved() / 1024**3

    log_memory_stats(member, output_dir, step="final")

    # Summary
    print(f"\n  {'=' * 50}")
    print(f"  Member {member} FP16 Memory Summary:")
    print(f"  {'=' * 50}")
    print(f"    Peak Allocated: {final_alloc:.2f} GB")
    print(f"    Peak Reserved: {final_reserv:.2f} GB")
    print(f"    Max During Inference: {max_memory_step:.2f} GB")
    print(f"    Threshold (24GB): {MEMORY_THRESHOLD_GB} GB")

    if exceeded_threshold:
        print(f"    Status: FAILED - Memory exceeded {MEMORY_THRESHOLD_GB} GB")
    else:
        print(f"    Status: PASSED - Memory stayed within {MEMORY_THRESHOLD_GB} GB")
        margin = MEMORY_THRESHOLD_GB - max_memory_step
        print(f"    Safety Margin: {margin:.2f} GB remaining")

    return {
        'member': member,
        'success': len(created_files) > 0,
        'file_size_mb': total_size,
        'peak_allocated_gb': final_alloc,
        'peak_reserved_gb': final_reserv,
        'max_inference_gb': max_memory_step,
        'steps': step_count,
        'exceeded_threshold': exceeded_threshold,
        'memory_readings': memory_readings
    }


def print_gpu_info():
    """Print GPU device information."""
    if torch.cuda.is_available():
        device = torch.cuda.current_device()
        props = torch.cuda.get_device_properties(device)
        total_gb = props.total_memory / 1024**3
        print(f"\nGPU Information:")
        print(f"  Device: {props.name}")
        print(f"  Total Memory: {total_gb:.2f} GB")
        print(f"  CUDA Version: {torch.version.cuda}")
        print(f"  PyTorch Version: {torch.__version__}")

        if total_gb < 24:
            print(f"  WARNING: GPU has less than 24GB VRAM!")
        elif total_gb >= 24 and total_gb < 40:
            print(f"  Target GPU class: 24GB (A10G, RTX 4090, RTX 3090)")
        else:
            print(f"  Note: GPU has {total_gb:.0f}GB - testing 24GB compatibility")

        return total_gb
    else:
        print("No CUDA device available!")
        return 0


def write_summary(output_dir, results, total_time, gpu_memory_gb, precision, num_chunks):
    """Write profiling summary to file."""
    summary_file = os.path.join(output_dir, SUMMARY_FILE)

    # Determine overall pass/fail
    all_passed = all(not r['exceeded_threshold'] for r in results)

    with open(summary_file, 'w') as f:
        f.write("=" * 70 + "\n")
        f.write("AIFS ENS v1 - FP16 Optimized Memory Profiling Summary\n")
        f.write("=" * 70 + "\n\n")

        # Overall status banner
        if all_passed:
            f.write("*" * 70 + "\n")
            f.write("*  STATUS: PASSED - Inference fits within 24GB VRAM                  *\n")
            f.write("*" * 70 + "\n\n")
        else:
            f.write("!" * 70 + "\n")
            f.write("!  STATUS: FAILED - Memory exceeded 24GB threshold                   !\n")
            f.write("!" * 70 + "\n\n")

        f.write(f"Date: {datetime.datetime.now().isoformat()}\n")
        f.write(f"GPU Total Memory: {gpu_memory_gb:.2f} GB\n")
        f.write(f"PyTorch Version: {torch.__version__}\n")
        f.write(f"CUDA Version: {torch.version.cuda}\n\n")

        f.write("Optimization Configuration:\n")
        f.write("-" * 70 + "\n")
        f.write(f"  Precision: FP{precision} ({'half' if precision == '16' else 'full'})\n")
        f.write(f"  ANEMOI_INFERENCE_NUM_CHUNKS: {num_chunks}\n")
        f.write(f"  PYTORCH_CUDA_ALLOC_CONF: expandable_segments:True\n")
        f.write(f"  Target Memory: <= {MEMORY_THRESHOLD_GB} GB\n\n")

        f.write("Test Configuration:\n")
        f.write("-" * 70 + "\n")
        f.write(f"  Members profiled: {len(results)}\n")
        f.write(f"  Total time: {total_time:.2f} seconds\n\n")

        f.write("Per-Member Results:\n")
        f.write("-" * 70 + "\n")
        for r in results:
            status = "PASS" if not r['exceeded_threshold'] else "FAIL"
            f.write(f"  Member {r['member']} [{status}]:\n")
            f.write(f"    Peak Allocated: {r['peak_allocated_gb']:.2f} GB\n")
            f.write(f"    Peak Reserved: {r['peak_reserved_gb']:.2f} GB\n")
            f.write(f"    Max During Inference: {r['max_inference_gb']:.2f} GB\n")
            f.write(f"    Steps Completed: {r['steps']}\n")
            f.write(f"    Output Size: {r['file_size_mb']:.2f} MB\n")
            if not r['exceeded_threshold']:
                margin = MEMORY_THRESHOLD_GB - r['max_inference_gb']
                f.write(f"    Safety Margin: {margin:.2f} GB\n")
            f.write("\n")

        # Aggregate stats
        if results:
            max_alloc = max(r['peak_allocated_gb'] for r in results)
            max_inference = max(r['max_inference_gb'] for r in results)

            f.write("Aggregate Statistics:\n")
            f.write("-" * 70 + "\n")
            f.write(f"  Maximum Peak Allocated: {max_alloc:.2f} GB\n")
            f.write(f"  Maximum During Inference: {max_inference:.2f} GB\n")
            f.write(f"  Target Threshold: {MEMORY_THRESHOLD_GB} GB\n")

            if max_inference <= MEMORY_THRESHOLD_GB:
                margin = MEMORY_THRESHOLD_GB - max_inference
                f.write(f"  Overall Margin: {margin:.2f} GB\n")
                f.write(f"\n  CONCLUSION: Safe for 24GB GPUs (A10G, RTX 4090, RTX 3090)\n")
            else:
                excess = max_inference - MEMORY_THRESHOLD_GB
                f.write(f"  Excess Memory: {excess:.2f} GB over threshold\n")
                f.write(f"\n  RECOMMENDATION: Increase NUM_CHUNKS to 32 or 64\n")

        f.write("\nOutput Files:\n")
        f.write("-" * 70 + "\n")
        f.write(f"  Memory Snapshot: {SNAPSHOT_FILE}\n")
        f.write(f"  Memory Log CSV: {MEMORY_LOG_FILE}\n")
        f.write(f"  This Summary: {SUMMARY_FILE}\n")

    print(f"\nSummary written to: {summary_file}")
    return all_passed


def main():
    """Main function for FP16 optimized profiling."""
    parser = argparse.ArgumentParser(
        description='Profile FP16-optimized AIFS ENS inference for 24GB GPUs',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick test with default settings (FP16 + 16 chunks)
  python pytorch_profile_fp16.py

  # Test with more aggressive chunking
  python pytorch_profile_fp16.py --chunks 32

  # Test specific configuration
  python pytorch_profile_fp16.py --chunks 16 --lead-time 144

  # Verify multiple members
  python pytorch_profile_fp16.py --members 1 2 3

Target GPUs (24GB VRAM):
  - NVIDIA A10G
  - NVIDIA RTX 4090
  - NVIDIA RTX 3090

Expected Results:
  - FP16 + 16 chunks: ~20-22 GB (safe margin)
  - FP16 + 32 chunks: ~18-20 GB (extra margin)
        """
    )

    parser.add_argument('--members', type=int, nargs='+', default=DEFAULT_ENSEMBLE_MEMBERS,
                        help='Ensemble members to profile (default: 1)')
    parser.add_argument('--lead-time', type=int, default=DEFAULT_LEAD_TIME,
                        help='Forecast lead time in hours (default: 72)')
    parser.add_argument('--output-dir', type=str, default=DEFAULT_OUTPUT_DIR,
                        help='Output directory for profiles')
    parser.add_argument('--chunks', type=int, default=DEFAULT_NUM_CHUNKS,
                        help='ANEMOI_INFERENCE_NUM_CHUNKS value (default: 16)')
    parser.add_argument('--precision', type=str, default=DEFAULT_PRECISION,
                        choices=['16', '32'],
                        help='Inference precision (default: 16 for FP16)')
    parser.add_argument('--date', type=str, default=None,
                        help='Forecast date YYYYMMDD (default: 20251127)')
    parser.add_argument('--pickle-dir', type=str, default="/scratch/input_states",
                        help='Directory containing input pickle files')
    parser.add_argument('--no-pickle', action='store_true',
                        help='Download data live instead of using pickle files')
    parser.add_argument('--threshold', type=float, default=23.0,
                        help='Memory threshold in GB (default: 23.0 for 24GB GPU)')

    args = parser.parse_args()

    # Update global variables based on args
    global USE_PICKLE_FILES, PICKLE_INPUT_DIR, MEMORY_THRESHOLD_GB
    USE_PICKLE_FILES = not args.no_pickle
    PICKLE_INPUT_DIR = args.pickle_dir
    MEMORY_THRESHOLD_GB = args.threshold

    # Parse date
    if args.date:
        DATE = datetime.datetime.strptime(args.date, "%Y%m%d")
    else:
        DATE = datetime.datetime(2025, 11, 27, 0, 0)

    print("=" * 70)
    print("AIFS ENS v1 - FP16 Optimized Memory Profiling")
    print("Target: 24GB GPU Compatibility Test")
    print("=" * 70)

    print(f"\nForecast Date: {DATE}")
    print(f"Members to profile: {args.members}")
    print(f"Lead time: {args.lead_time} hours")
    print(f"Output directory: {args.output_dir}")

    # Print GPU info
    gpu_memory_gb = print_gpu_info()

    # Set memory optimization environment variables
    print("\n" + "-" * 70)
    print("Applying Memory Optimizations:")
    print("-" * 70)
    print(f"  ANEMOI_INFERENCE_NUM_CHUNKS = {args.chunks}")
    print(f"  PYTORCH_CUDA_ALLOC_CONF = expandable_segments:True")
    print(f"  Precision = FP{args.precision}")
    print(f"  Memory Threshold = {MEMORY_THRESHOLD_GB} GB")

    os.environ['ANEMOI_INFERENCE_NUM_CHUNKS'] = str(args.chunks)
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Load model with FP16 precision
    print("\n" + "=" * 70)
    print(f"Loading model with FP{args.precision} precision...")
    print("=" * 70)

    checkpoint = {"huggingface": "ecmwf/aifs-ens-1.0"}

    if args.precision == "16":
        runner = SimpleRunner(checkpoint, device="cuda", precision="16")
        print("Model loaded with FP16 (half precision) - optimized for 24GB GPUs")
    else:
        runner = SimpleRunner(checkpoint, device="cuda")
        print("Model loaded with FP32 (full precision)")

    # Start memory snapshot recording
    print("\n" + "-" * 70)
    print("Starting PyTorch CUDA memory history recording...")
    print("-" * 70)
    torch.cuda.memory._record_memory_history(max_entries=100000)

    # Track results
    results = []
    start_time = time.time()

    # Run profiling
    for member in args.members:
        print(f"\n{'=' * 70}")
        print(f"Profiling ensemble member {member} (FP{args.precision} + {args.chunks} chunks)")
        print(f"{'=' * 70}")

        try:
            result = run_ensemble_member_fp16(
                runner, DATE, member, args.output_dir, args.lead_time
            )
            results.append(result)

        except torch.cuda.OutOfMemoryError as e:
            print(f"\nOUT OF MEMORY ERROR for member {member}!")
            print(f"Error: {str(e)}")
            print(f"\nRecommendation: Increase --chunks to 32 or 64")
            results.append({
                'member': member,
                'success': False,
                'exceeded_threshold': True,
                'peak_allocated_gb': gpu_memory_gb,
                'peak_reserved_gb': gpu_memory_gb,
                'max_inference_gb': gpu_memory_gb,
                'steps': 0,
                'file_size_mb': 0,
                'memory_readings': []
            })

        except Exception as e:
            print(f"Error profiling member {member}: {str(e)}")
            import traceback
            traceback.print_exc()
            continue

    total_time = time.time() - start_time

    # Dump memory snapshot
    print("\n" + "-" * 70)
    print("Saving PyTorch memory snapshot...")
    print("-" * 70)

    snapshot_path = os.path.join(args.output_dir, SNAPSHOT_FILE)
    torch.cuda.memory._dump_snapshot(snapshot_path)
    torch.cuda.memory._record_memory_history(enabled=None)
    print(f"Saved: {snapshot_path}")

    # Write summary
    all_passed = write_summary(
        args.output_dir, results, total_time, gpu_memory_gb,
        args.precision, args.chunks
    )

    # Final summary with clear pass/fail indication
    print("\n" + "=" * 70)
    if all_passed:
        print("TEST PASSED: FP16 inference fits within 24GB VRAM")
        print("=" * 70)
        if results:
            max_mem = max(r['max_inference_gb'] for r in results)
            margin = MEMORY_THRESHOLD_GB - max_mem
            print(f"  Peak memory: {max_mem:.2f} GB")
            print(f"  Safety margin: {margin:.2f} GB")
            print(f"\n  Safe for: A10G, RTX 4090, RTX 3090")
    else:
        print("TEST FAILED: Memory exceeded 24GB threshold")
        print("=" * 70)
        print(f"\n  Recommendations:")
        print(f"    1. Increase chunks: --chunks 32")
        print(f"    2. More aggressive: --chunks 64")
        print(f"    3. Check for memory leaks in snapshot")

    print(f"\nTotal time: {total_time:.2f} seconds")
    print(f"\nOutput files in {args.output_dir}/:")
    print(f"  - {SNAPSHOT_FILE}")
    print(f"  - {MEMORY_LOG_FILE}")
    print(f"  - {SUMMARY_FILE}")

    # Exit with appropriate code
    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
PyTorch Memory Profiling for AIFS ENS v1

This script profiles GPU memory usage during AIFS ensemble inference WITHOUT FP16 optimization.
It generates memory snapshots that can be visualized at https://pytorch.org/memory_viz

Based on multi_run_AIFS_ENS_v1.py but configured for profiling baseline (FP32) memory usage.

Usage:
    python pytorch_profile.py --members 1 --lead-time 72 --output-dir /scratch/profile_outputs

Output files:
    - aifs_ens_memory_snapshot.pickle: Full memory snapshot for pytorch.org/memory_viz
    - aifs_ens_gpu_mem_per_member.csv: Per-member peak memory statistics
    - profile_summary.txt: Human-readable profiling summary
"""

import datetime
from collections import defaultdict
import os
import time
import pickle
import argparse

import numpy as np
import earthkit.data as ekd
import earthkit.regrid as ekr
import torch

from anemoi.inference.runners.simple import SimpleRunner
from anemoi.inference.outputs.gribfile import GribFileOutput

from ecmwf.opendata import Client as OpendataClient

# Configuration
PARAM_SFC = ["10u", "10v", "2d", "2t", "msl", "skt", "sp", "tcw"]
PARAM_SFC_FC = ["lsm", "z", "slor", "sdor"]
PARAM_SOIL = ["sot"]
PARAM_PL = ["gh", "t", "u", "v", "w", "q"]
LEVELS = [1000, 925, 850, 700, 600, 500, 400, 300, 250, 200, 150, 100, 50]
SOIL_LEVELS = [1, 2]

# Default profiling configuration (can be overridden via CLI)
DEFAULT_ENSEMBLE_MEMBERS = [1]  # Profile single member by default
DEFAULT_LEAD_TIME = 72  # Short forecast for profiling
DEFAULT_OUTPUT_DIR = "/scratch/profile_outputs"

# Pickle file configuration
USE_PICKLE_FILES = True
PICKLE_INPUT_DIR = "/scratch/input_states"

# Profiling output files
SNAPSHOT_FILE = "aifs_ens_memory_snapshot.pickle"
MEMORY_LOG_FILE = "aifs_ens_gpu_mem_per_member.csv"
SUMMARY_FILE = "profile_summary.txt"


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


def log_memory_stats(member, output_dir, step=None):
    """Log current GPU memory statistics."""
    alloc_gb = torch.cuda.max_memory_allocated() / 1024**3
    reserv_gb = torch.cuda.max_memory_reserved() / 1024**3
    current_gb = torch.cuda.memory_allocated() / 1024**3

    log_file = os.path.join(output_dir, MEMORY_LOG_FILE)

    # Create header if file doesn't exist
    if not os.path.exists(log_file):
        with open(log_file, "w") as f:
            f.write("timestamp,member,step,allocated_GB,reserved_GB,current_GB\n")

    step_str = str(step) if step is not None else "final"
    with open(log_file, "a") as f:
        f.write(f"{datetime.datetime.utcnow().isoformat()},{member},{step_str},"
                f"{alloc_gb:.3f},{reserv_gb:.3f},{current_gb:.3f}\n")

    return alloc_gb, reserv_gb, current_gb


def run_ensemble_member_with_profiling(runner, date, member, output_dir, lead_time):
    """Run forecast for a single ensemble member with detailed memory profiling."""

    # Reset peak memory stats for this member
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()

    # Log initial memory state
    print(f"\n  Initial GPU memory:")
    init_alloc, init_reserv, _ = log_memory_stats(member, output_dir, step="init")
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

    # Calculate steps per file (72-hour periods)
    hours_per_file = 72
    steps_per_file = hours_per_file // runner.time_step

    created_files = []
    total_size = 0
    memory_peaks = []

    print(f"  Running forecast for member {member} ({lead_time} hours)...")
    step_count = 0
    current_file_step = 0
    grib_output = None
    outputs_initialized = False

    # Run with no_grad for inference
    with torch.no_grad():
        for state in runner.run(input_state=input_state, lead_time=lead_time):
            # Check if we need to start a new file
            if current_file_step == 0:
                if grib_output is not None:
                    grib_output.close()
                    outputs_initialized = False

                start_hour = step_count * runner.time_step
                end_hour = min(start_hour + hours_per_file, lead_time)
                grib_file = f"{output_dir}/profile_forecast_{date_str}_member{member:03d}_h{start_hour:03d}-{end_hour:03d}.grib"

                grib_output = GribFileOutput(runner, path=grib_file)

            if not outputs_initialized:
                grib_output.open(state)
                outputs_initialized = True

            grib_output.write_step(state)
            step_count += 1
            current_file_step += 1

            # Log memory every step during profiling
            alloc, reserv, current = log_memory_stats(member, output_dir, step=step_count)
            memory_peaks.append(alloc)

            # Print progress every 2 steps (12 hours)
            if step_count % 2 == 0:
                print(f"    Step {step_count} ({step_count * 6}h): "
                      f"Allocated={alloc:.2f}GB, Reserved={reserv:.2f}GB")

            if current_file_step >= steps_per_file or step_count * runner.time_step >= lead_time:
                grib_output.close()
                outputs_initialized = False

                if os.path.exists(grib_file):
                    file_size = os.path.getsize(grib_file) / (1024 * 1024)
                    created_files.append(os.path.basename(grib_file))
                    total_size += file_size

                current_file_step = 0

    # Close final file if still open
    if grib_output is not None and outputs_initialized:
        grib_output.close()

    # Final memory stats
    torch.cuda.empty_cache()
    final_alloc = torch.cuda.max_memory_allocated() / 1024**3
    final_reserv = torch.cuda.max_memory_reserved() / 1024**3

    # Log final stats
    log_memory_stats(member, output_dir, step="final")

    print(f"\n  Member {member} Memory Summary:")
    print(f"    Peak Allocated: {final_alloc:.2f} GB")
    print(f"    Peak Reserved: {final_reserv:.2f} GB")
    print(f"    Max during inference: {max(memory_peaks):.2f} GB")

    return {
        'member': member,
        'success': len(created_files) > 0,
        'file_size_mb': total_size,
        'peak_allocated_gb': final_alloc,
        'peak_reserved_gb': final_reserv,
        'max_inference_gb': max(memory_peaks),
        'steps': step_count
    }


def print_gpu_info():
    """Print GPU device information."""
    if torch.cuda.is_available():
        device = torch.cuda.current_device()
        props = torch.cuda.get_device_properties(device)
        print(f"\nGPU Information:")
        print(f"  Device: {props.name}")
        print(f"  Total Memory: {props.total_memory / 1024**3:.2f} GB")
        print(f"  CUDA Version: {torch.version.cuda}")
        print(f"  PyTorch Version: {torch.__version__}")
        return props.total_memory / 1024**3
    else:
        print("No CUDA device available!")
        return 0


def write_summary(output_dir, results, total_time, gpu_memory_gb):
    """Write profiling summary to file."""
    summary_file = os.path.join(output_dir, SUMMARY_FILE)

    with open(summary_file, 'w') as f:
        f.write("=" * 70 + "\n")
        f.write("AIFS ENS v1 - PyTorch Memory Profiling Summary\n")
        f.write("=" * 70 + "\n\n")

        f.write(f"Date: {datetime.datetime.now().isoformat()}\n")
        f.write(f"GPU Total Memory: {gpu_memory_gb:.2f} GB\n")
        f.write(f"PyTorch Version: {torch.__version__}\n")
        f.write(f"CUDA Version: {torch.version.cuda}\n\n")

        f.write("Configuration:\n")
        f.write(f"  Precision: FP32 (full precision, no optimization)\n")
        f.write(f"  Members profiled: {len(results)}\n")
        f.write(f"  Total time: {total_time:.2f} seconds\n\n")

        f.write("Per-Member Results:\n")
        f.write("-" * 70 + "\n")
        for r in results:
            f.write(f"  Member {r['member']}:\n")
            f.write(f"    Peak Allocated: {r['peak_allocated_gb']:.2f} GB\n")
            f.write(f"    Peak Reserved: {r['peak_reserved_gb']:.2f} GB\n")
            f.write(f"    Max During Inference: {r['max_inference_gb']:.2f} GB\n")
            f.write(f"    Steps: {r['steps']}\n")
            f.write(f"    Output Size: {r['file_size_mb']:.2f} MB\n\n")

        # Calculate aggregates
        if results:
            max_alloc = max(r['peak_allocated_gb'] for r in results)
            max_reserv = max(r['peak_reserved_gb'] for r in results)
            avg_alloc = sum(r['peak_allocated_gb'] for r in results) / len(results)

            f.write("Aggregate Statistics:\n")
            f.write("-" * 70 + "\n")
            f.write(f"  Maximum Peak Allocated: {max_alloc:.2f} GB\n")
            f.write(f"  Maximum Peak Reserved: {max_reserv:.2f} GB\n")
            f.write(f"  Average Peak Allocated: {avg_alloc:.2f} GB\n")
            f.write(f"  GPU Utilization: {max_alloc/gpu_memory_gb*100:.1f}%\n\n")

        f.write("Output Files:\n")
        f.write("-" * 70 + "\n")
        f.write(f"  Memory Snapshot: {SNAPSHOT_FILE}\n")
        f.write(f"  Memory Log CSV: {MEMORY_LOG_FILE}\n")
        f.write(f"  This Summary: {SUMMARY_FILE}\n\n")

        f.write("Next Steps:\n")
        f.write("-" * 70 + "\n")
        f.write("  1. Upload memory snapshot to https://pytorch.org/memory_viz\n")
        f.write("  2. Analyze allocation patterns and fragmentation\n")
        f.write("  3. Identify largest tensor allocations\n")
        f.write("  4. Compare with FP16 profiling results\n")

    print(f"\nSummary written to: {summary_file}")


def main():
    """Main function for profiling AIFS ensemble inference."""
    parser = argparse.ArgumentParser(
        description='Profile GPU memory usage for AIFS ENS inference (FP32 baseline)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Profile single member with short forecast (default)
  python pytorch_profile.py

  # Profile specific members
  python pytorch_profile.py --members 1 2 3

  # Profile with longer forecast
  python pytorch_profile.py --members 1 --lead-time 144

  # Custom output directory
  python pytorch_profile.py --output-dir /scratch/my_profile

Output:
  - aifs_ens_memory_snapshot.pickle: Upload to https://pytorch.org/memory_viz
  - aifs_ens_gpu_mem_per_member.csv: Per-step memory statistics
  - profile_summary.txt: Human-readable summary
        """
    )

    parser.add_argument('--members', type=int, nargs='+', default=DEFAULT_ENSEMBLE_MEMBERS,
                        help='Ensemble members to profile (default: 1)')
    parser.add_argument('--lead-time', type=int, default=DEFAULT_LEAD_TIME,
                        help='Forecast lead time in hours (default: 72)')
    parser.add_argument('--output-dir', type=str, default=DEFAULT_OUTPUT_DIR,
                        help='Output directory for profiles and forecasts')
    parser.add_argument('--date', type=str, default=None,
                        help='Forecast date YYYYMMDD (default: 20251127)')
    parser.add_argument('--pickle-dir', type=str, default="/scratch/input_states",
                        help='Directory containing input pickle files')
    parser.add_argument('--no-pickle', action='store_true',
                        help='Download data live instead of using pickle files')

    args = parser.parse_args()

    # Update globals based on args
    global USE_PICKLE_FILES, PICKLE_INPUT_DIR
    USE_PICKLE_FILES = not args.no_pickle
    PICKLE_INPUT_DIR = args.pickle_dir

    # Parse date
    if args.date:
        DATE = datetime.datetime.strptime(args.date, "%Y%m%d")
    else:
        DATE = datetime.datetime(2025, 11, 27, 0, 0)

    print("=" * 70)
    print("AIFS ENS v1 - PyTorch Memory Profiling (FP32 Baseline)")
    print("=" * 70)
    print(f"\nForecast Date: {DATE}")
    print(f"Members to profile: {args.members}")
    print(f"Lead time: {args.lead_time} hours")
    print(f"Output directory: {args.output_dir}")
    print(f"Using pickle files: {USE_PICKLE_FILES}")

    # Print GPU info
    gpu_memory_gb = print_gpu_info()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Load model WITHOUT memory optimizations (FP32 baseline)
    print("\n" + "=" * 70)
    print("Loading model in FP32 (full precision) mode...")
    print("NOTE: This requires ~50GB VRAM. Use for profiling baseline only.")
    print("=" * 70)

    checkpoint = {"huggingface": "ecmwf/aifs-ens-1.0"}
    runner = SimpleRunner(checkpoint, device="cuda")  # No precision="16" for baseline
    print("Model loaded successfully in FP32 mode!")

    # Start PyTorch memory snapshot recording
    print("\n" + "-" * 70)
    print("Starting PyTorch CUDA memory history recording...")
    print("-" * 70)
    torch.cuda.memory._record_memory_history(max_entries=100000)

    # Track results
    results = []
    start_time = time.time()

    # Run profiling for each member
    for member in args.members:
        print(f"\n{'=' * 70}")
        print(f"Profiling ensemble member {member}")
        print(f"{'=' * 70}")

        try:
            result = run_ensemble_member_with_profiling(
                runner, DATE, member, args.output_dir, args.lead_time
            )
            results.append(result)

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
    print(f"Saved CUDA memory snapshot to: {snapshot_path}")
    print(f"Upload to https://pytorch.org/memory_viz for visualization")

    # Write summary
    write_summary(args.output_dir, results, total_time, gpu_memory_gb)

    # Final summary
    print("\n" + "=" * 70)
    print("Profiling Complete!")
    print("=" * 70)
    print(f"Total time: {total_time:.2f} seconds")
    print(f"Members profiled: {len(results)}")

    if results:
        max_mem = max(r['peak_allocated_gb'] for r in results)
        print(f"Peak memory usage: {max_mem:.2f} GB")
        print(f"GPU utilization: {max_mem/gpu_memory_gb*100:.1f}%")

    print(f"\nOutput files in {args.output_dir}/:")
    print(f"  - {SNAPSHOT_FILE} (upload to pytorch.org/memory_viz)")
    print(f"  - {MEMORY_LOG_FILE} (per-step CSV data)")
    print(f"  - {SUMMARY_FILE} (human-readable summary)")


if __name__ == "__main__":
    main()

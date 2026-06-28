#!/usr/bin/env python3
"""
PyTorch Memory Profiling for AIFS ENS v2.0 — FP16.

Profiles GPU memory during **aifs-ens-2.0** ensemble inference with FP16 + chunking,
to size the GPU/precision/chunks for a production run. Clone of
``fp16FahamuAIFSv1/pytorch_profile_fp16.py`` with the v2 checkpoint and the 112-field
v2 input contract; the PyTorch CUDA-snapshot machinery is unchanged.

Method follows aifs-ens-1.0 Discussion #17 and the ECMWF "Anemoi Profiling" demo:
wrap one member's inference between ``torch.cuda.memory._record_memory_history()`` and
``_dump_snapshot()`` and report ``max_memory_allocated`` / ``max_memory_reserved``.

NOTE: aifs-ens-2.0 is larger than v1 — the v1 ~23 GB-reserved result is only a baseline;
re-measure here. The inference stack (anemoi-inference 0.8.3, torch 2.7, flash-attn
2.7.4) only exists on the v2 GPU env (see README.md).

Usage:
    # Stage one v2 pkl first, then:
    python pytorch_profile_fp16_v2.py --chunks 16 --members 1 --lead-time 72 \
        --pickle-dir /scratch/input_states --threshold 23.0
    # Tighter memory: --chunks 32 (or 64).  Live download instead of pkl: --no-pickle

Output files (in --output-dir):
    - aifs_ens_fp16_v2_memory_snapshot.pickle : snapshot for pytorch.org/memory_viz
    - aifs_ens_fp16_v2_gpu_mem.csv            : per-step memory statistics
    - fp16_v2_profile_summary.txt             : human-readable summary
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

# v2.0 parameter lists (from run_AIFS_ENS_v2.0.ipynb)
PARAM_SFC = ["10u", "10v", "2d", "2t", "msl", "skt", "sp", "tcw", "sd"]
PARAM_SFC_FC = ["lsm", "z", "slor", "sdor"]
PARAM_SOIL = ["vsw", "sot"]
PARAM_WAVE = ["wmb", "h1012", "h1214", "h1417", "h1721",
              "h2125", "h2530", "mwd", "cdww", "mwp", "swh"]
PARAM_PL = ["gh", "t", "u", "v", "w", "q"]
LEVELS = [1000, 925, 850, 700, 600, 500, 400, 300, 250, 200, 150, 100, 50, 10]
SOIL_LEVELS = [1, 2]
SOIL_RENAME = {"sot_1": "stl1", "sot_2": "stl2",
               "vsw_1": "swvl1", "vsw_2": "swvl2"}
G = 9.80665

# Default profiling configuration
DEFAULT_ENSEMBLE_MEMBERS = [1]
DEFAULT_LEAD_TIME = 72  # Short forecast for quick testing
DEFAULT_OUTPUT_DIR = "/scratch/profile_fp16_v2_outputs"

# Pickle file configuration
USE_PICKLE_FILES = True
PICKLE_INPUT_DIR = "/scratch/input_states"

# FP16 Optimization defaults
DEFAULT_PRECISION = "16"
DEFAULT_NUM_CHUNKS = 16

# Output files
SNAPSHOT_FILE = "aifs_ens_fp16_v2_memory_snapshot.pickle"
MEMORY_LOG_FILE = "aifs_ens_fp16_v2_gpu_mem.csv"
SUMMARY_FILE = "fp16_v2_profile_summary.txt"
PLOT_FILE = "aifs_ens_fp16_v2_memory_timeline.png"

# Memory threshold for 24GB GPU (with safety margin)
MEMORY_THRESHOLD_GB = 23.0


def get_open_data(date, param, levelist=[], number=None, constant=False,
                  source="ecmwf", **kwargs):
    """Retrieve + N320-interpolate ECMWF Open Data (v2 live-mode fallback)."""
    fields = defaultdict(list)
    dates = [date] if constant else [date - datetime.timedelta(hours=6), date]
    for d in dates:
        if number is None:
            data = ekd.from_source("ecmwf-open-data", date=d, param=param,
                                   levelist=levelist, source=source, **kwargs)
        else:
            kwargs.setdefault("stream", "enfo")
            data = ekd.from_source("ecmwf-open-data", date=d, param=param,
                                   levelist=levelist, number=[number],
                                   source=source, **kwargs)
        for f in data:
            if not levelist and f.metadata("levtype") == "pl":
                continue
            assert f.to_numpy().shape == (721, 1440)
            values = np.roll(f.to_numpy(), -f.shape[1] // 2, axis=1)
            values = ekr.interpolate(values, {"grid": (0.25, 0.25)}, {"grid": "N320"})
            name = f"{f.metadata('param')}_{f.metadata('levelist')}" if levelist \
                else f.metadata("param")
            fields[name].append(values)
            if constant:
                fields[name].append(values)
    for name, values in fields.items():
        fields[name] = np.stack(values)
    return fields


WAVE_DIR_COMPONENTS = ("cos_mwd", "sin_mwd")


def drop_unencodable_wave_dir(state):
    """Drop the model's ``cos_mwd``/``sin_mwd`` outputs before GRIB encoding.

    The v2 input prep decomposes mean wave direction ``mwd`` into circular
    components ``cos_mwd``/``sin_mwd``; the model emits them back. They are *not*
    GRIB parameters, so ``GribFileOutput`` dies with ``ConceptNoMatchError:
    Concept no match``. Recombining them into ``mwd`` doesn't help either —
    anemoi's ``GribFileOutput`` only writes variables the checkpoint declares
    (``typed_variables``), and ``mwd`` is not one (``KeyError: 'mwd'``).

    So we drop the two components here; every other output field (incl. the real
    wave params ``swh``/``mwp``/``cdww``) encodes fine, and the downstream
    extraction (Step 3) only needs ``tp``/``msl``/``2t``. If wave *direction* is
    ever needed, recover it from the raw output as ``degrees(atan2(sin, cos))``.
    The yielded state is a fresh per-step dict, so popping does not affect the
    autoregressive rollout.
    """
    fields = state["fields"]
    for name in WAVE_DIR_COMPONENTS:
        fields.pop(name, None)
    return state


def load_input_state_from_pickle(member, pickle_dir):
    """Load the pre-built 112-field input state for one member."""
    pickle_file = os.path.join(pickle_dir, f"input_state_member_{member:03d}.pkl")
    if not os.path.exists(pickle_file):
        raise FileNotFoundError(f"Pickle file not found: {pickle_file}")
    with open(pickle_file, "rb") as f:
        input_state = pickle.load(f)
    print(f"    Loaded from pickle: {pickle_file}")
    print(f"    Date: {input_state['date']}")
    print(f"    Fields: {len(input_state['fields'])}")
    return input_state


def get_input_fields(date, number, source="ecmwf"):
    """Live-download the v2.0 input fields for one member (fallback path)."""
    fields = {}
    fields.update(get_open_data(date, param=PARAM_SFC, number=number,
                                levtype="sfc", source=source))
    fields.update(get_open_data(date, param=PARAM_SFC_FC, constant=True,
                                levtype="sfc", source=source))
    fields.update(get_open_data(date, param=PARAM_WAVE, number=number,
                                stream="wave" if not number else "waef", source=source))
    soil = get_open_data(date, param=PARAM_SOIL, levelist=SOIL_LEVELS,
                         number=number, source=source)
    for k, v in soil.items():
        fields[SOIL_RENAME[k]] = v
    fields.update(get_open_data(date, param=PARAM_PL, levelist=LEVELS,
                                number=number, source=source))

    # v2.0 transforms
    mwd_rad = np.deg2rad(fields.pop("mwd"))
    fields["cos_mwd"] = np.cos(mwd_rad)
    fields["sin_mwd"] = np.sin(mwd_rad)
    fields.pop("q_10", None)
    mask = np.equal(fields["lsm"][0], 0)
    for f in ("sd", "swvl1", "swvl2"):
        fields[f][:, mask] = np.nan
    for level in LEVELS:
        fields[f"z_{level}"] = fields.pop(f"gh_{level}") * G
    return fields


def log_memory_stats(member, output_dir, step=None, threshold_gb=None):
    """Log current GPU memory statistics and check threshold."""
    threshold_gb = MEMORY_THRESHOLD_GB if threshold_gb is None else threshold_gb
    alloc_gb = torch.cuda.max_memory_allocated() / 1024**3
    reserv_gb = torch.cuda.max_memory_reserved() / 1024**3
    current_gb = torch.cuda.memory_allocated() / 1024**3

    log_file = os.path.join(output_dir, MEMORY_LOG_FILE)
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
    """Run forecast with FP16 + detailed memory profiling (v2.0)."""
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()

    print(f"\n  Initial GPU memory:")
    init_alloc, init_reserv, _ = log_memory_stats(member, output_dir, step="init")
    print(f"    Allocated: {init_alloc:.2f} GB, Reserved: {init_reserv:.2f} GB")

    if USE_PICKLE_FILES:
        print(f"  Loading initial conditions from pickle for member {member}...")
        input_state = load_input_state_from_pickle(member, PICKLE_INPUT_DIR)
        date = input_state["date"]
    else:
        print(f"  Retrieving initial conditions for member {member}...")
        fields = get_input_fields(date, member)
        input_state = dict(date=date, fields=fields)

    date_str = date.strftime("%Y%m%d_%H%M")

    runner.time_step = 6
    runner.lead_time = lead_time
    runner.reference_date = date

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

    with torch.no_grad():
        for state in runner.run(input_state=input_state, lead_time=lead_time):
            drop_unencodable_wave_dir(state)   # cos_mwd/sin_mwd are not GRIB params
            if current_file_step == 0:
                if grib_output is not None:
                    grib_output.close()
                    outputs_initialized = False
                start_hour = step_count * runner.time_step
                end_hour = min(start_hour + hours_per_file, lead_time)
                grib_file = (f"{output_dir}/fp16_v2_forecast_{date_str}_"
                             f"member{member:03d}_h{start_hour:03d}-{end_hour:03d}.grib")
                grib_output = GribFileOutput(runner, path=grib_file)

            if not outputs_initialized:
                grib_output.open(state)
                outputs_initialized = True

            grib_output.write_step(state)
            step_count += 1
            current_file_step += 1

            alloc, reserv, current = log_memory_stats(member, output_dir, step=step_count)
            memory_readings.append({"step": step_count, "hour": step_count * 6,
                                    "allocated": alloc, "reserved": reserv,
                                    "current": current})
            if current > max_memory_step:
                max_memory_step = current
            if current > MEMORY_THRESHOLD_GB:
                exceeded_threshold = True
                status = "EXCEEDED"
            else:
                status = "OK"
            print(f"    Step {step_count:2d} ({step_count * 6:3d}h): "
                  f"Current={current:.2f}GB, Peak={alloc:.2f}GB [{status}]")

            if current_file_step >= steps_per_file or step_count * runner.time_step >= lead_time:
                grib_output.close()
                outputs_initialized = False
                if os.path.exists(grib_file):
                    file_size = os.path.getsize(grib_file) / (1024 * 1024)
                    created_files.append(os.path.basename(grib_file))
                    total_size += file_size
                current_file_step = 0

    if grib_output is not None and outputs_initialized:
        grib_output.close()

    torch.cuda.empty_cache()
    final_alloc = torch.cuda.max_memory_allocated() / 1024**3
    final_reserv = torch.cuda.max_memory_reserved() / 1024**3
    log_memory_stats(member, output_dir, step="final")

    print(f"\n  {'=' * 50}")
    print(f"  Member {member} FP16 (v2) Memory Summary:")
    print(f"  {'=' * 50}")
    print(f"    Peak Allocated: {final_alloc:.2f} GB")
    print(f"    Peak Reserved: {final_reserv:.2f} GB")
    print(f"    Max During Inference: {max_memory_step:.2f} GB")
    print(f"    Threshold (24GB): {MEMORY_THRESHOLD_GB} GB")
    if exceeded_threshold:
        print(f"    Status: FAILED - Memory exceeded {MEMORY_THRESHOLD_GB} GB")
    else:
        print(f"    Status: PASSED - within {MEMORY_THRESHOLD_GB} GB "
              f"(margin {MEMORY_THRESHOLD_GB - max_memory_step:.2f} GB)")

    return {
        "member": member,
        "success": len(created_files) > 0,
        "file_size_mb": total_size,
        "peak_allocated_gb": final_alloc,
        "peak_reserved_gb": final_reserv,
        "max_inference_gb": max_memory_step,
        "steps": step_count,
        "exceeded_threshold": exceeded_threshold,
        "memory_readings": memory_readings,
    }


def print_gpu_info():
    """Print GPU device information."""
    if not torch.cuda.is_available():
        print("No CUDA device available!")
        return 0
    props = torch.cuda.get_device_properties(torch.cuda.current_device())
    total_gb = props.total_memory / 1024**3
    print(f"\nGPU Information:")
    print(f"  Device: {props.name}")
    print(f"  Total Memory: {total_gb:.2f} GB")
    print(f"  CUDA Version: {torch.version.cuda}")
    print(f"  PyTorch Version: {torch.__version__}")
    if total_gb < 24:
        print(f"  WARNING: GPU has less than 24GB VRAM!")
    elif total_gb < 40:
        print(f"  Target GPU class: 24GB (A10G, RTX 4090, RTX 3090)")
    else:
        print(f"  Note: GPU has {total_gb:.0f}GB - testing 24GB compatibility")
    return total_gb


def write_memory_plot(output_dir, results, gpu_memory_gb, precision, num_chunks):
    """Render a per-step VRAM timeline PNG from the collected memory readings.

    One line per metric (current / peak-allocated / peak-reserved) per member,
    with the GPU-total and threshold drawn as reference lines. Saved as a PNG so
    it can drop straight into docs (complements the interactive
    ``*_memory_snapshot.pickle`` for pytorch.org/memory_viz).
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  (matplotlib not installed — skipping PNG; pip install matplotlib)")
        return None

    plotted = [r for r in results if r.get("memory_readings")]
    if not plotted:
        print("  (no per-step readings — skipping PNG)")
        return None

    fig, ax = plt.subplots(figsize=(10, 6))
    for r in plotted:
        hrs = [m["hour"] for m in r["memory_readings"]]
        ax.plot(hrs, [m["current"] for m in r["memory_readings"]],
                marker="o", ms=3, label=f"m{r['member']} current")
        ax.plot(hrs, [m["allocated"] for m in r["memory_readings"]],
                marker="^", ms=3, ls="--", label=f"m{r['member']} peak-alloc")
        ax.plot(hrs, [m["reserved"] for m in r["memory_readings"]],
                marker="s", ms=3, ls=":", label=f"m{r['member']} peak-reserved")
    ax.axhline(gpu_memory_gb, color="red", lw=1.2, label=f"GPU total {gpu_memory_gb:.1f} GB")
    ax.axhline(MEMORY_THRESHOLD_GB, color="orange", lw=1.0, ls="--",
               label=f"threshold {MEMORY_THRESHOLD_GB:.0f} GB")
    ax.set_xlabel("Forecast lead time (hours)")
    ax.set_ylabel("GPU memory (GB)")
    ax.set_title(f"AIFS-ENS-2.0 FP{precision} VRAM — {num_chunks} chunks "
                 f"({plotted[0].get('peak_reserved_gb', 0):.1f} GB peak reserved)")
    ax.set_ylim(0, max(gpu_memory_gb, MEMORY_THRESHOLD_GB) * 1.05)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7, ncol=2, loc="upper left")
    fig.tight_layout()
    plot_path = os.path.join(output_dir, PLOT_FILE)
    fig.savefig(plot_path, dpi=130)
    plt.close(fig)
    print(f"Memory timeline PNG: {plot_path}")
    return plot_path


def write_summary(output_dir, results, total_time, gpu_memory_gb, precision, num_chunks):
    """Write profiling summary to file."""
    summary_file = os.path.join(output_dir, SUMMARY_FILE)
    # PASS only if at least one member ran AND every member both completed
    # (produced output) and stayed under threshold. An empty results list — or a
    # member that errored before any step — is a FAIL, not a silent PASS.
    all_passed = bool(results) and all(
        r.get("success") and not r["exceeded_threshold"] for r in results)

    with open(summary_file, "w") as f:
        f.write("=" * 70 + "\n")
        f.write("AIFS ENS v2.0 - FP16 Memory Profiling Summary\n")
        f.write("=" * 70 + "\n\n")
        if all_passed:
            f.write("*  STATUS: PASSED - Inference fits within 24GB VRAM\n\n")
        else:
            f.write("!  STATUS: FAILED - Memory exceeded 24GB threshold\n\n")

        f.write(f"Date: {datetime.datetime.now().isoformat()}\n")
        f.write(f"Checkpoint: ecmwf/aifs-ens-2.0\n")
        f.write(f"GPU Total Memory: {gpu_memory_gb:.2f} GB\n")
        f.write(f"PyTorch Version: {torch.__version__}\n")
        f.write(f"CUDA Version: {torch.version.cuda}\n\n")

        f.write("Optimization Configuration:\n")
        f.write("-" * 70 + "\n")
        f.write(f"  Precision: FP{precision} ({'half' if precision == '16' else 'full'})\n")
        f.write(f"  ANEMOI_INFERENCE_NUM_CHUNKS: {num_chunks}\n")
        f.write(f"  PYTORCH_CUDA_ALLOC_CONF: expandable_segments:True\n")
        f.write(f"  Target Memory: <= {MEMORY_THRESHOLD_GB} GB\n\n")

        f.write("Per-Member Results:\n")
        f.write("-" * 70 + "\n")
        for r in results:
            status = "PASS" if not r["exceeded_threshold"] else "FAIL"
            f.write(f"  Member {r['member']} [{status}]: "
                    f"peak_alloc={r['peak_allocated_gb']:.2f}GB "
                    f"peak_reserved={r['peak_reserved_gb']:.2f}GB "
                    f"max_infer={r['max_inference_gb']:.2f}GB "
                    f"steps={r['steps']} out={r['file_size_mb']:.1f}MB\n")

        if results:
            max_inference = max(r["max_inference_gb"] for r in results)
            f.write("\nAggregate:\n")
            f.write("-" * 70 + "\n")
            f.write(f"  Maximum During Inference: {max_inference:.2f} GB "
                    f"(threshold {MEMORY_THRESHOLD_GB} GB)\n")
            if max_inference <= MEMORY_THRESHOLD_GB:
                f.write(f"  CONCLUSION: Safe for 24GB GPUs "
                        f"(margin {MEMORY_THRESHOLD_GB - max_inference:.2f} GB)\n")
            else:
                f.write(f"  RECOMMENDATION: raise NUM_CHUNKS to 32/64 "
                        f"(excess {max_inference - MEMORY_THRESHOLD_GB:.2f} GB)\n")

        f.write("\nOutput Files:\n")
        f.write("-" * 70 + "\n")
        f.write(f"  Memory Snapshot: {SNAPSHOT_FILE}\n")
        f.write(f"  Memory Log CSV: {MEMORY_LOG_FILE}\n")
        f.write(f"  This Summary: {SUMMARY_FILE}\n")

    print(f"\nSummary written to: {summary_file}")
    return all_passed


def main():
    parser = argparse.ArgumentParser(
        description="Profile FP16 AIFS ENS v2.0 inference for 24GB GPUs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick test (FP16 + 16 chunks), one member staged at --pickle-dir
  python pytorch_profile_fp16_v2.py --chunks 16 --members 1 --lead-time 72

  # More aggressive chunking
  python pytorch_profile_fp16_v2.py --chunks 32

  # Live download instead of a staged pkl
  python pytorch_profile_fp16_v2.py --no-pickle --date 20260611

Target GPUs (24GB VRAM): A10G, RTX 4090, RTX 3090.
Note: aifs-ens-2.0 is larger than v1 — re-measure; v1 fit ~23 GB at 16 chunks.
        """
    )
    parser.add_argument("--members", type=int, nargs="+", default=DEFAULT_ENSEMBLE_MEMBERS,
                        help="Ensemble members to profile (default: 1)")
    parser.add_argument("--lead-time", type=int, default=DEFAULT_LEAD_TIME,
                        help="Forecast lead time in hours (default: 72)")
    parser.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR,
                        help="Output directory for profiles")
    parser.add_argument("--chunks", type=int, default=DEFAULT_NUM_CHUNKS,
                        help="ANEMOI_INFERENCE_NUM_CHUNKS value (default: 16)")
    parser.add_argument("--precision", type=str, default=DEFAULT_PRECISION,
                        choices=["16", "32"],
                        help="Inference precision (default: 16 for FP16)")
    parser.add_argument("--date", type=str, default=None,
                        help="Forecast date YYYYMMDD (default: 20251127)")
    parser.add_argument("--pickle-dir", type=str, default="/scratch/input_states",
                        help="Directory containing v2 input pickle files")
    parser.add_argument("--no-pickle", action="store_true",
                        help="Download data live instead of using pickle files")
    parser.add_argument("--threshold", type=float, default=23.0,
                        help="Memory threshold in GB (default: 23.0 for 24GB GPU)")
    args = parser.parse_args()

    global USE_PICKLE_FILES, PICKLE_INPUT_DIR, MEMORY_THRESHOLD_GB
    USE_PICKLE_FILES = not args.no_pickle
    PICKLE_INPUT_DIR = args.pickle_dir
    MEMORY_THRESHOLD_GB = args.threshold

    DATE = (datetime.datetime.strptime(args.date, "%Y%m%d") if args.date
            else datetime.datetime(2025, 11, 27, 0, 0))

    print("=" * 70)
    print("AIFS ENS v2.0 - FP16 Memory Profiling (24GB GPU compatibility)")
    print("=" * 70)
    print(f"\nForecast Date: {DATE}")
    print(f"Members to profile: {args.members}")
    print(f"Lead time: {args.lead_time} hours")
    print(f"Output directory: {args.output_dir}")

    gpu_memory_gb = print_gpu_info()

    print("\n" + "-" * 70)
    print("Applying Memory Optimizations:")
    print(f"  ANEMOI_INFERENCE_NUM_CHUNKS = {args.chunks}")
    print(f"  PYTORCH_CUDA_ALLOC_CONF = expandable_segments:True")
    print(f"  Precision = FP{args.precision}")
    print(f"  Memory Threshold = {MEMORY_THRESHOLD_GB} GB")

    os.environ["ANEMOI_INFERENCE_NUM_CHUNKS"] = str(args.chunks)
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    os.makedirs(args.output_dir, exist_ok=True)

    print("\n" + "=" * 70)
    print(f"Loading model ecmwf/aifs-ens-2.0 with FP{args.precision} precision...")
    print("=" * 70)
    checkpoint = {"huggingface": "ecmwf/aifs-ens-2.0"}
    if args.precision == "16":
        runner = SimpleRunner(checkpoint, device="cuda", precision="16")
        print("Model loaded with FP16 (half precision) - optimized for 24GB GPUs")
    else:
        runner = SimpleRunner(checkpoint, device="cuda")
        print("Model loaded with FP32 (full precision)")

    print("\nStarting PyTorch CUDA memory history recording...")
    torch.cuda.memory._record_memory_history(max_entries=100000)

    results = []
    start_time = time.time()
    for member in args.members:
        print(f"\n{'=' * 70}")
        print(f"Profiling member {member} (FP{args.precision} + {args.chunks} chunks)")
        print(f"{'=' * 70}")
        try:
            results.append(run_ensemble_member_fp16(
                runner, DATE, member, args.output_dir, args.lead_time))
        except torch.cuda.OutOfMemoryError as e:
            print(f"\nOUT OF MEMORY for member {member}: {e}")
            print(f"Recommendation: increase --chunks to 32 or 64")
            results.append({"member": member, "success": False, "exceeded_threshold": True,
                            "peak_allocated_gb": gpu_memory_gb, "peak_reserved_gb": gpu_memory_gb,
                            "max_inference_gb": gpu_memory_gb, "steps": 0, "file_size_mb": 0,
                            "memory_readings": []})
        except Exception as e:
            print(f"Error profiling member {member}: {e}")
            import traceback
            traceback.print_exc()
            # Record the failure so it surfaces in the summary / exit code,
            # instead of silently leaving `results` empty (which read as PASS).
            results.append({"member": member, "success": False,
                            "exceeded_threshold": True, "peak_allocated_gb": 0.0,
                            "peak_reserved_gb": 0.0, "max_inference_gb": 0.0,
                            "steps": 0, "file_size_mb": 0, "memory_readings": [],
                            "error": str(e)})
            continue

    total_time = time.time() - start_time

    print("\nSaving PyTorch memory snapshot...")
    snapshot_path = os.path.join(args.output_dir, SNAPSHOT_FILE)
    try:
        torch.cuda.memory._dump_snapshot(snapshot_path)
        print(f"Saved: {snapshot_path}")
    except Exception as e:
        # _dump_snapshot can raise (e.g. RuntimeError: stoi) under
        # expandable_segments:True or when no allocations were recorded. The
        # PNG/CSV already capture the profile, so don't fail the run over this.
        print(f"  (snapshot dump skipped: {e})")
    torch.cuda.memory._record_memory_history(enabled=None)

    all_passed = write_summary(args.output_dir, results, total_time, gpu_memory_gb,
                               args.precision, args.chunks)
    write_memory_plot(args.output_dir, results, gpu_memory_gb, args.precision, args.chunks)

    print("\n" + "=" * 70)
    if all_passed:
        print("TEST PASSED: FP16 v2.0 inference fits within 24GB VRAM")
        if results:
            max_mem = max(r["max_inference_gb"] for r in results)
            print(f"  Peak memory: {max_mem:.2f} GB | "
                  f"margin: {MEMORY_THRESHOLD_GB - max_mem:.2f} GB | safe for A10G/4090/3090")
    else:
        print("TEST FAILED: Memory exceeded 24GB threshold")
        print("  Try: --chunks 32 (or 64), or inspect the snapshot for leaks")
    print(f"\nTotal time: {total_time:.2f} seconds")
    print(f"Output files in {args.output_dir}/: {SNAPSHOT_FILE}, {MEMORY_LOG_FILE}, "
          f"{SUMMARY_FILE}, {PLOT_FILE}")

    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()

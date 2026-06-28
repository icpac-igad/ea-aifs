#!/usr/bin/env python3
"""Minimal AIFS-ENS-2.0 FP16 load test: run the full forecast loop on the GPU
and log peak VRAM per step. Output is consumed (not GRIB-encoded) so the test
isolates compute + memory from the cos_mwd/sin_mwd GRIB-param issue."""
import os, sys, time, pickle, argparse
import numpy as np
import torch

os.environ.setdefault("ANEMOI_INFERENCE_NUM_CHUNKS", "16")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

from anemoi.inference.runners.simple import SimpleRunner

ap = argparse.ArgumentParser()
ap.add_argument("--pkl", default="/scratch/input_states/input_state_member_001.pkl")
ap.add_argument("--lead-time", type=int, default=72)
ap.add_argument("--chunks", type=int, default=16)
ap.add_argument("--precision", default="16")
ap.add_argument("--threshold", type=float, default=23.0)
args = ap.parse_args()
os.environ["ANEMOI_INFERENCE_NUM_CHUNKS"] = str(args.chunks)

props = torch.cuda.get_device_properties(0)
total_gb = props.total_memory / 1024**3
print(f"GPU: {props.name} | total {total_gb:.2f} GB | torch {torch.__version__} | cuda {torch.version.cuda}")
print(f"Config: FP{args.precision} | chunks {args.chunks} | lead_time {args.lead_time}h | threshold {args.threshold} GB")

with open(args.pkl, "rb") as f:
    input_state = pickle.load(f)
date = input_state["date"]
print(f"Input: {len(input_state['fields'])} fields | date {date}")

torch.cuda.reset_peak_memory_stats()
torch.cuda.empty_cache()

print(f"Loading ecmwf/aifs-ens-2.0 (FP{args.precision})...")
runner = SimpleRunner({"huggingface": "ecmwf/aifs-ens-2.0"}, device="cuda", precision=args.precision)
runner.time_step = 6
runner.lead_time = args.lead_time
runner.reference_date = date

peak_step = 0.0
exceeded = False
t0 = time.time()
n = 0
with torch.no_grad():
    for state in runner.run(input_state=input_state, lead_time=args.lead_time):
        n += 1
        cur = torch.cuda.memory_allocated() / 1024**3
        alloc = torch.cuda.max_memory_allocated() / 1024**3
        reserv = torch.cuda.max_memory_reserved() / 1024**3
        peak_step = max(peak_step, cur)
        flag = "EXCEEDED" if cur > args.threshold else "OK"
        if cur > args.threshold:
            exceeded = True
        print(f"  Step {n:2d} ({n*6:3d}h): current={cur:.2f}GB peak_alloc={alloc:.2f}GB "
              f"peak_reserved={reserv:.2f}GB [{flag}]", flush=True)
        # consume/discard output — no GRIB encoding
        _ = next(iter(state["fields"].values()))

dt = time.time() - t0
alloc = torch.cuda.max_memory_allocated() / 1024**3
reserv = torch.cuda.max_memory_reserved() / 1024**3
print("=" * 60)
print(f"STEPS COMPLETED : {n} (expected {args.lead_time // 6})")
print(f"PEAK ALLOCATED  : {alloc:.2f} GB")
print(f"PEAK RESERVED   : {reserv:.2f} GB")
print(f"MAX CURRENT     : {peak_step:.2f} GB")
print(f"GPU TOTAL       : {total_gb:.2f} GB")
print(f"WALL TIME       : {dt:.1f} s  ({dt/max(n,1):.2f} s/step)")
ok = (n == args.lead_time // 6) and not exceeded and reserv <= total_gb
print(f"RESULT          : {'PASS' if ok else 'FAIL'} "
      f"(reserved {reserv:.2f} vs threshold {args.threshold} / total {total_gb:.2f} GB)")
sys.exit(0 if ok else 1)

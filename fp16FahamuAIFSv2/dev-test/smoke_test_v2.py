#!/usr/bin/env python3
"""Smoke test: download one member's v2 input pkl from GCS, run aifs-ens-2.0 (FP16)
for a short lead time, and report VRAM + output GRIB — WITHOUT uploading anything.

Validates the v2 GPU env end-to-end (checkpoint load, rollout, GRIB encode incl. the
cos_mwd/sin_mwd drop) before the full 50-member / 960h production run. This is the
script used to bring up the v2 env on the L4 (see ../LOAD_TEST_RESULTS.md).

    # default: member 1 of 20260625_0000, 72h, 16 chunks
    python dev-test/smoke_test_v2.py
    python dev-test/smoke_test_v2.py --date 20260625_0000 --member 1 --lead-time 72 --chunks 16

Needs an Ampere+ GPU and the v2 software env (torch 2.7 / anemoi-inference 0.8.3 /
flash-attn 2.7.4 / earthkit-regrid 0.5.1), plus coiled-data.json (one dir up).
Set HF_HOME and CC=/usr/bin/gcc before running.
"""
import argparse
import glob
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
V2 = os.path.dirname(HERE)
SHARED = os.path.join(os.path.dirname(V2), "shared")
sys.path.insert(0, SHARED)
sys.path.insert(0, V2)

from download_pkl_from_gcs import (  # noqa: E402
    download_from_gcs, verify_pickle_file, GCS_BUCKET,
)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--date", default="20260625_0000")
    p.add_argument("--member", type=int, default=1)
    p.add_argument("--lead-time", type=int, default=72)
    p.add_argument("--chunks", type=int, default=16)
    p.add_argument("--bucket", default=GCS_BUCKET)
    p.add_argument("--gcs-input-subpath", default="input_v2")
    p.add_argument("--service-account", default=os.path.join(V2, "coiled-data.json"))
    p.add_argument("--input-dir", default="/scratch/input_states")
    p.add_argument("--output-dir", default="/scratch/smoke_v2")
    args = p.parse_args()

    os.makedirs(args.input_dir, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)
    os.environ["ANEMOI_INFERENCE_NUM_CHUNKS"] = str(args.chunks)
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    # 1. download the member's input pkl
    blob = f"{args.date}/{args.gcs_input_subpath}/input_state_member_{args.member:03d}.pkl"
    local = os.path.join(args.input_dir, f"input_state_member_{args.member:03d}.pkl")
    if not (os.path.exists(local) and verify_pickle_file(local)[0]):
        print(f"[DL] {blob} -> {local}")
        if not download_from_gcs(args.bucket, blob, local, args.service_account):
            raise SystemExit("download failed")
    ok, msg = verify_pickle_file(local)
    print(f"[DL] pkl valid: {ok} ({msg})")

    # 2. load model + run a short forecast (no upload)
    import fp16_multi_run_AIFS_ENS_v2 as m
    m.PICKLE_INPUT_DIR = args.input_dir
    m.LEAD_TIME = args.lead_time
    from anemoi.inference.runners.simple import SimpleRunner
    import torch

    print(f"[MODEL] loading ecmwf/aifs-ens-2.0 (FP16, {args.chunks} chunks)...")
    t0 = time.time()
    runner = SimpleRunner({"huggingface": "ecmwf/aifs-ens-2.0"}, device="cuda", precision="16")
    print(f"[MODEL] loaded in {time.time()-t0:.1f}s")

    t1 = time.time()
    success, size = m.run_ensemble_member(runner, None, args.member, args.output_dir)
    dt = time.time() - t1

    print(f"\n[RESULT] success={success} size={size:.1f}MB elapsed={dt:.1f}s")
    print(f"[VRAM] max_allocated={torch.cuda.max_memory_allocated()/1e9:.2f}GB "
          f"max_reserved={torch.cuda.max_memory_reserved()/1e9:.2f}GB")
    files = sorted(glob.glob(os.path.join(args.output_dir, f"*member{args.member:03d}*.grib")))
    print(f"[FILES] {len(files)} grib files:")
    for f in files:
        print("   ", os.path.basename(f), f"{os.path.getsize(f)/1e6:.1f}MB")
    sys.exit(0 if success and files else 1)


if __name__ == "__main__":
    main()

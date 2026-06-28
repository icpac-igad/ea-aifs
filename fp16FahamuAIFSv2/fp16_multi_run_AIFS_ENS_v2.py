#!/usr/bin/env python3
"""
Multi-Run AIFS ENS v2.0 — FP16 inference for ensemble members.

Runs ECMWF's **aifs-ens-2.0** data-driven model (checkpoint ``ecmwf/aifs-ens-2.0``)
for multiple ensemble members using the anemoi-inference package, writing one GRIB
file per 72-hour window per member.

This is the v2 counterpart of ``fp16FahamuAIFSv1/fp16_multi_run_AIFS_ENS_v1.py``.
Differences:
  * Checkpoint ``ecmwf/aifs-ens-2.0`` (vs v1).
  * v2 input contract (112 fields): +snow depth, +soil moisture, +wave group
    (``mwd``→``cos_mwd``/``sin_mwd``), pressure level 10 hPa, ``q_10`` dropped,
    land-sea mask on ``sd``/``swvl1``/``swvl2``.
  * No ERA5T 74→92 adaptation (that path is v1/era5t-only).

Operationally the input is read from a pre-built pkl (the 112-field state produced by
``ecmwf_opendata_pkl_input_aifsens_v2.py``); live download is a fallback.

NOTE: the inference stack (anemoi-inference 0.8.3, anemoi-models 0.11.2, torch 2.7,
flash-attn 2.7.4) only exists on the GPU image — see fp16FahamuAIFSv2.md. The
``runner.run`` / ``GribFileOutput`` / ``SimpleRunner`` calls below mirror the v1 runner
and must be smoke-tested once on the v2 GPU env (anemoi API may have drifted).
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

from anemoi.inference.runners.simple import SimpleRunner
from anemoi.inference.outputs.printer import print_state
from anemoi.inference.outputs.gribfile import GribFileOutput

from ecmwf.opendata import Client as OpendataClient

# ---------------------------------------------------------------------------
# v2.0 parameter lists (from run_AIFS_ENS_v2.0.ipynb)
# ---------------------------------------------------------------------------
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

# Multi-run configuration (overridden by the orchestrator)
ENSEMBLE_MEMBERS = list(range(1, 10))
LEAD_TIME = 960  # Hours (40 days)
OUTPUT_DIR = "/scratch/ensemble_outputs"

# Pickle file configuration
USE_PICKLE_FILES = True
PICKLE_INPUT_DIR = "/scratch/input_states"

# GPU Memory Optimization
GPU_MEMORY_OPTIMIZATION = True
INFERENCE_PRECISION = "16"        # "16" = FP16, "32" = FP32
INFERENCE_NUM_CHUNKS = 16         # higher -> less VRAM, slower (tune for v2)


def get_open_data(date, param, levelist=[], number=None, constant=False,
                  source="ecmwf", **kwargs):
    """Retrieve + N320-interpolate ECMWF Open Data (v2 live-mode fallback).

    Mirrors ``ecmwf_opendata_pkl_input_aifsens_v2.get_open_data``: stacks
    [date-6h, date]; constants are fetched once and replicated.
    """
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
    Concept no match``. Recombining into ``mwd`` doesn't help — anemoi's
    ``GribFileOutput`` only writes variables the checkpoint declares
    (``typed_variables``), and ``mwd`` is not one (``KeyError: 'mwd'``).

    So we drop the two components; every other output field (incl. the real wave
    params ``swh``/``mwp``/``cdww``) encodes fine, and Step 3 only extracts
    ``tp``/``msl``/``2t``. Recover wave direction if needed as
    ``degrees(atan2(sin, cos))`` from the raw output. The yielded state is a
    fresh per-step dict, so popping does not affect the autoregressive rollout.
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
    print(f"    ✅ Loaded pickle: {pickle_file}")
    print(f"    Date: {input_state['date']} | Fields: {len(input_state['fields'])}")
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


def run_ensemble_member(runner, date, member, output_dir):
    """Run forecast for one member, saving a GRIB file every 72 hours.

    Identical control flow to the v1 runner — only the input contract changed.
    """
    if USE_PICKLE_FILES:
        print(f"Loading initial conditions from pickle for member {member}...")
        input_state = load_input_state_from_pickle(member, PICKLE_INPUT_DIR)
        date = input_state["date"]
    else:
        print(f"Retrieving initial conditions for member {member}...")
        fields = get_input_fields(date, member)
        input_state = dict(date=date, fields=fields)

    date_str = date.strftime("%Y%m%d_%H%M")

    # Context properties on the runner (extends Context)
    runner.time_step = 6
    runner.lead_time = LEAD_TIME
    runner.reference_date = date

    hours_per_file = 72
    steps_per_file = hours_per_file // runner.time_step  # 72/6 = 12

    created_files = []
    total_size = 0

    print(f"Running forecast for member {member}...")
    step_count = 0
    current_file_step = 0
    grib_output = None
    outputs_initialized = False

    for state in runner.run(input_state=input_state, lead_time=LEAD_TIME):
        drop_unencodable_wave_dir(state)   # cos_mwd/sin_mwd are not GRIB params
        if current_file_step == 0:
            if grib_output is not None:
                grib_output.close()
                outputs_initialized = False
            start_hour = step_count * runner.time_step
            end_hour = min(start_hour + hours_per_file, LEAD_TIME)
            grib_file = (f"{output_dir}/aifs_ens_forecast_{date_str}_"
                         f"member{member:03d}_h{start_hour:03d}-{end_hour:03d}.grib")
            grib_output = GribFileOutput(runner, path=grib_file)
            print(f"  Creating file: {os.path.basename(grib_file)}")

        if not outputs_initialized:
            grib_output.open(state)
            outputs_initialized = True

        grib_output.write_step(state)
        step_count += 1
        current_file_step += 1

        if step_count % 4 == 0:
            print(f"  Member {member}: {step_count * 6} hours completed")

        if current_file_step >= steps_per_file or step_count * runner.time_step >= LEAD_TIME:
            grib_output.close()
            outputs_initialized = False
            if os.path.exists(grib_file):
                file_size = os.path.getsize(grib_file) / (1024 * 1024)
                created_files.append(os.path.basename(grib_file))
                total_size += file_size
                print(f"  ✓ Completed: {os.path.basename(grib_file)} ({file_size:.2f} MB)")
            else:
                print(f"  ✗ Error: File not created: {os.path.basename(grib_file)}")
            current_file_step = 0

    if grib_output is not None and outputs_initialized:
        grib_output.close()

    if created_files:
        print(f"✓ Member {member} completed: {len(created_files)} files, "
              f"total {total_size:.2f} MB")
        return True, total_size
    print(f"✗ Error: No output files created for member {member}")
    return False, 0


def main():
    """Standalone multi-member run (the orchestrator is the usual entry point)."""
    DATE = datetime.datetime(2025, 11, 27, 0, 0)
    print("Initial date is", DATE)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if GPU_MEMORY_OPTIMIZATION:
        print(f"GPU memory optimization: ON | "
              f"{'FP16' if INFERENCE_PRECISION == '16' else 'FP32'} | "
              f"chunks={INFERENCE_NUM_CHUNKS}")
        os.environ["ANEMOI_INFERENCE_NUM_CHUNKS"] = str(INFERENCE_NUM_CHUNKS)
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    print("\nLoading model ecmwf/aifs-ens-2.0 ...")
    checkpoint = {"huggingface": "ecmwf/aifs-ens-2.0"}
    if GPU_MEMORY_OPTIMIZATION and INFERENCE_PRECISION == "16":
        runner = SimpleRunner(checkpoint, device="cuda", precision="16")
    else:
        runner = SimpleRunner(checkpoint, device="cuda")
    print("Model loaded.\n")

    times, ok = [], []
    for member in ENSEMBLE_MEMBERS:
        t0 = time.time()
        print(f"\n{'='*60}\nMember {member}\n{'='*60}")
        try:
            success, _ = run_ensemble_member(runner, DATE, member, OUTPUT_DIR)
            if success:
                ok.append(member)
        except Exception as e:
            print(f"✗ member {member} failed: {e}")
            continue
        times.append(time.time() - t0)
        print(f"member {member}: {times[-1]:.1f}s")

    print(f"\n{'='*60}\nDone: {len(ok)}/{len(ENSEMBLE_MEMBERS)} members"
          + (f", total {sum(times)/60:.1f} min" if times else ""))


if __name__ == "__main__":
    main()

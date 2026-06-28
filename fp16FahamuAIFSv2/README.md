# fp16FahamuAIFSv2 (AIFS ENS v2.0 / FP16 / ECMWF Open Data)

New model version targeting **AIFS-ENS-2.0** (checkpoint `ecmwf/aifs-ens-2.0`), run
at FP16. This folder implements the **v2.0 input data preparation** (Step 1) and the
**v2 GPU runner + orchestrator** (Step 2). Post-processing/submission (Steps 3–5) reuse
the `shared/` CLIs once v2 GRIB output exists. The Step 2 code is written but **not yet
smoke-tested on a GPU** (the inference stack only exists on the GPU image — see the
GPU-runner caveats below).

Reference notebook: `run_AIFS_ENS_v2.0.ipynb`
(https://huggingface.co/ecmwf/aifs-ens-2.0/blob/main/run_AIFS_ENS_v2.0.ipynb).

## Scoping — what changed vs `*AIFSv1`

Compared against `shared/ecmwf_opendata_pkl_input_aifsens.py`:

| Group | v1 | **v2.0** |
|-------|----|----------|
| Surface | `10u,10v,2d,2t,msl,skt,sp,tcw` | **+ `sd`** (snow depth) |
| Constants | `lsm,z,slor,sdor` | same |
| Soil | `sot` → `stl1,stl2` | **`vsw,sot`** → `stl1,stl2,**swvl1,swvl2**` |
| Wave | *(none)* | **11 new** `wmb,h1012,h1214,h1417,h1721,h2125,h2530,mwd,cdww,mwp,swh` (stream `waef`/`wave`) |
| Pressure levels | 13 (`1000…50`) | **14 (`1000…10`)** |
| PL params | `gh,t,u,v,w,q` | same, **drop `q_10`** |
| Transforms | `gh→z` | `gh→z` **+ `mwd→cos_mwd/sin_mwd`** **+ LSM-mask `sd,swvl1,swvl2`→NaN over sea** |
| **Field count** | ~92 | **112** (verified) |
| Inference env | regrid 0.4.0 / anemoi (v1) | regrid **0.5.1**, anemoi-models **0.11.2**, anemoi-inference **0.8.3**, torch **2.7**, flash-attn **2.7.4** |
| Checkpoint | `ecmwf/aifs-ens-1.0` | **`ecmwf/aifs-ens-2.0`** |

Retrieval mechanics are unchanged: download 0.25° open data over `[date-6h, date]`,
roll −180..180 → 0..360, interpolate to **N320** with `earthkit-regrid`, stack the two
timesteps. Constants (`lsm,z,slor,sdor`) are still fetched once and replicated.

## Scripts

| Script | Step | Role |
|--------|------|------|
| `ecmwf_opendata_pkl_input_aifsens_v2.py` | 1 (CPU) | ECMWF Open Data → 112-field input-state pkl per member, upload to GCS |
| `fp16_automate_aifs_gpu_pipeline_v2.py` | 2 (GPU) | Orchestrator: per-member download → FP16 inference → upload → cleanup. Loads `ecmwf/aifs-ens-2.0` once. |
| `fp16_multi_run_AIFS_ENS_v2.py` | 2 (GPU) | AIFS-ENS-2.0 FP16 runner (`run_ensemble_member`), 72h-chunked GRIB. Imported by the orchestrator. |
| `pytorch_profile_fp16_v2.py` | 2 (GPU) | VRAM profiler for aifs-ens-2.0 (FP16 + chunks); PyTorch CUDA memory snapshot. See *GPU Memory Profiling*. |

The 112-field set: 9 surface + 4 constants + 4 soil (`stl1/2`,`swvl1/2`) + 12 wave
(`mwd`→`cos_mwd/sin_mwd`) + 83 pressure-level (6 params × 14 levels − `q_10`).

## Run

```bash
# --- Step 1: input prep (CPU / ETL machine) ---
# Latest open-data date, all 50 members, upload to gs://…/<date>/input_v2/
python fp16FahamuAIFSv2/ecmwf_opendata_pkl_input_aifsens_v2.py --members 1-50
# Pin a date / subset / skip upload while testing:
python fp16FahamuAIFSv2/ecmwf_opendata_pkl_input_aifsens_v2.py \
    --date 20260611 --members 1 --no-upload --keep-local

# --- Step 2: GPU inference (Ampere+ GPU, v2 software env) ---
python fp16FahamuAIFSv2/fp16_automate_aifs_gpu_pipeline_v2.py \
    --date 20260611_0000 --members 1-50 --lead-time 960
# reads  gs://…/20260611_0000/input_v2/  ->  writes gs://…/20260611_0000/fp16_v2_forecasts/
```

- **Step 1 output:** `gs://aifs-aiquest-us-20251127/<date>_0000/input_v2/input_state_member_NNN.pkl`
  (kept separate from v1's `input/` so both versions coexist). Requires `coiled-data.json`
  unless `--no-upload`. CPU only. `--source` picks the mirror (`ecmwf`/`azure`/`aws`/`google`).
- **Step 2 output:** `gs://…/<date>_0000/fp16_v2_forecasts/aifs_ens_forecast_<date>_memberNNN_h*.grib`.
  Needs an Ampere+ GPU and the v2 software env (below).
- **Steps 3–5:** reuse `shared/aifs_n320_grib_1p5defg_nc_cli.py` (point
  `--gcs-input-subpath fp16_v2_forecasts`), then the `shared/` quintile + submission CLIs
  (submit under a new `AIWQ_MODEL_NAME_V2`).

## GPU software environment (Step 2)

The inference stack is **not** in the `aifs-etl` ETL env — build a dedicated GPU
software env / coiled image pinned to the notebook versions:

```
anemoi-inference==0.8.3  anemoi-models==0.11.2  anemoi-utils==0.4.35.post3
torch==2.7.0  torch-geometric==2.6.1
earthkit-regrid==0.5.1  ecmwf-opendata==0.3.29  'earthkit-data<1.0.0'
flash-attn==2.7.4.post1  # cu12torch2.7 wheel
```

## GPU Memory Profiling (Step 2)

Before locking a GPU/precision/chunk choice for v2, profile peak VRAM with the PyTorch
CUDA memory snapshot — the method from
[aifs-ens-1.0 Discussion #17](https://huggingface.co/ecmwf/aifs-ens-1.0/discussions/17)
and the ECMWF ["Anemoi Profiling" demo](https://events.ecmwf.int/event/466/timetable/).
**aifs-ens-2.0 is a larger model than v1, so the v1 numbers are only a starting point —
v2 must be re-profiled on its own GPU env.**

Use **`pytorch_profile_fp16_v2.py`** (cloned from the v1 fp16 profiler, swapped to the
`ecmwf/aifs-ens-2.0` checkpoint and the 112-field v2 input). It wraps a single-member
inference between `torch.cuda.memory._record_memory_history()` and `_dump_snapshot()` and
reports `max_memory_allocated` / `max_memory_reserved`.

**Steps**

1. **Stage one v2 input pkl** on the GPU box (so the profiler skips the download):
   ```bash
   gsutil cp gs://aifs-aiquest-us-20251127/<date>_0000/input_v2/input_state_member_001.pkl \
       /scratch/input_states/
   ```
   (or use `--no-pickle` to fetch live).
2. **Run the profiler** for the precision/chunks under test (start at FP16 + 16 chunks):
   ```bash
   python fp16FahamuAIFSv2/pytorch_profile_fp16_v2.py --chunks 16 --members 1 \
       --lead-time 72 --pickle-dir /scratch/input_states --threshold 23.0
   # tighter memory: --chunks 32 (or 64)
   ```
3. **Read the peaks**: the script prints `max_memory_allocated` / `max_memory_reserved`
   (GB) and a summary; it flags a fail if reserved exceeds `--threshold` (default 23 GB)
   and suggests raising `--chunks`.
4. **Inspect the snapshot** at <https://pytorch.org/memory_viz> (upload the emitted
   `*_memory_snapshot.pickle` / `*_gpu_mem.csv`) to see the allocation timeline.
5. **Pick the config**: choose the smallest GPU whose VRAM clears the measured *reserved*
   peak, then set the same `INFERENCE_PRECISION` / `INFERENCE_NUM_CHUNKS` in
   `fp16_automate_aifs_gpu_pipeline_v2.py` (and `fp16_multi_run_AIFS_ENS_v2.py`).

**v1 baseline for contrast (AIFS-ENS v1.0, Discussion #17)** — re-measure for v2:

| Config | Peak allocated | Peak reserved | Fits 24 GB? |
|--------|----------------|---------------|-------------|
| FP32 (full) | — | >34 GB (real total >48 GB w/ CUDA workspace) | ❌ needs A100/H100 |
| FP16 (`precision="16"`) + `NUM_CHUNKS=16` | ~20 GB | ~23 GB | ✅ L4 / A10G / RTX 4090 |

## Caveats / TODO

- **earthkit-regrid:** the v2 notebook pins **0.5.1**; the repo `aifs-etl` env has
  **0.4.0**. The N320 interpolation call is unchanged, but bump the env to 0.5.1 for an
  exact match before a production run.
- **Wave stream availability:** wave fields come from `waef` (ensemble) / `wave`
  (control). If a member 404s on the wave group, check open-data wave availability for
  that cycle.
- **LSM mask:** derived here from the already-N320 `lsm` field (`lsm==0`), instead of the
  notebook's separate `lsm.grib` file — equivalent and avoids a side artifact.
- **GPU runner not yet smoke-tested.** The Step 2 code mirrors the v1 runner's proven
  inference loop, but the anemoi-inference jump (v1 → **0.8.3**) may have drifted APIs.
  Verify on the v2 GPU env, in order:
  1. `SimpleRunner(checkpoint, device="cuda", precision="16")` still accepts those kwargs.
  2. `runner.time_step` / `runner.lead_time` / `runner.reference_date` settable as attrs.
  3. `GribFileOutput(runner, path=…)` + `.open/.write_step/.close` unchanged.
  4. `from anemoi.inference.outputs.printer import print_state` path unchanged.
  5. Re-profile **VRAM** for aifs-ens-2.0 in FP16 (bigger model) — may exceed 24 GB on
     L4 → raise `ANEMOI_INFERENCE_NUM_CHUNKS` or use A100.
- **Confirm v2 output params:** Step 3 extracts `tp`, `msl`, `2t`. aifs-ens-2.0 adds wave
  outputs but should retain these — inspect one member's GRIB before wiring Steps 3–5.
- **Ensemble noise:** v2 is inherently stochastic (model injects noise per run); keep the
  50 IC-perturbed members. Exact reproducibility needs the determinism flags from the
  notebook (`CUBLAS_WORKSPACE_CONFIG`, `torch.use_deterministic_algorithms`).

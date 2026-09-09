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
| `ecmwf_opendata_pkl_input_aifsens_v2.py` | 1 (CPU) | ECMWF Open Data → 112-field input-state pkl per member, upload to GCS. Two backends via `--fetch`: **`index`** (default — one byte range per field, 16 concurrent, eccodes, `google` mirror; ~5× faster) or `earthkit` (original `earthkit-data` path, `aws`/`ecmwf` only). Bit-identical output. |
| `fp16_automate_aifs_gpu_pipeline_v2.py` | 2 (GPU) | Orchestrator: per-member download → FP16 inference → upload → cleanup. Loads `ecmwf/aifs-ens-2.0` once. |
| `fp16_multi_run_AIFS_ENS_v2.py` | 2 (GPU) | AIFS-ENS-2.0 FP16 runner (`run_ensemble_member`), 72h-chunked GRIB. Imported by the orchestrator. |
| `pytorch_profile_fp16_v2.py` | 2 (GPU) | VRAM profiler for aifs-ens-2.0 (FP16 + chunks); PyTorch CUDA memory snapshot. See *GPU Memory Profiling*. |

The 112-field set: 9 surface + 4 constants + 4 soil (`stl1/2`,`swvl1/2`) + 12 wave
(`mwd`→`cos_mwd/sin_mwd`) + 83 pressure-level (6 params × 14 levels − `q_10`).

## Run

```bash
# --- Step 1: input prep (CPU / ETL machine) ---
# All 50 members, upload to gs://…/<date>/input_v2/. --date is REQUIRED
# (latest-date detection was removed); defaults to --fetch index --source google.
python fp16FahamuAIFSv2/ecmwf_opendata_pkl_input_aifsens_v2.py \
    --date 20260611 --members 1-50
# Subset / skip upload while testing:
python fp16FahamuAIFSv2/ecmwf_opendata_pkl_input_aifsens_v2.py \
    --date 20260611 --members 1 --no-upload --keep-local
# Fall back to the original earthkit-data backend (aws/ecmwf only):
python fp16FahamuAIFSv2/ecmwf_opendata_pkl_input_aifsens_v2.py \
    --date 20260611 --members 1-50 --fetch earthkit --source aws

# --- Step 2: GPU inference (Ampere+ GPU, v2 software env) ---
python fp16FahamuAIFSv2/fp16_automate_aifs_gpu_pipeline_v2.py \
    --date 20260611_0000 --members 1-50 --lead-time 960
# reads  gs://…/20260611_0000/input_v2/  ->  writes gs://…/20260611_0000/fp16_v2_forecasts/
```

- **Step 1 output:** `gs://aifs-aiquest-us-20251127/<date>_0000/input_v2/input_state_member_NNN.pkl`
  (kept separate from v1's `input/` so both versions coexist). Requires `coiled-data.json`
  unless `--no-upload`. CPU only.

  **Retrieval backend (`--fetch`).** The default is **`index`**: read the `.index`
  sidecar, then fetch **one single byte range per field**, 16 concurrent
  (`INDEX_MAX_WORKERS`), decoding with eccodes. It defaults to the **`google`**
  mirror and needs neither `earthkit-data` nor `ecmwf-opendata`.
  **50 members: 2.9 h (20260806) vs 14.5 h (20260730).**

  `--fetch earthkit` keeps the original `earthkit.data.from_source("ecmwf-open-data")`
  path as a fallback (`aws`/`ecmwf` only). It is slow because `ecmwf-opendata` merges a
  param group's byte ranges into **one combined multi-range request** and runs the groups
  serially — a single `503 SlowDown` loses the whole request and costs a 120 s backoff.
  That batching is also why `--source google` fails on this path (GCS answers multi-range
  GETs with `400 InvalidArgument`); the mirror itself is fine, and serves single-range
  requests normally. Both backends produce **bit-identical** pkls (verified field-by-field
  on 6 members across 2 dates and both mirrors).
- **Step 2 output:** `gs://…/<date>_0000/fp16_v2_forecasts/aifs_ens_forecast_<date>_memberNNN_h*.grib`.
  Needs an Ampere+ GPU and the v2 software env (below).
- **Steps 3–5:** reuse the `shared/` CLIs with the **`--v2`** flag (added for this model).
  It switches the GCS subpaths to the `fp16_v2_*` namespace and the submission model name,
  leaving v1/fp16 behaviour untouched.

  **ETL software env (Steps 3–5).** Build a CPU env pinned to the **v2 `earthkit-regrid
  0.5.1`** (the same regrid version the v2 input-prep/inference uses; the v1 `aifs-etl`
  env's 0.4.0 also works for regrid but the v2 pipeline standardises on 0.5.1):

  ```bash
  micromamba create -y -n aifs-etl -c conda-forge python=3.12.7 \
      "earthkit-data<1.0.0" "earthkit-regrid=0.5.1" google-cloud-storage xarray netcdf4 icechunk python-dotenv
  micromamba activate aifs-etl

  # REQUIRED for the quintile (3b) + submission (3c/3d) steps — not on conda-forge:
  pip install AI_WQ_package python-dotenv
  ```

  > **Pin `earthkit-data<1.0.0`.** earthkit-data 1.0.0 changed the source API — `from_source(...)`
  > returns a non-iterable `GribData`, so the input-prep's `for f in data:` loop breaks
  > (`TypeError: 'GribData' object is not iterable`; `len()` raises `ImportError: cannot import
  > name 'convert_array'` against earthkit-utils 0.3.0). The unpinned `earthkit-data` in the
  > create line silently resolves to 1.0.0. Pin `<1.0.0` (solves to 0.20.0) as above.
  > Fix an already-drifted env with:
  > `micromamba install -n aifs-etl -c conda-forge 'earthkit-data<1.0.0' 'earthkit-regrid=0.5.1'`.

  > **If a run hangs at `surface fields...` (input prep), don't `^Z` it — that's the cause.**
  > earthkit-data serialises open-data downloads with a file lock
  > (`/tmp/earthkit-data-cloudenv/e-odretriever-*.cache.lock`). A **suspended** (`^Z`) or wedged
  > prep process keeps that lock, so every *new* run blocks forever at the first download with no
  > output. Recover with: `pgrep -af ecmwf_opendata_pkl_input_aifsens_v2` → `kill -CONT <pid>;
  > kill -9 <pid>` for any stuck/stopped jobs, then `rm -f /tmp/earthkit-data-cloudenv/*.cache.lock`,
  > then re-run. To stop a run cleanly use Ctrl-C (SIGINT), not Ctrl-Z. The first `ekr.interpolate`
  > call also does a one-time N320 matrix download (tens of seconds) — that's normal, not a hang.

  > **AWS S3 `503 Slow Down` can stall input prep for hours — restart it, don't wait.**
  > The mirror is hardcoded to AWS S3. S3 occasionally throttles the pressure-level burst
  > (14 levels × 6 params × 2 timesteps) with its `SlowDown` code, and `multiurl` then retries
  > **500 times at 120 s** — ~16 h of zero progress. Seen 2026-07-23: 1 h 40 m, **0 pkls**, log
  > repeating `Recovering from HTTP error [503 Slow Down], attempt N of 500`.
  > It is **not** an outage — plain requests to
  > `ecmwf-forecasts.s3.eu-central-1.amazonaws.com` returned 200 in <1 s throughout; the
  > process had just wedged itself in a retry loop.
  > **Fix:** `pkill -f ecmwf_opendata_pkl_input_aifsens_v2`, then re-run. earthkit's cache keeps
  > the already-downloaded fields, so the in-flight member finishes in seconds.
  > **Detect it:** pkl count stops rising while `503 Slow Down` repeats in the log.
  > (Mirror health check, if ever needed: `google` and `ecmwf` were both healthy; **`azure` is
  > broken — HTTP 409**. There is no `--source` flag; `create_input_state(..., source=)` still
  > takes one in code.)

  > **Cleanup between cycles.** A finished cycle costs ~630 GB
  > (`icechunk_v2` ~584 GB + `input_states` ~48 GB + `nc_1p5deg` ~2 GB). Reclaim it with
  > `python cleanup_aifs_run.py` (dry-run) → `--yes`. It keeps `aiwq/*.nc` (the quintile +
  > climatology files), protects the newest `--keep-latest` cycles, and refuses to touch a
  > cycle a running process is using.

  > **Why this matters:** the quintile CLI (3b) needs `AI_WQ_package` to fetch the 20-yr
  > quintile **climatology** from the AI-WQ server. It logs in as
  > `ftplib.FTP('ftp.ecmwf.int', 'ai_weather_quest', password)` — so it **does need
  > `AIWQ_PASSWORD`** (from `.env`). (The host also allows anonymous login, but the
  > climatology is fetched with that account, not anonymously.)
  > If the package is missing it does **not** error loudly — it prints
  > `AI_WQ_package not available, using local climatology files`, silently falls back to
  > local files that don't exist, and ends with `No quintile data calculated`. If you see
  > that, the fix is `pip install AI_WQ_package`, **not** a server/FTP problem.
  >
  > **`530 Login authentication failed`** means `AIWQ_PASSWORD` was never loaded, not that
  > the password is wrong. `load_dotenv()` used to search upward from the *script's*
  > directory (`shared/`), which never holds the `.env` — so running
  > `python ../shared/ensemble_quintile_analysis_cli.py` from this folder still missed it.
  > Both CLIs now resolve `.env` from the **working directory** first
  > (`find_dotenv(usecwd=True)`), so "run from this folder" behaves as documented.

  Then run from this folder so `coiled-data.json` and `.env` resolve:

  ```bash
  # 3a — regrid:  reads <date>_0000/fp16_v2_forecasts/  ->  writes <date>_0000/fp16_v2_1p5deg_nc/
  python ../shared/aifs_n320_grib_1p5defg_nc_cli.py --date 20260625_0000 --members 1-50 --v2

  # 3b — quintiles:  reads fp16_v2_1p5deg_nc/  ->  writes ensemble_quintile_probabilities_20260625_v2.nc
  python ../shared/ensemble_quintile_analysis_cli.py --date 20260625 --v2

  # 3c — submit:  uses ..._v2.nc + the v2 model name
  python ../shared/forecast_submission_cli.py --date 20260625 --v2          # add --dry-run to validate

  # 3d — out-of-window only:  build + zip the per-variable/week AI-WQ files for a manual upload
  python ../shared/aiwq_individual_files_cli.py --date 20260625 --v2
  ```

### Fast re-runs & out-of-window submissions (3b–3d optimized)

For faster re-runs of Steps 3b–3d (climatology → quintiles → individual files → zip), or
when rerunning after a network error:

```bash
# --- First run: full quintile pipeline ---
python ../shared/ensemble_quintile_analysis_cli.py --date 20260625 --v2

# --- If climatology FTP download fails, use direct FTP downloader ---
python download_climatology.py --date 20260625 --output-dir ./

# --- Fast re-run: load existing ensemble from icechunk store (avoids 50-member re-download) ---
python ../shared/ensemble_quintile_analysis_cli.py --date 20260625 --v2 --skip-ensemble

# --- Individual files + zip (as before) ---
AIWQ_TEAM_NAME=Fahamu AIWQ_MODEL_NAME_FP16=fp16FahamuAIFSv2 AIWQ_PASSWORD=<pwd> \
  python ../shared/aiwq_individual_files_cli.py --date 20260625 --v2

# --- Cleanup intermediates (ensemble_nc_files, aiwq_individual_<date>, optionally icechunk_store) ---
python cleanup_aiwq_intermediates.py --date 20260625                  # remove individual files only
python cleanup_aiwq_intermediates.py --all                           # full cleanup (keep .zip + quintile file)
python cleanup_aiwq_intermediates.py --all --dry-run                 # preview what will be deleted
```

**Output location:** `aiwq_submission_<date>_<team>_<model>.zip` is saved in the current working directory
(typically `fp16FahamuAIFSv2/`) alongside the quintile file.

**Performance notes:**

| Step | First run | With `--skip-ensemble` |
|------|-----------|----------------------|
| 3b: Quintiles | ~2–3 min (50-member ensemble download + calculation) | ~1 min (load from cache + calculation) |
| FTP climatology | ~1–2 min (or retried with backoff) | Skipped if files exist locally |
| 3d: Individual files + zip | ~30 sec | ~30 sec |
| **Total** | **~3–5 min** | **~1–2 min** |

**Intermediate file sizes (can be cleaned up after zipping):**

- `ensemble_nc_files/`: ~6.5 GB (50 members × ~130 MB each) — deleted by `cleanup_aiwq_intermediates.py`
- `aiwq_individual_<date>/`: ~30 MB (6 .nc files) — deleted by `cleanup_aiwq_intermediates.py`
- `ensemble_icechunk_store/`: ~10 GB (lazy-loaded zarr format) — kept for re-runs, delete with `--all`
- `aiwq_submission_*.zip`: ~700 KB — **keep this for submission**

**Climatology troubleshooting:**

If `ensemble_quintile_analysis_cli.py` fails during climatology download with `530 Login authentication failed`:
1. The direct FTP downloader (`download_climatology.py`) often succeeds where the AI_WQ_package fails
   (different FTP session handling).
2. Use `download_climatology.py` as a workaround, then retry the quintile CLI with `--skip-ensemble`.
3. Files are cached locally and reused across runs, so the FTP step only runs once per date pair.

  > **Submission window:** the live endpoint (3c) only accepts a forecast start date within
  > its window (`start_date` → `start_date + 3 days`). Outside it you get
  > *"You are not allowed to submit a forecast for … at this point in time"* — this is a
  > closed window, not an auth/FTP failure. For archival or a manual portal upload of a
  > past-window date, use **3d**: it reuses the exact submission prep but writes each
  > populated DataArray to `{var}_{date}_p{week}_{team}_{model}.nc` under
  > `./aiwq_individual_<date>/` and bundles them into
  > `aiwq_submission_<date>_<team>_<model>.zip` (it does **not** submit).

  Submission model name resolves as `AIWQ_MODEL_NAME_V2` → `AIWQ_MODEL_NAME_FP16` →
  `AIWQ_MODEL_NAME`. This folder's `.env` sets `AIWQ_MODEL_NAME_FP16=fp16FahamuAIFSv2`, so
  `--v2` submits under **`fp16FahamuAIFSv2`** without needing `AIWQ_MODEL_NAME_V2`.
  The Coiled service account `coiled-data.json` (here:
  `coiled-data@sewaa-416306`) provides GCS access; AI-WQ climatology + submission use `.env`.

  > **v2 regrid note (handled in code):** aifs-ens-2.0 GRIB has an inconsistent
  > pressure-level dimension (`t/u/v/w/z` = 14 levels incl. 10 hPa, but `q` = 13 because
  > `q_10` is dropped). A naïve full `to_xarray()` fails with *"inconsistent dimension
  > levelist 13 != 14"*. `aifs_n320_grib_1p5defg_nc_cli.py` selects the surface params
  > (`msl/tp/2t`) **before** conversion, which avoids the conflict.

### Why v2 regrid is ~7× faster than the old v1 timing

Selecting the 3 surface params before interpolation is also a large speedup, and explains
the big drop in per-member wall-clock vs the historical v1 figure:

| Regrid path | Fields interpolated per 72 h file | Per-member time |
|-------------|-----------------------------------|-----------------|
| Original (full FieldList → extract after) | **~1416** (mostly pressure levels: `q/t/u/v/w/z` × 13–14 lvl × 12 steps) | **~7.6 min** (v1 doc) / 5.7 min (v2 env, member 001) |
| Surface-only (`fl.sel(param=['msl','tp','2t'])` → interpolate) | **36** (3 vars × 12 steps) | **~1.1 min** (measured, 50-member run) |

`earthkit-regrid` applies a sparse N320→1.5° matrix multiply **per field**, so cost scales
with field count. The old path regridded the *entire* GRIB (~1416 fields) and only then
kept `tp/msl/2t` — ~40× more interpolation than needed. Selecting the 3 surface params up
front makes GCS download the new floor. Measured on the same 2-vCPU ETL box: member 001
went **5.7 → 1.1 min** (full vs surface-only, identical env), and the full 50-member v2 run
held a steady **1.1–1.3 min/member**.

**This is not a v1-vs-v2 model difference** — it's the pre-selection optimisation in the
shared CLI. It falls back to the full FieldList only if the select fails, so **v1/fp16
runs get the same speedup** if re-run on the current script.

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

# AI Forecast Submission Workflow

This document describes the complete sequential workflow for AI forecast submission using the AIFS ensemble system. The workflow spans from initial condition preparation to final forecast submission.

## Overview

The AI forecast submission process consists of 3 main steps across different computing environments (CPU/ETL and GPU) to produce ensemble weather forecasts and submit them for evaluation.

### Workflow Index
1. **Initial Condition Preparation** (ETL Machine) → `ecmwf_opendata_pkl_input_aifsens.py`
2. **GPU Inference** (GPU Machines)
   - FP32 (A100 GPU) → `automate_aifs_gpu_pipeline.py`
   - FP16 (G2 GPU) → `fp16_automate_aifs_gpu_pipeline.py`
3. **Post-Processing & Submission** (ETL Machine)
   - Regrid → `aifs_n320_grib_1p5defg_nc_cli.py`
   - Quintile Analysis → `ensemble_quintile_analysis_cli.py`
   - Forecast Submission → `forecast_submission_cli.py`

---

## ETL Environment Setup

The ETL (non-GPU) machine is used for **Step 1** and **Step 3**. Start the Coiled notebook:

```bash
coiled notebook start --name p2-aifs-etl-20260129 --vm-type n2-standard-2 --software aifs-etl-v2 --workspace=gcp-sewaa-nka --region us-east5
```

### Software Installation

Install the required environment using micromamba:

```bash
micromamba create -n aifs-etl -c conda-forge python=3.12.7 \
  && eval "$(micromamba shell hook --shell bash)" \
  && micromamba activate aifs-etl \
  && micromamba install -c conda-forge earthkit-data ecmwf-opendata \
  && pip install gcsfs s3fs earthkit-regrid==0.4.0 google-cloud-storage icechunk AI_WQ_package \
  && sudo apt update && sudo apt install nano
```

### Credentials Setup

Copy the `.env.example` file to `.env` and fill in your credentials:

```bash
cp .env.example .env
nano .env
```

Example `.env` contents:

```
AIWQ_TEAM_NAME=Fahamu
AIWQ_MODEL_NAME=FahamuAIFSv1
AIWQ_MODEL_NAME_FP16=FahamuAIFSv1_fp16
AIWQ_PASSWORD=your_password_here
```

A GCS service account key file (`coiled-data.json`) is also required for cloud storage access.

---

## Step 1: Initial Condition Preparation (ETL Machine)

**File:** `ecmwf_opendata_pkl_input_aifsens.py`

```bash
python ecmwf_opendata_pkl_input_aifsens.py
```

- **Purpose:** Download and preprocess ECMWF open data for ensemble members 1-50
- **Environment:** ETL machine (CPU-only, `n2-standard-2`)
- **Input:** ECMWF open data (surface, soil, pressure level parameters)
- **Output:** Pickle files uploaded to GCS bucket (`gs://aifs-aiquest-us-20251127/YYYYMMDD_0000/input/`)
- **Requires:** `coiled-data.json` (GCS service account key)

---

## Step 2: GPU Inference

### Step 2a: FP32 Inference (A100 GPU)

Start the GPU notebook:

```bash
coiled notebook start --name p1-gpu-aifs-20260129 --vm-type a2-ultragpu-1g --software east5-us-flashattn-dockerv1 --workspace=gcp-sewaa-nka --region us-east5 --disk-size 60
```

**File:** `automate_aifs_gpu_pipeline.py`

```bash
python automate_aifs_gpu_pipeline.py --date 20260129_0000 --members 1-50
```

- **Purpose:** Run AIFS-ENS model at full FP32 precision for all ensemble members
- **Environment:** A100 GPU (`a2-ultragpu-1g`, ~80GB VRAM)
- **Processing:** One member at a time (download → inference → upload → cleanup) to minimise storage usage
- **Output:** GRIB files uploaded to `gs://aifs-aiquest-us-20251127/YYYYMMDD_0000/forecasts/`

**Required files on the GPU machine:**

| File | Purpose |
|------|---------|
| `automate_aifs_gpu_pipeline.py` | Main pipeline orchestrator |
| `fp32_multi_run_AIFS_ENS_v1.py` | AIFS model runner (FP32) |
| `download_pkl_from_gcs.py` | GCS download utility |
| `upload_aifs_gpu_output_grib_gcs.py` | GCS upload utility |
| `coiled-data.json` | GCS service account key |

**SHUTDOWN GPU notebook after completion** to avoid unnecessary costs.

### Step 2b: FP16 Inference (G2 GPU)

Start the GPU notebook:

```bash
coiled notebook start --name p2-fp16-20260129 --vm-type g2-standard-12 --software flashattn-dockerv1 --workspace=gcp-sewaa-nka --region us-east4 --disk-size 400
```

**File:** `fp16_automate_aifs_gpu_pipeline.py`

```bash
python fp16_automate_aifs_gpu_pipeline.py --date 20260129_0000 --members 1-50
```

- **Purpose:** Run AIFS-ENS model at FP16 (half precision), reducing VRAM from ~50GB to <24GB
- **Environment:** G2 GPU (`g2-standard-12`)
- **Processing:** Same per-member pipeline as FP32 version
- **Output:** GRIB files uploaded to `gs://aifs-aiquest-us-20251127/YYYYMMDD_0000/fp16_forecasts/`

**Required files on the GPU machine:**

| File | Purpose |
|------|---------|
| `fp16_automate_aifs_gpu_pipeline.py` | Main pipeline orchestrator (FP16) |
| `fp16_multi_run_AIFS_ENS_v1.py` | AIFS model runner (FP16) |
| `download_pkl_from_gcs.py` | GCS download utility |
| `upload_aifs_gpu_output_grib_gcs.py` | GCS upload utility |
| `coiled-data.json` | GCS service account key |

**SHUTDOWN GPU notebook after completion.**

---

## Step 3: Post-Processing & Submission (ETL Machine)

Use the same ETL machine from Step 1:

```bash
coiled notebook start --name p2-aifs-etl-20260129 --vm-type n2-standard-2 --software aifs-etl-v2 --workspace=gcp-sewaa-nka --region us-east5
```

### Step 3a: Forecast Download & Regrid

**File:** `aifs_n320_grib_1p5defg_nc_cli.py`

```bash
python aifs_n320_grib_1p5defg_nc_cli.py --date 20260129

# For FP16:
python aifs_n320_grib_1p5defg_nc_cli.py --date 20260129 --fp16
```

- **Purpose:** Download GRIB files from GCS and regrid from N320 to 1.5 degree NetCDF
- **Output:** NetCDF files in `gs://aifs-aiquest-us-20251127/YYYYMMDD_0000/1p5deg_nc/` (or `fp16_1p5deg_nc/`)

### Step 3b: Ensemble Quintile Analysis

**File:** `ensemble_quintile_analysis_cli.py`

```bash
# FP32 mode (uses icechunk by default for memory efficiency)
python ensemble_quintile_analysis_cli.py --date 20260129

# FP16 mode
python ensemble_quintile_analysis_cli.py --date 20260129 --fp16
```

- **Purpose:** Download ensemble NetCDF from GCS, retrieve climatology, calculate quintile probabilities
- **Output:** `ensemble_quintile_probabilities_YYYYMMDD.nc` (or `_fp16.nc`)
- **Requires:** `.env` file with `AIWQ_PASSWORD` for climatology retrieval, `coiled-data.json` for GCS access

### Step 3c: Forecast Submission

**File:** `forecast_submission_cli.py`

```bash
# FP32 submission
python forecast_submission_cli.py --date 20260129

# FP16 submission
python forecast_submission_cli.py --date 20260129 --fp16

# Dry run (validate without submitting)
python forecast_submission_cli.py --date 20260129 --dry-run
```

- **Purpose:** Submit quintile probabilities to AI Weather Quest competition
- **Requires:** `.env` file with `AIWQ_TEAM_NAME`, `AIWQ_MODEL_NAME`, and `AIWQ_PASSWORD`
- **Submits:** 3 variables (mslp, pr, tas) x 2 weeks = 6 forecasts per run

---

## Data Flow

```
ECMWF Open Data → Pickle Files → GCS (YYYYMMDD_0000/input/)
                                        ↓
                              GPU Inference (FP32/FP16)
                                        ↓
                              GCS (YYYYMMDD_0000/forecasts/ or fp16_forecasts/)
                                        ↓
                              Regrid N320 → 1.5deg NetCDF
                                        ↓
                              GCS (YYYYMMDD_0000/1p5deg_nc/ or fp16_1p5deg_nc/)
                                        ↓
                              Quintile Analysis → Submission
```

## Storage Strategy
- **GCS Bucket:** `aifs-aiquest-us-20251127`
- **Path Structure:**
  - Input pickle files: `YYYYMMDD_0000/input/`
  - FP32 GRIB forecasts: `YYYYMMDD_0000/forecasts/`
  - FP16 GRIB forecasts: `YYYYMMDD_0000/fp16_forecasts/`
  - FP32 NetCDF outputs: `YYYYMMDD_0000/1p5deg_nc/`
  - FP16 NetCDF outputs: `YYYYMMDD_0000/fp16_1p5deg_nc/`
- **Service Account:** `coiled-data.json` for GCS access

## GPU Memory Optimization

| GPU | VRAM | Pipeline Script | Precision | Chunks |
|-----|------|----------------|-----------|--------|
| A100 (80GB) | 80GB | `automate_aifs_gpu_pipeline.py` | FP32 | Default |
| A100 (40GB) | 40GB | `automate_aifs_gpu_pipeline.py` | FP32 | 8 |
| G2 (L4 24GB) | 24GB | `fp16_automate_aifs_gpu_pipeline.py` | FP16 | 16 |
| A10G | 24GB | `fp16_automate_aifs_gpu_pipeline.py` | FP16 | 16 |
| RTX 4090 | 24GB | `fp16_automate_aifs_gpu_pipeline.py` | FP16 | 16 |

**Reference:** [HuggingFace Discussion #17](https://huggingface.co/ecmwf/aifs-ens-1.0/discussions/17)

## Ensemble Configuration
- **Members:** 1-50
- **Forecast Length:** 792 hours (33 days)
- **Meteorological Parameters:** pr, mslp, tas

---

## ERA5T Pipeline (CEDA Data Source)

An alternative pipeline using ERA5T data from the CEDA archive instead of ECMWF Open Data.
This enables forecasts initialized from dates not covered by ECMWF Open Data (which only
retains the most recent ~24h). ERA5T has a ~1 week lag from real time.

For full technical documentation, see [`era5tFp16FahamuAIFSv1.md`](era5tFp16FahamuAIFSv1.md).

### Key Differences from Standard Pipeline

| Aspect | Standard (ECMWF Open Data) | ERA5T (CEDA) |
|--------|---------------------------|--------------|
| Input fields | 92 fields | 74 fields (adapted to 92 at inference) |
| Members | 1-50 | 0-9 (10 EDA members) |
| Lead time | 792h (33 days) | 960h (40 days) |
| Data source | ECMWF Open Data (latest only) | CEDA ERA5T archive (~1 week lag) |
| Auth | None | CEDA Bearer token (`ceda_token` in `.env`) |

### ERA5T Workflow

**Step 1: Create pkl files from CEDA** (ETL Machine)

```bash
uv run ceda_era5t_pkl_input_aifsens.py
```

Edit `DATE` in the script to set the initialization date. Requires `ceda_token` in `.env`.
Output: `gs://aifs-aiquest-us-20251127/era5t/YYYYMMDD/input_state_member_00*.pkl`

**Step 2: GPU Inference** (GPU Machine, >=24GB VRAM)

```bash
python era5t_fp16_automate_aifs_gpu_pipeline.py \
    --date YYYYMMDD_0000 \
    --members 0-9 \
    --gcs-input-prefix era5t/YYYYMMDD \
    --gcs-output-subpath era5t_fp16_forecasts \
    --lead-time 960
```

Note: `--date` is the target forecast date folder, `--gcs-input-prefix` points to the
ERA5T init date pkl files. For example, init date 20260227 → target date 20260305.

**Step 3: GRIB to 1.5deg NetCDF** (ETL Machine)

```bash
python era5t_aifs_n320_grib_1p5deg_nc_cli.py \
    --date YYYYMMDD_0000 \
    --members 0-9 \
    --gcs-input-subpath era5t_fp16_forecasts \
    --gcs-output-subpath era5t_fp16_1p5deg_nc \
    --init-date YYYYMMDD
```

`--init-date` must match the ERA5T initialization date used in the GRIB filenames.

**Step 4: Quintile Analysis** (ETL Machine)

```bash
python era5t_ensemble_quintile_analysis_cli.py --date YYYYMMDD --members 0-9 --fp16
```

**Step 5: Submit** (ETL Machine)

```bash
python era5t_forecast_submission_cli.py --date YYYYMMDD
```

### ERA5T Scripts Reference

| Script | Purpose |
|--------|---------|
| `ceda_era5t_pkl_input_aifsens.py` | CEDA ERA5T → pkl (74 fields, 10 members) |
| `era5t_fp16_automate_aifs_gpu_pipeline.py` | GPU pipeline orchestrator (FP16) |
| `era5t_fp16_multi_run_AIFS_ENS_v1.py` | FP16 inference with field adaptation (74→92) |
| `era5t_aifs_n320_grib_1p5deg_nc_cli.py` | GRIB → 1.5deg NetCDF regridding |
| `era5t_ensemble_quintile_analysis_cli.py` | Quintile probability calculation |
| `era5t_forecast_submission_cli.py` | AI Weather Quest submission |

### ERA5T Execution Times and Costs

| Script | Time | Environment | Cost |
|--------|------|-------------|------|
| `ceda_era5t_pkl_input_aifsens.py` | ~14 min | CPU (n2-standard-2) | ~$0.04 |
| `era5t_fp16_automate_aifs_gpu_pipeline.py` | ~2.5 hours | GPU (g2-standard-12) | ~$5-7 |
| `era5t_aifs_n320_grib_1p5deg_nc_cli.py` | ~1 hour | CPU (n2-standard-2) | ~$0.24 |
| `era5t_ensemble_quintile_analysis_cli.py` | ~10 min | CPU (n2-standard-2) | ~$0.04 |
| `era5t_forecast_submission_cli.py` | ~5 min | CPU (n2-standard-2) | ~$0.02 |

## Dependencies

### Core Packages
- `anemoi-inference`: ECMWF AI model runner
- `earthkit-data`: ECMWF data handling
- `earthkit-regrid`: Data regridding (v0.4.0)
- `google-cloud-storage`: GCS operations
- `icechunk`: Memory-efficient ensemble processing
- `AI_WQ_package`: Forecast submission and evaluation
- `python-dotenv`: Credential management from `.env` file

### Authentication
- GCS service account key (`coiled-data.json`) for cloud storage access
- `.env` file with AI Weather Quest credentials (team name, model name, password)

## Script Execution Times and Costs

| Script | Execution Time | Environment | Cost (USD) | Notes |
|--------|----------------|-------------|------------|-------|
| `ecmwf_opendata_pkl_input_aifsens.py` | 2-2.5 hours | CPU (n2-standard-2) | ~$0.48-0.60 | Data preprocessing and GCS upload |
| `automate_aifs_gpu_pipeline.py` | 6.5-7 hours | GPU (a2-ultragpu-1g) | ~$35-42 | FP32, 50 members, per-member processing |
| `fp16_automate_aifs_gpu_pipeline.py` | 6.5-7 hours | GPU (g2-standard-12) | ~$15-20 | FP16, 50 members, reduced cost GPU |
| `aifs_n320_grib_1p5defg_nc_cli.py` | 4-4.5 hours | CPU (n2-standard-2) | ~$0.96-1.08 | GRIB regridding and processing |
| `ensemble_quintile_analysis_cli.py` | 15 minutes | CPU (n2-standard-2) | ~$0.06 | Ensemble analysis |
| `forecast_submission_cli.py` | 5 minutes | CPU (n2-standard-2) | ~$0.02 | Submission validation |

## Troubleshooting: HuggingFace Model Download Hangs

### Symptom

The GPU inference pipeline hangs indefinitely at model checkpoint download:

```
Running forecast for member 0...
Fetching 7 files:   0%|          | 0/7 [00:00<?, ?it/s]
```

This occurs inside `runner.run()` when the `anemoi-inference` library attempts to download the `ecmwf/aifs-ens-1.0` model weights (~3-4 GB) from HuggingFace Hub. The model metadata loads quickly during `SimpleRunner()` init, but the large checkpoint blob download stalls.

### Root Cause

The HuggingFace `huggingface_hub` downloader can hang due to:

1. **Network throttling or rate limiting** on unauthenticated requests from cloud VMs
2. **Incomplete downloads with stale lock files** preventing retry (a previous failed/killed download leaves `.incomplete` and `.lock` files in the cache)
3. **No `HF_TOKEN` set**, causing anonymous download which is subject to stricter rate limits

### Diagnosis

```bash
# Check for incomplete downloads and stale locks
ls -la ~/.cache/huggingface/hub/models--ecmwf--aifs-ens-1.0/blobs/
# Look for files ending in .incomplete

ls -la ~/.cache/huggingface/hub/.locks/models--ecmwf--aifs-ens-1.0/
# Look for .lock files with recent timestamps

# Check HF token
echo $HF_TOKEN
```

### Fix: Clear Stale Cache and Retry

```bash
# 1. Kill the stuck process
kill $(pgrep -f era5t_fp16_automate_aifs_gpu_pipeline)

# 2. Remove incomplete downloads and stale locks
rm -f ~/.cache/huggingface/hub/models--ecmwf--aifs-ens-1.0/blobs/*.incomplete
rm -f ~/.cache/huggingface/hub/.locks/models--ecmwf--aifs-ens-1.0/*.lock

# 3. Set HF token to avoid rate limiting
export HF_TOKEN="your_huggingface_token"

# 4. Re-run the pipeline
python era5t_fp16_automate_aifs_gpu_pipeline.py --date YYYYMMDD_0000 --members 0-4 ...
```

### Recommended: Pre-cache Model in Docker Image

The most reliable solution is to bake the HuggingFace model into the Docker image used for GPU inference. This eliminates runtime downloads entirely, avoids network dependency during forecast runs, and ensures reproducible deployments.

#### Approach 1: Dockerfile with Pre-downloaded Model

Add the model download step to the GPU Docker image build:

```dockerfile
FROM your-base-gpu-image:latest

# Install huggingface_hub for download
RUN pip install huggingface_hub

# Pre-download the AIFS-ENS model checkpoint into the HF cache
# This caches all 7 files (~3-4 GB) at build time
ARG HF_TOKEN
RUN python -c "\
from huggingface_hub import snapshot_download; \
snapshot_download('ecmwf/aifs-ens-1.0', token='${HF_TOKEN}')"

# The model is now cached at /root/.cache/huggingface/hub/models--ecmwf--aifs-ens-1.0/
# anemoi-inference will find it automatically without any network calls
```

Build with:

```bash
docker build --build-arg HF_TOKEN=hf_your_token -t aifs-gpu-cached:latest .
```

#### Approach 2: Volume Mount from GCS

If Docker image size is a concern (~3-4 GB added), pre-download the model to a persistent disk or GCS bucket and mount it:

```bash
# Pre-download once to a persistent location
python -c "
from huggingface_hub import snapshot_download
snapshot_download('ecmwf/aifs-ens-1.0', cache_dir='/mnt/model-cache/huggingface')
"

# Mount at runtime
export HF_HOME=/mnt/model-cache/huggingface
python era5t_fp16_automate_aifs_gpu_pipeline.py ...
```

#### Approach 3: Coiled Software Environment with Cached Model

For Coiled-managed GPU notebooks, include the model download in the software environment setup so it is available when the notebook starts:

```bash
# During software environment creation, ensure model is cached
python -c "from huggingface_hub import snapshot_download; snapshot_download('ecmwf/aifs-ens-1.0')"
```

### Why Docker Pre-caching is Preferred

| Approach | Pros | Cons |
|----------|------|------|
| Runtime download | No image size increase | Slow startup (~10-30 min), network dependent, can hang |
| Docker pre-cache | Zero startup delay, no network needed, fully reproducible | Larger image (~3-4 GB), requires rebuild for model updates |
| Volume mount | Flexible, shared across instances | Requires persistent disk setup, mount configuration |

For operational forecast pipelines where reliability and speed matter, **Docker pre-caching is strongly recommended**. It converts a flaky runtime network dependency into a deterministic build-time step.

---


## Forecast Run History

| S.No | Date | Ensemble Members | Status | Notes |
|------|------|------------------|--------|-------|
| 1 | 2025-08-21 | 50 | Completed | Full ensemble run |
| 2 | 2025-08-28 | 50 | Completed | Full ensemble run |
| 3 | 2025-09-04 | 50 | Completed | Full ensemble run |
| 4 | 2025-09-11 | 48 | Completed | Reduced members due to GPU memory issue |
| 5 | 2025-09-18 | 20 | Completed | Time exceeded to download from opendata |

## Acknowledgements

This work was funded in part by:

1. Hazard modeling, impact estimation, climate storylines for event catalogue
   on drought and flood disasters in the Eastern Africa (E4DRR) project.
   https://icpac-igad.github.io/e4drr/ United Nations | Complex Risk Analytics
   Fund (CRAF'd) on the activity 2.3.3 Experiment generative AI for EPS(Ensemble Prediction Systems):
   Explore the application of Generative AI (cGAN) in bias correction and
   downscaling of EPS data in an operational setup.
2. The Strengthening Early Warning Systems for Anticipatory Action (SEWAA)
   Project. https://cgan.icpac.net/

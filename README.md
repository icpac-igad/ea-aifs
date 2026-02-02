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

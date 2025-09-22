# AI Forecast Submission Workflow

This document describes the complete sequential workflow for AI forecast submission using the AIFS ensemble system. The workflow spans from initial condition preparation to final forecast submission.

## Overview

The AI forecast submission process consists of 7 sequential steps that move data between different computing environments (CPU, GPU, and cloud storage) to produce ensemble weather forecasts and submit them for evaluation.

## Sequential Workflow Steps

### Step 1: Initial Condition Preparation (CPU)
In the NON GPU notebook 
```bash
coiled notebook start --name aifs-etl --vm-type n2-standard-2 --software aifs-etl-v2 --workspace=gcp-sewaa-nka
```
**File:** `ecmwf_opendata_pkl_input_aifsens.py`
- **Purpose:** Download and preprocess ECMWF open data for ensemble members 1-50
- **Environment:** CPU-only machine
- **Input:** ECMWF open data (surface, soil, pressure level parameters)
- **Output:** Pickle files containing preprocessed input states
- **Action:** Upload pickle files to GCS bucket for GPU access

### Step 2: Transfer to GPU Environment 
```bash 
coiled notebook start --name p1-aifs-20250918 --vm-type a2-ultragpu-1g --software flashattn-dockerv1 --workspace=gcp-sewaa-nka --region europe-west4
```
In the GPU notebook bash 
```bash 
python download_pkl_from_gcs.py --date 20250828_0000 --members 1-25 --output-dir /scratch/input_states
```
**File:** `download_pkl_from_gcs.py`
- **Purpose:** Download preprocessed input states to GPU machine
- **Environment:** GPU machine
- **Input:** Pickle files from GCS bucket
- **Output:** Local input state files ready for model execution
- **Action:** Avoids time-consuming data retrieval on expensive GPU instances

### Step 3: AI Model Execution (GPU)
In the GPU notebook bash 
```bash 
python multi_run_AIFS_ENS_v1.py 
```
**File:** `multi_run_AIFS_ENS_v1.py` 
- **Purpose:** Run ECMWF's aifs-ens-v1 model for 50 ensemble members
- **Environment:** GPU machine (requires anemoi-inference package)
- **Input:** Preprocessed input states from Step 2
- **Output:** GRIB forecast files for each ensemble member
- **Action:** Generate 792-hour forecasts using AI model

### Step 4: GPU Output Upload
In the GPU notebook bash 
```bash
python upload_aifs_gpu_output_grib_gcs.py --bucket aifs-aiquest --directory /scratch/ensemble_outputs/ --prefix forecasts/20250904/ --members 1-10 --threads 15

SHUTDOWN GPU notebook 
```
**File:** `upload_aifs_gpu_output_grib_gcs.py`
- **Purpose:** Upload generated GRIB files to cloud storage  
- **Environment:** GPU machine
- **Input:** GRIB forecast files from Step 3
- **Output:** GRIB files uploaded to GCS bucket
- **Action:** Multi-threaded upload for faster transfer, then shutdown GPU

### Step 5: Forecast Download regrid from N320 into 1.5 deg (ETL Environment)
In the NON GPU notebook
```bash
python aifs_n320_grib_1p5defg_nc.py 
```
**File:** `aifs_n320_grib_1p5defg_nc.py`
- **Purpose:** Download forecast GRIB files for post-processing
- **Environment:** Non-GPU ETL machine (e.g., Coiled notebook)
- **Input:** GRIB files from GCS bucket  
- **Output:** Local GRIB files for analysis
- **Action:** Download specific time ranges (h432-792) for ensemble members


### Step 6: Ensemble Analysis and Submission
In the Non-GPU notebook
```bash
python ensemble_quintile_analysis.py 
```
**File:** `ensemble_quintile_analysis.py`
- **Purpose:** Analyze ensemble forecasts and submit to AI forecast competition
- **Environment:** ETL machine  
- **Input:** Validated GRIB files from Step 5
- **Output:** Quintile analysis and forecast submission
- **Action:** Calculate ensemble statistics, retrieve climatology, and call `submit_forecast()` function

### Step 7: Forecast Submission 
In the Non-GPU notebook 
```bash
Use forecast_submission_20250918.ipynb
```

## Key Components

### Data Flow
1. **ECMWF Open Data** → **Pickle Files** → **GCS Bucket**
2. **GCS Bucket** → **GPU Input** → **AI Model** → **GRIB Output**
3. **GRIB Output** → **GCS Bucket** → **ETL Environment** → **Analysis & Submission**

### Computing Environments
- **CPU Machine:** Data preprocessing and initial condition preparation
- **GPU Machine:** AI model execution (expensive, time-limited)
- **ETL Machine:** Post-processing, analysis, and submission (cost-effective)

### Storage Strategy
- **GCS Buckets:** Intermediate storage between computing environments
- **Multi-threading:** Fast upload/download of large GRIB files
- **Service Account:** Authentication for GCS access (`coiled-data-e4drr_202505.json`)

## Configuration Parameters

### Ensemble Configuration
- **Members:** 1-50 (input preparation) / 1-9 (analysis)
- **Forecast Length:** 792 hours (33 days)
- **Time Ranges:** h432-504, h504-576, h576-648, h648-720, h720-792

### Meteorological Parameters
- **Surface:** pr, mslp, tas

## Dependencies

### Core Packages
- `anemoi-inference`: ECMWF AI model runner
- `earthkit-data`: ECMWF data handling
- `earthkit-regrid`: Data regridding
- `google-cloud-storage`: GCS operations
- `AI_WQ_package`: Forecast submission and evaluation

### Authentication
- GCS service account key for cloud storage access
- AIWQ_SUBMIT_PWD environment variable for forecast submission

## Final Output
The workflow culminates with the execution of `submit_forecast()` function in `ensemble_quintile_analysis.py:504`, which submits the processed ensemble forecast to the AI weather forecasting competition.

## Script Execution Times and Costs

| Script | Execution Time | Environment | Cost (USD) | Notes |
|--------|----------------|-------------|------------|--------|
| `ecmwf_opendata_pkl_input_aifsens.py` | 2-2.5 hours | CPU (n2-standard-2) | ~$0.48-0.60 | Data preprocessing and GCS upload |
| `download_pkl_from_gcs.py` | 10-15 minutes | GPU (a2-ultragpu-1g) | ~$0.50-0.75 | Fast GCS download to GPU local storage |
| `multi_run_AIFS_ENS_v1.py` | 6 hours | GPU (a2-ultragpu-1g) | ~$5-6 | 50 ensemble members AI inference |
| `upload_aifs_gpu_output_grib_gcs.py` | 5 minutes | GPU (a2-ultragpu-1g) | ~$0.25 | Multi-threaded GRIB upload |
| `aifs_n320_grib_1p5defg_nc.py` | 4-4.5 hours | CPU (n2-standard-2) | ~$0.96-1.08 | GRIB regridding and processing |
| `ensemble_quintile_analysis.py` | 15 minutes | CPU (n2-standard-2) | ~$0.06 | Ensemble analysis and submission |
| `forecast_submission_20250918.ipynb` | 5 minutes | CPU (n2-standard-2) | ~$0.02 | Final submission validation |

**Total Cost per Forecast Run: ~$30-40 USD**
- GPU Operations: ~$5-6 per hour
- CPU Operations: ~$2-3

## Forecast Run History

| S.No | Date | Ensemble Members | Status | Notes |
|------|------|------------------|--------|--------|
| 1 | 2025-08-21 | 50 | Completed | Full ensemble run |
| 2 | 2025-08-28 | 50 | Completed | Full ensemble run |
| 3 | 2025-09-04 | 50 | Completed | Full ensemble run |
| 4 | 2025-09-11 | 48 | Completed | Reduced members due to GPU memery issue |
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


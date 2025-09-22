# AI Forecast Submission Workflow

This document describes the complete sequential workflow for AI forecast submission using the AIFS ensemble system. The workflow spans from initial condition preparation to final forecast submission.

## Overview

The AI forecast submission process consists of 7 sequential steps that move data between different computing environments (CPU, GPU, and cloud storage) to produce ensemble weather forecasts and submit them for evaluation.

## Sequential Workflow Steps

### Step 1: Initial Condition Preparation (CPU)
**File:** `ecmwf_opendata_pkl_input_aifsens.py`
- **Purpose:** Download and preprocess ECMWF open data for ensemble members 1-50
- **Environment:** CPU-only machine
- **Input:** ECMWF open data (surface, soil, pressure level parameters)
- **Output:** Pickle files containing preprocessed input states
- **Action:** Upload pickle files to GCS bucket for GPU access

### Step 2: Transfer to GPU Environment  
**File:** `download_pkl_from_gcs.py`
- **Purpose:** Download preprocessed input states to GPU machine
- **Environment:** GPU machine
- **Input:** Pickle files from GCS bucket
- **Output:** Local input state files ready for model execution
- **Action:** Avoids time-consuming data retrieval on expensive GPU instances

### Step 3: AI Model Execution (GPU)
**File:** `multi_run_AIFS_ENS_v1.py` 
- **Purpose:** Run ECMWF's aifs-ens-v1 model for 50 ensemble members
- **Environment:** GPU machine (requires anemoi-inference package)
- **Input:** Preprocessed input states from Step 2
- **Output:** GRIB forecast files for each ensemble member
- **Action:** Generate 792-hour forecasts using AI model

### Step 4: GPU Output Upload
**File:** `upload_aifs_gpu_output_grib_gcs.py`
- **Purpose:** Upload generated GRIB files to cloud storage  
- **Environment:** GPU machine
- **Input:** GRIB forecast files from Step 3
- **Output:** GRIB files uploaded to GCS bucket
- **Action:** Multi-threaded upload for faster transfer, then shutdown GPU

### Step 5: Forecast Download (ETL Environment)
**File:** `download_grib_from_gcs.py`
- **Purpose:** Download forecast GRIB files for post-processing
- **Environment:** Non-GPU ETL machine (e.g., Coiled notebook)
- **Input:** GRIB files from GCS bucket  
- **Output:** Local GRIB files for analysis
- **Action:** Download specific time ranges (h432-792) for ensemble members

### Step 6: Forecast Validation
**File:** `aifs_792hr_forecast_grib_check_vars.py`
- **Purpose:** Validate and check forecast GRIB file variables
- **Environment:** ETL machine
- **Input:** Downloaded GRIB files from Step 5
- **Output:** Validated forecast data ready for analysis
- **Action:** Quality control and variable verification

### Step 7: Ensemble Analysis and Submission
**File:** `ensemble_quintile_analysis.py`
- **Purpose:** Analyze ensemble forecasts and submit to AI forecast competition
- **Environment:** ETL machine  
- **Input:** Validated GRIB files from Step 6
- **Output:** Quintile analysis and forecast submission
- **Action:** Calculate ensemble statistics, retrieve climatology, and call `submit_forecast()` function

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

# AI Forecast Submission Workflow

This document describes the complete sequential workflow for AI forecast submission using the AIFS ensemble system. The workflow spans from initial condition preparation to final forecast submission.

## Overview

The AI forecast submission process consists of 7 sequential steps that move data between different computing environments (CPU, GPU, and cloud storage) to produce ensemble weather forecasts and submit them for evaluation.

### Workflow Index
1. **Initial Condition Preparation** → `ecmwf_opendata_pkl_input_aifsens.py`
2. **Transfer to GPU Environment** → `download_pkl_from_gcs.py`
3. **AI Model Execution** → `multi_run_AIFS_ENS_v1.py`
4. **GPU Output Upload** → `upload_aifs_gpu_output_grib_gcs.py`
5. **Forecast Download & Regrid** → `aifs_n320_grib_1p5defg_nc.py`
6. **Ensemble Analysis** → `ensemble_quintile_analysis.py`
7. **Forecast Submission** → `forecast_submission_20250918.ipynb`

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
coiled notebook start --name p1-aifs-20250918 --vm-type a2-ultragpu-1g --software flashattn-dockerv1 --workspace=gcp-sewaa-nka --region us-east5
```
In the GPU notebook bash
```bash
python download_pkl_from_gcs.py --date 20251127_0000 --members 1-50 --output-dir /scratch/input_states
```
**File:** `download_pkl_from_gcs.py`
- **Purpose:** Download preprocessed input states to GPU machine
- **Environment:** GPU machine
- **Input:** Pickle files from GCS bucket (`gs://aifs-aiquest-us-20251127/YYYYMMDD_0000/input/`)
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
python upload_aifs_gpu_output_grib_gcs.py --bucket aifs-aiquest-us-20251127 --directory /scratch/ensemble_outputs/ --prefix 20251127_0000/forecasts/ --members 1-50 --threads 15

SHUTDOWN GPU notebook
```
**File:** `upload_aifs_gpu_output_grib_gcs.py`
- **Purpose:** Upload generated GRIB files to cloud storage
- **Environment:** GPU machine
- **Input:** GRIB forecast files from Step 3
- **Output:** GRIB files uploaded to GCS bucket (`gs://aifs-aiquest-us-20251127/YYYYMMDD_0000/forecasts/`)
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
1. **ECMWF Open Data** → **Pickle Files** → **GCS Bucket** (`YYYYMMDD_0000/input/`)
2. **GCS Bucket** → **GPU Input** → **AI Model** → **GRIB Output** → **GCS Bucket** (`YYYYMMDD_0000/forecasts/`)
3. **GCS Bucket** → **ETL Environment** → **NetCDF** → **GCS Bucket** (`YYYYMMDD_0000/1p5deg_nc/`) → **Analysis & Submission**

### Computing Environments
- **CPU Machine:** Data preprocessing and initial condition preparation
- **GPU Machine:** AI model execution (expensive, time-limited)
- **ETL Machine:** Post-processing, analysis, and submission (cost-effective)

### Storage Strategy
- **GCS Bucket:** `aifs-aiquest-us-20251127` (us-east5 region)
- **Path Structure:**
  - Input pickle files: `YYYYMMDD_0000/input/`
  - GRIB forecasts: `YYYYMMDD_0000/forecasts/`
  - NetCDF outputs: `YYYYMMDD_0000/1p5deg_nc/`
- **Multi-threading:** Fast upload/download of large GRIB files
- **Service Account:** Authentication for GCS access (`coiled-data.json`)

## Configuration Parameters

### Ensemble Configuration
- **Members:** 1-50 (input preparation) / 1-9 (analysis)
- **Forecast Length:** 792 hours (33 days)
- **Time Ranges:** h432-504, h504-576, h576-648, h648-720, h720-792

### Meteorological Parameters
- **Surface:** pr, mslp, tas

### GPU Memory Optimization

The AIFS-ENS model requires ~50GB VRAM at full precision, limiting it to high-end GPUs (A100, H100).
The following settings in `multi_run_AIFS_ENS_v1.py` reduce memory usage to <24GB, enabling use on more
accessible GPUs like RTX 4090 or A10G.

**Reference:** [HuggingFace Discussion #17](https://huggingface.co/ecmwf/aifs-ens-1.0/discussions/17)

#### Configuration Options

```python
# In multi_run_AIFS_ENS_v1.py
GPU_MEMORY_OPTIMIZATION = True   # Enable memory optimizations
INFERENCE_PRECISION = "16"       # "16" (FP16) or "32" (FP32)
INFERENCE_NUM_CHUNKS = 16        # 8, 16, 32, or 64
```

#### Memory Reduction Techniques

| Setting | Description | Memory Impact |
|---------|-------------|---------------|
| `INFERENCE_PRECISION="16"` | Use FP16 (half precision) inference | ~40-50% reduction |
| `INFERENCE_NUM_CHUNKS=16` | Split computation into chunks | ~20-30% reduction |
| `PYTORCH_CUDA_ALLOC_CONF` | Enable expandable memory segments | Reduces fragmentation |

#### Recommended Configurations by GPU

| GPU | VRAM | Settings |
|-----|------|----------|
| A100 (80GB) | 80GB | `GPU_MEMORY_OPTIMIZATION=False` (full precision) |
| A100 (40GB) | 40GB | `PRECISION="16"`, `CHUNKS=8` |
| A10G | 24GB | `PRECISION="16"`, `CHUNKS=16` |
| RTX 4090 | 24GB | `PRECISION="16"`, `CHUNKS=16` |
| RTX 3090 | 24GB | `PRECISION="16"`, `CHUNKS=32` |

#### Trade-offs

- **FP16 Precision:** Minimal accuracy impact for weather forecasting; slight differences in extreme values
- **Chunking:** Higher chunk counts reduce memory but increase inference time (~5-15% slower)
- **Recommended:** Start with `PRECISION="16"` and `CHUNKS=16`, adjust if needed

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
| `download_pkl_from_gcs.py` | 10-15 minutes | GPU (a2-ultragpu-1g) | ~$1.25-1.50 | Fast GCS download to GPU local storage |
| `multi_run_AIFS_ENS_v1.py` | 6.5-7 hours | GPU (a2-ultragpu-1g) | ~$35-42 | 50 ensemble members AI inference |
| `upload_aifs_gpu_output_grib_gcs.py` | 5 minutes | GPU (a2-ultragpu-1g) | ~$0.50 | Multi-threaded GRIB upload |
| `aifs_n320_grib_1p5defg_nc.py` | 4-4.5 hours | CPU (n2-standard-2) | ~$0.96-1.08 | GRIB regridding and processing |
| `ensemble_quintile_analysis.py` | 15 minutes | CPU (n2-standard-2) | ~$0.06 | Ensemble analysis and submission |
| `forecast_submission_20250918.ipynb` | 5 minutes | CPU (n2-standard-2) | ~$0.02 | Final submission validation |

**Total Cost per Forecast Run: ~$39-46 USD**
- GPU Operations: ~$37-44 (7+ hours at $5-6/hour)
- CPU Operations: ~$1.5-2 (7+ hours at $0.24/hour)

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


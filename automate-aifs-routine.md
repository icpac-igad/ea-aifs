# Automated AIFS GPU Pipeline

This document describes the automated workflow for running the AIFS ensemble forecast system on GPU machines, consolidating three separate scripts into one lean orchestration script.

## Critical Design: Per-Member Processing

**IMPORTANT:** The pipeline processes **ONE ensemble member at a time** through the complete cycle before moving to the next member. This is crucial to avoid storage space issues on GPU machines.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    PER-MEMBER PROCESSING LOOP                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   FOR EACH ENSEMBLE MEMBER (1 to 50):                                       │
│                                                                              │
│   ┌──────────────────────────────────────────────────────────────────────┐  │
│   │                                                                      │  │
│   │   STEP 1: Download         STEP 2: Inference        STEP 3: Upload  │  │
│   │   ┌─────────────┐          ┌─────────────┐          ┌─────────────┐ │  │
│   │   │ GCS → .pkl  │ ──────►  │ GPU Model   │ ──────►  │ .grib → GCS │ │  │
│   │   │ (1 file)    │          │ → .grib     │          │ (11 files)  │ │  │
│   │   └─────────────┘          └─────────────┘          └──────┬──────┘ │  │
│   │                                                            │        │  │
│   │                            STEP 4: Cleanup                 │        │  │
│   │                            ┌─────────────┐                 │        │  │
│   │                            │ Delete .pkl │ ◄───────────────┘        │  │
│   │                            │ Delete .grib│                          │  │
│   │                            └─────────────┘                          │  │
│   │                                                                      │  │
│   └──────────────────────────────────────────────────────────────────────┘  │
│                                                                              │
│   → REPEAT FOR NEXT MEMBER                                                  │
│                                                                              │
│   Storage at any time: ~1 pkl (~500MB) + ~11 grib (~2GB) = ~2.5GB          │
│   NOT: 50 pkl + 550 grib files = ~125GB (would fill storage!)              │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Why Per-Member Processing?

| Approach | Storage Required | Risk |
|----------|-----------------|------|
| **Batch (all members)** | ~125 GB | Storage fills up, pipeline fails |
| **Per-member (current)** | ~2.5 GB | Minimal footprint, always safe |

Each ensemble member generates:
- 1 pickle input file (~500 MB)
- 11 GRIB output files (792h ÷ 72h chunks) (~2 GB total)

With 50 members processed in batch mode, you'd need ~125 GB of local storage. Per-member processing keeps storage under 3 GB at all times.

## Scripts Integration

### Original Scripts (3 separate files)

| Script | Purpose | Environment |
|--------|---------|-------------|
| `download_pkl_from_gcs.py` | Download pickle input files from GCS | GPU |
| `fp32_multi_run_AIFS_ENS_v1.py` | Run AIFS-ENS model for ensemble members | GPU |
| `upload_aifs_gpu_output_grib_gcs.py` | Upload GRIB forecasts to GCS | GPU |

### Integrated Script (1 lean orchestration file)

| Script | Purpose | Imports From |
|--------|---------|--------------|
| `automate_aifs_gpu_pipeline.py` | Orchestrate all 3 steps | All three scripts above |

## GCS Bucket Structure

**Bucket:** `aifs-aiquest-us-20251127`

```
gs://aifs-aiquest-us-20251127/
├── YYYYMMDD_0000/
│   ├── input/                          # From ecmwf_opendata_pkl_input_aifsens.py
│   │   ├── input_state_member_001.pkl
│   │   ├── input_state_member_002.pkl
│   │   ├── ...
│   │   └── input_state_member_050.pkl
│   │
│   └── forecasts/                      # From automate_aifs_gpu_pipeline.py
│       ├── aifs_ens_forecast_YYYYMMDD_0000_member001_h000-072.grib
│       ├── aifs_ens_forecast_YYYYMMDD_0000_member001_h072-144.grib
│       ├── ...
│       └── aifs_ens_forecast_YYYYMMDD_0000_member050_h720-792.grib
```

### Path Format

- **Date format:** `YYYYMMDD_0000` (e.g., `20251127_0000`)
- **Input path:** `{date}/input/input_state_member_{NNN}.pkl`
- **Output path:** `{date}/forecasts/aifs_ens_forecast_{date}_member{NNN}_h{start}-{end}.grib`

## Usage

### Basic Usage

```bash
# Run full pipeline for 50 ensemble members
python automate_aifs_gpu_pipeline.py --date 20251127_0000 --members 1-50
```

### Examples

```bash
# Run specific members only
python automate_aifs_gpu_pipeline.py --date 20251127_0000 --members 1,5,10,25

# Run a subset (useful for testing or resuming)
python automate_aifs_gpu_pipeline.py --date 20251127_0000 --members 26-50

# Use more upload threads for faster GCS uploads
python automate_aifs_gpu_pipeline.py --date 20251127_0000 --members 1-50 --upload-threads 10

# Custom directories
python automate_aifs_gpu_pipeline.py --date 20251127_0000 --members 1-50 \
    --input-dir /scratch/inputs \
    --output-dir /scratch/outputs
```

### Command Line Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--date` | Required | Date string (YYYYMMDD_0000) |
| `--members` | `1-50` | Member range (1-50 or 1,2,3) |
| `--bucket` | `aifs-aiquest-us-20251127` | GCS bucket name |
| `--service-account` | `coiled-data.json` | Service account JSON |
| `--input-dir` | `/scratch/input_states` | Local input directory |
| `--output-dir` | `/scratch/ensemble_outputs` | Local output directory |
| `--upload-threads` | `5` | Upload threads per member |

**Note:** Cleanup is automatic after each member. No `--cleanup` flag needed.

## Pipeline Steps (Per Member)

For **each ensemble member**, these 4 steps execute sequentially:

### Step 1: Download Single .pkl File

Downloads ONE pickle file from GCS for the current member.

**Source script:** `download_pkl_from_gcs.py`

**Functions used:**
- `download_from_gcs()` - Download single file from GCS
- `verify_pickle_file()` - Validate pickle file integrity

**Behavior:**
- Downloads: `gs://{bucket}/{date}/input/input_state_member_{NNN}.pkl`
- Validates file integrity after download
- Skips download if valid local file exists

### Step 2: Run AIFS-ENS Model (GPU Inference)

Executes the AIFS model for the current member only.

**Source script:** `fp32_multi_run_AIFS_ENS_v1.py`

**Functions used:**
- `load_input_state_from_pickle()` - Load input from pickle
- `run_ensemble_member()` - Run forecast for one member

**Note:** The model is loaded ONCE at pipeline start and reused for all members.

**Output:** 11 GRIB files (792h ÷ 72h chunks):
```
aifs_ens_forecast_{date}_member{NNN}_h000-072.grib
aifs_ens_forecast_{date}_member{NNN}_h072-144.grib
...
aifs_ens_forecast_{date}_member{NNN}_h720-792.grib
```

### Step 3: Upload .grib Files to GCS

Uploads GRIB files for the current member to GCS.

**Source script:** `upload_aifs_gpu_output_grib_gcs.py`

**Class used:**
- `GCSGribUploaderMultiThreaded` - Thread-safe parallel uploader

**Destination:** `gs://{bucket}/{date}/forecasts/`

**Features:**
- Multi-threaded upload (default: 5 threads per member)
- 32MB upload chunks for efficiency

### Step 4: Cleanup Local Files

**CRITICAL:** Removes BOTH input and output files for the current member.

**Files deleted:**
- `input_state_member_{NNN}.pkl` (input)
- All `*_member{NNN}_*.grib` files (outputs)

This step runs **always**, even if upload had partial failures, to prevent storage accumulation.

## Complete Workflow

### End-to-End Process

```
CPU Machine (ETL)                    GPU Machine                     CPU Machine (ETL)
       │                                  │                                │
       │ ecmwf_opendata_pkl_...py         │                                │
       │ ─────────────────────────►       │                                │
       │ (Creates pkl, uploads to         │                                │
       │  GCS /input/)                    │                                │
       │                                  │                                │
       │                     automate_aifs_gpu_pipeline.py                 │
       │                                  │                                │
       │                        Step 1: Download pkl                       │
       │                        Step 2: Run AIFS model                     │
       │                        Step 3: Upload GRIB                        │
       │                                  │                                │
       │                                  │ ────────────────────────────►  │
       │                                  │ (GRIB files in GCS /forecasts/)│
       │                                  │                                │
       │                                  │                    aifs_n320_grib_1p5defg_nc.py
       │                                  │                    ensemble_quintile_analysis.py
       │                                  │                    forecast_submission.ipynb
```

### Typical Run Sequence

```bash
# 1. On CPU machine: Prepare input states
python ecmwf_opendata_pkl_input_aifsens.py

# 2. On GPU machine: Run automated pipeline
python automate_aifs_gpu_pipeline.py --date 20251127_0000 --members 1-50 --cleanup

# 3. Shutdown GPU machine to save costs

# 4. On CPU machine: Post-process and submit
python aifs_n320_grib_1p5defg_nc.py
python ensemble_quintile_analysis.py
```

## Architecture Benefits

### Code Reuse

The integrated script imports functions from existing scripts:

```python
# From download_pkl_from_gcs.py
from download_pkl_from_gcs import (
    download_from_gcs,
    parse_member_range,
    verify_pickle_file,
)

# From upload_aifs_gpu_output_grib_gcs.py
from upload_aifs_gpu_output_grib_gcs import (
    GCSGribUploaderMultiThreaded
)

# From fp32_multi_run_AIFS_ENS_v1.py (lazy loaded)
from fp32_multi_run_AIFS_ENS_v1 import (
    load_input_state_from_pickle,
    run_ensemble_member,
)
```

### Advantages

| Aspect | Benefit |
|--------|---------|
| **Per-member processing** | Prevents storage overflow on GPU machines |
| **Single command** | One script to run entire GPU workflow |
| **Code reuse** | No duplication, imports from existing scripts |
| **Maintainability** | Fix bugs in one place, propagates everywhere |
| **Automatic cleanup** | Deletes local files after each member |
| **Error resilience** | Failed member doesn't block others |
| **Progress tracking** | ETA and per-member timing |

### Per-Member Processing Flow

```
Model loaded once (reused for all members)
           │
           ▼
┌──────────────────────────────────────┐
│ Member 1:                            │
│   Download pkl → Inference → Upload  │
│   → Cleanup (delete pkl + grib)      │
└──────────────────────────────────────┘
           │
           ▼
┌──────────────────────────────────────┐
│ Member 2:                            │
│   Download pkl → Inference → Upload  │
│   → Cleanup (delete pkl + grib)      │
└──────────────────────────────────────┘
           │
           ▼
          ...
           │
           ▼
┌──────────────────────────────────────┐
│ Member 50:                           │
│   Download pkl → Inference → Upload  │
│   → Cleanup (delete pkl + grib)      │
└──────────────────────────────────────┘
           │
           ▼
      Pipeline complete
```

## Storage Requirements

### Per-Member Processing (Current Design)

| Item | Size | Notes |
|------|------|-------|
| 1 pkl file | ~500 MB | Downloaded, then deleted |
| 11 grib files | ~2 GB | Generated, uploaded, then deleted |
| **Max at any time** | **~2.5 GB** | Minimal footprint |

### Comparison: Batch Processing (NOT Used)

| Item | Size | Notes |
|------|------|-------|
| 50 pkl files | ~25 GB | All downloaded first |
| 550 grib files | ~100 GB | All generated before upload |
| **Total required** | **~125 GB** | Would overflow most GPU instances |

**Conclusion:** Per-member processing reduces storage needs from ~125 GB to ~2.5 GB.

## Estimated Costs and Times

| Step | Time | Cost (a2-ultragpu-1g) |
|------|------|----------------------|
| Download (per member) | ~15 sec | included |
| Model (per member) | ~8 min | included |
| Upload (per member) | ~30 sec | included |
| **Total (50 members)** | **~7 hours** | **~$37-44** |

## Troubleshooting

### Common Issues

1. **Service account not found**
   ```
   ERROR: Service account not found: coiled-data.json
   ```
   Ensure `coiled-data.json` is in the working directory.

2. **GPU out of memory**
   Enable memory optimization in `fp32_multi_run_AIFS_ENS_v1.py`:
   ```python
   GPU_MEMORY_OPTIMIZATION = True
   INFERENCE_PRECISION = "16"
   INFERENCE_NUM_CHUNKS = 16
   ```

3. **Download failures**
   Check GCS bucket permissions and network connectivity.
   The pipeline will report which members failed - rerun with just those members.

4. **Partial run recovery**
   If pipeline fails at member N, simply restart with remaining members:
   ```bash
   # Original run failed at member 25
   # Restart from member 25 onwards
   python automate_aifs_gpu_pipeline.py --date 20251127_0000 --members 25-50
   ```

5. **Storage issues**
   Per-member processing prevents storage overflow. If you still see issues:
   - Check that cleanup is running (look for `[CLEANUP]` in logs)
   - Verify `/scratch` has at least 5GB free space
   - Check for orphan files from previous failed runs:
     ```bash
     rm -f /scratch/input_states/*.pkl
     rm -f /scratch/ensemble_outputs/*.grib
     ```

## Dependencies

```
anemoi-inference
earthkit-data
earthkit-regrid
google-cloud-storage
numpy
```

## File Reference

| File | Location | Purpose |
|------|----------|---------|
| `automate_aifs_gpu_pipeline.py` | GPU machine | Orchestration script |
| `download_pkl_from_gcs.py` | GPU machine | Download functions |
| `fp32_multi_run_AIFS_ENS_v1.py` | GPU machine | Model execution |
| `upload_aifs_gpu_output_grib_gcs.py` | GPU machine | Upload functions |
| `ecmwf_opendata_pkl_input_aifsens.py` | CPU machine | Input preparation |
| `coiled-data.json` | Both | GCS service account key |

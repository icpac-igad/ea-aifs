# ERA5T FP16 Fahamu AIFS v1 Pipeline

End-to-end pipeline for running AIFS ensemble forecasts using CEDA ERA5T data
with FP16 precision inference. Covers pkl creation through AI Weather Quest submission.

## Pipeline Overview

```
Step 1: CEDA ERA5T → pkl (74 fields, 10 members)
Step 2: pkl → GRIB (GPU inference, FP16, 960h forecast, 74→92 field adaptation)
Step 3: GRIB → 1.5° NetCDF (regrid N320 → 1.5°, extract msl/tp/2t)
Step 4: NetCDF → Quintile probabilities
Step 5: Submit to AI Weather Quest
```

## Key Differences from Standard Pipeline

| Aspect | Standard (ECMWF Open Data) | ERA5T (CEDA) |
|--------|---------------------------|--------------|
| Input fields | 92 fields | 74 fields (adapted to 92 at inference) |
| Members | 1-50 | 0-9 (10 EDA members) |
| Surface params | 10u,10v,2d,2t,msl,skt,sp,tcw | 10u,10v,2t,msl,tcwv |
| PL params | t,q,z,u,v,w × 13 levels | t,q,z,u,v × 13 levels |
| Soil temps | stl1, stl2 | not available |
| Constants | lsm,z,slor,sdor | lsm,z,slor,sdor (from ECMWF Open Data) |
| Lead time | 792h (33 days) | 960h (40 days) |
| Init date | Same as forecast date | Different (e.g., 20260227 init → 20260305 target) |

## Field Adaptation (74 → 92)

The `adapt_era5t_fields()` function in `era5t_fp16_multi_run_AIFS_ENS_v1.py`
automatically derives the 18 missing fields at inference time:

| Derived Field | Source | Method |
|---------------|--------|--------|
| tcw | tcwv | Rename (vapor is dominant component) |
| sp | msl | Copy (surface pressure ≈ mean sea level pressure) |
| 2d | 2t | 2t - 2K (conservative dewpoint depression) |
| skt | 2t | Copy (skin temp ≈ 2m temp) |
| stl1, stl2 | 2t | Copy (soil temp ≈ surface temp) |
| w × 13 levels | — | Zeros (vertical velocity is small) |

---

## Step 1: Create pkl Files from CEDA ERA5T

Downloads ERA5T EDA surface fields (per-member) and deterministic pressure level
fields from CEDA, interpolates model levels to pressure levels, and packages
into pkl files with 74 fields each.

```bash
python ceda_era5t_pkl_input_aifsens.py
```

- **Input**: CEDA ERA5T archive (requires `ceda_token` env var)
- **Output**: `gs://aifs-aiquest-us-20251127/era5t/YYYYMMDD/input_state_member_000.pkl` ... `_009.pkl`
- **Config**: Edit `DATE` in script (e.g., `datetime(2026, 2, 27, 0, 0)`)
- **Fields**: 5 EDA surface + 65 PL (deterministic, shared) + 4 constants = 74

## Step 2: GPU Inference (pkl → GRIB)

Runs AIFS-ENS-1.0 model in FP16 precision. Downloads pkl from GCS, runs
inference with field adaptation, uploads GRIB output, cleans up. One member
at a time to minimize storage.

```bash
python era5t_fp16_automate_aifs_gpu_pipeline.py \
    --date 20260305_0000 \
    --members 0-9 \
    --gcs-input-prefix era5t/20260227 \
    --gcs-output-subpath era5t_fp16_forecasts \
    --lead-time 960
```

- **Input**: `gs://aifs-aiquest-us-20251127/era5t/20260227/input_state_member_*.pkl`
- **Output**: `gs://aifs-aiquest-us-20251127/20260305_0000/era5t_fp16_forecasts/*.grib`
- **Requires**: GPU with ≥24GB VRAM
- **Time**: ~15-20 min per member (960h forecast)

### Arguments

| Argument | Description | Example |
|----------|-------------|---------|
| `--date` | Target forecast date (GCS folder) | `20260305_0000` |
| `--members` | Member range | `0-9` |
| `--gcs-input-prefix` | GCS path to pkl files | `era5t/20260227` |
| `--gcs-output-subpath` | GCS subfolder for GRIB output | `era5t_fp16_forecasts` |
| `--lead-time` | Forecast hours | `960` |

## Step 3: GRIB → 1.5° NetCDF

Regrids N320 GRIB files to 1.5° regular lat/lon and extracts msl, tp, 2t
variables. Processes each member in a subprocess for clean file handle management.

**Important**: `--init-date` must match the ERA5T initialization date used in
the GRIB filenames (not the `--date` folder).

```bash
python era5t_aifs_n320_grib_1p5deg_nc_cli.py \
    --date 20260305_0000 \
    --members 0-9 \
    --gcs-input-subpath era5t_fp16_forecasts \
    --gcs-output-subpath era5t_fp16_1p5deg_nc \
    --init-date 20260227
```

- **Input**: `gs://aifs-aiquest-us-20251127/20260305_0000/era5t_fp16_forecasts/aifs_ens_forecast_20260227_0000_member*_h*.grib`
- **Output**: `gs://aifs-aiquest-us-20251127/20260305_0000/era5t_fp16_1p5deg_nc/aifs_ensemble_forecast_1p5deg_member*.nc`
- **Time ranges**: Days 18-40 (h432-h960) in 72h chunks

### Arguments

| Argument | Description | Example |
|----------|-------------|---------|
| `--date` | GCS folder date | `20260305_0000` |
| `--init-date` | ERA5T init date (for GRIB filenames) | `20260227` |
| `--members` | Member range | `0-9` |
| `--gcs-input-subpath` | GRIB input subfolder | `era5t_fp16_forecasts` |
| `--gcs-output-subpath` | NetCDF output subfolder | `era5t_fp16_1p5deg_nc` |
| `--no-upload` | Skip GCS upload (for testing) | — |

## Step 4: Quintile Analysis

Downloads ensemble NetCDF files from GCS, computes weekly mean/sum per member,
and calculates quintile probabilities against 20-year climatology.

```bash
python era5t_ensemble_quintile_analysis_cli.py \
    --date 20260305 \
    --members 0-9 \
    --fp16
```

Or with a custom GCS prefix:

```bash
python era5t_ensemble_quintile_analysis_cli.py \
    --date 20260305 \
    --members 0-9 \
    --gcs-prefix 20260305_0000/era5t_fp16_1p5deg_nc/
```

- **Input**: `gs://aifs-aiquest-us-20251127/20260305_0000/era5t_fp16_1p5deg_nc/*.nc`
- **Output**: `era5t_ensemble_quintile_probabilities_20260305_fp16.nc`
- **Variables**: mslp (msl), pr (tp), tas (2t) × week1, week2

## Step 5: Forecast Submission

Submits quintile probabilities to AI Weather Quest for all variables and weeks.

```bash
python era5t_forecast_submission_cli.py --date 20260305
```

Dry run (no actual submission):

```bash
python era5t_forecast_submission_cli.py --date 20260305 --dry-run
```

### Environment Variables (.env file)

```
AIWQ_TEAM_NAME=YourTeamName
AIWQ_MODEL_NAME_ERA5T=FahamuAIFSv1_ERA5T
AIWQ_MODEL_NAME=FahamuAIFSv1
AIWQ_PASSWORD=YourPassword
```

---

## GCS Path Structure

```
gs://aifs-aiquest-us-20251127/
├── era5t/
│   └── 20260227/                          # Step 1 output (pkl files)
│       ├── input_state_member_000.pkl
│       ├── input_state_member_001.pkl
│       └── ...
└── 20260305_0000/
    ├── era5t_fp16_forecasts/              # Step 2 output (GRIB files)
    │   ├── aifs_ens_forecast_20260227_0000_member000_h000-072.grib
    │   ├── aifs_ens_forecast_20260227_0000_member000_h072-144.grib
    │   └── ...
    └── era5t_fp16_1p5deg_nc/              # Step 3 output (NetCDF files)
        ├── aifs_ensemble_forecast_1p5deg_member000.nc
        ├── aifs_ensemble_forecast_1p5deg_member001.nc
        └── ...
```

## Scripts Reference

| Script | Purpose |
|--------|---------|
| `ceda_era5t_pkl_input_aifsens.py` | CEDA ERA5T → pkl (74 fields, 10 members) |
| `era5t_fp16_automate_aifs_gpu_pipeline.py` | GPU pipeline orchestrator |
| `era5t_fp16_multi_run_AIFS_ENS_v1.py` | FP16 inference with field adaptation |
| `era5t_aifs_n320_grib_1p5deg_nc_cli.py` | GRIB → 1.5° NetCDF regridding |
| `era5t_ensemble_quintile_analysis_cli.py` | Quintile probability calculation |
| `era5t_forecast_submission_cli.py` | AI Weather Quest submission |
| `download_pkl_from_gcs.py` | Shared: pkl download + verification (threshold ≥70) |
| `upload_aifs_gpu_output_grib_gcs.py` | Shared: multi-threaded GRIB upload |

## Quick Run (All Steps)

```bash
# Step 1: Create pkl files (run on CPU machine with CEDA access)
python ceda_era5t_pkl_input_aifsens.py

# Step 2: GPU inference (run on GPU machine, ≥24GB VRAM)
python era5t_fp16_automate_aifs_gpu_pipeline.py --date 20260305_0000 --members 0-9 --gcs-input-prefix era5t/20260227 --gcs-output-subpath era5t_fp16_forecasts --lead-time 960

# Step 3: GRIB to NetCDF (run on CPU machine)
python era5t_aifs_n320_grib_1p5deg_nc_cli.py --date 20260305_0000 --members 0-9 --gcs-input-subpath era5t_fp16_forecasts --gcs-output-subpath era5t_fp16_1p5deg_nc --init-date 20260227

# Step 4: Quintile analysis
python era5t_ensemble_quintile_analysis_cli.py --date 20260305 --members 0-9 --fp16

# Step 5: Submit
python era5t_forecast_submission_cli.py --date 20260305
```

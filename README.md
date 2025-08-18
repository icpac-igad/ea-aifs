# ea-aifs
ECMWF AIFS ensemble model run routines for AI weather forecasting competition.

## AI Forecast Submission Workflow

This repository contains a complete workflow for generating and submitting AI weather forecasts using ECMWF's AIFS ensemble model. The process involves 7 sequential steps spanning CPU preprocessing, GPU model execution, and cloud-based post-processing to produce ensemble forecasts for competition submission.

**Quick Overview:**
1. Download ECMWF data → preprocess to pickle files → upload to GCS
2. GPU: Download pickle files → run AI model → generate GRIB forecasts → upload to GCS  
3. ETL: Download GRIB files → validate → analyze ensemble → submit forecasts

📋 **[Complete Workflow Documentation](docs/AI_FORECAST_WORKFLOW.md)**

## Core Workflow Scripts

### Sequential Execution Order:
1. `ecmwf_opendata_pkl_input_aifsens.py` - Prepare initial conditions (CPU)
2. `download_pkl_from_gcs.py` - Transfer to GPU environment  
3. `multi_run_AIFS_ENS_v1.py` - Run AI model for 50 ensemble members (GPU)
4. `upload_aifs_gpu_output_grib_gcs.py` - Upload forecast outputs to cloud
5. `download_grib_from_gcs.py` - Download for post-processing (ETL)
6. `aifs_792hr_forecast_grib_check_vars.py` - Validate forecast data
7. `ensemble_quintile_analysis.py` - Analyze and submit forecasts (**final submission**)

## Documentation

See the [docs](docs/) folder for detailed documentation:

- **[AI FORECAST WORKFLOW](docs/AI_FORECAST_WORKFLOW.md)** - Complete workflow guide
- [CONTEXT EXPLANATION](docs/CONTEXT_EXPLANATION.md)
- [DETAILED GRIB SAVING EXPLANATION](docs/DETAILED_GRIB_SAVING_EXPLANATION.md)
- [DETAILED NETCDF SAVING EXPLANATION](docs/DETAILED_NETCDF_SAVING_EXPLANATION.md)
- [EARTHKIT DATA USAGE EXPLANATION](docs/EARTHKIT_DATA_USAGE_EXPLANATION.md)
- [FORECAST OUTPUTS DOCUMENTATION](docs/FORECAST_OUTPUTS_DOCUMENTATION.md)

## Development & Testing

Unit tests and development utilities are located in the `unittests/` folder:

## Jupyter Notebooks

- `multi_run_AIFS_ENS_v1.ipynb` - Interactive notebook for multi-run AIFS ensemble
- `run_AIFS_ENS_v1.ipynb` - Interactive notebook for single AIFS ensemble run

## Docker Environment

The project includes a Docker environment in the `env/` folder for running AIFS models with CUDA support. See [docs/](docs/) for Docker setup and Coiled integration notes. 

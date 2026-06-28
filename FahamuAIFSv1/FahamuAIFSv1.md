# FahamuAIFSv1 (FP32 / ECMWF Open Data)

The baseline Fahamu pipeline: AIFS-ENS-1.0 ensemble forecasts driven by **ECMWF Open
Data** initial conditions, run at **full FP32 precision** on an A100-class GPU
(`a2-ultragpu-1g`, ~80 GB VRAM). 50 members, 792 h (33-day) forecast. Submits to AI
Weather Quest under model name `FahamuAIFSv1` (`.env`: `AIWQ_MODEL_NAME`).

This model shares Steps 1, 3, 4 and 5 with [`../fp16FahamuAIFSv1`](../fp16FahamuAIFSv1/fp16FahamuAIFSv1.md);
they differ only in the GPU inference step (FP32 vs FP16). The ERA5T variant is fully
separate — see [`../era5tFp16FahamuAIFSv1`](../era5tFp16FahamuAIFSv1/era5tFp16FahamuAIFSv1.md).

## Scripts

| Script | Step | Role |
|--------|------|------|
| `FahamuAIFSv1/automate_aifs_gpu_pipeline.py` | 2 | GPU pipeline orchestrator (download → inference → upload → cleanup, per member) |
| `FahamuAIFSv1/fp32_multi_run_AIFS_ENS_v1.py` | 2 | AIFS-ENS model runner (FP32) — imported by the orchestrator |
| `FahamuAIFSv1/pytorch_profile_fp32.py` | — | FP32 GPU memory/throughput profiling helper |
| `shared/ecmwf_opendata_pkl_input_aifsens.py` | 1 | ECMWF Open Data → input-state pkl (members 1–50) |
| `shared/aifs_n320_grib_1p5defg_nc_cli.py` | 3 | GRIB → 1.5° NetCDF regrid (run **without** `--fp16`) |
| `shared/ensemble_quintile_analysis_cli.py` | 4 | Weekly quintile probabilities vs 20-yr climatology |
| `shared/forecast_submission_cli.py` | 5 | AI Weather Quest submission |
| `shared/aiwq_individual_files_cli.py` | 5 (alt) | Build + zip AI-WQ individual files without live submission |
| `shared/download_pkl_from_gcs.py` | — | Shared: pkl download + verification |
| `shared/upload_aifs_gpu_output_grib_gcs.py` | — | Shared: multi-threaded GRIB upload |

GCS output: GRIB → `YYYYMMDD_0000/forecasts/`, NetCDF → `YYYYMMDD_0000/1p5deg_nc/`.

## Quick Run

Run from the repo root. (On the GPU machine, copy the `FahamuAIFSv1/` and `shared/`
scripts plus the `coiled-data.json` GCS key alongside each other.)

```bash
# Step 1 — Initial conditions (ETL machine, CPU)
python shared/ecmwf_opendata_pkl_input_aifsens.py

# Step 2 — FP32 GPU inference (A100, ~80GB VRAM)
python FahamuAIFSv1/automate_aifs_gpu_pipeline.py --date 20260129_0000 --members 1-50

# Step 3 — GRIB → 1.5° NetCDF (ETL machine; FP32 = no --fp16)
python shared/aifs_n320_grib_1p5defg_nc_cli.py --date 20260129          # add --max-workers N on multi-core hosts

# Step 4 — Quintile analysis
python shared/ensemble_quintile_analysis_cli.py --date 20260129

# Step 5 — Submit (within init date → init+3 days)
python shared/forecast_submission_cli.py --date 20260129                 # --dry-run to validate
```

**Required `.env`:** `AIWQ_TEAM_NAME`, `AIWQ_MODEL_NAME=FahamuAIFSv1`, `AIWQ_PASSWORD`
(plus `coiled-data.json` for GCS). **Shut the GPU notebook down after Step 2.**

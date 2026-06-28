# fp16FahamuAIFSv1 (FP16 / ECMWF Open Data)

The half-precision Fahamu pipeline: AIFS-ENS-1.0 ensemble forecasts driven by **ECMWF
Open Data** initial conditions, run at **FP16** so the model fits in <24 GB VRAM on a
cheaper GPU (G2 `g2-standard-12` / L4, also A10G, RTX 4090). 50 members, 792 h (33-day)
forecast. Submits to AI Weather Quest under model name `FahamuAIFSv1_fp16`
(`.env`: `AIWQ_MODEL_NAME_FP16`).

Identical to [`../FahamuAIFSv1`](../FahamuAIFSv1/FahamuAIFSv1.md) except for the GPU
inference step (FP16 vs FP32) and the `--fp16` flag on Steps 3–5, which routes the
`fp16_*` GCS subfolders. The ERA5T variant is fully separate — see
[`../era5tFp16FahamuAIFSv1`](../era5tFp16FahamuAIFSv1/era5tFp16FahamuAIFSv1.md).

## Scripts

| Script | Step | Role |
|--------|------|------|
| `fp16FahamuAIFSv1/fp16_automate_aifs_gpu_pipeline.py` | 2 | GPU pipeline orchestrator, FP16 (download → inference → upload → cleanup, per member) |
| `fp16FahamuAIFSv1/fp16_multi_run_AIFS_ENS_v1.py` | 2 | AIFS-ENS model runner (FP16) — imported by the orchestrator |
| `fp16FahamuAIFSv1/pytorch_profile_fp16.py` | — | FP16 GPU memory/throughput profiling helper |
| `shared/ecmwf_opendata_pkl_input_aifsens.py` | 1 | ECMWF Open Data → input-state pkl (members 1–50) |
| `shared/aifs_n320_grib_1p5defg_nc_cli.py` | 3 | GRIB → 1.5° NetCDF regrid (run with `--fp16`) |
| `shared/ensemble_quintile_analysis_cli.py` | 4 | Weekly quintile probabilities vs 20-yr climatology (`--fp16`) |
| `shared/forecast_submission_cli.py` | 5 | AI Weather Quest submission (`--fp16`) |
| `shared/aiwq_individual_files_cli.py` | 5 (alt) | Build + zip AI-WQ individual files without live submission (`--fp16`) |
| `shared/download_pkl_from_gcs.py` | — | Shared: pkl download + verification |
| `shared/upload_aifs_gpu_output_grib_gcs.py` | — | Shared: multi-threaded GRIB upload |

GCS output: GRIB → `YYYYMMDD_0000/fp16_forecasts/`, NetCDF → `YYYYMMDD_0000/fp16_1p5deg_nc/`.

## Quick Run

Run from the repo root. (On the GPU machine, copy the `fp16FahamuAIFSv1/` and `shared/`
scripts plus the `coiled-data.json` GCS key alongside each other.)

```bash
# Step 1 — Initial conditions (ETL machine, CPU) — same opendata pkl as FP32
python shared/ecmwf_opendata_pkl_input_aifsens.py

# Step 2 — FP16 GPU inference (G2/L4, <24GB VRAM)
python fp16FahamuAIFSv1/fp16_automate_aifs_gpu_pipeline.py --date 20260129_0000 --members 1-50

# Step 3 — GRIB → 1.5° NetCDF (ETL machine; --fp16)
python shared/aifs_n320_grib_1p5defg_nc_cli.py --date 20260129 --fp16     # add --max-workers N on multi-core hosts

# Step 4 — Quintile analysis
python shared/ensemble_quintile_analysis_cli.py --date 20260129 --fp16

# Step 5 — Submit (within init date → init+3 days)
python shared/forecast_submission_cli.py --date 20260129 --fp16           # --dry-run to validate
# or build + zip without submitting:
python shared/aiwq_individual_files_cli.py --date 20260129 --fp16
```

**Required `.env`:** `AIWQ_TEAM_NAME`, `AIWQ_MODEL_NAME_FP16=FahamuAIFSv1_fp16`,
`AIWQ_PASSWORD` (plus `coiled-data.json` for GCS). **Shut the GPU notebook down after
Step 2.**

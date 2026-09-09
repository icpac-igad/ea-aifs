# AIFS ENS v2.0 Run Commands — 20260716

**Date:** 20260716 (Thursday) · **Members:** 1–50 · **Box:** local RTX 5000 Ada, 30 GB.

Entire cycle — **pkl creation → GPU inference → regrid → quintiles → submission-ready
NetCDF** — run **on the local box, no GCS at any step** (input pkls made locally, inference
to a local Icechunk store, downstream read from local files). This is the first cycle where
Step 1 was also generated locally (20260709 pulled existing pkls from GCS).

See [`run_commands_20260709.md`](run_commands_20260709.md) for the flag rationale,
[`LOCAL_GPU_RUN.md`](LOCAL_GPU_RUN.md) for the env, and
[`ICECHUNK_COMMIT_CADENCE.md`](ICECHUNK_COMMIT_CADENCE.md) for the store layout.

---

## ⏱ End-to-end timing (pkl creation → submission-ready NetCDF)

| Step | What | Wall time | Clock (measured) | Output |
|------|------|-----------|------------------|--------|
| **1** | pkl creation, 50 members (local) | **10 h 27 m** | 12:19 → 22:44 (Jul 16) | 50 × 0.97 GB = **48 GB** |
| **2** | GPU inference → Icechunk, 50 members | **3 h 55 m** | 19:03 → 22:58 (Jul 16) | **584 GB** store |
| **3a** | regrid 432–792 h → 1.5° NetCDF | **8 m 49 s** | 04:05 → 04:14 (Jul 17) | 50 × 199 MB = **9.8 GB** |
| **3b** | quintiles → **submission NetCDF** | **2 m 03 s** | 04:14 → 04:16 (Jul 17) | **6.7 MB** |
| 3c | submission dry-run (no submit) | ~1 s | 04:16 (Jul 17) | 6/6 validated |

- **End-to-end elapsed (Step 1 start → submission-ready NetCDF): 15 h 58 m** (12:19 Jul 16 → 04:16 Jul 17).
- **Net processing ≈ 10 h 51 m** — the elapsed figure includes a **~5 h 08 m idle gap** (22:58 → 04:05)
  during which the GPU was running an *unrelated* cGAN training and the operator was away; no AIFS
  work happened in that window.
- **The "submission-ready NetCDF"** is `ensemble_quintile_probabilities_20260716_v2.nc` (6.7 MB) —
  the file 3c submits.

### Why the numbers look the way they do
- **Steps 1 & 2 overlapped.** Inference started at 19:03 while Step 1 was at 46/50, consuming pkls
  as they were written (pipelined) — so Step 2 finished only ~14 min after Step 1.
- **Step 1 was ~2× its usual pace** (10.4 h vs a solo ~5 h). Early members ran ~5–6 min each; once
  GPU inference started at 19:03 it competed for network/CPU and the open-data downloads for the
  last members slowed to ~70–100 min each. Running Step 1 alone (or before Step 2) avoids this.
- **3a is 9 min, not "hours."** The old GRIB path pulled ~8 GB of GRIB per member from GCS and
  regridded ~1416 fields; the Icechunk path reads N320 locally and regrids only the 3 surface
  params (`msl/tp/2t`) → **10.6 s/member** (identical to 20260709's 8 m 37 s — verified byte-for-byte
  same NetCDF sizes).

---

## Environment

```bash
export MAMBA_ROOT_PREFIX=/tank/projects/micromamba
PY=/tank/projects/micromamba/envs/aifs-gpu/bin/python
export HF_HOME=/tank/projects/hf_cache
BASE=/tank/projects/aifs-run/20260716_0000
cd /home/ezra/ea-aifs/fp16FahamuAIFSv2      # so .env + coiled-data.json resolve
```

## Step 1 — input pkls, local (no GCS upload)

```bash
$PY ecmwf_opendata_pkl_input_aifsens_v2.py --date 20260716 --members 1-50 \
    --no-upload --keep-local --out-dir $BASE/input_states
```
50/50 saved, each exactly 971,413,663 B (uniform size = all complete). Runs in the `aifs-gpu`
env (it has the ETL deps). Mirror is hardcoded to AWS S3 — there is no `--source` flag.

## Step 2 — GPU inference → local Icechunk store

Differs from 20260709 in two ways, because the pkls are **local** (not in GCS):
**no `--gcs-fetch`** (would 404) and **no `--cleanup-pkl`** (local-only pkls, ~10 min each to
regenerate; 1.7 TB free). Run as a resumable loop so any member whose pkl isn't ready yet is
retried next pass (Step 1 may still be finishing).

```bash
# one pass (the loop just repeats this with --skip-existing until 50/50):
$PY run_local_icechunk_v2.py \
    --date 20260716_0000 --members 1-50 --lead-time 792 \
    --input-dir $BASE/input_states --store $BASE/icechunk_v2 \
    --n-members 50 --commit-every 1 --float-size f4 \
    --write-hours 432-792 --skip-existing
```
Result: **50/50 members, 3052 commits, 584 GB**, tag `cycle-20260716_0000` → 50/50
(clean tag — the whole ensemble ran in one driver, so no partial-snapshot tag trap). A single
pass completed all 50 (inference reached each member after its pkl already existed).

## Step 3 — downstream (all CPU, no GCS; ran alongside the cGAN GPU job)

```bash
# 3a — regrid 432-792h -> 1.5° NetCDF, straight from the Icechunk store
$PY ../shared/aifs_n320_grib_1p5defg_nc_cli.py \
    --date 20260716_0000 --members 1-50 --v2 --no-upload \
    --source icechunk --icechunk-store $BASE/icechunk_v2 \
    --icechunk-tag cycle-20260716_0000 --output-dir $BASE/nc_1p5deg

# 3b — quintiles. --work-dir homes ALL per-cycle artifacts under $BASE/aiwq (quintile
# file + climatology + temp store), NOT the repo folder. Still run from fp16FahamuAIFSv2/
# so .env resolves -> AIWQ_PASSWORD.
$PY ../shared/ensemble_quintile_analysis_cli.py \
    --date 20260716 --v2 --local-nc-dir $BASE/nc_1p5deg --work-dir $BASE/aiwq

# 3c — submit (dry-run first). --output-dir points at the same folder for the quintile file.
# Window is start_date -> +3 days (20260716-20260719).
$PY ../shared/forecast_submission_cli.py --date 20260716 --v2 \
    --output-dir $BASE/aiwq --dry-run          # then drop --dry-run to submit
```

3a: 50/50 NetCDFs. 3b: `ensemble_quintile_probabilities_20260716_v2.nc`, dims
`(time_week=2, quintile=5, lat=121, lon=240)`, vars `2t`/`tp`/`msl` — all **in [0,1] and
summing to 1** across the 5 bins. 3b's climatology FTP fetch (valid dates 20260803, 20260810)
needs `AIWQ_PASSWORD` from `.env`; both CLIs now resolve `.env` from the working directory
(`find_dotenv(usecwd=True)`). 3c dry-run: **6/6**, Team `Fahamu`, Model `fp16FahamuAIFSv2`.

> **`--work-dir` (new).** `ensemble_quintile_analysis_cli.py` used to scatter its outputs
> (quintile `.nc`, 6 climatology `.nc`, ~1.6 GB temp `ensemble_icechunk_store/`) into the
> current directory. `--work-dir DIR` homes all three under `DIR`; `--output-dir`/`--clim-dir`
> still override individually and everything defaults to `--work-dir` (which defaults to `./`,
> so old behaviour is unchanged). Point `3b --work-dir` and `3c --output-dir` at `$BASE/aiwq`
> to keep the whole cycle's artifacts in one place next to `icechunk_v2/`.

---

## Status

- **Steps 1 → 3b: ✅ complete and verified.** Submission-ready file:
  `/tank/projects/aifs-run/20260716_0000/aiwq/ensemble_quintile_probabilities_20260716_v2.nc`
  (relocated from the repo folder via `--work-dir`).
- **3c: dry-run only (6/6).** The live submission is a separate, explicit step (outward/irreversible);
  run `forecast_submission_cli.py --date 20260716 --v2` (no `--dry-run`), launched **detached**, then
  verify server-side that the 6 files land under `ftp.ecmwf.int:/forecast_submissions/20260716/`.

## Outputs

| Path | Size | What |
|------|------|------|
| `/tank/projects/aifs-run/20260716_0000/input_states/` | 48 GB | 50 input pkls |
| `/tank/projects/aifs-run/20260716_0000/icechunk_v2/` | 584 GB | Icechunk store, tag `cycle-20260716_0000` |
| `/tank/projects/aifs-run/20260716_0000/nc_1p5deg/` | 9.8 GB | 50 × 1.5° NetCDF (432–792 h) |
| `/tank/projects/aifs-run/20260716_0000/aiwq/` | ~1.6 GB | AI-WQ artifacts (via `--work-dir`): 6 climatology `.nc`, temp store, and ↓ |
| `…/aiwq/ensemble_quintile_probabilities_20260716_v2.nc` | 6.7 MB | **submission file** (3c reads it via `--output-dir …/aiwq`) |

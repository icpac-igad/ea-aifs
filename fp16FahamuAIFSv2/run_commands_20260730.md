# AIFS ENS v2.0 Run Commands — 20260730

**Date:** 20260730 (Thursday) · **Members:** 1–50 · **Box:** local RTX 5000 Ada, 30 GB.
**Outcome: ✅ submitted** — all 6 files on `ftp.ecmwf.int/forecast_submissions/20260730/`
(`{mslp,pr,tas} × {p1,p2}`, `Fahamu / fp16FahamuAIFSv2`).

Same fully-local pipeline as [`run_commands_20260723.md`](run_commands_20260723.md)
(no GCS: local pkls → local Icechunk store → local downstream). This cycle ran **clean** —
the 20260723 incidents (climatology-dir bug, FTP retry) did not recur.

---

## ⏱ End-to-end timing

| Step | What | Wall time | Output |
|------|------|-----------|--------|
| **1** | pkl creation, 50 members (local) | **14 h 30 m** (52 184 s) | 42 GB |
| **2** | GPU inference → Icechunk | **3 h 55 m** (14 124 s) → ~283 s/member | 583 GB |
| **3a** | regrid 432–792 h → 1.5° NetCDF | **8 m 44 s** | 2.0 GB |
| **3b** | quintiles → submission NetCDF | **~2 m** (clean, one pass) | 1.6 GB (`aiwq/`) |
| **3c** | **live submit** | **6/6 in 247 s, 0 retries** | 6 files on FTP |

Step 1 was slow (14.5 h) — worse AWS S3 throttling than 20260723's 7.3 h, same root cause
(the path to AWS eu-central-1, ~1–3 Mbps; see the 20260723 runbook's bandwidth note). It
still completed with `EXIT=0`, 50/50, no manual restart.

---

## Commands (as run)

```bash
export MAMBA_ROOT_PREFIX=/tank/projects/micromamba
PY=/tank/projects/micromamba/envs/aifs-gpu/bin/python
export HF_HOME=/tank/projects/hf_cache
BASE=/tank/projects/aifs-run/20260730_0000
cd /home/ezra/ea-aifs/fp16FahamuAIFSv2

# --- Step 1: input pkls, local (setsid so a session exit can't kill the ~14h job) ---
setsid nohup bash -c "cd $PWD; \
  $PY -u ecmwf_opendata_pkl_input_aifsens_v2.py --date 20260730 --members 1-50 \
    --no-upload --keep-local --out-dir $BASE/input_states" \
  > /tank/projects/step1_20260730.log 2>&1 </dev/null &

# --- Step 2: GPU inference -> local Icechunk store (setsid) ---
setsid nohup bash -c "cd $PWD; export HF_HOME=$HF_HOME; \
  $PY -u run_local_icechunk_v2.py --date 20260730_0000 --members 1-50 --lead-time 792 \
    --input-dir $BASE/input_states --store $BASE/icechunk_v2 \
    --n-members 50 --commit-every 1 --float-size f4 --write-hours 432-792 --skip-existing" \
  > /tank/projects/run_0730_all.log 2>&1 </dev/null &

# --- Step 3a: regrid from the Icechunk store ---
$PY ../shared/aifs_n320_grib_1p5defg_nc_cli.py --date 20260730_0000 --members 1-50 --v2 \
    --no-upload --source icechunk --icechunk-store $BASE/icechunk_v2 \
    --icechunk-tag cycle-20260730_0000 --output-dir $BASE/nc_1p5deg

# --- Step 3b: quintiles (--work-dir homes quintile + climatology + temp store under aiwq/) ---
$PY ../shared/ensemble_quintile_analysis_cli.py --date 20260730 --v2 \
    --local-nc-dir $BASE/nc_1p5deg --work-dir $BASE/aiwq

# --- Step 3c: submit (window 20260730 -> 20260802 inclusive; submitted on 08-02) ---
$PY ../shared/forecast_submission_cli.py --date 20260730 --v2 --output-dir $BASE/aiwq --dry-run
$PY ../shared/forecast_submission_cli.py --date 20260730 --v2 --output-dir $BASE/aiwq   # live
```

Steps 3a/3b/3c were chained in one script (`run_0730_downstream.sh`) that stops on the first
failure; the live 3c was a separate explicit step after the dry-run.

---

## Notes

- **Clean run.** The 20260723 climatology-dir bug is fixed (`c6bf748`) — 3b downloaded the
  climatology (valid dates 20260817/20260824) into `aiwq/` and found it, one pass. 3c
  submitted 6/6 with no FTP retry. Submitted on **08-02**, the last day of the window.
- **`setsid` from the start** for both long jobs (Step 1 ~14.5 h, Step 2 ~4 h) — the
  20260723 lesson (a `nohup`-only job was killed by session teardown).
- **Disk pressure.** Two full ~586 GB stores now coexist (20260723 + 20260730). Both are
  submitted, so purge the older with
  `python cleanup_aifs_run.py --date 20260723 --yes` (keeps `aiwq/*.nc`).

## Outputs

| Path | Size | What |
|------|------|------|
| `…/20260730_0000/input_states/` | 42 GB | 50 input pkls |
| `…/20260730_0000/icechunk_v2/` | 583 GB | Icechunk store, tag `cycle-20260730_0000` (50/50) |
| `…/20260730_0000/nc_1p5deg/` | 2.0 GB | 50 × 1.5° NetCDF (432–792 h) |
| `…/20260730_0000/aiwq/ensemble_quintile_probabilities_20260730_v2.nc` | 6.7 MB | **submitted file** |

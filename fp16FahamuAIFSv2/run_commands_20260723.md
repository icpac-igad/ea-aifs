# AIFS ENS v2.0 Run Commands — 20260723

**Date:** 20260723 (Thursday) · **Members:** 1–50 · **Box:** local RTX 5000 Ada, 30 GB.
**Outcome: ✅ submitted** — all 6 files on `ftp.ecmwf.int/forecast_submissions/20260723/`
(`{mslp,pr,tas} × {p1,p2}`, `Fahamu / fp16FahamuAIFSv2`).

Same fully-local pipeline as [`run_commands_20260716.md`](run_commands_20260716.md)
(no GCS: local pkls → local Icechunk store → local downstream). This cycle hit four
incidents worth recording; each is noted inline and summarised under **Incidents**.

---

## ⏱ End-to-end timing

| Step | What | Wall time | Output |
|------|------|-----------|--------|
| **1** | pkl creation, 50 members (local) | **7 h 19 m** (26 333 s) | 41 GB |
| **2** | GPU inference → Icechunk | run 1 partial (9/50, killed) + **resume 3 h 20 m** (12 004 s, 41 members) → ~293 s/member | 586 GB |
| **3a** | regrid 432–792 h → 1.5° NetCDF | **8 m 53 s** | 2.0 GB |
| **3b** | quintiles → submission NetCDF | ~2 m (after the climatology-dir fix) | 1.6 GB (`aiwq/`) |
| **3c** | **live submit** | 5/6 in 242 s + 1 retry → **6/6** | 6 files on FTP |

Both Step 1 and Step 2 ran without their usual overlap this time (Step 1 finished long
before Step 2 started), so there is no single meaningful "end-to-end" wall figure — the
cycle was spread across 2026-07-23 → 07-26 with operator gaps and the incidents below.

---

## Commands (as run)

```bash
export MAMBA_ROOT_PREFIX=/tank/projects/micromamba
PY=/tank/projects/micromamba/envs/aifs-gpu/bin/python
export HF_HOME=/tank/projects/hf_cache
BASE=/tank/projects/aifs-run/20260723_0000
cd /home/ezra/ea-aifs/fp16FahamuAIFSv2      # so .env resolves

# --- Step 1: input pkls, local ---
$PY ecmwf_opendata_pkl_input_aifsens_v2.py --date 20260723 --members 1-50 \
    --no-upload --keep-local --out-dir $BASE/input_states

# --- Step 2: GPU inference -> local Icechunk store (run DETACHED via setsid; see Incidents) ---
setsid nohup bash -c "cd $PWD; export HF_HOME=$HF_HOME; \
  $PY -u run_local_icechunk_v2.py --date 20260723_0000 --members 1-50 --lead-time 792 \
    --input-dir $BASE/input_states --store $BASE/icechunk_v2 \
    --n-members 50 --commit-every 1 --float-size f4 --write-hours 432-792 --skip-existing" \
  > /tank/projects/run_0723_all.log 2>&1 </dev/null &

# --- Step 3a: regrid from the Icechunk store ---
$PY ../shared/aifs_n320_grib_1p5defg_nc_cli.py --date 20260723_0000 --members 1-50 --v2 \
    --no-upload --source icechunk --icechunk-store $BASE/icechunk_v2 \
    --icechunk-tag cycle-20260723_0000 --output-dir $BASE/nc_1p5deg

# --- Step 3b: quintiles (--work-dir homes quintile + climatology + temp store under aiwq/) ---
$PY ../shared/ensemble_quintile_analysis_cli.py --date 20260723 --v2 \
    --local-nc-dir $BASE/nc_1p5deg --work-dir $BASE/aiwq

# --- Step 3c: submit (window 20260723 -> 20260726 inclusive; was OPEN on 07-26) ---
$PY ../shared/forecast_submission_cli.py --date 20260723 --v2 --output-dir $BASE/aiwq --dry-run
$PY ../shared/forecast_submission_cli.py --date 20260723 --v2 --output-dir $BASE/aiwq   # live
# re-run a single failed one:
$PY ../shared/forecast_submission_cli.py --date 20260723 --v2 --output-dir $BASE/aiwq \
    --variables tas --weeks 1
```

---

## Incidents (all resolved)

1. **Step 1 AWS S3 `503 Slow Down` stall.** First launch wedged for 1 h 40 m with **0 pkls**,
   looping `Recovering from HTTP error [503 Slow Down], attempt N of 500` (500 × 120 s ≈ 16 h
   of nothing). Not an outage — direct requests to the bucket returned 200 in <1 s.
   **Killed and re-ran; earthkit's cache made the in-flight member finish in seconds.** The
   successful retry then took 7 h 19 m. (Bandwidth note below.)

2. **Step 2 inference killed after 9/50.** The first Step-2 process died with **no exit line**
   (not OOM — the "163 GB used" was reclaimable ZFS ARC; not a crash — no traceback). Cause:
   session/terminal teardown reaching the process group despite `nohup`. **Relaunched under
   `setsid`** (own session, PPID=1) and `--skip-existing` resumed from member 10 — the 9 done
   members were preserved. Use `setsid` for the long detached jobs.

3. **Step 3b "No quintile data calculated" (a bug I introduced).** The `--work-dir` change
   directed reads to `clim_dir` but `download_all_quintiles` still let AI_WQ write the
   climatology to the CWD → calc looked in `aiwq/`, found nothing. Only surfaced here because
   20260723's valid dates (20260810/20260817) were fresh (20260716's were pre-placed).
   **Fixed** (commit `c6bf748`): thread `dest=clim_dir` → `retrieve_20yr_quintile_clim(
   local_destination=…)`.

4. **Step 3c one transient FTP failure.** `tas_week1` failed with `Failed: 200 TYPE is now
   8-bit binary` (a normal FTP status line mis-caught as an error during new-file creation) —
   5/6 landed. A single-target re-run (`--variables tas --weeks 1`) succeeded; **6/6 verified
   server-side.** The submission window (`start_date → +3 days`) was **inclusive of 07-26** —
   the server accepted with "forecast submitted within competition time window".

> **Bandwidth (measured 2026-07-24, idle box):** the slowness is the **path to AWS S3
> eu-central-1**, not the machine. AWS sustained **1–3 Mbps** for GRIB pulls while the
> **Google** mirror did **11–27 Mbps** (6–10× faster) from the same box; `ecmwf` was healthy,
> `azure` broken (HTTP 409). The gigabit NIC and uplink are fine (Google proves it). AWS stays
> hardcoded by choice, but this is why Step 1 takes ~7 h here.

## Outputs

| Path | Size | What |
|------|------|------|
| `…/20260723_0000/input_states/` | 41 GB | 50 input pkls |
| `…/20260723_0000/icechunk_v2/` | 586 GB | Icechunk store, tag `cycle-20260723_0000` (50/50) |
| `…/20260723_0000/nc_1p5deg/` | 2.0 GB | 50 × 1.5° NetCDF (432–792 h) |
| `…/20260723_0000/aiwq/ensemble_quintile_probabilities_20260723_v2.nc` | 6.7 MB | **submitted file** |

Reclaim the heavy artifacts once satisfied: `python cleanup_aifs_run.py --date 20260723 --yes`
(keeps `aiwq/*.nc`).

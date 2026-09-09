# AIFS ENS v2.0 Run Commands — 20260813 (first ECBox cycle)

**Date:** 20260813 (Thursday) · **Members:** 1–50 · **Box:** local RTX 5000 Ada, 30 GB.
**Outcome: ✅ submitted 6/6 via ECBox** — `sites.ecmwf.int` → space `ecbox` / site
`AI_Weather_Quest` / `forecast_submissions/20260813` (`{mslp,pr,tas} × {p1,p2}`,
`Fahamu / fp16FahamuAIFSv2`).

**This is the first cycle on the AI-WQ FTP→ECBox migration.** Steps 1–3b are the same as
[`run_commands_20260806.md`](run_commands_20260806.md) (proto/GCS Step 1, local Icechunk,
local downstream); **only 3c changed** — and two small code fixes were needed for AI-WQ 3.29.

---

## ⏱ End-to-end timing

| Step | What | Wall time | Output |
|------|------|-----------|--------|
| **1** | pkl creation, 50 members (proto, **GCS**) | **2 h 01 m** (7 286 s) → 145 s/member | 46 GB |
| **2** | GPU inference → Icechunk | **4 h 00 m** (14 426 s) → ~285 s/member | 582 GB |
| **3a** | regrid 432–792 h → 1.5° NetCDF | **~9 m** | 2.0 GB |
| **3b** | quintiles → submission NetCDF | **~2 m** (after the AI-WQ-rename fix) | 1.6 GB (`aiwq/`) |
| **3c** | **live submit via ECBox** | **1 h 10 m** (4 213 s) → **~702 s (~11.7 min) / file** | 6 files on ECBox |

> **ECBox 3c is ~18× slower than FTP was** (~11.7 min/file vs ~40 s on FTP; 1 h 10 m vs ~4 m
> for all 6). **Start 3c well before the window closes.** The window is `start → +3 days`
> (20260813 → 20260816); this run submitted on 08-15 with ~1 h to spare, which was tight.

---

## The FTP → ECBox migration (one-time setup)

```bash
PY=/tank/projects/micromamba/envs/aifs-gpu/bin/python
# 1. AI-WQ >= 3.28 (routes fc dates >= 2026-08-13 to ECBox internally):
$PY -m pip install --upgrade AI-WQ-package            # -> 3.29
# 2. sites-toolkit (ECBox transport; import module name is `sites`):
$PY -m pip install sites-toolkit -i https://get.ecmwf.int/repository/pypi-all/simple
# 3. Put the ECBox TOKEN (emailed, ~140 chars) in fp16FahamuAIFSv2/.env as:
#    ecbox=<token>            (NOT the old 9-char FTP AIWQ_PASSWORD)
```

**How it works (AI-WQ 3.29, verified from source):**
`AI_WQ_forecast_submission(...)` has `ftp_closure_date = datetime(2026,8,13)`; for
`fc_date < closure` it uses FTP, else ECBox: `Site.from_space_and_name(space='ecbox',
name='AI_Weather_Quest')` + `Authenticator.from_token(token=password)`. **The `password`
argument is the ECBox token for ECBox dates** — no separate script.

**Two code fixes committed for the 3.29 era** (needed on every future cycle):
- `93b0caa` — `get_credentials` resolves the token `AIWQ_ECBOX_TOKEN → ecbox → AIWQ_PASSWORD`,
  so the `.env` `ecbox` key is used for submission (FTP password stays the fallback).
- `c8176ee` — AI-WQ 3.29 renamed the climatology function
  `retrieve_20yr_quintile_clim → _quantile_clim`; 3b called the old name and failed silently
  ("All 5 attempts failed → No quintile data calculated"). Climatology itself still downloads
  over FTP with the 9-char password; only submission uses ECBox.

> **Dependency note:** installing sites-toolkit downgraded `requests` to 2.32.5 (conflicts
> with `google-api-core` ≥2.33). Harmless for the local/GCS-mirror/ECBox paths used here.

---

## Commands (as run)

```bash
export MAMBA_ROOT_PREFIX=/tank/projects/micromamba
PY=/tank/projects/micromamba/envs/aifs-gpu/bin/python
export HF_HOME=/tank/projects/hf_cache
BASE=/tank/projects/aifs-run/20260813_0000

# --- Step 1: input pkls via the GCS mirror (proto), setsid ---
cd /home/ezra/ea-aifs/s3_grib_pkl
setsid nohup bash -c "cd $PWD; export PYTHONUNBUFFERED=1; \
  uv run --script s3_grib_pkl_input_aifsens_v2_proto.py \
    --date 20260813_0000 --members 1-50 --source gcs --skip-existing \
    --out $BASE/input_states" > /tank/projects/proto_gcs_20260813_all50.log 2>&1 </dev/null &
# proto writes proto_input_state_member_NNN.pkl; the runner expects input_state_member_NNN.pkl:
cd $BASE/input_states && for f in proto_input_state_member_*.pkl; do ln -sf "$f" "${f#proto_}"; done

# --- Free an old submitted store first if disk is tight (mind the space) ---
cd /home/ezra/ea-aifs/fp16FahamuAIFSv2
$PY cleanup_aifs_run.py --date <oldest-submitted> --yes

# --- Step 2: GPU inference -> local Icechunk store (setsid) ---
setsid nohup bash -c "cd $PWD; export HF_HOME=$HF_HOME; \
  $PY -u run_local_icechunk_v2.py --date 20260813_0000 --members 1-50 --lead-time 792 \
    --input-dir $BASE/input_states --store $BASE/icechunk_v2 \
    --n-members 50 --commit-every 1 --float-size f4 --write-hours 432-792 --skip-existing" \
  > /tank/projects/run_0813_all.log 2>&1 </dev/null &

# --- Step 3a/3b ---
$PY ../shared/aifs_n320_grib_1p5defg_nc_cli.py --date 20260813_0000 --members 1-50 --v2 \
    --no-upload --source icechunk --icechunk-store $BASE/icechunk_v2 \
    --icechunk-tag cycle-20260813_0000 --output-dir $BASE/nc_1p5deg
$PY ../shared/ensemble_quintile_analysis_cli.py --date 20260813 --v2 \
    --local-nc-dir $BASE/nc_1p5deg --work-dir $BASE/aiwq

# --- Step 3c: submit (AI-WQ 3.29 auto-routes to ECBox for fc date >= 08-13) ---
$PY ../shared/forecast_submission_cli.py --date 20260813 --v2 --output-dir $BASE/aiwq --dry-run
$PY ../shared/forecast_submission_cli.py --date 20260813 --v2 --output-dir $BASE/aiwq   # live, ~1h10m
```

Run 3c **detached** (`setsid`) — at ~12 min/file it easily exceeds a foreground shell timeout,
which would otherwise orphan it mid-submit.

## Verify / manage

- **Browser:** `https://sites.ecmwf.int` → `ecbox` / `AI_Weather_Quest` /
  `forecast_submissions/20260813/` → the 6 `.nc` files.
- **Rotate the token** in the `sites.ecmwf.int` web UI (it is a bearer token, not a
  password), then update the `ecbox` key in `.env`. (Rotate soon — the earlier `Ws6…` token
  is exposed in `2026-06-30-aifs-ebox-login2.txt`.)

## Outputs

| Path | Size | What |
|------|------|------|
| `…/20260813_0000/icechunk_v2/` | 582 GB | Icechunk store, tag `cycle-20260813_0000` (50/50) |
| `…/20260813_0000/nc_1p5deg/` | 2.0 GB | 50 × 1.5° NetCDF (432–792 h) |
| `…/20260813_0000/aiwq/ensemble_quintile_probabilities_20260813_v2.nc` | 6.7 MB | **submitted file** |

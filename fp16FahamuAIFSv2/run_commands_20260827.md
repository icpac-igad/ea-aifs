# AIFS ENS v2.0 Run Commands — 20260827 (N320, submitted)

**Date:** 20260827 (Thursday) · **Members:** 1–50 · **Box:** local RTX 5000 Ada, 30 GB.
**Outcome: ✅ complete and submitted.** A single **N320** rollout (583 GB, 432–792 h) with
3a/3b on top and **6/6 files submitted 2026-08-30 06:21 UTC**, 17 h 38 m inside the window.
Verification weeks **2026-09-14** and **2026-09-21**; scoreable from **2026-09-21**.

> **One grid this time.** 20260820 carried both an O96 corpus and an N320 store because the
> first rollout was O96-only and had to be redone. This cycle went straight to N320 — the
> graded route — so there is no second store and no risk of reading the wrong one.

---

## ⏱ End-to-end timing

| Step | What | Wall time | Output |
|------|------|-----------|--------|
| **0** | open-data availability pre-check | seconds | AIFS v2 RUNNABLE — sfc 9/9, 14/14 levels, 11/11 wave |
| **1** | pkl creation, 50 members (proto, **GCS**) | **2 h 31 m** (151.3 min) → **180.8 s/member** | 42 GB, 50/50, 0 failed |
| **2** | GPU inference → **N320** Icechunk | **4 h 01 m** (14:43→18:44) → **288.7 s/member** | 583 GB, 50/50, 0 failed, tagged |
| **3a** | regrid 432–792 h → 1.5° NetCDF | **8.5 m** | 2.1 GB, 50/50, 0 failed |
| **3b** | quintiles → AI-WQ NetCDF | **~2 m** | 6.7 MB |
| **3c** | **live submit via ECBox** | **1 h 06 m** (05:15→06:21 UTC) → ~11 min/file | **6/6 submitted, 0 failed** |

Step 2 ran at the N320 baseline (288.7 s/member vs the 287.6 s reference on 20260820), with
nothing competing for the GPU. Per-member spread 259.2–366.8 s.

**Step 1 was ~43 % slower than the 20260820 reference** (180.8 s/member vs 126.5 s). Nothing
local changed; this is GCS read throughput and is worth watching rather than acting on.

---

## Commands (as run)

```bash
PY=/tank/projects/micromamba/envs/aifs-gpu/bin/python
BASE=/tank/projects/aifs-run/20260827_0000
```

### Step 0 — is the cycle published?

```bash
cd /home/ezra/ea-aifs/fp16FahamuAIFSv2/run-pre50r1-dates
$PY check_open_data_inputs.py --date 20260827 --source gcs
# -> AIFS v2: RUNNABLE   sfc 9/9, sol 2/2, pl 6/6, levels 14/14, wave 11/11
```

### Free the disk first — this cycle needed 630 GB and 246 GB were free

```bash
$PY cleanup_aifs_run.py --date 20260806            # dry-run plan
$PY cleanup_aifs_run.py --date 20260806 --yes      # 587.8 GB reclaimed
```

Purges `icechunk_v2`, `nc_1p5deg` and `aiwq/ensemble_icechunk_store`; **keeps `aiwq/*.nc`**
(7 files — the quintile product plus climatology), which is all that is needed to score a
submitted cycle. 20260806 was already submitted and scores off the file on the server, so
losing its N320 store costs only the ability to re-derive it.

Check the arithmetic before choosing a victim: **a cycle needs ~630 GB** (42 GB pkls +
583 GB store), and only an N320 store is big enough to free that in one action.

### Step 1 — input pkls via the GCS mirror

```bash
mkdir -p $BASE/input_states
cd /home/ezra/ea-aifs/s3_grib_pkl
setsid nohup bash -c "cd $PWD; export PYTHONUNBUFFERED=1; \
  uv run --script s3_grib_pkl_input_aifsens_v2_proto.py \
    --date 20260827_0000 --members 1-50 --source gcs --skip-existing \
    --out $BASE/input_states" > /tank/projects/proto_gcs_20260827_all50.log 2>&1 </dev/null &
```

**Then the symlinks** — the builder writes `proto_input_state_member_NNN.pkl`, the runner
opens `input_state_member_NNN.pkl`:

```bash
cd $BASE/input_states && for f in proto_input_state_member_*.pkl; do ln -sf "$f" "${f#proto_}"; done
ls input_state_member_*.pkl | wc -l                       # must print 50
for f in input_state_member_*.pkl; do [ -e "$f" ] || echo "DANGLING $f"; done   # and resolve
```

Counting the symlinks is not enough — `ln -sf` happily creates a link to a missing target.
Check they **resolve**. Skipping this step entirely is what made 20260820 exit in 2 seconds
with `DONE: 0 written, 0 skipped, 50 failed`.

### Step 2 — GPU inference → N320 Icechunk

```bash
setsid nohup bash -c "cd $PWD; export HF_HOME=/tank/projects/hf_cache; \
  $PY -u run_local_icechunk_v2.py --date 20260827_0000 --members 1-50 --lead-time 792 \
    --input-dir $BASE/input_states --store $BASE/icechunk_v2 \
    --n-members 50 --commit-every 1 --float-size f4 --write-hours 432-792 --skip-existing" \
  > /tank/projects/run_0827_n320.log 2>&1 </dev/null &
```

> **`HF_HOME=/tank/projects/hf_cache` is not optional and not guessable.** This run was
> first launched with `/tank/projects/hf_home` — a *different*, half-populated cache that
> also contains a `models--ecmwf--aifs-ens-2.0` directory. The run does not fail: it stalls
> silently re-downloading the model from HF unauthenticated, GPU at 0 %, with nothing in
> the log after `Fields: 112`. Nine minutes were lost before it was caught. There are three
> plausible cache paths on this box (`hf_cache`, `hf_home`, `~/.cache/huggingface`) and only
> `hf_cache` is complete.
>
> Recovery: kill the run, **delete the empty store it left** (15 KB, repo + first commit, no
> schema — relaunching over it makes the grid guard compare against a schema-less
> directory), then relaunch.

```
Write window: 432-792h -> storing 61/132 steps (46%); others computed but not stored (NaN)
    [SCHEMA] init 50x132x542080, 120 vars (f4), time_chunk=1
[TAG] cycle-20260827_0000 -> G6MXAQVY4K55RK2P3W70 (50/50 members)
DONE: 50 written, 0 skipped, 0 failed of 50 members
```

**542,080 in the schema line is the grid check** — O96 prints 40,320. The other early signal
is the GPU: it should sit at ~11.8 GB and 99–100 % within a minute or two of launch.

### Post-run validation — open the store, don't trust the log

```python
repo = icechunk.Repository.open(icechunk.local_filesystem_storage(f"{BASE}/icechunk_v2"))
g = zarr.open_group(repo.readonly_session(tag="cycle-20260827_0000").store, mode="r")
```

| check | result |
|---|---|
| tags | `{'cycle-20260827_0000'}` — one store, one tag |
| `2t` shape / chunks | `(50, 132, 542080)` / `(1, 1, 542080)` — **N320** |
| variables | 124 |
| stored window | steps 71–131 = **61/132** (h432–792) |
| members 1 / 26 / 50 | first and last stored step finite |

### Steps 3a / 3b

```bash
$PY ../shared/aifs_n320_grib_1p5defg_nc_cli.py --date 20260827_0000 --members 1-50 --v2 \
    --no-upload --source icechunk --icechunk-store $BASE/icechunk_v2 \
    --source-grid n320 --icechunk-tag cycle-20260827_0000 --output-dir $BASE/nc_1p5deg
$PY ../shared/ensemble_quintile_analysis_cli.py --date 20260827 --v2 \
    --local-nc-dir $BASE/nc_1p5deg --work-dir $BASE/aiwq
```

3a echoes its provenance at runtime — check it rather than assuming:

```
📖 Icechunk store: /tank/projects/aifs-run/20260827_0000/icechunk_v2 (tag=cycle-20260827_0000)
  🔄 Reading N320 fields from Icechunk and regridding...
```

3b downloaded climatology for **20260914** and **20260921** over FTP, six files, all real
downloads — not the silent `All 5 attempts failed` no-op that bit 20260813 (fixed in
`c8176ee`).

### Step 3c — submit

```bash
setsid nohup bash -c "cd $PWD; \
  $PY -u ../shared/forecast_submission_cli.py --date 20260827 --v2 --output-dir $BASE/aiwq" \
  > /tank/projects/run_0827_3c.log 2>&1 </dev/null &
```

```
[1/6] mslp_week1 ✅   [3/6] pr_week1 ✅   [5/6] tas_week1 ✅
[2/6] mslp_week2 ✅   [4/6] pr_week2 ✅   [6/6] tas_week2 ✅
Successful: 6   Failed: 0
```

`Creating directory '20260827'` on the first file confirms a fresh remote directory.

**Pre-flight offline, because `--dry-run` proves nothing about the transport** — it
short-circuits before `AI_WQ_create_empty_dataarray`, so it exercises neither the ECBox
token nor the AI-WQ checks. What does work without transmitting:

| check | how | result |
|---|---|---|
| window open | `check_fc_submission.check_forecast_data_window('20260827')` | pass |
| team + model registered | `AI_WQ_create_empty_dataarray(...)` — hits the server, uploads nothing | pass |
| arrays | `prepare_aiwq_submission` → `(5,121,240)`, finite, ∈[0,1], Σ=1.000000 | 6/6 pass |
| template fit | returned coords `quintile = [0.2 … 1.0]` | pass |

The ECBox token is only proven by the first live file. **Rotate it** in the
`sites.ecmwf.int` web UI and update the `ecbox` key in `.env` — the earlier token is exposed
in `2026-06-30-aifs-ebox-login2.txt`.

---

## Outputs

| Path | Size | What |
|------|------|------|
| `…/20260827_0000/input_states/` | 42 GB | 50 pkls + 50 symlinks |
| `…/20260827_0000/icechunk_v2/` | **583 GB** | 124 vars × 50 members × **61/132 steps** (h432–792), tag `cycle-20260827_0000` |
| `…/20260827_0000/nc_1p5deg/` | 2.1 GB | 50 × 1.5° NetCDF, from the N320 store |
| `…/20260827_0000/aiwq/ensemble_quintile_probabilities_20260827_v2.nc` | 6.7 MB | the AI-WQ product |
| ECBox `AI_Weather_Quest/forecast_submissions/20260827/` | 6 files | the submission |
| `/tank/projects/{proto_gcs_20260827_all50,run_0827_n320,run_0827_3a,run_0827_3b,run_0827_3c}.log` | — | Step 1 / 2 / 3a / 3b / 3c logs |
| [`fp16FahamuAIFSv2_20260827.txt`](fp16FahamuAIFSv2_20260827.txt) | — | 3b + 3c transcript ([how it is built](RUN_LOGS_AND_TRANSCRIPTS.md)) |

Disk after the run: **201 GB free**. `icechunk_v2` has not been manifest-compacted — the
N320 chunk payload dominates, so the O96 trick returns proportionally much less.

---

## The 58-hour gap — the one thing to fix before the next cycle

Step 2 finished **2026-08-27 18:44 UTC**. 3a did not start until **2026-08-29 22:33**, and
3c completed **2026-08-30 06:21** — with 17 h 38 m left on a 3-day window. Nothing was
wrong: the rollout completed while nothing was watching, and the cycle simply sat there.

It came out fine only because the window is wide. **3a and 3b together are 11 minutes and
have no external dependencies** — no submission, no irreversible action, nothing needing a
human. They should be chained onto Step 2's `DONE:` line in the launcher so that a finished
rollout always leaves a submission-ready product, and the only remaining human decision is
3c.

| | |
|---|---|
| Step 2 done | 2026-08-27 18:44 UTC |
| 3a started | 2026-08-29 22:33 UTC — **+58 h** |
| 3c done | 2026-08-30 06:21 UTC |
| window closed | 2026-08-30 23:59 UTC — **17 h 38 m spare** |

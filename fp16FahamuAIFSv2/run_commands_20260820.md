# AIFS ENS v2.0 Run Commands — 20260820 (O96 archive + N320 rerun)

**Date:** 20260820 (Thursday) · **Members:** 1–50 · **Box:** local RTX 5000 Ada, 30 GB.
**Outcome: ✅ two complete stores.** An O96 corpus (102 GB, 0–792 h, run 2026-08-21) and
an **N320 store** (583 GB, 432–792 h, run 2026-08-22) with 3a/3b products on top of it.
**Submitted 2026-08-23 17:04 UTC — 6/6 files, 0 failed**, from the **N320** store.
The AI-WQ window for this cycle was `20260820 0000 UTC → 20260823 2359 UTC`
(`check_fc_submission.check_forecast_data_window`: Thursday + 3 days); 3c finished with
~7 h to spare. Scoreable from **2026-09-14**.

> **The two stores are different forecasts.** aifs-ens-2.0 is stochastic, so the N320
> rerun from the same pkls is a different realisation of the same ICs — not a recoding of
> the O96 corpus. Never mix them in one product, and never treat the pair as a
> grid-vs-grid comparison; that experiment is 20260813 (see the end of this file).

## Why there is an N320 rerun at all

The cycle was first run as the first *live* O96 inference (`--grid o96`), following
[`O96-icechunk-store/README.md`](O96-icechunk-store/README.md) §7 — but **without the
tier-B `--native-store` sidecar**. The chain script that launched it passed only
`--grid o96 --store …/icechunk_o96`, so **no N320 data was written at all**, and Step 3a
(which reads N320) had nothing to read.

> **And they should not be back-filled from O96.** The O96 route is currently
> **unvalidated for skill** — see
> [`RPSS_LEADERBOARD_DISCREPANCY.md`](O96-icechunk-store/forecast-evaluation/RPSS_LEADERBOARD_DISCREPANCY.md).
> For 20260709 the submitted **N320** forecast scored **+0.071 / +0.055 / +0.106**
> (tas / mslp / pr, week1, official leaderboard) while the **O96** rerun of the same
> cycle scored **−0.035 / +0.023 / +0.060** — worse on all three. The scorer is
> validated (it reproduces the leaderboard to three decimals on the submitted file), so
> the gap is in the forecast. It confounds the regrid route with a different stochastic
> draw and is not resolved until `settle_grid_question.sh` runs on 20260813 (from
> 2026-08-31).
>
> **Consequence: anything graded goes back to N320.** For this cycle that meant a full
> N320 rerun (Step 2N below) from the pkls, which had survived. From the *next* cycle on,
> run Step 2 once with `--grid o96 --native-store … --native-vars msl,tp,2t` so the 12 GB
> N320 sidecar is written alongside the O96 corpus in a single rollout — tier B in
> README §7 — and no rerun is ever needed.

Step 1 is unchanged from [`run_commands_20260813.md`](run_commands_20260813.md).

---

## ⏱ End-to-end timing

| Step | What | Wall time | Output |
|------|------|-----------|--------|
| **0** | open-data availability pre-check | seconds | 14/14 levels, 11/11 wave — RUNNABLE |
| **1** | pkl creation, 50 members (proto, **GCS**) | **1 h 46 m** (106.1 min) → **126.5 s/member** | 44 GB, 50/50, 0 failed |
| **2a** | GPU inference → O96 Icechunk, members 1–39 | **3 h 16 m** (02:38→05:54) → ~291 s/member | killed by SIGTERM at member 40 |
| **2b** | resume, `--skip-existing`, members 40–50 | **1 h 21 m** (13:49→15:10) → ~404 s/member | tagged `cycle-20260820_0000`, `PASS` |
| **2c** | manifest compaction + GC | ~10 m | 159 GB → **102 GB**, retagged `-c` |
| **2N** | **N320 rerun** from the same pkls (2026-08-22) | **4 h 05 m** (09:35→13:40) → **287.6 s/member** | 583 GB, 50/50, 0 failed, tagged |
| **3a** | regrid 432–792 h → 1.5° NetCDF (from N320) | **8.5 m** | 2.1 GB, 50/50, 0 failed |
| **3b** | quintiles → AI-WQ NetCDF | **~2 m** | 6.7 MB |
| **3c** | **live submit via ECBox** | **1 h 12 m** (15:52→17:04 UTC) → ~12 min/file | **6/6 submitted, 0 failed** |

Step 2's total GPU time for 50 members is **~4 h 37 m**. Members in the resume pass were
slower (~404 s vs ~291 s) because the source.coop upload of 20260212 was running
concurrently — GPU-bound and network-bound jobs coexist, but not for free on I/O.

---

## Commands (as run)

```bash
export MAMBA_ROOT_PREFIX=/tank/projects/micromamba
PY=/tank/projects/micromamba/envs/aifs-gpu/bin/python
export HF_HOME=/tank/projects/hf_cache
BASE=/tank/projects/aifs-run/20260820_0000
```

### Step 0 — is the cycle published?

```bash
cd /home/ezra/ea-aifs/fp16FahamuAIFSv2/run-pre50r1-dates
$PY check_open_data_inputs.py --date 20260820 --source gcs
# -> AIFS v2: RUNNABLE   sfc 9/9, 14/14 pressure levels, 11/11 wave
```

20260820 is post-cutover, so this is the **standard operational builder**, not the
pre-50r1 one. `t` and `t-6h` (2026-08-19 18z) must both be published.

### Step 1 — input pkls via the GCS mirror

```bash
mkdir -p $BASE/input_states
cd /home/ezra/ea-aifs/s3_grib_pkl
setsid nohup bash -c "cd $PWD; export PYTHONUNBUFFERED=1; \
  uv run --script s3_grib_pkl_input_aifsens_v2_proto.py \
    --date 20260820_0000 --members 1-50 --source gcs --skip-existing \
    --out $BASE/input_states" > /tank/projects/proto_gcs_20260820_all50.log 2>&1 </dev/null &
```

**Then create the symlinks — this is the step that bit this run.** The builder writes
`proto_input_state_member_NNN.pkl`; the runner opens `input_state_member_NNN.pkl`:

```bash
cd $BASE/input_states && for f in proto_input_state_member_*.pkl; do ln -sf "$f" "${f#proto_}"; done
ls input_state_member_*.pkl | wc -l    # must print 50
```

The backfill driver did this automatically; running pkl creation standalone does not.
Skipping it makes Step 2 exit in **2 seconds** with
`DONE: 0 written, 0 skipped, 50 failed` and leaves an empty store behind — delete that
store before relaunching, or the grid guard will compare against a schema-less directory.

### Step 2 — GPU inference → **O96** Icechunk store

Free disk first: this cycle needs ~160 GB pre-compaction, and it landed on a box at 90 %.
Check `df -h /tank/projects` and release spent `input_states` of completed cycles before
launching (`../cleanup_aifs_run.py --date <oldest-submitted> --yes` for a whole run).

```bash
cd /home/ezra/ea-aifs/fp16FahamuAIFSv2
setsid nohup bash -c "cd $PWD; export HF_HOME=$HF_HOME; \
  $PY -u run_local_icechunk_v2.py --date 20260820_0000 --members 1-50 --lead-time 792 \
    --input-dir $BASE/input_states \
    --grid o96 --store $BASE/icechunk_o96 \
    --n-members 50 --commit-every 1 --float-size f4 --skip-existing" \
  > /tank/projects/run_0820_o96.log 2>&1 </dev/null &
```

**This is exactly what was run, and it is the command to change next time** — add
`--native-store $BASE/icechunk_n320_aiwq --native-vars msl,tp,2t` for the tier-B sidecar
that keeps the graded path alive (+12 GB, +~17 s/member).

Note what is **absent** and deliberately so: no `--write-hours`. At O96 the whole
0–792 h corpus is ~100 GB, so the 432–792 h window — an N320-era compromise — is not
carried over. All 132 steps are stored.

Before committing 50 members on a new cycle, README §7 asks for a one-member smoke test
(`--members 1 … --no-tag` into `icechunk_o96_smoke`, then `validate_o96_run.py`, ~5 min).

### Step 2 — the interruption, and how it resumed

The first pass was terminated by **SIGTERM at member 40** (the GPU was handed to a cGAN
training run). The store was left at **39/50 members, untagged** — which is correct
behaviour: the tag is written only at 50/50, because Icechunk tags are immutable and
tombstoned, so a tag on a partial ensemble would permanently point at NaNs.

Resuming is the same command; `--skip-existing` does the right thing:

```
--- Member 039: [SKIP] already complete in store
To process: 11 member(s); skipped 39
--- Member 040 (1/11) ---
...
[TAG] cycle-20260820_0000 -> K80AX9X9AMK6S4FF0060 (50/50 members)
DONE: 11 written, 39 skipped, 0 failed of 50 members
```

The half-written member 40 was redone from scratch, not patched — its orphaned chunks
stayed in the store until GC (below).

### Step 2 — validate before trusting the store

```bash
cd /home/ezra/ea-aifs/fp16FahamuAIFSv2/O96-icechunk-store
$PY validate_o96_run.py --o96 $BASE/icechunk_o96 --tag cycle-20260820_0000 --members 1-50
```

```
O96 store : 120 vars, 2t (50, 132, 40320) chunks (1, 1, 40320)
  tags: {'cycle-20260820_0000'}
1. write window: steps 0-131 (h6-792), 132/132 stored
2. all 50 member(s) finite at steps [0, 131]
PASS
```

(With a sidecar, add `--sidecar $BASE/icechunk_n320_aiwq` — that is the check that proves
the in-flight regrid, since both stores then come from the same rollout.)

### Step 2c — compact the manifests

`rewrite_manifests` consolidates 792 k per-chunk manifest files into ~124, then the old
snapshots are expired and garbage-collected. The procedure and the reasoning are in
[`MANIFEST_COMPACTION.md`](O96-icechunk-store/MANIFEST_COMPACTION.md) §6 — the driver used
here was a session scratch script, not a committed CLI:

```bash
$PY -c "
import icechunk
r = icechunk.Repository.open(icechunk.local_filesystem_storage('$BASE/icechunk_o96'))
print(r.rewrite_manifests('consolidate manifests for archival', branch='main'))"
# then: fingerprint against cycle-20260820_0000, tag cycle-20260820_0000-c, expire + GC
```

**Fingerprint against the old tag before expiring anything** — up to that point the
compaction is free to undo, and not afterwards.

| | before | after |
|---|---|---|
| size | 159 GB | **102 GB** |
| objects | 1,615,720 | **805,436** |
| tag | `cycle-20260820_0000` | `cycle-20260820_0000-c` |

`chunks_deleted=5,880` here, against **0** on the uninterrupted cycles — those are the
orphans from the member-40 kill, legitimately unreferenced. Data fingerprints were
identical before and after GC and `validate_o96_run.py` still prints `PASS`, which is the
check that separates "benign orphan reclaim" from "GC ate live data".

---

## Outputs

| Path | Size | What |
|------|------|------|
| `…/20260820_0000/input_states/` | 44 GB | 50 pkls + 50 symlinks — kept; they fed both rollouts |
| `…/20260820_0000/icechunk_o96/` | 102 GB | 120 vars × 50 members × **132/132 steps**, tag `cycle-20260820_0000-c` |
| `…/20260820_0000/icechunk_v2/` | **583 GB** | 122 vars × 50 members × **61/132 steps** (h432–792), tag `cycle-20260820_0000` |
| `…/20260820_0000/nc_1p5deg/` | 2.1 GB | 50 × 1.5° NetCDF, **from the N320 store** |
| `…/20260820_0000/aiwq/ensemble_quintile_probabilities_20260820_v2.nc` | 6.7 MB | the AI-WQ product |
| ECBox `AI_Weather_Quest/forecast_submissions/20260820/` | 6 files | the submission — `{mslp,pr,tas}` × weeks 1–2 |
| `/tank/projects/{proto_gcs_20260820_all50,run_0820_n320,run_0820_3a,run_0820_3b,run_0820_3c}.log` | — | Step 1 / 2N / 3a / 3b / 3c logs |

Verification date for this cycle is **2026-09-14**; nothing here is scoreable before then.
`icechunk_v2` has **not** been manifest-compacted — the N320 chunk payload dominates, so
the O96 trick (§2c) returns proportionally much less.

---

## Step 2N — the N320 rerun (as run, 2026-08-22)

Step 1 was **not** repeated: the pkls were still on disk, so the rerun starts at inference.
This is the ordinary operational Step 2 — default grid, windowed write:

```bash
setsid nohup bash -c "cd $PWD; export HF_HOME=$HF_HOME; \
  $PY -u run_local_icechunk_v2.py --date 20260820_0000 --members 1-50 --lead-time 792 \
    --input-dir $BASE/input_states --store $BASE/icechunk_v2 \
    --n-members 50 --commit-every 1 --float-size f4 --write-hours 432-792 --skip-existing" \
  > /tank/projects/run_0820_n320.log 2>&1 </dev/null &
```

```
[SCHEMA] init 50x132x542080, 120 vars (f4), time_chunk=1
Write window: 432-792h -> storing 61/132 steps (46%)
[TAG] cycle-20260820_0000 -> 05KHG1NDP6B54VCKG1M0 (50/50 members)
DONE: 50 written, 0 skipped, 0 failed of 50 members
```

**542,080 values in the schema line is the grid check** — O96 prints 40,320. 287.6 s/member
over 50 members, i.e. the full N320 baseline, even though the source.coop upload ran
concurrently for the first two hours.

Post-run validation (open the store at the tag, don't trust the log):

| check | result |
|---|---|
| `2t` shape / chunks | `(50, 132, 542080)` / `(1, 1, 542080)` — N320 |
| variables | 122 |
| stored window | steps 71–131 = **61/132** (h432–792) |
| all 50 members finite at first and last stored step | **OK** |
| tag | `cycle-20260820_0000` |

## Steps 3a / 3b — and how the N320 grid is guaranteed

```bash
$PY ../shared/aifs_n320_grib_1p5defg_nc_cli.py --date 20260820_0000 --members 1-50 --v2 \
    --no-upload --source icechunk --icechunk-store $BASE/icechunk_v2 \
    --source-grid n320 --icechunk-tag cycle-20260820_0000 --output-dir $BASE/nc_1p5deg
$PY ../shared/ensemble_quintile_analysis_cli.py --date 20260820 --v2 \
    --local-nc-dir $BASE/nc_1p5deg --work-dir $BASE/aiwq
```

Now that a cycle holds **both** grids, picking the wrong one is a live risk. Four controls,
three of which fail loudly:

| control | value | if wrong |
|---|---|---|
| `--icechunk-store` | `…/icechunk_v2` | the path is the real control |
| `--source-grid` | `n320` (also the default) | only selects the earthkit input spec (`cli:527`); a mismatch feeds 40,320 values to a 542,080-wide matrix → **matmul dimension error** |
| `--icechunk-tag` | `cycle-20260820_0000` | **exists only on the N320 store** — compaction renamed the O96 tag to `…-c`, so the wrong store cannot resolve it |
| output provenance | written into the NetCDF attrs | `icechunk_store` / `icechunk_ref` record which store built the file |

The tag divergence is load-bearing but **accidental** — it exists because the O96 store was
compacted and the N320 store was not. Do not rely on it after a future compaction; the
`--source-grid` mismatch error and the store path are the durable guards.

There is no configuration that silently reads O96 while claiming N320. The one genuinely
silent path is passing `--source-grid o96` *and* the O96 store — i.e. deliberately asking
for the O96 route, which is what `nc_1p5deg_via_o96` was for on 20260813.

3b output, unchanged from any other cycle:

```
Available variables: ['2t_quintiles', 'msl_quintiles', 'tp_quintiles']
Valid dates: 20260907, 20260914
```

Climatology downloaded cleanly for both verification weeks over FTP — the
`retrieve_20yr_quintile_clim → _quantile_clim` rename fix (`c8176ee`) holding.

---

## Step 3c — the submission (as run, 2026-08-23)

```bash
BASE=/tank/projects/aifs-run/20260820_0000
PY=/tank/projects/micromamba/envs/aifs-gpu/bin/python

# dry run first — 6/6 would submit, 0 failed
$PY ../shared/forecast_submission_cli.py --date 20260820 --v2 --output-dir $BASE/aiwq --dry-run

# live, detached: ~12 min/file over ECBox exceeds any foreground shell timeout
setsid nohup bash -c "cd $PWD; \
  $PY -u ../shared/forecast_submission_cli.py --date 20260820 --v2 --output-dir $BASE/aiwq" \
  > /tank/projects/run_0820_3c.log 2>&1 < /dev/null &
```

```
[1/6] mslp_week1 ✅   [3/6] pr_week1 ✅   [5/6] tas_week1 ✅
[2/6] mslp_week2 ✅   [4/6] pr_week2 ✅   [6/6] tas_week2 ✅
Successful: 6   Failed: 0
```

**The dry run does not exercise the transport.** It short-circuits before
`AI_WQ_create_empty_dataarray`, so it proves neither the ECBox token nor the AI-WQ
checks. What can be verified offline, without transmitting, is worth doing instead —
all of this passed before launch:

| check | how |
|---|---|
| window open | `check_fc_submission.check_forecast_data_window('20260820')` |
| team + model registered | `AI_WQ_create_empty_dataarray(...)` — hits the server, uploads nothing |
| array shape / range / sums | `prepare_aiwq_submission` → `(5, 121, 240)`, finite, ∈[0,1], Σ=1.000000 |
| template fit | returned coords `quintile = [0.2 … 1.0]`, dims match the array |

The ECBox bearer token in `.env` is only proven by the first live file (`Updated CSV
uploaded to ECbox` in the log). **Rotate it** in the `sites.ecmwf.int` web UI and update
the `ecbox` key — the earlier token is exposed in `2026-06-30-aifs-ebox-login2.txt`.

Timing to plan against: **~12 min/file, ~1 h 12 m for six**, unchanged from 20260813's
1 h 10 m. Start 3c with hours of margin, never minutes.

---

## The grid question is still open

This cycle does **not** answer it. Its two stores are different draws, so comparing their
RPSS measures ensemble noise, not the regrid route. The clean comparison — one rollout,
both routes — is **20260813**, runnable from 2026-08-31:

```bash
cd /home/ezra/ea-aifs/fp16FahamuAIFSv2/O96-icechunk-store/forecast-evaluation
./settle_grid_question.sh          # refuses politely before 2026-08-31
```

Its inputs survived the 2026-08-22 disk reclaim (both quintile files verified present).
Its outcome decides whether future cycles need N320 at all beyond the 12 GB tier-B sidecar.

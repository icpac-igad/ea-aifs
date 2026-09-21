# AIFS ENS v2.0 Run Commands — 20260910 (tier B, interrupted and resumed, submitted)

**Date:** 20260910 (Thursday) · **Members:** 1–50 · **Box:** local RTX 5000 Ada, 30 GB.
**Outcome: ✅ submitted 6/6 on 2026-09-13 11:48 UTC**, with 12 h 11 m left in the window.
Verification weeks **2026-09-28** and **2026-10-05**; scoreable from **2026-10-05**.

> **This cycle nearly missed.** Step 1 ran at **2–3× its usual cost**, Step 2 was **killed
> silently at member 39** and sat dead for 24 h, and the whole thing finished on the last day
> of a 4-day window. Nothing was wrong with the forecast; everything that went wrong was
> operational, and both causes are still unexplained. Read §"What went wrong" before planning
> the next cycle.

---

## ⏱ End-to-end timing

| Step | What | Wall time | Output |
|------|------|-----------|--------|
| **0** | open-data availability pre-check | seconds | AIFS v2 RUNNABLE — 9/9 sfc, 14/14 levels, 11/11 wave |
| **1** | pkl creation, 50 members (proto, **GCS**) | **5 h 03 m** (303.2 min) → **363.1 s/member** ⚠️ | 42 GB, 50/50, 0 failed |
| **2a** | tier-B rollout, members 1–38 | **3 h 17 m** → 310.8 s/member | **killed at member 39**, untagged |
| **2b** | resume, `--skip-existing`, members 39–50 | **1 h 20 m** → 403.3 s/member | **50/50**, both stores tagged |
| **3a** | regrid 432–792 h → 1.5° NetCDF (from the sidecar) | **8.2 m** | 2.0 GB, 50/50, 0 failed |
| **3b** | quintiles → AI-WQ NetCDF | **~2 m** | 6.7 MB |
| **3c** | live submit via ECBox | **1 h 12 m** (10:36→11:48 UTC) | **6/6 submitted, 0 failed** |

Step 2 total GPU time was **4 h 37 m** across both passes, in line with 20260903's 4 h 38 m.
The resume ran slower per member (403 s vs 311 s) with nothing else on the GPU — unexplained,
and the same direction as Step 1's slowdown.

### Calendar, because the drift matters more than the durations

| | |
|---|---|
| init | 2026-09-10 00 UTC (Thursday) |
| Step 1 finished | 2026-09-11 12:11 UTC |
| Step 2 started | 2026-09-12 02:59 UTC |
| **Step 2 died** | **2026-09-12 06:21 UTC** at member 39 |
| **noticed** | **2026-09-13 09:00 UTC — 26 h 39 m later** |
| submitted | 2026-09-13 11:48 UTC |
| **window closed** | **2026-09-13 23:59 UTC** |

---

## Commands (as run)

```bash
PY=/tank/projects/micromamba/envs/aifs-gpu/bin/python
BASE=/tank/projects/aifs-run/20260910_0000
```

Steps 0, 1 and the symlinks are unchanged from
[`run_commands_20260903.md`](run_commands_20260903.md). Step 2 is the same tier-B invocation:

```bash
setsid nohup bash -c "cd $PWD; export HF_HOME=/tank/projects/hf_cache; \
  $PY -u run_local_icechunk_v2.py --date 20260910_0000 --members 1-50 --lead-time 792 \
    --input-dir $BASE/input_states \
    --grid o96 --store $BASE/icechunk_o96 \
    --native-store $BASE/icechunk_n320_aiwq \
    --native-vars msl,tp,2t,10u,10v,t_200,t_300,t_500,u_850,v_850 \
    --native-write-hours 432-792 \
    --n-members 50 --commit-every 1 --float-size f4 --skip-existing" \
  > /tank/projects/run_0910_tierb.log 2>&1 </dev/null &
```

### The resume — same command, new log

`--skip-existing` needs no other change. It reported exactly the right thing:

```
To process: 12 member(s); skipped 38
```

**It requires a member present in BOTH stores at each store's own final step**, so member 39 —
which had reached 720 h of 792 h when the process died — was **redone**, not accepted as
complete. That guarantee is why a silent kill mid-member is recoverable at all. Write the
resume to a *separate* log (`run_0910_tierb_resume.log`); the original still holds members 1–38
and is the only record of the first pass.

```
[TAG] icechunk_o96: cycle-20260910_0000 -> GJBT0JKSHJBZ9C60ACGG (50/50 members)
[TAG] icechunk_n320_aiwq: cycle-20260910_0000 -> EN0R5X5NEY3WE9T4RMH0 (50/50 members)
DONE: 12 written, 38 skipped, 0 failed of 50 members
```

**Two `[TAG]` lines** — first cycle where the sidecar tags itself (`5554586`). On 20260903 it
had to be tagged by hand afterwards.

### Post-run validation

Validate **after** a resume, not just after a clean run, and check the member that was redone:

| check | O96 corpus | N320 sidecar |
|---|---|---|
| tag resolves | ✅ `cycle-20260910_0000` | ✅ `cycle-20260910_0000` |
| `msl` shape | `(50, 132, 40320)` → O96 | `(50, 132, 542080)` → N320 |
| arrays | 124 | 14 (10 data + 4 coords) |
| stored steps | **132/132**, hours 0–792 | **61/132**, hours 432–792 |
| members 1 / 26 / **39** / 50 | finite at last step | finite at last step |

Member 39 is the one the kill left partial. It is finite in both stores, so the resume rewrote
it rather than leaving a half-written member behind the `--skip-existing` check.

### 3a — now with `--icechunk-tag`

```bash
$PY ../shared/aifs_n320_grib_1p5defg_nc_cli.py --date 20260910_0000 --members 1-50 --v2 \
    --no-upload --source icechunk --icechunk-store $BASE/icechunk_n320_aiwq \
    --source-grid n320 --icechunk-tag cycle-20260910_0000 --output-dir $BASE/nc_1p5deg
```

```
📖 Icechunk store: …/icechunk_n320_aiwq (tag=cycle-20260910_0000)
```

**`tag=`, not `branch=main`.** 20260903 had to omit the flag because the sidecar was untagged,
and passing a non-existent tag would have written it into the product's attributes even though
the branch was read. With the sidecar tagging itself, the product now records a real immutable
snapshot.

### 3b / 3c

```bash
$PY ../shared/ensemble_quintile_analysis_cli.py --date 20260910 --v2 \
    --local-nc-dir $BASE/nc_1p5deg --work-dir $BASE/aiwq
setsid nohup bash -c "cd $PWD; \
  $PY -u ../shared/forecast_submission_cli.py --date 20260910 --v2 --output-dir $BASE/aiwq" \
  > /tank/projects/run_0910_3c.log 2>&1 </dev/null &
```

Climatology downloaded for **20260928** and **20261005**, six files, all real downloads.

Offline pre-flight, all passing — `--dry-run` exercises neither the ECBox token nor the AI-WQ
checks, so it proves nothing:

| check | result |
|---|---|
| **provenance** | `…/icechunk_n320_aiwq @ **cycle-20260910_0000**` |
| window open | pass |
| team + model registered | pass |
| 6 arrays | `(5,121,240)`, finite, ∈[0,1], Σ=1.000000 |

```
[1/6] mslp_week1 ✅   [3/6] pr_week1 ✅   [5/6] tas_week1 ✅
[2/6] mslp_week2 ✅   [4/6] pr_week2 ✅   [6/6] tas_week2 ✅
Successful: 6   Failed: 0
```

---

## What went wrong

### 1. Step 2 was killed silently at member 39 — cause unknown

The log simply stops mid-member at `720h / 792h`, after `member 038` completed normally:

```
    600h / 792h (stored 100/132, 100 commits)
    720h / 792h (stored 120/132, 120 commits)
<end of file>
```

**No traceback, no Python exception, no `Killed` line, no OOM message.** A Python error would
have left a traceback; the absence of one means the process was terminated externally. Disk was
not the cause — 328 GB free at the time. `dmesg` was not readable from the session, so the
kernel log was never checked.

This is the **second** occurrence of the same signature: 20260820's O96 rollout died at member
40 and was attributed to SIGTERM. Two silent kills in the high-30s/40s of a 50-member rollout
is a pattern, not bad luck.

**Next time it happens, check before restarting** — the evidence is gone once the box is busy
again:

```bash
journalctl -k --since "-3h" | grep -iE "killed process|oom|segfault"
journalctl --user --since "-3h" | grep -iE "terminat|kill|timeout"
```

A plausible candidate not yet ruled out is a session or service manager reaping long-running
detached jobs. `setsid` survives the launching shell but not a cgroup-level cleanup.

### 2. Step 1 took 5 hours

| cycle | s/member |
|---|---|
| 20260820 | 126.5 |
| 20260827 | 180.8 |
| 20260903 | 183.5 |
| 20260909 | 130.3 |
| **20260910** | **363.1** |

Nothing failed — 50/50, 0 failed, 42 GB, all symlinks resolve — so this is throughput, not
correctness, and the GCS mirror is the obvious suspect. But it is **2× the worst previous
cycle**, and 20260909 ran at 130 s only a day earlier, so it is not a steady drift.

**The consequence is schedule, not data.** A 5 h Step 1 plus a 4.6 h Step 2 no longer fits in a
working day, which is how this cycle ended up finishing on the last day of its window.

### 3. The dead run sat for 26 h before anyone looked

Steps 3a and 3b are 10 minutes, have no external dependency and take no irreversible action.
**Chaining them onto Step 2's `DONE:` line** would have surfaced the failure within minutes
instead of a day — the absence of a `DONE:` line *is* the alarm. This is the fourth cycle in a
row where a finished or failed rollout waited on a human noticing; the previous three cost
22 h, 24 h and 58 h.

---

## Outputs

| Path | Size | What |
|------|------|------|
| `…/20260910_0000/input_states/` | 42 GB | 50 pkls + 50 symlinks |
| `…/20260910_0000/icechunk_o96/` | **162 GB** | 124 vars × 50 × **132/132 steps**, tag `cycle-20260910_0000` |
| `…/20260910_0000/icechunk_n320_aiwq/` | **52 GB** | 10 vars × 50 × 61/132 steps, tag `cycle-20260910_0000` |
| `…/20260910_0000/nc_1p5deg/` | 2.0 GB | 50 × 1.5° NetCDF, from the sidecar |
| `…/20260910_0000/aiwq/…_20260910_v2.nc` | 6.7 MB | the AI-WQ product, provenance by tag |
| ECBox `…/forecast_submissions/20260910/` | 6 files | the submission |
| `/tank/projects/{proto_gcs_20260910_all50,run_0910_tierb,run_0910_tierb_resume,run_0910_3a,run_0910_3b,run_0910_3c}.log` | — | step logs — note **two** Step 2 logs |
| [`fp16FahamuAIFSv2_20260910.txt`](fp16FahamuAIFSv2_20260910.txt) | — | 3b + 3c transcript |

Disk after: **264 GB free (87 %)**. Tier B is 214 GB of that, and
`cleanup_aifs_run.py` still cannot reclaim it — `HEAVY_PREFIXES` matches neither
`icechunk_o96` nor `icechunk_n320_aiwq`.

---

## Tropical-storm days — run before this store is purged

The tracker was run on this cycle on **2026-09-21**, contributing 100 samples to the
detector-native climatology:

```bash
$PY ts-mjo/ts_days.py --store $BASE/icechunk_n320_aiwq --tag cycle-20260910_0000 \
    --init 20260910 --out $BASE/ts_days_probs_20260910_tracked.nc
```

| week | ATL mean | NWP mean |
|---|---|---|
| 2026-09-28 | 2.2 | 8.2 |
| 2026-10-05 | 2.4 | 7.5 |

NWP's 6.2 in the previous cycle and 8.2 here are the low end of the sample —
the spread across cycles is real and is why a single cycle cannot be a climatology.

> **Retention rule.** The store is **52 GB** and will be purged; this product is
> **~20 KB** and is the entire scientific value of the cycle for this target. `cleanup_aifs_run.py` now
> **refuses to purge a cycle whose N320 store is present but whose TS product is not** — the
> store is the only source of the fields the tracker reads. See
> [`ts-mjo/TS_STORM_DAYS.md`](ts-mjo/TS_STORM_DAYS.md).

# AIFS ENS v2.0 Run Commands — 20260903 (tier B: O96 corpus + N320 sidecar, submitted)

**Date:** 20260903 (Thursday) · **Members:** 1–50 · **Box:** local RTX 5000 Ada, 30 GB.
**Outcome: ✅ complete and submitted.** The **first production tier-B cycle** — one rollout
writing an O96 corpus and an N320 sidecar side by side — with 3a/3b on the sidecar and
**6/6 files submitted 2026-09-05 11:21 UTC**, 1 d 12 h inside the window. Verification weeks
**2026-09-21** and **2026-09-28**; scoreable from **2026-09-28**.

> **209 GB instead of 583 GB, and the MJO target is no longer thrown away.** The previous
> shape stored 120 variables at N320 over hours 432–792 only. Tier B stores the **full
> 0–792 h at O96** plus a 10-variable N320 sidecar over the downstream window — a third of
> the disk, and days 8 and 15 survive, which MJO needs and the old window discarded.

---

## ⏱ End-to-end timing

| Step | What | Wall time | Output |
|------|------|-----------|--------|
| **0** | open-data availability pre-check | seconds | AIFS v2 RUNNABLE — sfc 9/9, 14/14 levels, 11/11 wave |
| **—** | purge `20260820` to make room | ~10 m | **633.4 GB reclaimed**, `aiwq/*.nc` kept |
| **1** | pkl creation, 50 members (proto, **GCS**) | **2 h 34 m** (153.6 min) → **183.5 s/member** | 42 GB, 50/50, 0 failed |
| **2** | GPU inference → **O96 + N320 sidecar** | **4 h 38 m** (03:37→08:15) → **334.3 s/member** | **209 GB**, 50/50, 0 failed, tagged |
| **3a** | regrid 432–792 h → 1.5° NetCDF (**from the sidecar**) | **8.3 m** | 2.0 GB, 50/50, 0 failed |
| **3b** | quintiles → AI-WQ NetCDF | **~2 m** | 6.7 MB |
| **3c** | **live submit via ECBox** | **1 h 15 m** (10:06→11:21 UTC) → ~12.5 min/file | **6/6 submitted, 0 failed** |

**334.3 s/member against the 287.6 s N320 baseline** — the sidecar costs ~47 s/member, close
to the ~18 s the 3-variable tier-B sidecar was measured at plus the extra seven variables.
Per-member spread 266.4–418.6 s.

Step 1 ran at 183.5 s/member, in line with 20260827's 180.8 and still well off the 126.5 s
seen on 20260820. Two cycles at ~182 s now; treat 126.5 as the outlier, not the norm.

---

## Commands (as run)

```bash
PY=/tank/projects/micromamba/envs/aifs-gpu/bin/python
BASE=/tank/projects/aifs-run/20260903_0000
```

### Step 0 — is the cycle published?

```bash
cd /home/ezra/ea-aifs/fp16FahamuAIFSv2/run-pre50r1-dates
$PY check_open_data_inputs.py --date 20260903 --source gcs
# -> AIFS v2: RUNNABLE   sfc 9/9, sol 2/2, pl 6/6, levels 14/14, wave 11/11
```

### Free the disk — 150 GB free, ~207 GB needed

```bash
$PY cleanup_aifs_run.py --date 20260820            # dry-run plan
$PY cleanup_aifs_run.py --date 20260820 --yes      # 633.4 GB reclaimed
```

> **⚠️ `cleanup_aifs_run.py` cannot reclaim a tier-B cycle.** Its `HEAVY_PREFIXES` is
> `("icechunk_v2", "fp16_v2_forecasts")` — it matches neither `icechunk_o96` nor
> `icechunk_n320_aiwq`. The 20260820 purge proved it: it reported "Deleted 4/4 paths" while
> leaving 105 GB behind (`icechunk_o96` 102 GB plus two `*_nwh` test stores). When 20260903
> is purged it will free the 42 GB of pkls and leave **209 GB of stores untouched**. Fix
> `HEAVY_PREFIXES` before the next disk squeeze.

### Step 1 — input pkls, then the symlinks

```bash
mkdir -p $BASE/input_states
cd /home/ezra/ea-aifs/s3_grib_pkl
setsid nohup bash -c "cd $PWD; export PYTHONUNBUFFERED=1; \
  uv run --script s3_grib_pkl_input_aifsens_v2_proto.py \
    --date 20260903_0000 --members 1-50 --source gcs --skip-existing \
    --out $BASE/input_states" > /tank/projects/proto_gcs_20260903_all50.log 2>&1 </dev/null &

cd $BASE/input_states && for f in proto_input_state_member_*.pkl; do ln -sf "$f" "${f#proto_}"; done
ls input_state_member_*.pkl | wc -l                                          # must print 50
for f in input_state_member_*.pkl; do [ -e "$f" ] || echo "DANGLING $f"; done  # and resolve
```

Count *and* resolve — `ln -sf` happily links to a missing target. Skipping the symlinks makes
Step 2 exit in 2 seconds with `DONE: 0 written, 0 skipped, 50 failed`.

### Step 2 — one rollout, two stores

```bash
setsid nohup bash -c "cd $PWD; export HF_HOME=/tank/projects/hf_cache; \
  $PY -u run_local_icechunk_v2.py --date 20260903_0000 --members 1-50 --lead-time 792 \
    --input-dir $BASE/input_states \
    --grid o96 --store $BASE/icechunk_o96 \
    --native-store $BASE/icechunk_n320_aiwq \
    --native-vars msl,tp,2t,10u,10v,t_200,t_300,t_500,u_850,v_850 \
    --native-write-hours 432-792 \
    --n-members 50 --commit-every 1 --float-size f4 --skip-existing" \
  > /tank/projects/run_0903_tierb.log 2>&1 </dev/null &
```

Three deliberate choices:

- **No `--write-hours` on the O96 store.** The corpus must stay complete. MJO's submission is
  a (9,4) array at days 8/15/22/29 and `check_data_characteristics` requires every column to
  sum to 1, so dropping days 8 and 15 makes MJO unsubmittable with no way back short of
  re-running inference.
- **`--native-write-hours 432-792`** — the sidecar only needs what 3a and the TS tracker read.
- **10 native vars, not the 3-var default.** The extra seven are what `ts_tracks.py` reads
  (`10u`, `10v`, `t_200/300/500`, `u_850`, `v_850`). At 3 vars the TS target would need the
  full N320 store again, defeating the point.

`HF_HOME=/tank/projects/hf_cache` is not optional and not guessable — see
[`run_commands_20260827.md`](run_commands_20260827.md), where the wrong cache stalled a run
silently for nine minutes.

```
Write window: ALL 132 steps stored
    [SCHEMA] init            50x132x40320,  120 vars (f4), time_chunk=1
    [SCHEMA] native sidecar  50x132x542080, 10 vars
[TAG] cycle-20260903_0000 -> XMA2VH69PHXC34T4FA2G (50/50 members)
DONE: 50 written, 0 skipped, 0 failed of 50 members
```

**Two schema lines is the tier-B check.** `40320` is O96, `542080` is N320. The O96 line must
say **ALL 132 steps**. The GPU should reach ~11.8 GB / 99 % within a minute or two.

Early on, the sidecar stays at ~160 KB while the O96 store grows into the hundreds of MB —
that is `--native-write-hours` working, not a fault. It starts filling once the rollout
passes hour 432.

### Post-run validation — open both stores

| check | O96 corpus | N320 sidecar |
|---|---|---|
| `msl` shape | `(50, 132, 40320)` → **O96** | `(50, 132, 542080)` → **N320** |
| variables | **124** | **14** (10 data + 4 coords) |
| stored steps | **132/132**, hours 6–792 | **61/132**, hours 432–792 |
| members 1 / 26 / 50 | finite at first and last | finite at first and last |
| tag | `cycle-20260903_0000` | `cycle-20260903_0000` — **see below** |

> **The sidecar was untagged on this run.** `run_local_icechunk_v2.py` tagged only the main
> store, so `icechunk_n320_aiwq` had `tags: []` and the O96 README §7 command
> (`--icechunk-tag cycle-<date>` against the sidecar) could not resolve. **Fixed in
> `5554586`** — both stores are now tagged, each checked for completeness against its own
> last kept step. The 20260903 sidecar was tagged retroactively after verifying all 50
> members were present; icechunk tags are immutable and deleted names are tombstoned, so a
> wrong target would be permanent.

### Steps 3a / 3b — 3a reads the **sidecar**, not the corpus

```bash
$PY ../shared/aifs_n320_grib_1p5defg_nc_cli.py --date 20260903_0000 --members 1-50 --v2 \
    --no-upload --source icechunk --icechunk-store $BASE/icechunk_n320_aiwq \
    --source-grid n320 --output-dir $BASE/nc_1p5deg
$PY ../shared/ensemble_quintile_analysis_cli.py --date 20260903 --v2 \
    --local-nc-dir $BASE/nc_1p5deg --work-dir $BASE/aiwq
```

**`--icechunk-tag` was deliberately omitted**, because the sidecar was untagged at the time.
That matters for more than resolution: 3a falls back to the branch when a tag is missing
(`cli:504`) but sets `icechunk_ref` from `self.icechunk_tag or ...` (`cli:601`), so passing a
non-existent tag writes that tag name into the product's attributes **even though the branch
was read** — provenance naming a ref that was never touched. Omitting it recorded the truth:

```
📖 Icechunk store: …/20260903_0000/icechunk_n320_aiwq (branch=main)
  🔄 Reading N320 fields from Icechunk and regridding...
```

With `5554586` in place, future cycles can pass the tag and get tag-based provenance.

3b downloaded climatology for **20260921** and **20260928** over FTP, six files, all real
downloads.

> **3b ran twice on this cycle** — once before the provenance change (`run_0903_3b.log`,
> 07:03) and once after (`run_0903_3b_prov.log`, 09:08). The submitted file's
> `processing_date` is 09:08:59, so the second is the one of record. See
> [`RUN_LOGS_AND_TRANSCRIPTS.md`](RUN_LOGS_AND_TRANSCRIPTS.md).

### Step 3c — submit

```bash
setsid nohup bash -c "cd $PWD; \
  $PY -u ../shared/forecast_submission_cli.py --date 20260903 --v2 --output-dir $BASE/aiwq" \
  > /tank/projects/run_0903_3c.log 2>&1 </dev/null &
```

```
[1/6] mslp_week1 ✅   [3/6] pr_week1 ✅   [5/6] tas_week1 ✅
[2/6] mslp_week2 ✅   [4/6] pr_week2 ✅   [6/6] tas_week2 ✅
Successful: 6   Failed: 0
```

**Pre-flight offline — `--dry-run` proves nothing about the transport.** It short-circuits
before `AI_WQ_create_empty_dataarray`, exercising neither the ECBox token nor the AI-WQ
checks. What works without transmitting, all of which passed:

| check | result |
|---|---|
| **provenance** — which store built this? | `…/icechunk_n320_aiwq @ branch:main` |
| window open | `check_forecast_data_window('20260903')` → pass |
| team + model registered | `AI_WQ_create_empty_dataarray(...)` — hits the server, uploads nothing |
| arrays | `(5,121,240)`, finite, ∈[0,1], Σ=1.000000 — 6/6 |
| template fit | coords `quintile = [0.2 … 1.0]` |

The first row is new this cycle (`8074bf6`): the quintile file now carries
`source_icechunk_store` / `source_icechunk_ref`, so on a cycle with two stores on disk you can
confirm **before** submitting that the product came from the N320 sidecar and not the O96
corpus. Previously that was unknowable from the file itself.

**Rotate the ECBox token** in the `sites.ecmwf.int` web UI and update the `ecbox` key in
`.env` — the earlier token is exposed in `2026-06-30-aifs-ebox-login2.txt`.

---

## Outputs

| Path | Size | What |
|------|------|------|
| `…/20260903_0000/input_states/` | 42 GB | 50 pkls + 50 symlinks |
| `…/20260903_0000/icechunk_o96/` | **158 GB** | 124 vars × 50 members × **132/132 steps** (0–792 h), tag `cycle-20260903_0000` |
| `…/20260903_0000/icechunk_n320_aiwq/` | **51 GB** | **10 vars** × 50 members × **61/132 steps** (432–792 h), N320 |
| `…/20260903_0000/nc_1p5deg/` | 2.0 GB | 50 × 1.5° NetCDF, from the sidecar |
| `…/20260903_0000/aiwq/ensemble_quintile_probabilities_20260903_v2.nc` | 6.7 MB | the AI-WQ product, now carrying provenance |
| ECBox `AI_Weather_Quest/forecast_submissions/20260903/` | 6 files | the submission |
| `/tank/projects/{proto_gcs_20260903_all50,run_0903_tierb,run_0903_3a,run_0903_3b,run_0903_3b_prov,run_0903_3c}.log` | — | step logs |
| [`fp16FahamuAIFSv2_20260903.txt`](fp16FahamuAIFSv2_20260903.txt) | 1038 lines | 3b + 3c transcript |

---

## What tier B buys, and what it does not

| | old (full N320) | tier B |
|---|---|---|
| disk per cycle | 583 GB | **209 GB** |
| O96 corpus | — | **132/132 steps**, 124 vars — MJO submittable |
| N320 | 120 vars, 61 steps | **10 vars**, 61 steps |
| gridded AI-WQ product | from N320 | from the sidecar, **bit-identical route** |
| TS tracker | from N320 | from the sidecar (its 7 extra vars) |

**What the sidecar does not carry.** Its 10 variables were chosen for the AI-WQ submission and
the TS tracker. It holds **no `q` at any level, no `w`, no `z`, and winds only at 850** — so
anything needing moisture, vertical motion or geopotential at 28 km must use the O96 corpus at
~112 km instead. That is the right trade for this cycle's targets, but it is a real constraint
on the evidence-node work: see
[`epistemic-reasoning-risk/EVIDENCE_NODES_OVER_33_DAYS.md`](epistemic-reasoning-risk/EVIDENCE_NODES_OVER_33_DAYS.md),
which prices raising those nodes to 28 km at ~309 GB for the full range.

`--native-vars` is an **inference-time** choice. Recovering a variable that was not written
costs a full 4½-hour rollout, so it is worth settling before the next cycle rather than after.

---

## The gaps this cycle exposed, in the order they should be closed

1. **`cleanup_aifs_run.py` cannot reclaim tier-B stores** — one line in `HEAVY_PREFIXES`.
   Bites at the next disk squeeze, and 209 GB is sitting behind it.
2. **The 22-hour idle gap.** Step 2 finished 09-04 08:15 UTC; 3a did not start until 09-05
   06:53. Third cycle running that a finished rollout waited on a human. 3a and 3b are
   11 minutes with no external dependency and no irreversible action — chaining them onto
   Step 2's `DONE:` line would leave every finished rollout submission-ready.
3. **`--native-vars` scope** — see above; decide before running 20260910.

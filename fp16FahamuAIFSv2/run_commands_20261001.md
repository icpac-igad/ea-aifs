# Run log — 20261001 (AIFS-ENS 2.0, tier B)

The cleanest rollout so far — Step 1 and Step 2 both ran first time with no intervention,
because the four traps from [`run_commands_20260924.md`](run_commands_20260924.md) were
handled up front in a chain script. **And then the submission silently did not happen.**
That is the whole lesson of this cycle: automating the compute paid off, automating the
*submission* did not.

---

## ⏱ End-to-end timing

| Step | What | Wall time | Output |
|------|------|-----------|--------|
| **0** | open-data pre-check | seconds | **RUNNABLE** — 9/9 sfc, 2/2 sol, 6/6 pl, 14/14 levels, 11/11 wave |
| **1** | pkl creation, 50 members (proto, **GCS**) | 04:41 → 06:34, **135.3 s/member** | 42 GB, **50/50** |
| **2** | tier-B rollout | 06:34 → 11:11, **4 h 37 m** | **50/50, 0 failed** |
| **3a** | regrid 432–792 h → 1.5° | **8.3 m** | 2.0 GB, 50/50 |
| **3b** | quintiles | **~3 m** | 6 arrays |
| **3c** | submit tas/mslp/pr | ~1 h 20 m | **6/6, 0 failed** |
| **TS** | tracker on the N320 sidecar | **~30 m** | 8.5 KB |
| **MJO** | VPM → RMM-fitted basis, O96 corpus | **~25 m** | 49 KB |
| **TS+MJO submit** | 3 files | ~35 m | **3/3** — but a day late, see below |

No purge was needed: 419 GB free at the start against ~261 GB required.

Stores: **O96 158 GB + N320 51 GB**, both tagged `cycle-20261001_0000`, 120/10 data arrays,
132/61 steps, members 1/26/50 finite at the last step. Disk after: 166 GB free.

**TS and MJO ran ~10× faster than on 20260924** (30/25 min against ~4 h 50 m). Nothing changed
in either script. The difference is I/O contention: last cycle they ran while 3c uploaded, each
other competed, and a 633 GB purge had just churned the pool. Here they had a quiet disk. Worth
knowing before reading a slow TS run as a problem.

---

## The chain script — what worked

`/tank/projects/chain_20261001.sh` ran **rename → Step 2 → 3a → 3b** unattended, with the
20260924 traps encoded rather than remembered:

- `input_states/` created before Step 1, since the proto script will not create it;
- `--date 20261001_0000`, because the bare `YYYYMMDD` dies in `strptime`;
- the `proto_` prefix stripped, **with an assertion** that `input_state_member_001.pkl` exists
  afterwards — without it, Step 2 reports `0 written, 50 failed` in seconds and looks like
  missing data;
- `HF_HOME=/tank/projects/hf_cache`, not `hf_home`;
- each stage gated on its own success string (`50 written`, `Successful: 50/50`, the submission
  file existing) so a silent partial cannot cascade.

All of it passed first time. This part of the automation earned its keep.

## The chain script — what failed, and why it was the wrong thing to automate

A second script, `submit_tsmjo_1001.sh`, was set to wait for 3c to release the shared
unique-ID CSV and then submit TS and MJO. **It never submitted.** Its log ends at

```
### 12:07 waiting for 3c (update_table_unique_identifies writes a shared CSV)
```

and ~17 h later the process was gone, 3c had finished 6/6, and
`AI_WQ_check_submission` reported all three files **ABSENT**. Resubmitted by hand the next day;
all three then landed, verified present on the server.

Three things to take from it:

1. **The window is what makes this dangerous.** 20261001's window closed **2026-10-04 23:59
   UTC** and the failure was caught at 2026-10-03 04:54 — ~43 h of margin. With a tighter
   window, or if nobody had asked for status, the TS and MJO entries would simply not exist.
2. **It failed silently.** No non-zero exit, no error line, nothing in the log after the first
   message. A `setsid` script that dies mid-wait leaves no trace distinguishable from one still
   waiting.
3. **Chaining a submission behind a long job is a bad trade.** It saves one check-in and buys an
   unmonitored failure mode. 20260924 submitted TS/MJO interactively, immediately after 3c, and
   worked first time. **Do that instead.**

**The rule this cycle establishes: always verify a submission against the server, never against
its own log.** `AI_WQ_check_submission` per variable and period is the check; the log saying
"submitted" is not. That is how this was caught at all.

---

## Commands (as run)

```bash
PY=/tank/projects/micromamba/envs/aifs-gpu/bin/python
BASE=/tank/projects/aifs-run/20261001_0000
mkdir -p $BASE/input_states

$PY -u s3_grib_pkl/s3_grib_pkl_input_aifsens_v2_proto.py \
    --date 20261001_0000 --members 1-50 --source gcs \
    --out $BASE/input_states --skip-existing
cd $BASE/input_states && for f in proto_input_state_member_*.pkl; do mv "$f" "${f#proto_}"; done

export HF_HOME=/tank/projects/hf_cache
$PY -u run_local_icechunk_v2.py --date 20261001_0000 --members 1-50 --lead-time 792 \
    --input-dir $BASE/input_states \
    --grid o96 --store $BASE/icechunk_o96 \
    --native-store $BASE/icechunk_n320_aiwq \
    --native-vars msl,tp,2t,10u,10v,t_200,t_300,t_500,u_850,v_850 \
    --native-write-hours 432-792 \
    --n-members 50 --commit-every 1 --float-size f4 --skip-existing

$PY -u ../shared/aifs_n320_grib_1p5defg_nc_cli.py --date 20261001_0000 --members 1-50 --v2 \
    --no-upload --source icechunk --icechunk-store $BASE/icechunk_n320_aiwq \
    --source-grid n320 --icechunk-tag cycle-20261001_0000 --output-dir $BASE/nc_1p5deg
$PY -u ../shared/ensemble_quintile_analysis_cli.py --date 20261001 --v2 \
    --local-nc-dir $BASE/nc_1p5deg --work-dir $BASE/aiwq
$PY -u ../shared/forecast_submission_cli.py --date 20261001 --v2 --output-dir $BASE/aiwq

$PY -u ts-mjo/ts_days.py --store $BASE/icechunk_n320_aiwq --tag cycle-20261001_0000 \
    --init 20261001 --out $BASE/ts_days_probs_20261001_tracked.nc
$PY -u ts-mjo/vpm_index.py --store $BASE/icechunk_o96 --init 20261001 \
    --clim /tank/projects/era5_vpm_clim/vpm_clim_1991_2020.npz \
    --eofs /tank/projects/era5_vpm_clim/rmm_basis.npz \
    --out $BASE/mjo_probs_20261001.nc
$PY -u submit_ts_mjo_cli.py --date 20261001 \
    --ts-nc $BASE/ts_days_probs_20261001_tracked.nc --tercile-dir $BASE/ts_terciles \
    --mjo-nc $BASE/mjo_probs_20261001.nc        # --dry-run first; then VERIFY on the server
```

Pre-flight before 3c: provenance `icechunk_n320_aiwq @ cycle-20261001_0000`; 6 arrays
`(2,5,121,240)`, finite, ∈[0,1], Σ=1 over the quintile axis; weeks 2026-10-19 / 2026-10-26.

---

## TS days — and a bound that makes "below" unreachable

| week | ATL mean | NWP mean |
|---|---|---|
| 2026-10-19 | 1.74 d | 7.08 d |
| 2026-10-26 | 1.60 d | 4.28 d |

14.1 tracks per member. Official bounds from
`/climatologies/2026/TS_20yrCLIM_WEEKLYTSDAYS_terciles_<validdate>.nc`:

| week | ATL | NWP |
|---|---|---|
| 2026-10-19 | 1 / 4 | 2 / 7 |
| 2026-10-26 | **0 / 5** | 2 / 6 |

Submitted:

| period | basin | below | near | above |
|---|---|---|---|---|
| 1 | ATL | 0.54 | 0.22 | 0.24 |
| | **NWP** | 0.08 | 0.36 | **0.56** |
| 2 | ATL | **0.00** | **0.90** | 0.10 |
| | NWP | 0.26 | 0.44 | 0.30 |

**ATL week 2's 0.90 on "near" is a property of the bounds, not of the forecast.** With a lower
tercile of **0**, `below` means `x < 0` — unreachable — so every member lands in `0 ≤ x < 5`.
A near-certain "near normal" that the model never asserted. Read the score accordingly, and
expect this whenever a published lower bound is 0.

**NWP week 1 is the familiar over-count**: mean 7.08 days against an upper tercile of 7, giving
P(above) = 0.56 — the same shape as the two verified checkpoints that were confidently wrong at
0.84 and 0.76. Week 2 is the first time NWP has *not* leaned on "above" in both windows
(P(above) 0.30, mean 4.28), which is worth watching rather than crediting yet.

## MJO phases — unchanged, still miscalibrated

Modal categories **7 / 7 / 8 / 8** at init +7/14/21/28 (valid 10-08, 10-15, 10-22, 10-29),
P(inactive) ≈ 0, **mean amplitude 3.07** against an observed 1.30 and P(amp<1) **0.01** against
0.37. Consistent with 2.05–2.84 over the previous four cycles: the §6g mean-state bias, not a
plumbing fault. `ts-mjo/MJO_METHOD.md` §5 still recommends against submitting; submitted on
instruction.

Bands kept at `/tank/projects/mjo_model_clim/bands_20261001.npz` — the ninth cycle in that set,
which is the model-climatology sample this bias needs (§5 of `MJO_METHOD.md`).

---

## Related

- [`run_commands_20260924.md`](run_commands_20260924.md) — the four Step 1 traps this cycle
  pre-empted, and the first TS/MJO submission.
- [`ts-mjo/MJO_METHOD.md`](ts-mjo/MJO_METHOD.md) — §5 on the model climatology.
- [`ts-mjo/TS_STORM_DAYS.md`](ts-mjo/TS_STORM_DAYS.md) — the official tercile bounds and the
  NWP over-count.

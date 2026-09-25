# Run log — 20260924 (AIFS-ENS 2.0, tier B)

The first cycle where **TS and MJO were submitted alongside tas/mslp/pr**. Also the cycle
where Step 1 had to be repaired before it would run at all: the environment had lost two
packages and the two scripts either side of Step 1 disagree about filenames.

---

## ⏱ End-to-end timing

| Step | What | Wall time | Output |
|------|------|-----------|--------|
| **0** | open-data pre-check | seconds | **RUNNABLE** — 9/9 sfc, 2/2 sol, 6/6 pl, 14/14 levels, 11/11 wave |
| **—** | purge `20260827` (after extracting its MJO bands) | ~12 m | **633.4 GB reclaimed**, 46 GB → 673 GB free |
| **1** | pkl creation, 50 members (proto, **GCS**) | **107.7 min** → **128.3 s/member** | 42 GB, 49/50 + 1 refill |
| **2** | tier-B rollout (O96 corpus + N320 sidecar) | **~4 h 31 m** | **50/50, 0 failed** — no silent kill |
| **3a** | regrid 432–792 h → 1.5° NetCDF | **8.3 m** | 2.0 GB, 50/50 |
| **3b** | quintiles → AI-WQ NetCDF | **~2 m** | 6 arrays |
| **3c** | submit tas/mslp/pr via ECBox | **1 h 19 m** | **6/6, 0 failed** |
| **TS** | tracker on the N320 sidecar | ~4 h 50 m | 8.5 KB |
| **MJO** | VPM → RMM-fitted basis, from the O96 corpus | ~4 h 45 m | 49 KB |
| **TS+MJO submit** | 3 files via ECBox | **~40 m** | **3/3, verified present** |

Step 1 at 128.3 s/member sits with 20260917's 131.1 and 20260820's 126.5. Step 2 finished
**01:59 UTC**, clear of the 06:00 `apt-daily-upgrade` window — by timing, not by protection.
**The sudo fix is still unapplied.**

---

## Four things broke before Step 1 would run

None of these are in any earlier run doc, and the first two make Step 1 unrunnable rather
than slow.

**1. `aifs-gpu` had lost `gribberish` and `obstore`.** `s3_grib_pkl_input_aifsens_v2_proto.py`
imports both. `gribberish` existed only in `cgan_env`, which has no `earthkit`, so **neither
environment could run Step 1**. Fixed with:

```bash
$PY -m pip install gribberish==1.6.0 obstore
```

Worth knowing because the fallback is the AWS path, which is **5× slower** — 14 h 30 m
against 2 h 55 m (`run_commands_20260806.md`).

**2. `--date` takes `YYYYMMDD_HHMM`, not `YYYYMMDD`.** The bare form dies in `strptime`
before doing anything.

**3. The script does not create its output directory.** `mkdir -p $BASE` is not enough —
`$BASE/input_states` must exist. Member 1 fetched all 112 fields, verified
`112/112, shapes={(2,542080)}`, and *then* failed on `open(out_pkl,"wb")`. Cost one member,
refilled in 2.1 min with `--members 1 --skip-existing`.

**4. The two scripts disagree on filenames — and the failure looks like missing data.**
Step 1 writes `proto_input_state_member_NNN.pkl`; `run_local_icechunk_v2.py` hardcodes
`input_state_member_NNN.pkl` (line 125) with no prefix option. The first Step 2 launch
reported `DONE: 0 written, 0 skipped, 50 failed` in seconds, every member a
`[SKIP] Pickle file not found`. Nothing warns about this in either script.

```bash
cd $BASE/input_states && for f in proto_input_state_member_*.pkl; do mv "$f" "${f#proto_}"; done
```

---

## Commands (as run)

```bash
PY=/tank/projects/micromamba/envs/aifs-gpu/bin/python
BASE=/tank/projects/aifs-run/20260924_0000
mkdir -p $BASE/input_states          # <- step 1 will NOT create this

# Step 1
$PY -u s3_grib_pkl/s3_grib_pkl_input_aifsens_v2_proto.py \
    --date 20260924_0000 --members 1-50 --source gcs \
    --out $BASE/input_states --skip-existing
cd $BASE/input_states && for f in proto_input_state_member_*.pkl; do mv "$f" "${f#proto_}"; done

# Step 2 — tier B, unchanged from 20260903
export HF_HOME=/tank/projects/hf_cache      # NOT hf_home
$PY -u run_local_icechunk_v2.py --date 20260924_0000 --members 1-50 --lead-time 792 \
    --input-dir $BASE/input_states \
    --grid o96 --store $BASE/icechunk_o96 \
    --native-store $BASE/icechunk_n320_aiwq \
    --native-vars msl,tp,2t,10u,10v,t_200,t_300,t_500,u_850,v_850 \
    --native-write-hours 432-792 \
    --n-members 50 --commit-every 1 --float-size f4 --skip-existing
```

Post-run validation, both stores at `cycle-20260924_0000`:

| check | O96 corpus | N320 sidecar |
|---|---|---|
| `msl` shape | `(50, 132, 40320)` | `(50, 132, 542080)` |
| arrays | 124 | 14 |
| stored steps | **132/132** (0–792 h) | **61/132** (432–792 h) |
| members 1 / 26 / 31 / 50 @ last step | finite, ~101.2 kPa | finite, ~101.2 kPa |
| size | 157 GB | 51 GB |

Pre-flight before 3c:

| check | result |
|---|---|
| provenance | `icechunk_n320_aiwq @ cycle-20260924_0000` |
| 6 arrays | `(2,5,121,240)` each, finite, ∈[0,1], Σ=1 over the quintile axis |
| weeks | 2026-10-12 (week 1), 2026-10-19 (week 2) |

---

## TS days — submitted, with a known bias

```bash
$PY ts-mjo/ts_days.py --store $BASE/icechunk_n320_aiwq --tag cycle-20260924_0000 \
    --init 20260924 --out $BASE/ts_days_probs_20260924_tracked.nc
```

| week | ATL mean | NWP mean |
|---|---|---|
| 2026-10-12 | 1.58 d | 7.60 d |
| 2026-10-19 | 1.68 d | 6.10 d |

Mean 13.5 tracks per member.

**The tercile bounds come from AI-WQ, not from us.** `/climatologies/2026/` on the FTP carries
`TS_20yrCLIM_WEEKLYTSDAYS_terciles_<validdate>.nc` — the official 20-year bounds the scoring
uses. Earlier cycles reported against a detector-native climatology instead, which is not the
same quantity:

| week | basin | **official** | detector-native |
|---|---|---|---|
| 2026-10-12 | ATL | 1 / 5 | 0 / 2 |
| | NWP | 3 / 7 | 5 / 8 |
| 2026-10-19 | ATL | 1 / 4 | 0 / 2 |
| | NWP | 2 / 7 | 5 / 8 |

Submitted probabilities:

| period | basin | P(below) | P(near) | P(above) |
|---|---|---|---|---|
| 1 (days 18–24) | ATL | 0.42 | 0.46 | 0.12 |
| | **NWP** | 0.16 | 0.26 | **0.58** |
| 2 (days 25–31) | ATL | 0.48 | 0.32 | 0.20 |
| | **NWP** | 0.16 | 0.38 | **0.46** |

**NWP is the known defect and the official bounds make it visible rather than worse.** The
official thresholds (3/7, 2/7) sit *below* the detector-native ones (5/8), so our over-count
lands on "above" — P(above) 0.58 and 0.46. Two earlier checkpoints had NWP confidently wrong
at P(above) 0.84 and 0.76 while ATL was confidently right at 0.92. Nothing about this cycle
fixes that; the fix is calibrating the detector, **not** choosing friendlier bounds.

**SWIO and SEIO are out of season and filled with a uniform 1/3.** AI-WQ's
`check_data_characteristics` activates ATL+NWP for months 6–11 and SWIO+SEIO for 12/1/2, and
**only active columns must sum to 1**. Their official bounds are 0/0, which under the binning
rule would send *all* probability to "above" — a confident forecast of an out-of-season basin.
Uniform 1/3 asserts nothing and is unscored either way.

## MJO phases — submitted, and knowingly miscalibrated

```bash
$PY ts-mjo/vpm_index.py --store $BASE/icechunk_o96 --init 20260924 \
    --clim /tank/projects/era5_vpm_clim/vpm_clim_1991_2020.npz \
    --eofs /tank/projects/era5_vpm_clim/rmm_basis.npz \
    --out $BASE/mjo_probs_20260924.nc
```

Reads the **O96 corpus**, not the sidecar: the sidecar has no `u_200`/`v_200`, and the MJO is
zonal wavenumber 1–3 so ~112 km is ample — the mirror of TS, where 28 km is essential.

Submitted as `(9,4)` at init +7/14/21/28 days — four single days, not weekly means, and one
file for all four lags (MJO is not split by forecast period).

| | 10-01 | 10-08 | 10-15 | 10-22 |
|---|---|---|---|---|
| modal category | 8 | 7 | 8 | 8 |
| P(inactive) | 0.00 | 0.00 | 0.00 | 0.02 |

**mean amplitude 2.84, P(amp<1) = 0.01, against an observed 1.30 and 0.37.** This is the §6g
bias: an *observed* climatology removed from a *model* field leaves the model's mean-state
offset in the anomaly (|offset|/std ≈ 0.67, stable across cycles), and dividing by ERA5's
standard deviations turns it into amplitude. `MJO_METHOD.md` §5 recommends **not** submitting
until a model climatology exists; it was submitted anyway on instruction. Expect category 0 to
be starved and the score to suffer.

The basis itself is sound — fitted directly to AI-WQ's official RMM, 98.4% correct octant on
held-out years. The failure is entirely on the forecast side.

```bash
$PY submit_ts_mjo_cli.py --date 20260924 \
    --ts-nc $BASE/ts_days_probs_20260924_tracked.nc --tercile-dir $BASE/ts_terciles \
    --mjo-nc $BASE/mjo_probs_20260924.nc          # --dry-run first
```

## What gets submitted, and the credential that does it

**Three submissions, not four: TS in both windows, MJO in window 1 only.** Confirmed against
`SON26_MJO_TS_successful_submissions.xlsx`, whose columns run
`Comp. week N, TS, window 1 | …, MJO, window 1 | …, TS, window 2` — there is no
*MJO, window 2* column. It matches the package: MJO builds one `(9,4)` covering lags
7/14/21/28 and is not split by forecast period.

That sheet also shows **`Fahamu / fp16FahamuAIFSv2` as `No` in every column** — so this is the
team's *first* TS and MJO submission, not a repeat. Only ~8 of the 52 registered models submit
these at all.

**Submission and retrieval need different credentials, and the error does not say so.**
Retrieval goes over FTP with `AIWQ_PASSWORD`; **submission goes via ECBox and needs the
`ecbox` token**. The package takes a single `password` argument for both —
`ftp_or_ecbox_loading` uses it as the FTP password and then as a Branca token — so the wrong
one fails with:

```
HTTP Code: 403 … {"message":"Not a Branca token","error":"Forbidden"}
  FAILED TS period 1: Could not list files from location '/forecast_submissions'.
```

An **auth** failure that reads like a **path** failure. The first TS/MJO attempt lost 0/3 to
it. `shared/forecast_submission_cli.py` already had the right precedence in a comment;
`submit_ts_mjo_cli.py` now matches it:

```python
pw = os.environ.get("AIWQ_ECBOX_TOKEN") or os.environ.get("ecbox") or os.environ["AIWQ_PASSWORD"]
```

Note the data was never the problem — the window check and `All data is between 0 and 1` both
passed on the failing attempt.

Verified on the server afterwards with `AI_WQ_check_submission`:

```
TS_20260924_p1_Fahamu_fp16FahamuAIFSv2.nc   exists
TS_20260924_p2_Fahamu_fp16FahamuAIFSv2.nc   exists
MJO_20260924_p1_Fahamu_fp16FahamuAIFSv2.nc  exists
```

**9 submissions for this cycle**: 6 gridded (tas/mslp/pr × 2 weeks) + TS × 2 + MJO × 1.

---

## Disk

Started at **46 GB free** — a cycle needs ~261 GB, so nothing could run. `20260827` was purged
for **633.4 GB** (576.9 icechunk_v2 + 45.2 input_states + 9.7 nc_1p5deg), taking free space to
673 GB; after this cycle's 208 GB of stores, ~410 GB remains.

**Its MJO bands were extracted first.** Every remaining store is VPM-ready, so purging one
destroys a model-climatology sample that cannot be recovered — the same retention logic
already codified for TS, but not yet enforced for MJO. Eight cycles now have bands in
`/tank/projects/mjo_model_clim/` (41 MB, 2026-05-14 → 2026-10-20), **outside `aifs-run`** so
`cleanup_aifs_run.py` cannot reach them.

Two standing cautions:

- **`20260514`'s source.coop upload is incomplete** (68,979 / 805,110 objects). It is 103 GB
  and looks like an easy target; deleting it strands that upload permanently.
- **`20260813` has an N320 store with no TS product**, so cleanup refuses it by design. That
  guard is correct.

---

## Related

- [`ts-mjo/MJO_METHOD.md`](ts-mjo/MJO_METHOD.md) — the MJO method, and §5 on why the forecast
  product is not yet submittable.
- [`ts-mjo/TS_STORM_DAYS.md`](ts-mjo/TS_STORM_DAYS.md) — the tracker, and the retention rule.
- [`run_commands_20260917.md`](run_commands_20260917.md) — the previous cycle; the silent kill.
- [`RUN_LOGS_AND_TRANSCRIPTS.md`](RUN_LOGS_AND_TRANSCRIPTS.md) — where the logs are and how
  the transcript is assembled.

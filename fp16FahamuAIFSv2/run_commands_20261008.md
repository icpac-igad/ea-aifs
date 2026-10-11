# Run log — 20261008 (AIFS-ENS 2.0, tier B)

9/9 submitted, and the cycle that found **the official submission checker reports false
failures**. The compute was uneventful; the verification was not.

---

## ⏱ End-to-end timing

| Step | What | Wall time | Output |
|------|------|-----------|--------|
| **0** | open-data pre-check | seconds | **RUNNABLE** — 9/9 sfc, 2/2 sol, 6/6 pl, 14/14 levels, 11/11 wave |
| **1** | pkl creation (run 2026-10-09, disk-limited) | **131.9 s/member** | 43 GB, **50/50** |
| **2** | tier-B rollout | 04:58 → 09:34, **4 h 36 m** | **50/50, 0 failed** |
| **3a** | regrid 432–792 h → 1.5° | 09:34 → 09:42 | 2.0 GB, 50/50 |
| **3b** | quintiles | 09:42 → 09:44 | 6 arrays |
| **3c** | submit tas/mslp/pr | → 12:37 | **6/6, 0 failed** |
| **TS** | tracker on the N320 sidecar | ~25 m | 8.5 KB |
| **MJO** | VPM → RMM-fitted basis, O96 corpus | ~20 m | 49 KB |
| **TS+MJO submit** | 3 files | ~2 h | **3/3** |

Stores: **O96 158 GB + N320 51 GB**, both tagged `cycle-20261008_0000`. Disk 361 GB free after.

Step 1 ran a day early because disk was short (116 GB against the ~209 GB Step 2 needs); ~457 GB
was freed before the rollout. **Step 2 ran straight through the 06:11 `apt-daily-upgrade`
window and survived** — fourth consecutive clean rollout. The sudo fix remains unapplied.

---

## The official submission checker reports false failures

`AI_WQ_check_submission` said **`tas` and `pr` were ABSENT** for all four files, while 3c
reported `Successful: 6, Failed: 0`. The log was right and the checker was wrong.

`forecast_submission.py` calls

```python
entries = content_manager.list(remote_path=remote_dir)
```

**once**, and ignores the `continuation_token` the response carries. ECBox pages at **100
entries**; `20261008` holds **261 files across 3 pages**. The first page ended at
`mslp_20261008_p2_NMCBridge_multical.nc`, so every filename sorting after it — all `pr_*` and
all `tas_*` — was invisible and reported missing.

The failure is **alphabetical**, which determines which verdicts can be trusted:

| filename | sorts | page 1? |
|---|---|---|
| `MJO_*`, `TS_*` | early (uppercase) | **yes — reliable** |
| `mslp_*`, `pr_*`, `tas_*` | late (lowercase) | **not necessarily** |

So an **ABSENT from the official checker is only meaningful for TS and MJO.** For the gridded
variables it carries no information.

`check_submissions_paged.py` follows the continuation token and checks all nine at once:

```
$ python check_submissions_paged.py --date 20261008
20261008: 261 files across 3 page(s)
  OK   tas p1 … OK  MJO p1
9/9 present
```

### This qualifies the rule from 20261001, it does not overturn it

`run_commands_20261001.md` concluded: *always verify a submission against the server, never
against its own log* — after a chained submitter died and left TS/MJO genuinely absent while
logging nothing. **That conclusion still holds**, and the 20261001 detection was sound:
`MJO_*` and `TS_*` sort early, so they were on page 1 and their ABSENT was real, which the
successful resubmission confirmed.

What changes is the instrument. **Verify against the server with a paginated check.** The
official single-page checker is fine for TS and MJO and unusable for tas/mslp/pr.

---

## Commands (as run)

```bash
PY=/tank/projects/micromamba/envs/aifs-gpu/bin/python
BASE=/tank/projects/aifs-run/20261008_0000

# Step 1 was run on 2026-10-09, a day ahead, because Step 2 had no room.
# NOTE the symlink convention, not a rename -- see CYCLE_DATA_INVENTORY.md.
mkdir -p $BASE/input_states
$PY -u s3_grib_pkl/s3_grib_pkl_input_aifsens_v2_proto.py \
    --date 20261008_0000 --members 1-50 --source gcs \
    --out $BASE/input_states --skip-existing
cd $BASE/input_states && for i in $(seq 1 50); do
  f=$(printf "input_state_member_%03d.pkl" $i)
  [ -f "$f" ] && [ ! -L "$f" ] && mv "$f" "proto_$f" && ln -s "proto_$f" "$f"
done

# Steps 2, 3a, 3b chained in /tank/projects/chain_20261008.sh -- COMPUTE ONLY
bash /tank/projects/chain_20261008.sh

# 3c, TS and MJO interactive
$PY -u ../shared/forecast_submission_cli.py --date 20261008 --v2 --output-dir $BASE/aiwq
$PY -u ts-mjo/ts_days.py --store $BASE/icechunk_n320_aiwq --tag cycle-20261008_0000 \
    --init 20261008 --out $BASE/ts_days_probs_20261008_tracked.nc
$PY -u ts-mjo/vpm_index.py --store $BASE/icechunk_o96 --init 20261008 \
    --clim /tank/projects/era5_vpm_clim/vpm_clim_1991_2020.npz \
    --eofs /tank/projects/era5_vpm_clim/rmm_basis.npz --out $BASE/mjo_probs_20261008.nc
$PY -u submit_ts_mjo_cli.py --date 20261008 \
    --ts-nc $BASE/ts_days_probs_20261008_tracked.nc --tercile-dir $BASE/ts_terciles \
    --mjo-nc $BASE/mjo_probs_20261008.nc
$PY check_submissions_paged.py --date 20261008      # NOT AI_WQ_check_submission
```

**Compute is chained; submission is not.** That split is deliberate — see the 20261001 failure.

Validation before 3c: O96 120 arrays / 132 steps / 40,320 pts; N320 10 arrays / 61 steps /
542,080 pts; both tagged; provenance `icechunk_n320_aiwq @ cycle-20261008_0000`; 6 arrays
`(2,5,121,240)` finite, ∈[0,1], Σ=1. Weeks 2026-10-26 / 2026-11-02.

---

## TS days — NWP hedges for the second cycle running

| week | ATL mean | NWP mean |
|---|---|---|
| 2026-10-26 | 0.96 d | 5.04 d |
| 2026-11-02 | 0.88 d | 2.78 d |

11.8 tracks per member. Official bounds:

| week | ATL | NWP |
|---|---|---|
| 2026-10-26 | **0** / 5 | 2 / 6 |
| 2026-11-02 | **0** / 4 | **1** / 5 |

Submitted:

| period | basin | below | near | above |
|---|---|---|---|---|
| 1 | ATL | **0.00** | **0.96** | 0.04 |
| | NWP | 0.20 | 0.34 | 0.46 |
| 2 | ATL | **0.00** | **0.94** | 0.06 |
| | NWP | 0.32 | 0.36 | 0.32 |

**ATL's lower bound is 0 in both weeks**, so "below" means `x < 0` — unreachable — and with
means near 0.9 days everything lands in `near`. The 0.96 and 0.94 are the bounds, **not model
confidence**. 20261001 had this in one window; here it is both.

**NWP has now hedged twice.** P(above) 0.46 and 0.32, against 0.56/0.46 last cycle and
0.58/0.46 before, with means falling 7.60 → 7.08 → 5.04 and 6.10 → 4.28 → 2.78. The detector
over-count is unchanged, but the forecast activity is dropping, so it no longer pins "above".
Two cycles is a trend worth watching and not yet evidence the bias is fixed.

## MJO phases — and the first eastward progression

Modal categories **7 → 8 → 8 → 1** at init +7/14/21/28 (valid 10-15, 10-22, 10-29, 11-05).
That wraps through phase 8 to 1, i.e. **eastward progression** — every previous cycle sat
near-stationary (5–6, then 7–8, then 7/7/8/8).

Do not read much into it yet: mean amplitude **2.92** against an observed 1.30 and
P(amp<1) **0.02** against 0.37. The §6g mean-state bias is unchanged, so the phase sequence is
being driven by a field whose amplitude is ~2× too large. Noted as an observation, not a
result.

Bands kept at `/tank/projects/mjo_model_clim/bands_20261008.npz` — the tenth cycle in that set.

---

## Related

- [`CYCLE_DATA_INVENTORY.md`](CYCLE_DATA_INVENTORY.md) — stores, grids, and the pkl symlink
  convention.
- [`run_commands_20261001.md`](run_commands_20261001.md) — the chained-submitter failure that
  produced the verify-against-the-server rule this cycle qualifies.
- [`ts-mjo/MJO_METHOD.md`](ts-mjo/MJO_METHOD.md) — §5, the model climatology.

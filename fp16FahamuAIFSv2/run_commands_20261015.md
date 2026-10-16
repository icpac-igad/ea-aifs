# Run log — 20261015 (AIFS-ENS 2.0, tier B)

9/9 submitted, verified against the server. The cycle where **the official TS bounds finally
pointed the NWP forecast downward** — and where `ts_days.py` was run without its tercile
directory, which is survivable but worth not repeating.

---

## ⏱ End-to-end timing

| Step | What | Wall time | Output |
|------|------|-----------|--------|
| **0** | open-data pre-check | seconds | **RUNNABLE** — 9/9 sfc, 2/2 sol, 6/6 pl, 14/14 levels, 11/11 wave |
| **1** | pkl creation | 03:49 → 06:26, **2 h 37 m** (187.4 s/member) | 42 GB, **50/50** |
| **2** | tier-B rollout | 06:26 → 11:07, **4 h 41 m** | **50 written, 0 skipped, 0 failed** |
| **3a** | regrid 432–792 h → 1.5° | 11:07 → 11:15, 8.4 m | 2.1 GB, 50/50 |
| **3b** | quintiles | 11:15 → 11:17 | 6 arrays |
| **3c** | submit tas/mslp/pr | 11:19 → 12:29, **1 h 10 m** | **6/6, 0 failed** |
| **TS** | tracker on the N320 sidecar | ~6 m | 20 KB, 11.4 tracks/member |
| **MJO** | VPM → RMM-fitted basis, O96 corpus | ~4 m | 70 KB |
| **TS+MJO submit** | 3 files | 12:31 → 13:04, 33 m | **3/3** |
| **verify** | `check_submissions_paged.py` | seconds | **9/9 present** |

Stores: **O96 160 GB** (`WN1X6559XCY2MXERY3R0`) **+ N320 52 GB** (`76NRNDJXCH6EZJW32CQG`),
both tagged `cycle-20261015_0000`. Disk **104 GB free** after; no purge was needed this cycle,
the first time in several.

Step 1 was 45 min slower than the 1 h 50 m estimate — 187 s/member against 132 s on 20261008.
GCS throughput, not a fault. ECBox was the real drag: **~12 min per file**, which is why 3c
alone took 70 min.

**Step 1 ran straight through the ~06:00 `apt-daily-upgrade` window and survived** — fifth
consecutive clean rollout. The `sudo systemctl disable --now apt-daily-upgrade.timer
apt-daily.timer` fix is **still unapplied**; five cycles have been saved by timing, not by
protection.

---

## Trap: `ts_days.py` without `--aiwq-tercile-dir`

The TS compute was launched in parallel with the 3c upload and the tercile files had not been
fetched yet, so `ts_days.py` fell back to **the ensemble's own terciles** and said so:

```
!! no --aiwq-tercile-dir / --tercile-clim: using the ensemble's own terciles,
   so the probabilities are self-referential and NOT submission-grade.
```

This was harmless **because `submit_ts_mjo_cli.py` rebins from the stored counts** against
whatever `--tercile-dir` holds — the probabilities in `ts_days_probs_*.nc` are not what gets
submitted. But the two differ enough that reading the `ts_days.py` log as the forecast would
have been wrong:

| week | basin | ts_days' own | official | P(below) self → official |
|---|---|---|---|---|
| 2026-11-02 | NWP | 0 / 3 | **1 / 5** | 0.00 → **0.48** |
| 2026-11-02 | ATL | 0 / 1 | 0 / 4 | 0.00 → 0.00 |

**Fetch the terciles before running `ts_days.py`**, so its log and the submission agree:

```python
from AI_WQ_package import retrieve_evaluation_data as red
red.retrieve_20yr_quantile_clim('20261102', 'TS', os.environ['AIWQ_PASSWORD'],
                                local_destination=f'{BASE}/ts_terciles')
```

There is no `retrieve_20yr_TS_clim`; the TS terciles come from
`retrieve_20yr_quantile_clim(..., variable='TS', ...)`, which maps to
`/climatologies/2026/TS_20yrCLIM_WEEKLYTSDAYS_terciles_<validdate>.nc` over **FTP**
(`AIWQ_PASSWORD`, not the ecbox token).

---

## Commands (as run)

```bash
PY=/tank/projects/micromamba/envs/aifs-gpu/bin/python
BASE=/tank/projects/aifs-run/20261015_0000

mkdir -p $BASE/input_states
$PY -u s3_grib_pkl/s3_grib_pkl_input_aifsens_v2_proto.py \
    --date 20261015_0000 --members 1-50 --source gcs \
    --out $BASE/input_states --skip-existing

# rename -> Step 2 -> 3a -> 3b chained in /tank/projects/chain_20261015.sh -- COMPUTE ONLY
bash /tank/projects/chain_20261015.sh

# 3c, TS and MJO interactive
$PY -u ../shared/forecast_submission_cli.py --date 20261015 --v2 --output-dir $BASE/aiwq
$PY -u ts-mjo/ts_days.py --store $BASE/icechunk_n320_aiwq --tag cycle-20261015_0000 \
    --init 20261015 --out $BASE/ts_days_probs_20261015_tracked.nc
$PY -u ts-mjo/vpm_index.py --store $BASE/icechunk_o96 --init 20261015 \
    --clim /tank/projects/era5_vpm_clim/vpm_clim_1991_2020.npz \
    --eofs /tank/projects/era5_vpm_clim/rmm_basis.npz \
    --dump-bands /tank/projects/mjo_model_clim/bands_20261015.npz \
    --out $BASE/mjo_probs_20261015.nc
$PY -u submit_ts_mjo_cli.py --date 20261015 \
    --ts-nc $BASE/ts_days_probs_20261015_tracked.nc --tercile-dir $BASE/ts_terciles \
    --mjo-nc $BASE/mjo_probs_20261015.nc --dry-run   # then without --dry-run
$PY check_submissions_paged.py --date 20261015       # NOT AI_WQ_check_submission
```

**Compute is chained; submission is not.** See the 20261001 failure for why.

Validation before 3c: 6 arrays `(2, 5, 121, 240)`, NaN 0, range ⊂ [0,1], Σ over quintile
= 1.000000 exactly; weeks 2026-11-02 / 2026-11-09. Global mean P(Q5) for `2t` is **0.446** —
the same warm shift against the 20-year climatology that every cycle has shown.

The paged check saw **57 files across 1 page**, so this cycle's verdicts needed no pagination.
That is a property of checking early in the window, not a change in the tool — by submission
close the date will hold hundreds of files and the official single-page checker will again
report false ABSENTs for `tas`/`pr`/`mslp`.

---

## TS days — the first below-leaning NWP forecast

| week | ATL mean | NWP mean |
|---|---|---|
| 2026-11-02 | 1.10 d | 2.38 d |
| 2026-11-09 | 0.98 d | 2.64 d |

11.4 tracks per member. Official bounds:

| week | ATL | NWP |
|---|---|---|
| 2026-11-02 | **0** / 4 | **1** / 5 |
| 2026-11-09 | **0** / 1 | **0** / 3 |

Submitted:

| period | basin | below | near | above |
|---|---|---|---|---|
| 1 | ATL | **0.00** | **0.90** | 0.10 |
| | NWP | **0.48** | 0.28 | 0.24 |
| 2 | ATL | **0.00** | 0.60 | 0.40 |
| | NWP | 0.00 | 0.62 | 0.38 |

SWIO and SEIO are out of season (bounds `0/0`), filled with uniform 1/3 by
`submit_ts_mjo_cli.py`; active columns sum to 1.

**Week 1 NWP is the first below-leaning TS forecast this pipeline has submitted.** Its lower
bound is 1 day and the ensemble mean is 2.38, so 48% of members land under it. Read against
the four-cycle trend this is the detector bias unwinding, not reversing:

| cycle | NWP w1 mean | NWP w1 P(above) |
|---|---|---|
| 20260924 | 7.60 d | 0.58 |
| 20261001 | 7.08 d | 0.56 |
| 20261008 | 5.04 d | 0.46 |
| **20261015** | **2.38 d** | **0.24** |

Forecast NWP activity has fallen monotonically for four cycles while the bounds stayed
comparable, so the over-count that used to pin "above" no longer does. **This is not evidence
the detector is calibrated** — it is evidence that a biased detector applied to a quiet basin
produces low counts. The fix is still detector calibration.

**ATL week 1 at P(near) = 0.90 is a bounds artefact, not confidence.** With a published lower
bound of 0, "below" means `x < 0` and is unreachable. Third cycle running.

## MJO phases — stationary in 8, and still miscalibrated

Modal category is **8 at every one of the 34 days** (days 6–7 dip to 7), so the submitted
sequence is 8 → 8 → 8 → 8 at init +7/14/21/28. No progression, after 20261008's 7 → 8 → 8 → 1.

Mean amplitude **2.71** against an observed 1.30; P(amp < 1) **0.018** against 0.37; the
forecast's own per-field σ runs 1.34–1.65× the basis values. The §5 / §6g mean-state bias is
unchanged, so the phase concentration is a field with roughly twice the amplitude it should
have — **the confidence is not real**. Submitted on standing instruction, flagged in the
dry-run output.

Bands kept at `/tank/projects/mjo_model_clim/bands_20261015.npz` — the eleventh cycle in that
set, and the reason no store may be purged before its bands are extracted.

---

## Related

- [`run_commands_20261008.md`](run_commands_20261008.md) — the paginated-checker finding.
- [`run_commands_20261001.md`](run_commands_20261001.md) — why submission is never chained.
- [`ts-mjo/TS_STORM_DAYS.md`](ts-mjo/TS_STORM_DAYS.md) — detector over-count.
- [`ts-mjo/MJO_METHOD.md`](ts-mjo/MJO_METHOD.md) — §5, the model climatology blocker.

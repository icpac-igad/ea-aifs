# MJO and tropical-storm-days diagnostics from the AIFS-ENS Icechunk store

Builds the two AI Weather Quest **non-gridded** targets out of a per-cycle Icechunk store
(50 members, 120+ variables, native N320). Neither is currently submittable; both files
below say exactly why, and what would change it.

| Target | Document | Status |
|---|---|---|
| **Tropical storm days** (`TS`) | **[`TS_STORM_DAYS.md`](TS_STORM_DAYS.md)** | detector + tracker work and match AI-WQ's quantity; blocked on an unestablished per-basin amplitude discrepancy. **Free checkpoint 2026-09-14.** |
| **MJO phase** (`MJO`) | **[`MJO_PHASE.md`](MJO_PHASE.md)** | no OLR in the model, so no RMM. VPM (χ200) is the one open route — and it needs the **full 0–792 h corpus**, not the windowed store. |

Split out of a single README on 2026-08-30; the two targets share almost nothing but the
store readers, and the TS discussion had grown to eight sections.

## Files

| file | what it is | target |
|---|---|---|
| `store_io.py` | shared store readers — written-step detection, valid times, basin masks, day/7-day-window grouping | both |
| `grid_ops.py` | reduced-Gaussian operators — relative vorticity without regridding, radius queries on the flat `values` vector | both |
| `ts_days.py` | driver: storm-day counts → tercile probabilities | TS |
| `ts_tracks.py` | cyclone centre detection and track linking | TS |
| `test_tercile_binning.py` | conformance test against AI-WQ's own scorer | TS |
| `mjo_index.py` | band anomalies → (with an EOF basis) index and phases | MJO |

## Quick start

```bash
PY=/tank/projects/micromamba/envs/aifs-gpu/bin/python
STORE=/tank/projects/aifs-run/20260820_0000/icechunk_v2
TAG=cycle-20260820_0000

# tropical storm days -> tercile probabilities against the official IBTrACS terciles
$PY ts_days.py --store $STORE --tag $TAG --init 20260820 \
    --aiwq-tercile-dir /tank/projects/ibtracs/clim --out ts_days_probs.nc

# MJO: stops after step 5 without an EOF basis, and says why
$PY mjo_index.py --store $STORE --tag $TAG --init 20260820 --dump-bands mjo_bands.npz
```

A cycle covering days 18–33 yields 16 days → 2 full 7-day windows, which is what AI-WQ
scores for TS. **MJO needs all four lags (day 8/15/22/29) and therefore a 0–792 h store** —
see [`MJO_PHASE.md`](MJO_PHASE.md) §2.

## The one thing both targets share

Neither diagnostic is self-contained in a forecast. Both need observational reference data
that only the AI-WQ package or ERA5 can supply — tercile boundaries for TS, an EOF basis
and a 120-day low-frequency mean for MJO. Each file's own table lists what, from where, and
why the forecast cannot supply it.

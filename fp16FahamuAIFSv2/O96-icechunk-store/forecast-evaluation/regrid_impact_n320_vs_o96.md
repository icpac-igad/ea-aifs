# Does the O96 hop cause the poor `tas` score?

**No.** It contributes ~3 % of the `tas` miscalibration; the other ~97 % is already in the
native N320 forecast.

The question arose because both scored cycles show negative `tas` RPSS
([20260212](evaluation_20260212.md) −0.074, [20260514](evaluation_20260514.md) −0.167)
while `mslp` scores well, and both were produced through `N320 → O96 → 1.5°` rather than
the production `N320 → 1.5°`.

## The test

20260813 holds **both** an N320-derived and an O96-derived 1.5° product built from *one*
50-member rollout — its O96 store was transcoded from its N320 store, then both went
through the identical downstream chain. The regrid route is therefore the only difference,
which is what makes the comparison clean.

```bash
python compare_regrid_routes.py --date 20260813 \
  --n320-dir /tank/projects/aifs-run/20260813_0000/aiwq \
  --o96-dir  /tank/projects/aifs-run/20260813_0000/aiwq_via_o96 \
  --n320-nc  /tank/projects/aifs-run/20260813_0000/nc_1p5deg \
  --o96-nc   /tank/projects/aifs-run/20260813_0000/nc_1p5deg_via_o96
```

## Result

Mean probability per quintile, both weeks pooled; climatology = 0.200 everywhere:

| `tas` | Q1 | Q2 | Q3 | Q4 | Q5 | sharpness |
|---|---|---|---|---|---|---|
| **N320** | 0.118 | 0.123 | 0.145 | 0.175 | **0.438** | **0.5164** |
| **O96** | 0.122 | 0.122 | 0.143 | 0.173 | **0.439** | **0.5266** |
| diff | +0.004 | −0.001 | −0.002 | −0.002 | +0.000 | **+0.0102** |

| variable | Δ sharpness (O96 − N320) | gap from climatology | regrid's share |
|---|---|---|---|
| **tas** | **+0.0102** | 0.316 | **3.2 %** |
| mslp | +0.0012 | 0.251 | 0.5 % |
| pr | +0.0005 | 0.206 | 0.2 % |

### Mechanism

Ensemble spread of the weekly aggregate the quintiles are cut from:

| variable | N320 (wk1 / wk2) | O96 (wk1 / wk2) | retained |
|---|---|---|---|
| tas | 1.63 / 1.54 K | 1.60 / 1.52 K | **98.7 %** |
| mslp | 485 / 440 Pa | 484 / 439 Pa | 99.7 % |
| pr | 14.9 / 12.6 mm | 14.1 / 12.0 mm | 95.0 % |

The O96 hop costs ~1.3 % of `tas` ensemble spread, which is what sharpens the
distribution. Real, consistent in sign — and far too small to matter.

Note `pr` loses the most spread (5 %) yet shifts sharpness least (0.2 %), because its
distribution is already close to flat. Spread loss only bites where the forecast is
committing.

## The regrid itself is not in question

Established separately, so this is not a numerical-error issue:

* `N320 → O96` is **bit-identical** to `earthkit.regrid.interpolate` (`max|d| = 0.0`)
* the O96 store round-trips within f4 rounding of the source
* `N320 → 1.5°` reproduces the shipped production NetCDF **bit-for-bit**

## What is actually causing it

The **native N320** forecast puts 0.438 on the warmest `tas` quintile and 0.556 of its
mass in the outer two — before any O96 hop. 20260514 shows the same shape (Q5 = 0.356).
This is intrinsic to the model's ensemble and consistent with the ESS ≈ 1.8–1.9,
redundancy ≈ 0.96 result in `../../epistemic-reasoning-risk/`.

**The lever on `tas` RPSS is ensemble dispersion, not the grid.** Reverting to N320 storage
would recover ~3 % of the calibration gap at 12× the disk cost.

## Two things this does not settle

1. **This is calibration, not RPSS.** 20260813 verifies 2026-08-31 / 09-07; until then the
   direct O96-vs-N320 *RPSS* comparison cannot be run. Worth running once it verifies —
   `evaluate_cycle.sh` scores the O96 route, and the N320 route can be scored from the
   `icechunk_n320_aiwq` sidecar with `--source-grid n320`.
2. **The warm shift may be partly a trend artifact.** The AI-WQ climatology is a 20-year
   distribution, so warming alone pushes a well-calibrated modern forecast toward Q5.
   Separating trend from over-confidence needs observed quintile frequencies across
   several cycles.

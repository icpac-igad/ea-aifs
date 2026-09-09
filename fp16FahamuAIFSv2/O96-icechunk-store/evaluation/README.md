# Evaluation scripts

One-off analysis scripts kept as the evidence behind the decisions in `../README.md`.
**Paths are hardcoded to `/tank/projects/aifs-run/20260813_0000`** — these were written
to answer a question once, not to be a reusable library. The reusable pieces are
`../o96_grid.py`, `../transcode_n320_to_o96.py` and `../validate_o96_run.py`.

| script | question it answered | headline result |
|---|---|---|
| `ab_fidelity.py` | how far does `N320→O96→1.5°` drift from `N320→1.5°`, per field? | `msl` 3.5 % of field sd, `2t` 3.6 %, `tp` 38 % |
| `ab_spread.py` | is that drift small next to the ensemble spread it must not distort? | error/spread median 0.02 (`msl`), 0.17 (`2t`); spread retained 99.7 / 97.8 / 93.8 % |
| `cmp_submissions.py` | how much does the submitted quintile file move? | modal quintile changes at 4.3 % (`msl`), 13.4 % (`2t`), 19.0 % (`tp`) of gridpoints |
| `quintile_yardstick.py` | is that a lot? compares against a 50-member bootstrap | **every O96 number is inside the ensemble's own sampling noise** |
| `bench_regrid_forms.py` | batched vs per-field sparse regrid | per-field is ~9-10× faster, bit-identical |
| `list_regrid_matrices.py` | which archive grids are even reachable from N320? | **exactly one Gaussian target exists: O96** — N96 and 1.0° both dead-end |

`list_regrid_matrices.py` is the one script here with no hardcoded paths — it just reads
earthkit's matrix database, so it stays useful if earthkit ships new matrices later.

`quintile_yardstick.py` re-implements the weekly aggregation and quintile counting from
`shared/ensemble_quintile_analysis_cli.py`; it self-checks against the shipped
submission file and reproduced it to `max|d| = 0.0` before the comparison was trusted.

Run them with the project env:

```bash
PY=/tank/projects/micromamba/envs/aifs-gpu/bin/python
$PY quintile_yardstick.py
```

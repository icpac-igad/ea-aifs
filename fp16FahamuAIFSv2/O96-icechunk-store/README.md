# O96 Icechunk store

**Question.** Step 2 writes every field on the native N320 grid (542,080 cells). Even
storing only the downstream window (432–792 h, 61 of 132 steps) that is **583 GB per
cycle**; the full 0–792 h rollout would be **~1.26 TB**. Can we store a regridded (~1°)
corpus instead — and is the coarse store still representative of what N320 gives us?

**Answer, measured on the 20260813 cycle (all 50 members, end to end).** Yes, and the
mechanism should be an **inline regrid in the inference loop** — not the
write-N320-then-delete-the-block scheme. Details below.

Result for 20260813: **57 GB replacing 583 GB (10.2×)**, with the graded AI-WQ
submission reproduced **bit-identically**.

## What is in this folder

| file | what it is |
|---|---|
| `o96_grid.py` | the regrid — O96 cell coordinates, the earthkit matrix, and `regrid_block`. The one place the grid is defined. |
| `transcode_n320_to_o96.py` | offline: convert an existing N320 store to O96, and cut the native N320 sidecar (`--native --vars msl,tp,2t`) |
| `validate_o96_run.py` | acceptance check to run before deleting anything |
| `test_writer_paths.py` | regression test: proves the live pipeline's default path is unchanged |
| [`MANIFEST_COMPACTION.md`](MANIFEST_COMPACTION.md) | **why 57 GB of manifest indexes 792,000 chunks in 17 MB**, and how to reclaim it |
| `forecast-evaluation/` | score a completed cycle's RPSS against AI-WQ observations, locally, no upload |
| `evaluation/` | the analysis scripts behind every number quoted here |

Two files in the **parent** directory were modified rather than copied here, because
they are the live pipeline and must not be forked:

- `../run_local_icechunk_v2.py` — gained `--grid {n320,o96}`, `--native-store`,
  `--native-vars`. Defaults are unchanged.
- `../icechunk_output.py` — `IcechunkMemberWriter` gained `regrid`, `var_filter` and
  chunk-aligned slab buffering. All no-ops under the current production config.

### The changes are opt-in — nothing happens unless you ask for it

| you want | flags | result |
|---|---|---|
| **the original N320 output** (default) | *none* — or `--grid n320` | byte-for-byte what the pipeline always wrote |
| **N320 off, O96 only** (tier C) | `--grid o96` | one O96 store, ~97 GB for the full corpus, no N320 anywhere |
| **O96 + the AI-WQ sidecar** (tier B, adopted) | `--grid o96 --native-store PATH` | O96 corpus + 12 GB N320 store for `msl,tp,2t` |

`--grid` defaults to `n320`, so an existing command line behaves exactly as before. The
new code is only reachable through these flags: with `--grid n320` and no
`--native-store`, `regrid` and `var_filter` are both `None` and the writer takes the
same branch it always did.

That claim is testable rather than a promise — `test_writer_paths.py` replays real N320
fields through the writer and asserts the default path reproduces the source
byte-for-byte, with no chunk-write amplification:

```bash
PY=/tank/projects/micromamba/envs/aifs-gpu/bin/python
$PY O96-icechunk-store/test_writer_paths.py
# -> PASS — the live pipeline's default behaviour is unchanged
```

The runner also refuses to write one grid into a store that already holds the other,
and it checks **before** loading the model rather than failing mid-rollout:

```
ERROR: the --store store already holds a N320 grid (542080 values) but --grid o96
       writes 40320.
```

### The three grids, and what they are called

Two different families are in play, which is why the second hop matters:

| | family | GRIB `gridType` | earthkit spec | points | spacing |
|---|---|---|---|---|---|
| **N320** | classic **reduced Gaussian** | `reduced_gg` | `{"grid": "N320"}` | 542,080 | 320 latitude rows per hemisphere, rows unevenly spaced, points per row vary |
| **O96** | **octahedral** reduced Gaussian | `reduced_gg` | `{"grid": "O96"}` | 40,320 | 96 rows per hemisphere, row *i* has `16+4i` points |
| **1.5°** (AI-WQ) | **regular latitude/longitude** | `regular_ll` | `{"grid": [1.5, 1.5]}` | 29,040 | 121 × 240, evenly spaced — *not* Gaussian at all |

So the AI-WQ target has no N-or-O style name; it is just a regular lat/lon grid given by
its increment. `N` = classic reduced Gaussian, `O` = octahedral reduced Gaussian, and
`F` (e.g. `F320`) would be a *full* Gaussian grid with every row the same length.

Both Gaussian grids are quasi-equal-area on an unstructured `values` axis, which is why
O96 substitutes for N320 without touching the store schema or any reader. A regular
lat/lon archive would not: it over-weights high latitudes in an unweighted cell mean,
and `earthkit-regrid` has no `1.0° → 1.5°` matrix to reach the AI-WQ grid from it.

---

## 1. Why O96, and not 1.0° regular lat/lon or N96

`O96` is an octahedral reduced Gaussian grid, 40,320 cells (~112 km), the same *kind*
of grid as N320 — quasi-equal-area cells on an unstructured `values` axis. Three
things follow, and all three break on a 1.0° regular lat/lon grid:

| | O96 | 1.0° regular |
|---|---|---|
| `earthkit-regrid` second hop to the AI-WQ 1.5° grid | **matrix exists** | **no matrix** — needs a hand-rolled interpolator with different numerics |
| Unweighted cell mean over a polygon (what `s2s_bn_evidence_prep.py` does) | unbiased — cells are equal-area | latitude-biased — needs cos(lat) weights added everywhere |
| Store schema / reader changes | none — still `(member, time, values)` with `latitude`/`longitude` coord arrays | new 2-D grid, every reader changes |
| Size, 120 vars × 50 members × 132 steps | **~102 GB** | ~164 GB |

O96 is also 1.6× smaller than 1.0° for a comparable effective resolution.

### Why not N96, or any other Gaussian grid — the real constraint is the FIRST hop

Enumerating all 2,518 matrices in earthkit-regrid's database settles it. **From N320
there are exactly 25 targets: 24 regular lat/lon grids, and exactly one Gaussian grid —
`O96`.** The complete list of `reduced_gg → reduced_gg` matrices in the whole database
is only ten pairs:

```
N1280 -> N320     N256 -> O1280    N256 -> O400     N320 -> O96      O1280 -> N320
O1280 -> O96      O2560 -> O1280   O2560 -> O320    O2560 -> O48     O2560 -> O96
```

So `N96` is *not* the problem people expect. `N96` is well supported in general — 48
matrices involve it, and **`N96 → 1.5°` exists** just as `O96 → 1.5°` does. The blocker
is that **`N320 → N96` does not exist**, and neither does any route to it
(`O96 → N96` is absent too). N96 is simply unreachable from the model grid.

That also corrects the shape of the argument for the 1.0° option: there the blocker
genuinely *is* the second hop (`N320 → 1.0°` exists, `1.0° → 1.5°` does not). Two
different failures, one conclusion.

And even if `N320 → N96` existed, O96 would still win — a "96" in each name does not mean
the same point count, and the octahedral grid is the *smaller* of the two:

| grid | family | points | full 120-var 50-member 132-step store |
|---|---|---|---|
| N320 | classic reduced Gaussian | 542,080 | ~1259 GB |
| **O96** | **octahedral** | **40,320** | **~94 GB** |
| N96 | classic reduced Gaussian | 50,662 | ~118 GB |

Same 96 latitude rows per hemisphere, but the classic reduction rule keeps 26 % more
points than the octahedral `16 + 4i` rule. O96 is the smaller, the reachable, and the
grid AIFS-ENS was pre-trained on.

Building our own `N320 → N96` matrix was not considered: the whole point of using
earthkit's matrix is that the archive is *bit-identical* to the interpolation the
production 1.5° path already performs. A hand-rolled operator throws that away.

### O96 coordinates

The store needs real `latitude`/`longitude` arrays. Octahedral O96 is 192 Gaussian
latitude rows; row `i` counted from the nearest pole carries `16 + 4i` equally spaced
longitudes (`2·Σ(16+4i) = 40320`). Derived analytically in
`transcode_n320_to_o96.py:o96_coords()` and validated against
`interpolate(n320_latitudes, N320, O96)` to < 0.006° — i.e. the point ordering matches
earthkit's exactly.

---

## 2. Storage

The O96 row is **measured** — the whole 20260813 cycle (120 vars, 50/50 members) was
transcoded and came to 45 GB. The 132-step column scales that; the 1.0° row uses the
same 0.733 compression that predicts 582 GB against the 583 GB the N320 store occupies.

| grid | 61 steps (432–792 h) | 132 steps (0–792 h) |
|---|---|---|
| N320 (today) | **583 GB** (measured) | ~1259 GB |
| 1.0° regular | ~76 GB | ~164 GB |
| **O96** | **45 GB** (measured) | **~97 GB** |

The full 0–792 h corpus on O96 costs **less than a sixth** of what today's *partial*
N320 store costs. Cheap enough to keep many cycles and to stop discarding hours 0–432.

---

## 3. Is the O96 corpus representative? (the A/B, run for real)

Two paths compared on the 20260813 cycle, all 50 members, all the way to the submitted
file:

- **A (production):** N320 → 1.5° → weekly aggregate → quintile probabilities
- **C (proposed):** N320 → **O96** → 1.5° → weekly aggregate → quintile probabilities

Path C was produced by patching only `_regrid_n320()` in a copy of
`shared/aifs_n320_grib_1p5defg_nc_cli.py`; everything downstream is the untouched
production code. The quintile reproduction used for the yardstick below matches the
shipped file to `max|d| = 0.0`.

### Field level, 1.5° (vs the ensemble spread it must not distort)

| var | regrid RMSE ÷ ensemble spread (median) | ensemble-mean shift | ensemble spread retained |
|---|---|---|---|
| `msl` | 0.02 | 1.7 Pa (spread 749 Pa) | 99.7 % |
| `2t` | 0.17 | 0.039 K (spread 1.6 K) | 97.8 % |
| `tp` | ~0.95 (p95) | 2.7e-5 m | 93.8 % |

Precipitation is where the extra hop bites: the O96 hop damps the *precipitation
ensemble spread by ~5–6 %*. `msl` and `2t` are essentially untouched.

The interpolation weights are non-negative, so no new negatives appear in
positive-definite fields (`tp`, `cp`, `sf`, `tcw` all checked — clean).

### Product level, and the yardstick that matters

Change in the submitted quintile probabilities, against a **bootstrap resample of the
same 50 members** — i.e. how much the product moves just from the ensemble's own
Monte-Carlo sampling:

| var / week | O96 archive: MAE, modal-quintile flips | 50-member bootstrap: MAE, flips |
|---|---|---|
| `msl` w1 | 0.0096, **4.4 %** | 0.035–0.042, 19–26 % |
| `msl` w2 | 0.0091, **4.3 %** | 0.033–0.041, 19–21 % |
| `2t` w1 | 0.028, **12.5 %** | 0.032–0.042, 15–19 % |
| `2t` w2 | 0.029, **14.3 %** | 0.029–0.037, 16–20 % |
| `tp` w1 | 0.033, **18.1 %** | 0.034–0.040, 20–22 % |
| `tp` w2 | 0.032, **19.9 %** | 0.037–0.047, 23–29 % |

**Every O96 number is inside the ensemble's own sampling noise.** Switching the archive
to O96 moves the submitted forecast less than re-drawing the 50 members does. (One
caveat: `2t` probability *RMSE* is a little higher than bootstrap, 0.067 vs 0.042–0.060,
because O96 error is spatially concentrated in complex terrain rather than diffuse —
MAE and flip rate are both below bootstrap.)

So an O96-only archive is defensible even for the graded product. It is not *free*,
which is why the tiers in §5 exist.

---

## 4. The mechanism: regrid inline, do not write-then-delete

The write-N320-block → regrid → delete-block → next-block scheme is the wrong tool:

- **Icechunk cannot cheaply free it.** Chunks are copy-on-write and are only removed by
  `expire_snapshots()` + `garbage_collect()`, and only once *no live snapshot* references
  them. In a rolling scheme the tip snapshot still references the block, so freeing it
  means overwriting the region with NaN, committing, expiring **every** older snapshot —
  destroying the per-step ACID history that `--commit-every 1` exists to provide — and
  then a full-manifest GC scan. Fragile, and it runs on every cycle.
- **It is not needed.** The regrid is a sparse matmul, and the whole point is that the
  N320 state is already in memory when the rollout yields it.

`earthkit.regrid.interpolate` costs ~9 ms per field, and most of that is per-call
overhead (grid-spec lookup, validation) rather than arithmetic. Fetch the matrix once
from `earthkit.regrid.db.find` and apply it directly:

```python
from earthkit.regrid.db import find as db_find
mat, _ = db_find({"grid": "N320"}, {"grid": "O96"}, method="linear")   # CSR 40320x542080
out = np.stack([mat @ f.astype("float64") for f in block])   # (120, 542080) -> (120, 40320)
```

**Apply it one field at a time.** The obvious batching — `(mat @ block.T).T` for a whole
step — is **~9-10x slower**, not faster: the matrix is extremely sparse (116,696 nnz in
40,320 x 542,080) and SciPy's sparse-times-dense-matrix kernel handles that far worse
than 120 sparse mat-vecs. This cost a wasted GPU run to discover; the first
implementation batched, and the GPU sat at **0 % utilisation** while a single core did
the regrid.

Measured on a real 120-field step, all **bit-identical** (`max|d| = 0.0`):

| | per field | 120-field step |
|---|---|---|
| `earthkit.interpolate` | 9.1 ms | ~1090 ms |
| batched `(mat @ block.T).T` | ~34 ms | ~4100 ms |
| **per-field loop** | **0.21 ms** | **25 ms** |

25 ms × 61 stored steps = **~1.5 s per member** of actual regrid. The end-to-end cost of
tier B is larger — **+17 s/member**, §7 — because the sidecar adds a second set of chunk
writes and a second commit per step. Either way N320 never touches the disk, and peak
memory is one step (120 × 542,080 × f4 ≈ 260 MB), not a 72 h block. Looping also avoids
materialising a 520 MB float64 copy per step.

**Chunking.** At O96 a 1-step chunk is only 161 KB, which is small; `--commit-every 12`
(72 h) gives ~1.9 MB chunks instead. This is where the "72 hours" instinct belongs: as
the *commit/chunk cadence*, not as a delete cycle. The transcoder defaults to 12; the
live runner stays at `--commit-every 1` to keep one ACID snapshot per model step, and
measurement showed chunk length barely moves the total size (185.2 MB at `time_chunk=12`
vs 185.4 MB at `time_chunk=1` for the same 12 steps).

**Gotcha, learned the hard way.** Writes must be aligned to the time chunk. Writing
step by step into a 12-step chunk makes zarr read-modify-write it 12 times, and because
Icechunk is copy-on-write every touch leaves a chunk object behind — a one-member test
that way cost **4.9 GB instead of ~1.1 GB**. Buffer a whole chunk and assign the slab
once (`transcode_n320_to_o96.py` does this).

---

## 5. Recommended tiers

The only thing O96 genuinely costs is **spatial-unit granularity in the epistemic /
BN diagnostics**. An N320 cell is ~941 km²; an O96 cell is ~12,650 km². The current
`s2s_bn_evidence_prep.py --min-cells 20` threshold (≈18,800 km²) is only **1.5 O96
cells** — a unit would need 253,000 km² to keep 20 cells. Sub-basin and admin-1 units
get coarse.

Only 14 variables feed those consumers: `2t msl tp ro swvl1 swvl2 q_700 t_500 t_700
t_850 v_850 w_500 w_700 z_500`.

| tier | what to keep at N320 | O96 corpus | total/cycle | fidelity |
|---|---|---|---|---|
| **A — safest** | 14 epistemic vars, 432–792 h (~57 GB) | 120 vars, 0–792 h (~97 GB) | **~154 GB** | everything currently computed is bit-identical |
| **B — ADOPTED** | 3 AI-WQ vars only, 432–792 h (**12 GB**, measured) | 120 vars, 0–792 h (~97 GB) | **~109 GB** | submission bit-identical; BN diagnostics move to O96 units |
| **C — leanest** | none | 120 vars, 0–792 h | **~97 GB** | submission moves within ensemble sampling noise (§3) |

Built for 20260813 at the current 432–792 h window: **45 GB O96 + 12 GB sidecar = 57 GB,
against the 583 GB N320 store — a 10.2× reduction with the graded product unchanged.**

Tier B buys a bit-identical graded submission for 12 GB — 2 % of today's store — so
there is no reason to take the §3 risk on the thing that is actually scored.

If tier B or C is adopted, rescale `--min-cells` (20 N320 cells → 2 O96 cells for the
same area) and expect small units to be dropped or noisy.

---

## 6. Converting the stores that already exist

`transcode_n320_to_o96.py` rewrites an existing N320 store onto O96, verifying against
earthkit as it goes. 20260813 is **done** (50/50 members, tagged, 45 GB); 20260806 is
still 632 GB of N320 and would transcode to ~45 GB the same way.

```bash
PY=/tank/projects/micromamba/envs/aifs-gpu/bin/python
BASE=/tank/projects/aifs-run/20260813_0000

# the O96 corpus (all 120 variables)
$PY transcode_n320_to_o96.py \
    --src $BASE/icechunk_v2 --dst $BASE/icechunk_o96 \
    --src-tag cycle-20260813_0000 --date 20260813_0000 --verify

# the tier-B native sidecar
$PY transcode_n320_to_o96.py \
    --src $BASE/icechunk_v2 --dst $BASE/icechunk_n320_aiwq \
    --native --vars msl,tp,2t \
    --src-tag cycle-20260813_0000 --date 20260813_0000

# ALWAYS before deleting the N320 store
$PY validate_o96_run.py --o96 $BASE/icechunk_o96 \
    --sidecar $BASE/icechunk_n320_aiwq --tag cycle-20260813_0000 --members 1-50
```

Read-bound on the source, ~2 min/member (~1.5–2 h for 50). A tag is written only when
all 50 members are present — Icechunk tags are immutable and tombstoned, so a tag on a
partial ensemble would permanently point at NaNs. Confirm `validate_o96_run.py` prints
`PASS` **before** releasing the N320 store with `../cleanup_aifs_run.py`.

### Acceptance, 20260813 (the check that licenses deleting the N320 store)

Both stores hold 50/50 members and carry the `cycle-20260813_0000` tag.

| check | result |
|---|---|
| inline verify, every transcoded member | 48/48 `max\|batched − earthkit\| = 0.0`, no mismatches/skips |
| all 50 members finite at first and last stored step, both stores | pass |
| random member/step/variable spot checks, O96 vs `interpolate(N320, O96)` | within f4 rounding |
| random spot checks, sidecar vs N320 source | bit-identical |
| O96 store via `xr.open_zarr` + the `s2s_bn_evidence_prep` access pattern | 1014 cells over the ICPAC box, all finite |
| **sidecar → 3a → all 50 × 1.5° NetCDFs vs shipped `nc_1p5deg/`** | **50/50 bit-identical** |
| **sidecar → 3a → 3b → `ensemble_quintile_probabilities_20260813_v2.nc`** | **bit-identical, `max\|d\| = 0.0` on all three variables** |

The graded submission is exactly reproducible from the 12 GB sidecar, so
`icechunk_v2` for 20260813 can be released.

---

## 7. Live inference (from the next cycle on)

`../run_local_icechunk_v2.py` gained `--grid` and `--native-store`. Tier B in one run —
the O96 corpus and the N320 sidecar are written side by side as the rollout streams,
and N320 never reaches disk. **Run these from the parent directory**
(`fp16FahamuAIFSv2/`), unchanged from the existing runbook apart from the three new
flags:

```bash
setsid nohup bash -c "cd $PWD; export HF_HOME=$HF_HOME; \
  $PY -u run_local_icechunk_v2.py --date 20260820_0000 --members 1-50 --lead-time 792 \
    --input-dir $BASE/input_states \
    --grid o96 --store $BASE/icechunk_o96 \
    --native-store $BASE/icechunk_n320_aiwq --native-vars msl,tp,2t \
    --n-members 50 --commit-every 1 --float-size f4 --skip-existing" \
  > /tank/projects/run_0820_all.log 2>&1 </dev/null &
```

Then point 3a at the **sidecar**, not the O96 store, so the submission path is unchanged:

```bash
$PY ../shared/aifs_n320_grib_1p5defg_nc_cli.py --date 20260820_0000 --members 1-50 --v2 \
    --no-upload --source icechunk --icechunk-store $BASE/icechunk_n320_aiwq \
    --icechunk-tag cycle-20260820_0000 --output-dir $BASE/nc_1p5deg
```

**Do not pass `--write-hours` on an O96 store — always store the full 0–792 h corpus.**
The 432–792 h window exists only because N320 made the full rollout unaffordable
(583 GB for 46 % of the steps). At O96 the whole corpus is ~97 GB, so there is no reason
to throw away hours 0–432 and no way to get them back short of re-running inference.
The narrow window is an N320-era compromise; it should not be carried over.

Defaults are unchanged: with no `--grid`/`--native-store` the runner writes exactly the
N320 store it always did (regression-tested field by field).

`--commit-every 1` stays the safe cadence. The writer now buffers a whole time chunk
before writing, so a larger `--commit-every` (e.g. 12 = 72 h) no longer causes the
copy-on-write blow-up described in §4 — but at O96 it saves little, and 1 keeps the
per-step ACID snapshots.

### Live GPU validation (member 1, 20260813 IC, production settings)

| run | wall time | GPU util |
|---|---|---|
| N320 baseline (the 50-member 20260813 cycle) | ~285 s/member | — |
| tier B, **batched** regrid (first attempt) | 434 s | **0 %** — starved by the regrid |
| tier B, per-field regrid (shipped) | **302 s** | 100 % |

**+17 s/member, ~6 %** over the N320 baseline — a 50-member cycle goes from ~4 h 00 m to
about 4 h 12 m, while the store drops from 583 GB to 57 GB.

Checks on the live run (`--grid o96 --native-store … --write-hours 432-792`):

- 61/132 steps stored, 61 commits; step 10 reads NaN, steps 71 and 131 finite — the
  write window is exactly right
- **O96 store == `interpolate(sidecar, N320, O96)` within f4 rounding** for `msl`, `tp`,
  `2t` at steps 71/72/90/131. Both stores come from the *same* rollout, so this is the
  check that actually proves the in-flight regrid; it does not depend on the model
  being reproducible.
- and it is not reproducible: re-running member 1 from the same pkl gives a different
  realisation (`2t` rms 2.8 K against a field sd of 14.2 K). **aifs-ens-2.0 is
  stochastic** — never expect a re-run to match an archived member bit for bit.

Reproduce that smoke test on any cycle before committing all 50 members:

```bash
# ~5 min on the RTX 5000 Ada
$PY -u run_local_icechunk_v2.py --date <DATE>_0000 --members 1 --lead-time 792 \
    --input-dir $BASE/input_states \
    --grid o96 --store $BASE/icechunk_o96_smoke \
    --native-store $BASE/icechunk_n320_smoke --native-vars msl,tp,2t \
    --n-members 50 --commit-every 1 --float-size f4 --no-tag

$PY O96-icechunk-store/validate_o96_run.py \
    --o96 $BASE/icechunk_o96_smoke --sidecar $BASE/icechunk_n320_smoke --members 1
```

If it prints `PASS` and `nvidia-smi` showed the GPU near 100 % during the run, go.

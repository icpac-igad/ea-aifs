# Path A — Direct Icechunk-on-GCS Output for AIFS-ENS-2.0

**Goal:** the GPU inference writes each ensemble member **straight into a transactional
Icechunk store on GCS during the rollout** — no local GRIB staging, no separate upload
step, no `/scratch` churn. This document specifies Path A in detail, then the way
forward for **parallel writers** (multiple members at once, incl. Cloud Run GPU jobs
writing separate members), and how the **downstream regridding + AI Weather Quest
(AI-WQ) routine** adapt.

> Scope note: Path A only. The GRIB path (`GribFileOutput`) stays available as a
> fallback during migration but is not the target end-state.

---

## 0. Verified groundwork (this was tested, not assumed)

| Thing | Result |
|---|---|
| `icechunk` / `zarr` in the v2 venv | `icechunk==2.0.6`, `zarr==3.2.1` (installed alongside the v2 stack) |
| anemoi native Zarr writer | `anemoi.inference.outputs.zarr.ZarrOutput` (0.8.3) writes `time`, `latitude`, `longitude` (native N320 cells) + one `(time, values)` array per variable into **any** Zarr `StoreLike` |
| GCS-backed Icechunk store | `icechunk.gcs_storage(bucket=..., prefix=..., service_account_file="coiled-data.json")` authenticates with the **existing** service account |
| End-to-end POC (real GCS) | Created a repo on `gs://aifs-aiquest-us-20251127/_icechunk_poc/…`, wrote a `(member,time,values)` ensemble with **one commit per member**, read it back with `xr.open_zarr`, inspected `ancestry` history, deleted the POC objects |
| Distributed-write API | `Session.fork()` → picklable `ForkSession`; `Session.merge(...)` / `icechunk.distributed.merge_sessions(...)`; single `commit()`; `rebase` + `BasicConflictSolver` as fallback |

---

## 1. Store layout (the schema)

One Icechunk **repository per forecast cycle**, e.g.
`gs://aifs-aiquest-us-20251127/20260625_0000/icechunk_v2`.

```
repo (branch: main)
├── coords
│   ├── latitude    (values)                 f4   # native N320 cell lat  (542080)
│   ├── longitude   (values)                 f4   # native N320 cell lon
│   ├── time        (time)                   i4   # seconds since reference_date
│   └── member      (member)                 i4   # 1..50
└── data variables  (one per model output field, ~120)
    ├── 2t          (member, time, values)   f4
    ├── msl         (member, time, values)   f4
    ├── tp          (member, time, values)   f4
    ├── swh, mwp, cdww, cos_mwd, sin_mwd …    f4   # ALL wave fields kept (see §1.3)
    └── z_500, t_850, u_250, …  (the 83 pressure-level fields)
```

- **`member`** — new axis (anemoi's stock `ZarrOutput` has no ensemble axis; we add it, see §2).
- **`time`** — `LEAD_TIME // timestep` steps (960/6 = 160; +1 if you also write step 0).
  Encoded exactly like `ZarrOutput`: `i4` seconds since `reference_date`, CF attrs.
- **`values`** — the **native N320 reduced-Gaussian cells (542080)**. No regridding at
  write time → lossless and compact. `latitude`/`longitude` 1-D coord arrays let any
  reader place the cells; regridding to 1.5° happens **downstream** (§4).

### 1.1 Chunking — the single most important choice
```
data vars:  chunks = (1, time, values)      # one member × all steps × all cells per chunk
coords:     latitude/longitude chunks=(values,), time chunks=(time,)
```
With **member-chunk size 1**, *every member owns whole, disjoint chunks*. No two members
ever touch the same chunk. This is what makes parallel writes (§3) conflict-free.

> ⚠️ **Time-axis caveat (measured — see [`ICECHUNK_COMMIT_CADENCE.md`](ICECHUNK_COMMIT_CADENCE.md)).**
> Putting the whole time axis in one chunk (`time` above) is only safe if you **buffer the
> member and write the chunk in one shot**. Writing it **step-by-step** (as the runner does)
> read-modify-writes the full chunk every step → **~11× amplification at 24 steps, tens-of-×
> at the real 160 steps**. If you want to commit every 72 h / 6 h, chunk the time axis to the
> commit window (`(1, 12, values)` / `(1, 1, values)`); if you keep the big chunk, buffer and
> write once. Commit *frequency* itself is cheap — the **chunk shape** is what governs write cost.

### 1.2 dtype & compression
- `float32` (`f4`) is the default and the **only safe blanket choice**.
- > ⚠️ **`float16` (`f2`) is NOT safe as a blanket dtype — measured.** The earlier advice
  > here ("`f2` halves the store with no real precision loss") was wrong. `float16` maxes
  > out at **65,504**, and **12 of the 120 model output fields exceed it**, so they would
  > silently become `±inf`:
  >
  > | field | max\|value\| |
  > |---|---|
  > | `ssrd` | 19,958,648 |
  > | `strd` | 10,039,377 |
  > | `z_10` … `z_400` | 315,762 … 75,402 |
  > | **`msl`** | **104,063** |
  > | `sp` | 103,924 |
  >
  > `msl` is one of the three fields the downstream regrid extracts (`tp`/`msl`/`2t`), so
  > this would corrupt the AI-WQ chain. Even *in-range* fields lose accuracy: `2t ≈ 318 K`
  > quantises to ~0.25 K steps in `f16` (the model runs FP16 on *normalised* values, not
  > raw Kelvin — that is why inference is fine but storage is not).
  >
  > If you need to halve the store, use **per-variable** dtypes (`f4` for the 12 above,
  > `f2` elsewhere) or CF-style `int16` + `scale_factor`/`add_offset` packing — not a
  > blanket `f2`. Better: cut *steps*, not precision (see §1.4).
- Add a Zarr v3 compressor (e.g. `zstd` level 3) in `create_array(..., compressors=…)`.
  GRIB chunks are ~1.6 GB each; compressed `f4` Zarr is comparable.

### 1.4 Store only the steps the downstream actually reads
`shared/aifs_n320_grib_1p5defg_nc_cli.py` reads exactly the **432–792 h** windows
(days 18–33; its `time_ranges` are `432-504, 504-576, 576-648, 648-720, 720-792`).
The rollout is autoregressive so every step must be **computed**, but steps outside that
window need not be **stored** — with `(1, 1, values)` chunks a skipped step allocates no
chunks and reads back as `NaN`.

Measured (member 001, `f4`, per-step commits): storing all 160 steps of a 960 h run =
**30 GB/member**; storing only 432–792 h of a 792 h run = **61/132 steps = 12 GB/member**.
Across 50 members that is **~600 GB instead of ~1.5 TB**. Use
`run_local_icechunk_v2.py --lead-time 792 --write-hours 432-792`. (Cloud Run's `app.py`
already uses `LEAD_TIME = 792`; the v2 default of 960 h overshoots downstream by 168 h.)

### 1.3 Keep the wave-direction fields (a data **gain** over GRIB)
The GRIB path was **forced to drop** `cos_mwd`/`sin_mwd` (`drop_unencodable_wave_dir`,
the `ConceptNoMatchError` fix) because they are not GRIB params. Icechunk/Zarr has **no
such limit** — store all ~120 fields, including the circular wave-direction components.
**For Path A, do *not* call `drop_unencodable_wave_dir`.** Recover direction downstream
as `degrees(atan2(sin_mwd, cos_mwd))` if needed.

---

## 2. Path A implementation (sequential — present setup)

The current runner already drives a GRIB output object by hand inside
`for state in runner.run(...)` (see `fp16_multi_run_AIFS_ENS_v2.run_ensemble_member`).
We swap that object for an Icechunk ensemble writer. Inference stays **one member at a
time** (no conflicts possible), but each member's chunks go **straight to GCS** and are
sealed with a per-member commit.

### 2.1 New writer (drop-in, ~50 lines)
```python
# fp16FahamuAIFSv2/icechunk_output.py
import numpy as np, zarr, icechunk

def open_repo(bucket, prefix, service_account_file):
    storage = icechunk.gcs_storage(bucket=bucket, prefix=prefix,
                                   service_account_file=service_account_file)
    return icechunk.Repository.open_or_create(storage)

def init_schema(repo, n_members, n_steps, n_values, var_names,
                latitudes, longitudes, ref_date, timestep_s, float_size="f4"):
    """One-time, committed BEFORE any member is written (and before any fork)."""
    s = repo.writable_session("main")
    root = zarr.group(s.store, overwrite=True)
    root.create_array("latitude",  shape=(n_values,),  chunks=(n_values,), dtype="f4",
                      dimension_names=("values",))[:] = latitudes
    root.create_array("longitude", shape=(n_values,),  chunks=(n_values,), dtype="f4",
                      dimension_names=("values",))[:] = longitudes
    t = root.create_array("time", shape=(n_steps,), chunks=(n_steps,), dtype="i4",
                          dimension_names=("time",))
    t[:] = (np.arange(1, n_steps + 1) * timestep_s).astype("i4")
    t.attrs["units"] = f"seconds since {ref_date}"; t.attrs["calendar"] = "gregorian"
    root.create_array("member", shape=(n_members,), chunks=(n_members,), dtype="i4",
                      dimension_names=("member",))[:] = np.arange(1, n_members + 1)
    for v in var_names:
        root.create_array(v, shape=(n_members, n_steps, n_values),
                          chunks=(1, n_steps, n_values), dtype=float_size,
                          dimension_names=("member", "time", "values"))
    s.commit(f"init schema: {n_members}×{n_steps}×{n_values}, {len(var_names)} vars")

class IcechunkMemberWriter:
    """Writes ONE member's rollout into the (member,time,values) arrays, then commits."""
    def __init__(self, repo, member_index, branch="main"):
        self.session = repo.writable_session(branch)
        self.root = zarr.open_group(self.session.store, mode="r+")
        self.m = member_index
        self.n = 0
    def write_step(self, state):                       # called per yielded state
        for name, value in state["fields"].items():    # keep ALL fields (no wave drop)
            self.root[name][self.m, self.n, :] = value
        self.n += 1
    def commit(self, member_number):
        return self.session.commit(f"member {member_number:03d}")
```

### 2.2 Runner change (replaces the GribFileOutput block)
```python
writer = IcechunkMemberWriter(repo, member_index=member - 1)
for state in runner.run(input_state=input_state, lead_time=LEAD_TIME):
    writer.write_step(state)        # straight to GCS; no GRIB, no drop_unencodable_wave_dir
writer.commit(member)               # one ACID snapshot per member
```

### 2.3 Orchestrator change
`fp16_automate_aifs_gpu_pipeline_v2.py` loses the **download / upload / cleanup** thirds
of `process_single_member`. The per-member loop collapses to *download pkl → infer (writes
to GCS) → commit*. (Input pkls are still pulled from `…/input_v2/`; only the **output**
side changes.) Add a `--output {grib,icechunk}` switch so the GRIB path remains a fallback.

### 2.4 Reproducibility
After all members commit, **tag the cycle**: `repo.create_tag("cycle-20260625_0000", snapshot_id=...)`.
The AI-WQ submission can then pin the exact immutable snapshot it scored.

---

## 3. Way forward — parallel writers to the **same** store

**Key Icechunk fact:** a branch's commits are *linear*. Two sessions **cannot** both
commit to `main` concurrently — the second must `rebase`. So "parallel members" is not
"N independent commits"; it is one of the two patterns below. Both are safe **because of
the `(1, time, values)` chunking** (members never share a chunk).

### 3.A Present: sequential inference, single writer
One process, one member at a time, `writable_session → write → commit`. Zero conflicts.
Nothing more needed. (This is §2.)

### 3.B Several members in parallel **inside one process/box** — cooperative fork/merge
Use when one GPU box (or one multi-GPU box) runs a few members concurrently.

```
init_schema(...) ; commit                     # metadata exists & is committed FIRST
session = repo.writable_session("main")
fork = session.fork()                          # picklable handle shared by all workers
                                               # each worker writes its own member region:
def write_member(fork, m_idx, member_state_stream):
    root = zarr.open_group(fork.store, mode="r+")
    for n, state in enumerate(member_state_stream):
        for name, val in state["fields"].items():
            root[name][m_idx, n, :] = val
    return fork                                # return the fork (carries the diff)

forks = [write_member(session.fork(), m, stream_m) for m in batch]   # threads/procs
session.merge(*forks)                          # combine disjoint regions
session.commit(f"members {batch[0]+1}-{batch[-1]+1}")   # ONE snapshot for the batch
```
Disjoint member regions + per-member chunks ⇒ the merge has **no conflicts**.

### 3.C Cloud Run GPU — separate jobs, each writing **separate members**
This is the target for horizontal scale: N Cloud Run GPU jobs, job *k* handles a member
subset, all writing into the **one** cycle store. Two viable designs:

**Design C1 — distributed fork/merge (recommended, single final snapshot).**
```
Coordinator (Cloud Run job / Workflows step):
  repo = open_repo(...)
  init_schema(...); commit                        # once
  session = repo.writable_session("main")
  for k, member_subset in enumerate(partitions):  # e.g. 50 members / 10 jobs = 5 each
      fork = session.fork()
      gcs_put(f".../forks/in/{k}.pkl", pickle(fork), member_subset)   # hand off

Worker job k (GPU):
  fork, members = unpickle(gcs_get(f".../forks/in/{k}.pkl"))
  load model once; for m in members: roll out, write fork.store[m_idx]
  gcs_put(f".../forks/out/{k}.pkl", pickle(fork))                     # return diff

Coordinator (after all workers finish):
  forks = [unpickle(gcs_get(f".../forks/out/{k}.pkl")) for k in range(N)]
  merge_sessions(session, *forks)                 # icechunk.distributed.merge_sessions
  sid = session.commit("all 50 members, cycle 20260625_0000")
  repo.create_tag("cycle-20260625_0000", snapshot_id=sid)
```
- `ForkSession` is explicitly picklable for exactly this scatter/gather.
- Workers never commit; they only accumulate chunk writes in their fork. Nothing is
  visible until the coordinator's single `merge + commit` → **atomic publication** of the
  whole ensemble. A worker that dies is simply **re-run** (its region was never published).
- Conflict-free by construction (disjoint members ⇒ disjoint chunks). No rebase needed.

**Design C2 — branch-per-member + sequential merge (no central coordinator during compute).**
```
Worker k for member m:
  s = repo.writable_session("main")              # branched from latest main
  ... write member m region ...
  try: s.commit(f"member {m:03d}")
  except ConflictError: s.rebase(icechunk.ConflictDetector()); s.commit(...)  # retry
```
Because members write disjoint chunks, a `rebase` against another member's commit
**always succeeds** (no overlapping chunks). Simpler to operate (each job is self-
contained) but commits are serialized, so you get N snapshots instead of one and a small
rebase-retry cost under contention. Use C2 if a central post-merge coordinator is
awkward in your Cloud Run topology; otherwise prefer **C1**.

**Operational guardrails (either design):**
- **Schema is created once, before any worker writes**, and committed — workers only ever
  write chunk *data*, never metadata ⇒ no metadata conflicts.
- Keep member-chunk size = 1. If you ever re-chunk along `member`, parallel safety is lost.
- Idempotent retries: re-running a worker re-writes the same disjoint region; safe.
- Set Icechunk storage retry/timeout settings (`StorageRetriesSettings`,
  `StorageTimeoutSettings`) for flaky Cloud Run networking.

---

## 4. Downstream — regridding + AI-WQ routine

Current chain (GRIB-based):
```
…/fp16_v2_forecasts/*.grib
  → aifs_n320_grib_1p5defg_nc_cli.py   (cfgrib read → ekr.interpolate N320→1.5° → per-member NetCDF)  → …/fp16_1p5deg_nc/
  → ensemble_quintile_analysis_cli.py  (weekly means, 5-bin quintile prob vs 20-yr clim)
  → forecast_submission_cli.py / aiwq_individual_files_cli.py  (AI-WQ {var}_{date}_p{week}_{team}_{model}.nc → submit)
```
Variable map (already in the regrid CLI): `pr→tp`, `tas→2t`, `mslp→msl`.

### 4.1 Regrid step reads the Icechunk store instead of GRIB

> ✅ **IMPLEMENTED** — `shared/aifs_n320_grib_1p5defg_nc_cli.py --source icechunk
> --icechunk-store PATH [--icechunk-tag TAG]`. Output is structurally identical to the
> GRIB path (validated: dims, coords, dtypes and the sparse `(time, step)` NaN block
> pattern all match), so Steps 3b–3d are unchanged. `--source grib` remains the default.
> Two gotchas found while wiring it: `earthkit-regrid` has **no batch mode** for N320
> (a leading dim raises `matmul: dimension mismatch`), so fields are regridded one step at
> a time; and a GRIB window `h432-504` holds hours **438..504**, not 432..504 — the store
> index is `hour/6 - 1` (confirmed by an offset sweep).

The regrid math is unchanged — only the **source** changes from "download GRIB + cfgrib"
to "open the cycle store and select the cells". The store already holds native N320
`values`, so `earthkit-regrid` runs on them directly:
```python
src = icechunk.Repository.open(gcs_storage(...)).readonly_session("main")   # or a tag
ds  = xr.open_zarr(src.store, consolidated=False, zarr_format=3)            # member,time,values
for m in members:
    for short in ("tp", "2t", "msl"):
        cells = ds[short].isel(member=m).values                  # (time, 542080) native N320
        ll = np.stack([ekr.interpolate(cells[t], {"grid": "N320"}, {"grid": [1.5, 1.5]})
                       for t in range(cells.shape[0])])           # (time, 121, 240) @1.5°
        ...
```
This removes the per-file GRIB download and the cfgrib dependency from Step 3. The
careful earthkit-regrid **SQLite cache isolation per worker** in the current CLI stays
exactly as is (each parallel slot keeps its own matrix cache).

### 4.2 Where the 1.5° output lands — two options
- **Option R1 (minimal change):** keep writing per-member NetCDF to `…/fp16_v2_1p5deg_nc/`
  exactly as today. The quintile + submission CLIs are then **untouched**. Recommended for
  the first cut — smallest downstream delta.
- **Option R2 (Icechunk all the way):** write a **second, derived** Icechunk store
  `…/icechunk_v2_1p5deg` with dims `(member, time, latitude, longitude)` on the regular
  1.5° grid, using the same fork/merge pattern (§3) since members are independent. The
  quintile step then reads this regular-grid store directly (it already works on
  `latitude`/`longitude` dims and an ensemble axis). Cleaner lineage and one place for
  reproducible tags; do this once R1 is proven.

### 4.3 Quintile analysis
Operates on the 1.5° regular grid (`latitude`, `longitude`) and an ensemble axis,
computing 5-bin probabilities against the 20-yr quintile climatology for the AI-WQ week
windows (the CLI uses `day 4 + 7·k` valid windows, i.e. the sub-seasonal weeks). Under:
- **R1** — unchanged (reads the per-member NetCDFs).
- **R2** — point it at the 1.5° store: `xr.open_zarr(store).sel(time=week_window).mean("time")`
  then the existing vectorized `calculate_grid_quintiles`. The ensemble axis is just the
  store's `member` dim → no per-file concat needed.

### 4.4 AI-WQ submission
`forecast_submission_cli.py` / `aiwq_individual_files_cli.py` build
`{variable}_{date}_p{week}_{teamname}_{modelname}.nc` and submit. **No change** beyond
pointing at the new model name (`AIWQ_MODEL_NAME_V2`) and, optionally, recording the
Icechunk **snapshot tag** used so a submission is exactly reproducible from object storage.

### 4.5 End-to-end lineage (Path A, target)
```
ECMWF Open Data ─► input_v2/*.pkl (CPU ETL, unchanged)
        │
        ▼  GPU inference (Cloud Run GPU jobs, members in parallel — §3.C)
   icechunk_v2  (member,time,values @ native N320, ALL ~120 fields)   ◄─ tag: cycle-<date>
        │
        ▼  regrid N320→1.5°  (ekr.interpolate; R1→NetCDF  |  R2→icechunk_v2_1p5deg)
   1.5° ensemble (member,time,lat,lon; tp/2t/msl)
        │
        ▼  weekly means + quintile prob vs 20-yr climatology
   AI-WQ quintile fields
        │
        ▼  {var}_{date}_p{week}_{team}_{model}.nc  ─►  AI Weather Quest submission
```

---

## 5. Migration plan (phased, GRIB stays a fallback)

1. **P1 — writer + sequential path.** Add `icechunk_output.py` + `--output icechunk` to the
   runner/orchestrator. Smoke-test 1 member, 72h → verify a committed store on GCS and an
   `xr.open_zarr` round-trip. (GRIB remains the default until P3.)
2. **P2 — regrid from store (R1).** Teach `aifs_n320_grib_1p5defg_nc_cli.py` a
   `--source icechunk` mode that reads the store and regrids to the **existing** NetCDF
   output. Quintile + submission untouched. Validate AI-WQ numbers match the GRIB path.
3. **P3 — parallel writers.** Implement §3.C Design C1 (fork → scatter to Cloud Run GPU →
   merge → single commit + tag). Flip inference default to Icechunk.
4. **P4 — Icechunk downstream (R2, optional).** Derived 1.5° store; quintiles read it
   directly. Retire the GRIB + per-file-NetCDF intermediates.

## 6. Dependencies / infra / auth
- GPU image: add `icechunk` (`>=2.0`) and `zarr>=3` (pulls `numcodecs`). **No `gcsfs`** —
  Icechunk has a native Rust GCS backend.
- Auth: existing `coiled-data.json` service account (verified writing to the bucket).
- Bucket unchanged (`aifs-aiquest-us-20251127`); new prefixes `…/icechunk_v2[ _1p5deg]`.
- Cloud Run: ensure jobs can read/write the bucket and exchange the small `ForkSession`
  pickles (a `…/forks/` prefix or the job payload).

## 7. Risks & decisions to confirm
- **`float16` vs `float32`** for the stored fields (size vs. exactness). Default `f4`;
  `f2` recommended for bulk.
- **R1 vs R2** for the 1.5° stage (NetCDF first, Icechunk later — recommended).
- **C1 vs C2** for Cloud Run parallelism (single-snapshot fork/merge vs branch-per-member
  + rebase). C1 preferred.
- **Garbage collection / expiry:** Icechunk keeps history; schedule `expire_snapshots` +
  `garbage_collect` (or tag-and-prune) so old cycles don't grow object storage unbounded.
- anemoi `ZarrOutput.close()` calls `consolidate_metadata` — **not used here**; Icechunk
  has its own manifest and we publish via `session.commit()`.

---

### Appendix — verified API quick-reference
```python
import icechunk, zarr, xarray as xr
storage = icechunk.gcs_storage(bucket="aifs-aiquest-us-20251127",
                               prefix="20260625_0000/icechunk_v2",
                               service_account_file="coiled-data.json")
repo = icechunk.Repository.open_or_create(storage)
s    = repo.writable_session("main"); store = s.store; s.commit("msg")
fork = s.fork()                                  # picklable -> Cloud Run workers
s.merge(fork_a, fork_b, ...)                     # or icechunk.distributed.merge_sessions(s, *forks)
s.rebase(icechunk.ConflictDetector())            # C2 retry path
ro   = repo.readonly_session("main")             # or .readonly_session(tag="cycle-…")
ds   = xr.open_zarr(ro.store, consolidated=False, zarr_format=3)
list(repo.ancestry(branch="main"))               # commit history
repo.create_tag("cycle-20260625_0000", snapshot_id=sid)
```

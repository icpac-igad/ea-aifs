# Icechunk commit cadence — per-member vs per-72h vs per-6h (evaluated & scoped)

**Question (scoped before running):** the present Path A design makes **one Icechunk
commit per member** after its rollout (see `ICECHUNK_PATH_A.md` §2.2/§2.4; the bring-up
transcript it came from is kept locally and is no longer in the repo). Can the store instead
be committed **every 72 h**, or **every 6 h** (each model step) if wanted? Is that a good
idea, and what has to change?

**Short answer.** Yes — committing every 72 h or 6 h is feasible and cheap **on its own**.
But commit *frequency* is the easy half. The half that actually matters is the **time-axis
chunk shape**, and the currently-specified chunking (`(1, n_steps, values)` — the whole time
axis in one chunk, `ICECHUNK_PATH_A.md` §1.1) is **the wrong shape for a step-by-step writer**
and causes large write amplification regardless of commit frequency. Fixing the chunking is
what unlocks cheap fine-grained commits.

---

## 1. The finding: chunk shape governs write cost, not commit count

The rollout writes **one timestep at a time** (`root[var][m, n, :] = value` for
`n = 0..n_steps-1`, per `IcechunkMemberWriter.write_step` / anemoi's `ZarrOutput`). A Zarr
**chunk is the atomic I/O unit** — writing a single-step sub-region of a chunk that spans
many steps forces a **full read-modify-write of the entire chunk on every step**. So a chunk
holding all `n_steps` is rewritten `n_steps` times per member.

Measured on this box (local Icechunk store, `icechunk 2.1.0`, N320 width `V=542080`,
`S=24` steps, `NV=3` vars, `f4`, incompressible random data → 156 MB logical; readback
verified on the last step in every case):

| Config | time-chunk | commit every | #commits | store on disk | objects | wall | **store / logical** |
|---|---|---|---|---|---|---|---|
| **P1** per-member, 1 big chunk *(current design)* | 24 steps | 24 (once) | 1 | **1804 MB** | 84 | 16.0 s | **11.6×** |
| **P2** per-72h, chunk aligned | 12 steps | 12 | 2 | 938 MB | 90 | 7.7 s | 6.0× |
| **P3** per-6h, chunk aligned | 1 step | 1 | 24 | **144 MB** | 222 | 1.3 s | **0.92×** |
| **P4** per-72h, chunk **mis**aligned | 24 steps | 12 | 2 | 1804 MB | 90 | 14.3 s | 11.6× |

Read this carefully:

- **Amplification tracks the time-chunk span, ≈ `time_chunk / 2`** (24→11.6×, 12→6.0×,
  1→~1×). It is **independent of how often you commit**: P1 (1 commit) and P4 (2 commits)
  produce the *identical* 1804 MB because both use the big all-time chunk.
- **Finer time-chunks are smaller *and* faster** (P3: 0.92×, 1.3 s vs P1: 11.6×, 16 s) —
  the opposite of the intuition that "more commits = more overhead."
- The current `(1, n_steps, values)` design (P1) already pays **11.6×** at just 24 steps.

### At production scale this is severe
The real rollout is **`n_steps = 160`** (960 h / 6 h). A `(1, 160, values)` chunk is
`160 × 542080 × 4 ≈ 347 MB` **per variable**, rewritten 160 times → tens-of-× amplification
per member. Per member the logical store is ~`160 × 542080 × ~120 vars × 4 B ≈ 42 GB` (`f4`)
or ~21 GB (`f2`); at ~tens-of-× that is **hundreds of GB–TB of writes per member**. On the
**GCS** target that is real egress/ingest cost and wall-clock, not just transient local disk
(which Icechunk GC would later reclaim, but the *I/O* is already spent).

---

## 2. The two clean designs (and how commit cadence falls out)

To write step-by-step **without** amplification, the time-chunk must be **small** (≤ the
amount you write between flushes). Two ways to get there:

### Design S — small time-chunks, stream + commit at cadence
Chunk the time axis to the commit window and write/commit per window:

| Want | time-chunk | write pattern | commit | amplification | objects/var/member |
|---|---|---|---|---|---|
| **commit every 6 h** | `(1, 1, V)` | write each step, commit each step | per step (160×) | ~1.0× | 160 |
| **commit every 72 h** | `(1, 12, V)` | write 12 steps, commit the window | per 72 h (13×) | ~6× *(or ~1× if you buffer — below)* | 13 |

### Design S+ — buffer the window in RAM, write the whole chunk once (best for 72 h)
For 72 h commits, **buffer one 72 h window (12 steps) in RAM**, then write the whole
`(1, 12, V)` chunk in a *single* assignment and commit. One write per chunk → **~1×
amplification**, durable every 72 h, only ~13 chunks/var/member. Buffer cost is trivial:
`12 × 542080 × ~120 × 4 ≈ 3.1 GB` RAM (this box has 220 GB). **This mirrors the current GRIB
cadence exactly** (one 72 h GRIB file ↔ one 72 h Icechunk commit) and is the recommended
target if you want intra-member durability.

### Design B — keep big chunks, but buffer the *whole member* and write once
If you specifically want the big `(1, n_steps, V)` chunk (fewest objects, best bulk-read),
you must **not** write it step-wise. Buffer the member's full array (~21–42 GB RAM) and write
each variable's chunk in one shot, then a single per-member commit. ~1× amplification, but
**no intra-member commits** — you only have data at member end. This is the amplification-free
version of the *current* per-member design.

> **Commit frequency itself is cheap** — extra commits add only small snapshot/manifest
> metadata (P1 vs P4 identical chunk bytes). Its real costs are (a) a longer snapshot history
> to garbage-collect (`expire_snapshots` + `garbage_collect`, already flagged in
> `ICECHUNK_PATH_A.md` §7) and (b) each commit is a durability/visibility point for readers.

---

## 2a. Decision: per-step (6 h) commits — IMPLEMENTED

The **72 h boundary was only a GRIB-file artifact** (chunk the rollout into 72 h files,
close each to flush memory). Icechunk needs no such thing — every `write_step` already
streams to the disk-backed session store, so RAM stays flat at any cadence. So the chosen
cadence is **one commit per model step = 6 h**, the natural inference granularity.

- **6 h is the model's native step** (confirmed from the actual run: GRIB output came out at
  steps `[6, 12, … 960]`). aifs-ens-2.0 yields one state every 6 h, so **6 h is the finest
  possible cadence — 3 h is not produced by the model** and cannot be committed.
- Per-step is also the **cheapest** config (chunk `(1,1,values)`, ~0.92×, no buffering).

Implemented in `icechunk_output.py` + `run_local_icechunk_v2.py`:
`--commit-every 1` (default) → `time_chunk=1`, commit every 6 h. **Smoke-tested** (member 001,
72 h): schema `50×12×542080`, 120 vars, **12 commits (one per 6 h step)**, chunks
`(1,1,542080)`, tag `cycle-…`, xarray readback correct (2t mean 289.7 K), store **2.3 GB vs
3.1 GB logical (0.74×) — no amplification**. Unwritten members read back as **NaN** (fill),
not 0. `--commit-every 2` → 12 h, `12` → 72 h (all with aligned `time_chunk`).

### Validated at full scale (member 001, **960 h / 160 steps**, per-step 6 h commits)

| | |
|---|---|
| Commits | **160** (one per 6 h step) + schema + init = **162 on `main`** |
| Wall clock | **494.9 s (8.25 min)** |
| Store | **30 GB** vs **41.6 GB logical** (`f4`) → **0.72× — no amplification** |
| Schema | `(50, 160, 542080)`, 120 vars, chunks `(1, 1, 542080)`, tag `cycle-20260702_0000` |
| Correctness | all 160 steps finite; h960 `2t` mean **289.2 K** |

**Cost of per-step commits:** the same 960 h member via the GRIB path took **394 s**, so 160
commits add **~101 s (~0.63 s/commit, +25 %)** — a modest price for a durable snapshot every
6 h and no upload leg. `--float-size f2` roughly halves the 30 GB (the model already runs FP16).

### All 50 members live in ONE store
The schema's `member` axis is sized by `--n-members` (default 50) and each member writes its
own slice, owning **disjoint chunks** (member-chunk size 1) — so members never conflict and
`--members 1-50` fills a single store. Verified: members 1/5/50 written into one
`(50, …)` store, schema inited once, unwritten members read **NaN**. Sequential inference ⇒
linear commits ⇒ zero conflicts (`ICECHUNK_PATH_A.md` §3.A).

`--skip-existing` makes a 50-member run **idempotent/resumable**: a member whose final step is
already present is skipped *before* the pkl load and inference, so an interrupted run picks up
where it stopped. Verified against the real 960 h store (member 1 → present, members 2/50 →
absent).

## 3. Recommendation

1. **Do not write the `(1, n_steps, values)` chunk step-by-step.** That is the one
   combination to avoid (it is what §1.1 currently specifies + how the writer writes).
2. **If you want per-72 h commits (recommended, matches the GRIB cadence):** Design **S+** —
   chunk `(1, 12, values)`, buffer each 72 h window, write the window chunk once, commit.
   ~1× amplification, durable every 72 h, ~13 chunks/var/member.
3. **If you want per-6 h commits:** Design **S** — chunk `(1, 1, values)`, write+commit each
   step. ~1× amplification, finest durability, most objects (160/var/member).
4. **If you want the current per-member commit only:** Design **B** — keep big chunks but
   buffer the whole member and write once (not step-wise).
5. Whatever the cadence, schedule Icechunk **GC** (`expire_snapshots` + `garbage_collect`)
   so history/objects don't grow unbounded.

Downstream is unaffected: `xr.open_zarr(session.store)` reads **any** chunking transparently,
so the N320→1.5° regrid and AI-WQ steps (`ICECHUNK_PATH_A.md` §4) don't care which cadence
was used — this is purely a write-path decision. (Larger time-chunks read marginally faster
for the "all steps of tp/msl/2t" access pattern, which is another reason to prefer the 72 h
window over per-6 h if reads dominate.)

---

## 4. Implementation (done — local writer)

Implemented in `icechunk_output.py` + `run_local_icechunk_v2.py`:

- **`init_schema(..., time_chunk=1)`** — data-var chunks are `(1, time_chunk, n_values)`
  (aligned to the commit cadence), with `fill_value=NaN` so unwritten members/steps read as
  missing rather than 0.
- **`IcechunkMemberWriter(..., commit_every=1)`** — commits every `commit_every` model steps
  and reopens a fresh session; `finalize()` flushes any tail. At `commit_every=1` /
  `time_chunk=1` each 6 h step is one whole chunk written once → no amplification.
- **`run_local_icechunk_v2.py --commit-every 1`** (default = 6 h; `2` = 12 h, `12` = 72 h).
  The runner loop is otherwise unchanged; the GRIB path stays the untouched fallback.

The Path A **GCS** twin (`open_repo_gcs`) takes the same `time_chunk`/`commit_every` — only the
storage backend differs. For `commit_every > 1` *without* re-chunking you'd hit intra-window
amplification; the aligned `time_chunk` avoids it (or buffer the window per Design S+).

*Evidence: `/tank/projects` probe, 2026-07-06, `icechunk 2.1.0` / `zarr 3.2.1`, RTX 5000 Ada
box. Numbers reproduce with the 4-config probe in this folder's scratch.*

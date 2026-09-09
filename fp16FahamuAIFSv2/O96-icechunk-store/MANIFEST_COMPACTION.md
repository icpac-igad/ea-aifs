# Why 120 variables × 132 global timesteps index in 17 MB

A 20260212 O96 store is 166 GB on disk, of which **57 GB was manifest** — 34 % of the
archive spent on bookkeeping. Consolidating it produces **124 manifests totalling
17.2 MB**, a 3,315× reduction, with byte-identical data.

Two questions follow, and they have different answers: *how can the index be so small?*
and *why was it ever so large?*

## 1. A manifest is an index, not data

The manifest does not contain field values. It maps a chunk's logical position to where
its bytes live:

```
(array, member, time)  ->  (object id, offset, length)
```

The store holds **120 arrays × 50 members × 132 steps = 792,000 chunks**, and the
consolidated manifests are 17.2 MB. That is:

```
17.2 MB / 792,000 = 21.7 bytes per chunk reference
```

22 bytes to record where a ~132 KB chunk of O96 field data lives. The data itself is
untouched at **105 GB in `chunks/`**. Nothing was compressed away — the index was simply
being stored ~3,300 times over.

Global coverage is irrelevant to the index size. O96 is 40,320 cells, but a whole
`(member, time)` slice of one variable is **one chunk**, so the grid contributes one
reference regardless of how many cells it holds. Index size scales with the *number of
chunks*, never with the number of gridpoints inside them.

## 2. Why it was 57 GB: manifest rewriting is O(n²) in commits

The runner uses `--commit-every 1`, giving one ACID commit per model step:

```
50 members × 132 steps = 6,600 commits (+2)  ->  6,602 commits
```

**Every commit re-serialises the complete chunk index for each array it touches.** Commit
*i* writes a manifest listing all *i* chunks known so far, not just the one it added. So
the manifest files number

```
6,602 commits × 120 arrays = 792,240  (measured: 792,004)
```

and their sizes sum quadratically:

```
Σ(i=1..6602) i × 22 B  ≈  479 MB per array
479 MB × 120 arrays    ≈  57.5 GB        (measured: 57.1 GB)
```

Measured manifest sizes match the picture — median 70 KB, max 145 KB, i.e. the last
commits each carry a near-complete index of ~6,600 references.

The consolidated form writes each array's index **once**:

```
6,600 chunks × 22 B = 145 KB per array × 120 = 17.4 MB   (measured: 17.2 MB)
```

Both predictions land within 1 % of measurement, so this is the mechanism, not a guess.

## 3. What this costs and what it buys

`--commit-every 1` is not a mistake — it buys a durable ACID snapshot per model step, so
an interrupted 5-hour rollout resumes at step granularity. That is worth real money
*during* inference. Once a cycle is complete and tagged, the per-step history has done its
job, and the 57 GB is pure carry cost.

So: **keep `--commit-every 1` for the run, compact afterwards.**

```python
repo.rewrite_manifests("consolidate manifests for archival", branch="main")
```

Takes ~1 second. It only *adds* — 124 new manifests, ~17 MB — and cannot remove the old
ones, because the cycle tag still references the snapshot that points at them.

## 4. Reclaiming the space is irreversible

To actually drop the 57 GB you must expire the old snapshots and garbage-collect. That
carries two permanent consequences:

* **The tag name is tombstoned.** Icechunk tags are immutable *and* deleted names can
  never be reused. `cycle-20260212_0000` cannot be recreated on this repo, so the
  compacted snapshot needs a new name and every reader that opens the old tag must be
  updated.
* **The per-step history goes.** 6,602 snapshots collapse to one. No data is lost — every
  chunk survives — but you can no longer open the store as it stood at step 47.

For a completed archive that will not be re-run, both are acceptable. For a cycle still
being written, neither is.

## 5. Result

| | before | after |
|---|---|---|
| manifests | 792,004 files, 57.1 GB | **124 files, 17.2 MB** |
| chunks | 792,002 files, 105 GB | unchanged |
| total objects | 1,603,813 | ~792,130 |
| store size | 166 GB | ~105 GB |
| data | — | **byte-identical** (5/5 sha256 fingerprints, 50/50 members, 132/132 steps) |

Halving the object count matters more than the 57 GB for the source.coop upload, which is
request-bound rather than bandwidth-bound at ~1.6 M objects.

## 6. Reproducing

```bash
PY=/tank/projects/micromamba/envs/aifs-gpu/bin/python
$PY -c "
import icechunk
r = icechunk.Repository.open(icechunk.local_filesystem_storage('<store>'))
print(r.rewrite_manifests('consolidate manifests for archival', branch='main'))"
```

Then verify fingerprints against the old tag *before* expiring anything — the compaction
is free to undo up to that point, and not afterwards.

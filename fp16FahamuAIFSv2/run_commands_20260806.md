# AIFS ENS v2.0 Run Commands — 20260806

**Date:** 20260806 (Thursday) · **Members:** 1–50 · **Box:** local RTX 5000 Ada, 30 GB.
**Outcome: ✅ submitted** — all 6 files verified on `ftp.ecmwf.int/forecast_submissions/20260806/`
(`{mslp,pr,tas} × {p1,p2}`, `Fahamu / fp16FahamuAIFSv2`).

Same fully-local pipeline as [`run_commands_20260730.md`](run_commands_20260730.md)
(no GCS: local pkls → local Icechunk store → local downstream), with **one change that
matters**: Step 1 was run with the new
[`s3_grib_pkl/s3_grib_pkl_input_aifsens_v2_proto.py`](../s3_grib_pkl/s3_grib_pkl_input_aifsens_v2_proto.py)
against the **Google mirror** instead of `ecmwf_opendata_pkl_input_aifsens_v2.py` against AWS.
**Step 1 went 14 h 30 m → 2 h 55 m (~5×).** See "Step 1 rewrite" below.

---

## ⏱ End-to-end timing

| Step | What | Wall time | Output |
|------|------|-----------|--------|
| **1** | pkl creation, 50 members (**proto, GCS**) | **2 h 55 m** (175.1 min) → 209 s/member | 46 GB |
| **2** | GPU inference → Icechunk | **3 h 43 m** → ~285 s/member | 582 GB |
| **3a** | regrid 432–792 h → 1.5° NetCDF | **8 m 48 s** | 2.0 GB |
| **3b** | quintiles → submission NetCDF | **~2 m** (clean, one pass) | 1.7 GB (`aiwq/`) |
| **3c** | **live submit** | 6/6 (1 retry) | 6 files on FTP |

For comparison, 20260730 on AWS: Step 1 **14 h 30 m**, Step 2 3 h 55 m, 3a 8 m 44 s.
Everything except Step 1 is unchanged — the Step 1 saving is the whole story of this cycle.

---

## Commands (as run)

```bash
export MAMBA_ROOT_PREFIX=/tank/projects/micromamba
PY=/tank/projects/micromamba/envs/aifs-gpu/bin/python
export HF_HOME=/tank/projects/hf_cache
BASE=/tank/projects/aifs-run/20260806_0000
cd /home/ezra/ea-aifs/fp16FahamuAIFSv2

# --- Step 1: input pkls, local, via the GCS mirror (NEW - see below) ---
# uv resolves the PEP-723 deps itself; no aifs-etl env needed.
cd /home/ezra/ea-aifs/s3_grib_pkl
setsid nohup bash -c "cd $PWD; export PYTHONUNBUFFERED=1; \
  uv run --script s3_grib_pkl_input_aifsens_v2_proto.py \
    --date 20260806_0000 --members 1-50 --source gcs --skip-existing \
    --out $BASE/input_states" \
  > /tank/projects/proto_gcs_20260806_all50.log 2>&1 </dev/null &

# --- Step 2: GPU inference -> local Icechunk store (setsid), unchanged ---
cd /home/ezra/ea-aifs/fp16FahamuAIFSv2
setsid nohup bash -c "cd $PWD; export HF_HOME=$HF_HOME; \
  $PY -u run_local_icechunk_v2.py --date 20260806_0000 --members 1-50 --lead-time 792 \
    --input-dir $BASE/input_states --store $BASE/icechunk_v2 \
    --n-members 50 --commit-every 1 --float-size f4 --write-hours 432-792 --skip-existing" \
  > /tank/projects/run_0806_all.log 2>&1 </dev/null &

# --- Step 3a: regrid from the Icechunk store ---
$PY ../shared/aifs_n320_grib_1p5defg_nc_cli.py --date 20260806_0000 --members 1-50 --v2 \
    --no-upload --source icechunk --icechunk-store $BASE/icechunk_v2 \
    --icechunk-tag cycle-20260806_0000 --output-dir $BASE/nc_1p5deg

# --- Step 3b: quintiles ---
$PY ../shared/ensemble_quintile_analysis_cli.py --date 20260806 --v2 \
    --local-nc-dir $BASE/nc_1p5deg --work-dir $BASE/aiwq

# --- Step 3c: submit (window 20260806 -> 20260809 inclusive; submitted on 08-08) ---
$PY ../shared/forecast_submission_cli.py --date 20260806 --v2 --output-dir $BASE/aiwq --dry-run
$PY ../shared/forecast_submission_cli.py --date 20260806 --v2 --output-dir $BASE/aiwq   # live
# one target failed with the known FTP status-line bug -> single-target re-run:
$PY ../shared/forecast_submission_cli.py --date 20260806 --v2 --output-dir $BASE/aiwq \
    --variables mslp --weeks 2
```

Console transcript of 3b/3c: [`fp16FahamuAIFSv2_20260806.txt`](fp16FahamuAIFSv2_20260806.txt).

---

## Step 1 rewrite: why 14.5 h → 2.9 h

**The old path.** `ecmwf_opendata_pkl_input_aifsens_v2.py` goes through
`earthkit.data.from_source("ecmwf-open-data", source="aws")`. Underneath, `ecmwf-opendata`
reads the `.index` sidecar, then **merges every matched field's byte range into ONE combined
`Range: bytes=a-b,c-d,e-f…` header** and issues ~9 giant requests per member, serialised
across the 5 param groups (concurrency is capped at the number of *dates*, i.e. 2). When AWS
returns `503 SlowDown` or resets, the **whole combined request** is lost and retried after a
hardcoded 120 s backoff. That is the entire 14.5 h.

**The new path.** The prototype reads the same `.index`, then fetches **one single byte range
per field**, 16 at a time in a thread pool. A throttled field costs only itself.

**Why `--source google` never worked through the old path** (and why that was a red herring):
GCS rejects multi-range GETs with `400 InvalidArgument: Multiple ranges are not supported`,
while S3 accepts them. So the mirror looked broken when it was the *batching* that was
incompatible. Ask GCS for **one range at a time** and it answers normally (HTTP 206) — and
much faster from this box. The bandwidth note in `run_commands_20260723.md` (AWS 1–3 Mbps vs
Google 11–27 Mbps) was right about the mirrors but was never validated on a real GRIB pull.

Measured on this box, same field, same bytes:

| | AWS | GCS |
|---|---|---|
| single field (`gh_700`, 505 kB) | 153.2 s | **1.7 s** |
| full member 1 | 255.3 s | **141.6 s** |
| full member 3 | 1586.4 s (26.4 min) | **130.2 s** (2.2 min) |
| 50 members | (14.5 h on 20260730) | **2.92 h** |

**Correctness.** The prototype's pkls are **bit-identical** to the production
`earthkit-data` pkls — verified field-by-field (112/112, `max_abs_diff=0`, NaN masks matched)
on 5 members across 2 dates (20260625 m1–m2, 20260806 m1–m3), for both mirrors. GCS and AWS
serve byte-identical `.index` and message bytes, so the offsets are interchangeable.

**Decoder split.** `gribberish` handles enfo/oper (PDT 1); the **wave** group is PDT **104**
with a bitmap and `gribberish` rejects it (`Only HorizontalAnalysisForecast templates are
supported`, still true in 1.6.0, the latest on PyPI), so wave falls back to `eccodes`.
Two things learned the hard way:

- `eccodes.codes_get_values()` returns the raw missing-value sentinel **9999** at land points
  unless you `codes_set(gid, "missingValue", np.nan)` **first**. Without it all 12 wave fields
  had `9999` over ~30 % of the grid where production had NaN — silently wrong, no error.
  Fixed in `6c47e96`.
- `eccodes` is **faster** than `gribberish` here (9.4 vs 15.1 ms/msg) and bitwise-identical on
  the same messages, so the dual-decoder split buys nothing but complexity. Decode is
  irrelevant anyway: **~10 ms of decode against 4 000–16 000 ms of fetch**; regrid (85 ms) costs
  ~9× more CPU than decode. A whole member is ~21 s of CPU — everything else is network wait.

**Not yet done:** the production `ecmwf_opendata_pkl_input_aifsens_v2.py` is **unchanged** and
still AWS-only. Step 1 for this cycle was run from the standalone prototype. Folding
single-range + GCS into the production script (or promoting the prototype) is the open item.

---

## Incidents

1. **Step 1 member spread.** Even on GCS the per-member time ranged **116.9 s → 2725.6 s**
   (the 45-minute outlier was one member hitting a bad patch). Mean 209 s. Still 50/50, no
   manual intervention.

2. **Step 2 resumed cleanly.** 3 members (1, 2, 3) were already in the store from an earlier
   partial run, so `--skip-existing` wrote 47 and skipped 3 — `0 failed`. The tag
   `cycle-20260806_0000` was applied only after all 50 were present (50/50), avoiding the
   20260709 tag trap (icechunk tombstones deleted tag names permanently).

3. **Step 3c `mslp_week2` failed with `200 TYPE is now 8-bit binary`** — a normal FTP status
   line mis-caught as an error while creating a new file. **This is the second cycle in a row**
   (20260723 hit it on `tas_week1`), so it is an intermittent bug in the submission wrapper's
   error handling, not a one-off. A single-target re-run succeeded; **6/6 verified server-side**
   by listing the FTP directory (the file's timestamp differs from the other five, confirming
   the retry is what landed it).

4. **Log filename collision.** Step 2 was logged to `/tank/projects/run_0730_all.log` — the
   *previous* cycle's name, reused. Contents are unambiguously 20260806, but the command block
   above writes `run_0806_all.log`; use that.

## Notes

- **`setsid` for both long jobs** (Step 1 ~2.9 h, Step 2 ~3.7 h) — the 20260723 lesson (a
  `nohup`-only job was killed by session teardown).
- **`PYTHONUNBUFFERED=1` when logging to a file.** Each `uv run` subprocess block-buffers
  stdout, so a running member shows *no* log output until it exits — which looks like a hang
  and isn't.
- **Disk.** `/tank/projects` was at 62 % with 805 GB free before the run; this cycle adds
  ~630 GB. Purge older submitted cycles with
  `python cleanup_aifs_run.py --date <YYYYMMDD> --yes` (keeps `aiwq/*.nc`).
- **Pre-existing warning, harmless for now:** `aifs_n320_grib_1p5defg_nc_cli.py:568` emits an
  xarray `FutureWarning` — `xr.concat` default `join='outer'` becomes `'exact'` in a future
  release and will then raise `ValueError` on the `step` coordinate. Worth fixing before an
  xarray bump.

## Outputs

| Path | Size | What |
|------|------|------|
| `…/20260806_0000/input_states/` | 46 GB | 50 input pkls (built via GCS proto) |
| `…/20260806_0000/icechunk_v2/` | 582 GB | Icechunk store, tag `cycle-20260806_0000` (50/50) |
| `…/20260806_0000/nc_1p5deg/` | 2.0 GB | 50 × 1.5° NetCDF (432–792 h) |
| `…/20260806_0000/aiwq/ensemble_quintile_probabilities_20260806_v2.nc` | 6.99 MB | **submitted file** |

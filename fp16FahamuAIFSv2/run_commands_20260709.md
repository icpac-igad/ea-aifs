# AIFS ENS v2.0 Run Commands - 20260709

**Date:** 20260709  
**Members:** 1-50 (all 50 ensemble members)  
**Steps:** 1 (CPU) input prep · 2 (GPU) inference — *as actually run on the local
RTX 5000 Ada box*, writing a local Icechunk store (no GCS upload, no GRIB).

For the full local-GPU environment build see [`LOCAL_GPU_RUN.md`](LOCAL_GPU_RUN.md);
for the store layout / commit cadence see [`ICECHUNK_PATH_A.md`](ICECHUNK_PATH_A.md)
and [`ICECHUNK_COMMIT_CADENCE.md`](ICECHUNK_COMMIT_CADENCE.md).

## Environment Setup

Two separate envs — the CPU ETL env (Step 1) and the GPU inference env (Step 2):

```bash
# Step 1 (CPU): input prep
micromamba activate aifs-etl

# Step 2 (GPU): inference — micromamba root prefix lives on the projects disk
export MAMBA_ROOT_PREFIX=/tank/projects/micromamba
PY=/tank/projects/micromamba/envs/aifs-gpu/bin/python
export HF_HOME=/tank/projects/hf_cache        # keep the ~2.4 GB checkpoint off $HOME
```

## PKL Creation Command

```bash
# Create input-state pickle files for 50 members
# Date: 20260709 | Source: AWS S3 mirror (hardcoded) | Upload to GCS: Yes
micromamba run -n aifs-etl python ecmwf_opendata_pkl_input_aifsens_v2.py \
  --date 20260709 \
  --members 1-50
```

> **No `--source` flag.** The open-data mirror is hardcoded to the **AWS S3** replica
> (the direct `ecmwf` portal is throttled to 500 simultaneous connections). Passing
> `--source` fails with `error: unrecognized arguments: --source`.

## Command Breakdown

| Parameter | Value | Description |
|-----------|-------|-------------|
| `--date` | `20260709` | Forecast initialization date (YYYYMMDD) |
| `--members` | `1-50` | All 50 ensemble members |
| (fixed) | mirror | AWS S3 — hardcoded, not selectable |
| (default) | `--gcs-subpath` | `input_v2` - GCS storage path |
| (default) | `--out-dir` | `input_states_v2/` - local output directory |

## Alternative Command Examples

```bash
# Subset members only (e.g., members 1-10 for testing)
micromamba run -n aifs-etl python ecmwf_opendata_pkl_input_aifsens_v2.py \
  --date 20260709 --members 1-10

# Specific members (e.g., 1, 5, 10, 25)
micromamba run -n aifs-etl python ecmwf_opendata_pkl_input_aifsens_v2.py \
  --date 20260709 --members 1,5,10,25

# Skip GCS upload (local testing only), keep the pkls on disk
micromamba run -n aifs-etl python ecmwf_opendata_pkl_input_aifsens_v2.py \
  --date 20260709 --members 1-50 --no-upload --keep-local
```

## Expected Output

The script will:
1. Download ECMWF Open Data for each member (date-6h and date)
2. Regrid from 0.25° to N320 resolution
3. Apply v2.0 transformations (112 fields including new wave params)
4. Save to `input_states_v2/input_state_member_NNN.pkl`
5. Upload to GCS: `gs://aifs-aiquest-us-20251127/20260709_0000/input_v2/input_state_member_NNN.pkl`

**Per-member processing time:** ~1-2 minutes each  
**Total time for 50 members:** ~50-100 minutes (depends on network speed)

## Output Files

```
input_states_v2/
├── input_state_member_001.pkl  (~0.97 GB)
├── input_state_member_002.pkl  (~0.97 GB)
│   ...
└── input_state_member_050.pkl  (~0.97 GB)
```

**Total local size:** ~48 GB (measured: 112 fields × 2 timesteps × 542,080 N320 cells, float64)  
**GCS location:** `gs://aifs-aiquest-us-20251127/20260709_0000/input_v2/`

## Step 2: GPU inference on the local box (Icechunk output)

For 20260709 the 50 pkls **already existed** in GCS, so Step 1 was not re-run — Step 2
fetched them on demand. Run from this folder so `coiled-data.json` resolves.

```bash
cd fp16FahamuAIFSv2
PY=/tank/projects/micromamba/envs/aifs-gpu/bin/python
export HF_HOME=/tank/projects/hf_cache
BASE=/tank/projects/aifs-run/20260709_0000
```

### 2a. Single-member test first

```bash
# fetches member 001 (0.971 GB) from GCS if absent, then runs it
$PY run_local_icechunk_v2.py \
  --date 20260709_0000 --members 1 --lead-time 792 \
  --input-dir $BASE/input_states --store $BASE/icechunk_v2 \
  --n-members 50 --commit-every 1 --float-size f4 \
  --write-hours 432-792 --skip-existing --gcs-fetch
```

Creates the store + schema (`50 × 132 × 542,080`, 120 fields) on the first member.
Re-running is a no-op thanks to `--skip-existing`.

### 2b. Full ensemble (members 1-50, one store, resumable)

```bash
$PY run_local_icechunk_v2.py \
  --date 20260709_0000 --members 1-50 --lead-time 792 \
  --input-dir $BASE/input_states --store $BASE/icechunk_v2 \
  --n-members 50 --commit-every 1 --float-size f4 \
  --write-hours 432-792 --skip-existing \
  --gcs-fetch --cleanup-pkl
```

Run it detached (it takes hours): `nohup … > /tank/projects/run_0709_all.log 2>&1 &`

### Why these flags

| Flag | Why |
|------|-----|
| `--lead-time 792` | Downstream reads at most 792 h. The v2 default of 960 h overshoots by 168 h of pure compute. (Cloud Run's `app.py` also uses 792.) |
| `--write-hours 432-792` | `shared/aifs_n320_grib_1p5defg_nc_cli.py` reads only the `432-504 … 720-792` windows (days 18-33). The rollout still **computes** every step; steps outside the window are simply not **stored** (no chunks → read back as `NaN`). Stores **61/132 steps** → **12 GB/member instead of ~30 GB**. |
| `--commit-every 1` | One ACID commit per 6 h model step (the native timestep). `time_chunk` is aligned to it, so writes never amplify. |
| `--float-size f4` | **Do not use `f2`.** 12 of 120 fields exceed float16's 65,504 max — incl. `msl` (104,063), which downstream needs — and would silently become `±inf`. See `ICECHUNK_PATH_A.md` §1.2. |
| `--skip-existing` | Idempotent resume: members already complete are skipped *before* any download or inference. |
| `--gcs-fetch` | Pulls each member's pkl from GCS, prefetching the **next** member during the current member's inference, so the ~92 s transfer is hidden. |
| `--cleanup-pkl` | Deletes each pkl after use — ~3 GB on disk at a time instead of ~48 GB. |
| `--n-members 50` | Sizes the store's `member` axis; all 50 members share **one** store with disjoint chunks. |

### Monitoring

`stdout` is block-buffered into the log, so `[OK]` lines lag by minutes. The **store is
the reliable progress signal**:

```bash
# how many members are complete + latest commit
$PY - <<'EOF'
import icechunk, zarr, numpy as np
r = icechunk.Repository.open_or_create(
    icechunk.local_filesystem_storage("/tank/projects/aifs-run/20260709_0000/icechunk_v2"))
print("latest commit:", list(r.ancestry(branch="main"))[0].message)
a = zarr.open_group(r.readonly_session("main").store, mode="r")["2t"]
done = [m + 1 for m in range(50) if np.isfinite(a[m, 131, 0])]
print(f"members complete: {len(done)}/50")
EOF

du -sh /tank/projects/aifs-run/20260709_0000/icechunk_v2
nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader
```

### Verify a member

```bash
$PY - <<'EOF'
import icechunk, zarr, numpy as np
r = icechunk.Repository.open_or_create(
    icechunk.local_filesystem_storage("/tank/projects/aifs-run/20260709_0000/icechunk_v2"))
rg = zarr.open_group(r.readonly_session("main").store, mode="r")
a = np.asarray(rg["2t"][0, :, 0]); fin = np.isfinite(a)
idx = np.where(fin)[0]
print("2t", rg["2t"].shape, "chunks", rg["2t"].chunks)
print(f"stored idx {idx.min()}..{idx.max()} = hours {(idx.min()+1)*6}..{(idx.max()+1)*6}")
for v in ("2t", "msl", "tp"):
    print(f"  {v} @432h mean {np.nanmean(rg[v][0,71,:]):,.2f}")
EOF
```

### Measured (this box: RTX 5000 Ada, 30 GB, FP16, 16 chunks)

| | |
|---|---|
| Per member | **~317 s** (~11.8 GB VRAM, 100 % GPU); download hidden behind inference |
| Store per member | **11.6 GB** (61 stored steps × 120 fields, `f4`) |
| Full ensemble | **~4.5 h**, **~580 GB** — fits the 1.4 TB `/tank/projects` |
| Commits | 61 per member (one per 6 h step) + schema + init |
| Output | `/tank/projects/aifs-run/20260709_0000/icechunk_v2`, tag `cycle-20260709_0000` |

> Storing all 160 steps of a 960 h run would be **30 GB/member → ~1.5 TB**, which does
> **not** fit. The `--write-hours` window is what makes the 50-member run viable here.

## Step 3a: regrid 432-792h → 1.5° NetCDF, reading the Icechunk store

`shared/aifs_n320_grib_1p5defg_nc_cli.py` gained `--source icechunk`: it reads the native
N320 fields straight from the store (no GRIB, no GCS download) and writes the **identical**
NetCDF the GRIB path produced, so Steps 3b–3d are unchanged.

```bash
$PY ../shared/aifs_n320_grib_1p5defg_nc_cli.py \
  --date 20260709_0000 --members 1-50 --v2 --no-upload \
  --source icechunk \
  --icechunk-store /tank/projects/aifs-run/20260709_0000/icechunk_v2 \
  --icechunk-tag cycle-20260709_0000-final
```

Drop `--no-upload` to publish to `<date>/fp16_v2_1p5deg_nc/`. Prefer `--icechunk-tag` over
`--icechunk-branch main` for reproducibility — but **use the `-final` tag** (see the tag note
below). With `--source icechunk --no-upload` no GCS credentials are needed at all.

**Output contract (validated against the GRIB path, member 001):** dims
`(member=1, time=5, step=60, latitude=121, longitude=240)`, vars `2t`/`msl`/`tp` `float64`,
`step` = `timedelta64[us]` hours 438..792, lat `90..-90`, lon `0..358.5`. `xr.concat` outer-joins
on `step`, so each `time` block holds only its own 12 steps and the rest are `NaN` (80 % NaN
overall) — exactly why 3b does `.isel(time=t).sum(dim='step', skipna=True)`. The NaN/valid
block pattern is **identical** to the GRIB path.

> Values differ from a GRIB run of the "same" member because aifs-ens-2.0 injects noise per
> run — two inferences of one member are different stochastic realisations. Verified: the
> difference grows with lead time (0.44 K @ h6 → 0.90 K @ h72 → 2.4 K @ h438), and the step
> alignment `index = hour/6 - 1` is confirmed by an offset sweep.

### Fallback: the GRIB path (no Icechunk)

`fp16_automate_aifs_gpu_pipeline_v2.py` still works and is untouched. For a local run,
`--no-upload --keep-local` skip the GCS upload and the cleanup:

```bash
$PY fp16_automate_aifs_gpu_pipeline_v2.py \
  --date 20260709_0000 --members 1 --lead-time 960 \
  --input-dir $BASE/input_states --output-dir $BASE/fp16_v2_forecasts_local \
  --no-upload --keep-local
```
(Member 001 / 960 h → 14 GRIB files, 20.2 GB, 394 s.)

## Troubleshooting

| Issue | Solution |
|-------|----------|
| "earthkit-data serialises open-data downloads with a file lock" (hangs) | Use Ctrl-C (not Ctrl-Z) to stop; remove lock: `rm -f /tmp/earthkit-data-cloudenv/*.cache.lock` |
| `error: unrecognized arguments: --source` | The `--source` flag was removed; the AWS S3 mirror is hardcoded. Drop the flag. |
| Network timeout / slow downloads | Simply re-run; the mirror is already the faster AWS S3 replica |
| "coiled-data.json not found" warning | GCS upload skipped (local-only mode) — expected if running without GCS credentials |
| Member fails verification | Check network connectivity; member will be skipped and flagged in summary |
| **Step 2:** `ImportError: Could not import huggingface_hub` | anemoi fetches the checkpoint through it. `pip install huggingface_hub` (it is missing from the README's GPU stack list). |
| **Step 2:** truncated pkl (e.g. 56 MB instead of 0.97 GB) | A killed download left a partial file. `--gcs-fetch` now downloads to `*.part` and renames only after the byte count matches the blob. |
| **Step 2:** log shows no progress for minutes | `stdout` is block-buffered into the log. Query the store's `ancestry()` instead (see *Monitoring*). |
| **Step 2:** store much larger than expected | Check `--write-hours` / `--commit-every`. A big `time_chunk` written step-by-step read-modify-writes the whole chunk each step (~11.6× amplification at 24 steps). See `ICECHUNK_COMMIT_CADENCE.md`. |
| **Step 2:** `[TAG] skipped (tag already exists, tags are immutable)` | A previous partial run already tagged the cycle. Icechunk tags are immutable **and deleted names are tombstoned — never reusable**, so do *not* delete to "fix" it. The runner now only tags once **all** `--n-members` are present; read a partial store via `branch="main"`, not a tag. |

## Links

- **Reference:** [AIFS ENS v2.0 Notebook](https://huggingface.co/ecmwf/aifs-ens-2.0/blob/main/run_AIFS_ENS_v2.0.ipynb)
- **Input prep script:** `fp16FahamuAIFSv2/ecmwf_opendata_pkl_input_aifsens_v2.py`
- **Local GPU runner:** `fp16FahamuAIFSv2/run_local_icechunk_v2.py`
- **v2 README:** `fp16FahamuAIFSv2/README.md`
- **Local GPU env + benchmark:** `fp16FahamuAIFSv2/LOCAL_GPU_RUN.md`

---

**Status (local GPU run, 20260709_0000): ✅ COMPLETE**

- Input: 50/50 pkls present in `gs://aifs-aiquest-us-20251127/20260709_0000/input_v2/`
  (0.971 GB each) — Step 1 not re-run.
- Step 2: member 001 validated first, then members 2-50 into the same store with
  `--skip-existing --gcs-fetch --cleanup-pkl`.
- Result: **49 written, 1 skipped, 0 failed**, wall clock **14,219 s (3.95 h)**
  (~317 s/member). Store **584 GB** at
  `/tank/projects/aifs-run/20260709_0000/icechunk_v2`.
- Verified: **3052 commits** on `main` (= 2 + 61 × 50), **50/50 members** each with
  exactly **61 stored steps** (indices 71..131 = hours 432..792), shape
  `(50, 132, 542080)`, chunks `(1, 1, 542080)`.

> **Read the full ensemble via `branch="main"` or the tag `cycle-20260709_0000-final`.**
> The tag `cycle-20260709_0000` was created by the *single-member test run* and points at a
> snapshot where **only member 001 is present** (the other 49 read as `NaN`). Icechunk tags
> are immutable and deleted names are tombstoned, so it cannot be repointed. The runner now
> tags only when the whole ensemble is complete.

```python
import icechunk, zarr
r = icechunk.Repository.open_or_create(icechunk.local_filesystem_storage(
    "/tank/projects/aifs-run/20260709_0000/icechunk_v2"))
ro = r.readonly_session(tag="cycle-20260709_0000-final")   # or branch="main"
ds = zarr.open_group(ro.store, mode="r")
```

# Running AIFS-ENS-2.0 v2 FP16 inference on a **local GPU**

How the `fp16FahamuAIFSv2` Step-2 inference was made to run **on-box** (no Coiled, no
Cloud Run), writing output to **local disk** instead of GCS. Covers the exact env build,
the scripts used, the code changes made, the run commands, and a wall-clock **benchmark
vs the Cloud Run (L4) path**.

Everything lives on the roomy projects disk:

| Thing | Path |
|---|---|
| micromamba root prefix | `/tank/projects/micromamba` |
| env | `aifs-gpu` (Python 3.12) → `/tank/projects/micromamba/envs/aifs-gpu/bin/python` |
| micromamba binary | `/home/ezra/bin/micromamba` |
| HF checkpoint cache | `/tank/projects/hf_cache` (`HF_HOME`) |
| run workspace | `/tank/projects/aifs-run/<date>/` |

---

## 0. Hardware

| | |
|---|---|
| GPU | **NVIDIA RTX 5000 Ada Generation**, 30 GB VRAM, compute capability **8.9 (sm_89)** |
| Driver / CUDA | 580.159.03 / CUDA 13.0 (runs cu12 wheels fine, backward-compatible) |
| Host | 220 GB RAM, `/tank/projects` 1.9 TB free (group-`users` writable) |

30 GB clears the FP16 / 24 GB bar with headroom, so `INFERENCE_NUM_CHUNKS=16` at
`precision="16"` is comfortable (no chunk bump needed).

---

## 1. Environment build (exact commands)

There was **no** conda/mamba/micromamba on the box and the base Python had no torch/anemoi,
so the GPU env was built from scratch.

### 1.1 Bootstrap micromamba

```bash
cd /home/ezra
export MAMBA_ROOT_PREFIX=/home/ezra/micromamba          # (binary lands in ~/bin)
curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xvj bin/micromamba
/home/ezra/bin/micromamba --version                     # 2.8.1
```

### 1.2 Create the env on the projects disk

```bash
export MAMBA_ROOT_PREFIX=/tank/projects/micromamba
/home/ezra/bin/micromamba create -y -n aifs-gpu -c conda-forge python=3.12 pip
```

### 1.3 Install the GPU stack (pip, inside the env)

```bash
PY=/tank/projects/micromamba/envs/aifs-gpu/bin/python
$PY -m pip install \
  torch==2.7.0 torch-geometric==2.6.1 \
  anemoi-inference==0.8.3 anemoi-models==0.11.2 anemoi-utils==0.4.35.post3 \
  earthkit-regrid==0.5.1 "earthkit-data<1.0.0" ecmwf-opendata==0.3.29 \
  icechunk zarr google-cloud-storage python-dotenv netcdf4 xarray
```

Resolved versions (key ones): **torch 2.7.0+cu126, anemoi-inference 0.8.3,
anemoi-models 0.11.2, anemoi-utils 0.4.35.post3, earthkit-regrid 0.5.1,
earthkit-data 0.20.0, ecmwf-opendata 0.3.29, icechunk 2.1.0, zarr 3.2.1,
google-cloud-storage 3.12.0, numpy 2.5.0**.

### 1.4 flash-attn — pick the wheel by torch's C++11 ABI

Building flash-attn from source takes an hour+, so use the prebuilt wheel — but it must
match `(cuda, torch, python, C++11-ABI)`. **torch 2.7 wheels use `cxx11abi = TRUE`**
(older torch used FALSE), so check it first:

```bash
$PY -c "import torch; print(torch._C._GLIBCXX_USE_CXX11_ABI)"   # -> True
```

Then fetch and install the matching wheel (`cu12torch2.7cxx11abiTRUE-cp312`):

```bash
cd /tank/projects
WHL="flash_attn-2.7.4.post1+cu12torch2.7cxx11abiTRUE-cp312-cp312-linux_x86_64.whl"
curl -sL -o "$WHL" \
  "https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1%2Bcu12torch2.7cxx11abiTRUE-cp312-cp312-linux_x86_64.whl"
$PY -m pip install --no-deps "$WHL"     # ~415 MB wheel; --no-deps: torch already present
```

> ⚠️ Save the wheel under its **full** name — pip rejects `flash_attn.whl`
> (*"Invalid wheel filename (wrong number of parts)"*).

### 1.5 huggingface_hub — REQUIRED (not in the README's list)

anemoi downloads the `ecmwf/aifs-ens-2.0` checkpoint through `huggingface_hub`. Without it
the runner dies at the first step with
*`ImportError: Could not import huggingface_hub`*. Install it:

```bash
$PY -m pip install huggingface_hub
```

### 1.6 Verify the stack

```bash
$PY - <<'EOF'
import torch, flash_attn
print("torch", torch.__version__, "| cuda", torch.cuda.is_available(), "|", torch.cuda.get_device_name(0))
from flash_attn import flash_attn_func
q=torch.randn(1,64,8,64,device="cuda",dtype=torch.float16); print("flash_attn OK", tuple(flash_attn_func(q,q,q).shape))
import anemoi.inference, anemoi.models, earthkit.regrid, icechunk, zarr
print("anemoi-inference", anemoi.inference.__version__, "| icechunk", icechunk.__version__, "| zarr", zarr.__version__)
EOF
```

Expected:
```
torch 2.7.0+cu126 | cuda True | NVIDIA RTX 5000 Ada Generation
flash_attn OK (1, 64, 8, 64)
anemoi-inference 0.8.3 | icechunk 2.1.0 | zarr 3.2.1
```

---

## 2. Scripts & code

### 2.1 Existing (unchanged logic)
| Script | Role |
|---|---|
| `fp16_automate_aifs_gpu_pipeline_v2.py` | Orchestrator: per-member download → FP16 inference → upload → cleanup. Loads `ecmwf/aifs-ens-2.0` once. |
| `fp16_multi_run_AIFS_ENS_v2.py` | The runner (`run_ensemble_member`): 72h-chunked GRIB via `GribFileOutput`. |

### 2.2 New (added for local + Icechunk)
| Script | Role |
|---|---|
| `icechunk_output.py` | **New.** Local-filesystem Icechunk writer — `open_repo_local()`, `init_schema()`, `IcechunkMemberWriter`. Adapts `ICECHUNK_PATH_A.md` §2.1 (`gcs_storage`) to `icechunk.local_filesystem_storage(path)`. Native N320 `(member,time,values)` schema, one commit/member, keeps ALL fields (incl. `cos_mwd`/`sin_mwd`). |
| `run_local_icechunk_v2.py` | **New.** Fully-local runner: load model once → read local pkl → inference straight into a local Icechunk store → commit. No GCS, no GRIB, no upload/cleanup. |

### 2.3 Change made to `fp16_automate_aifs_gpu_pipeline_v2.py`

Added two flags for local runs; **default behavior is unchanged** (still downloads/uploads/
cleans up when the flags are absent):

- `--no-upload` — skip the GCS upload; keep the GRIB in `--output-dir`. Also relaxes the
  service-account requirement when the input pkl is already staged locally.
- `--keep-local` — do not delete the pkl/GRIB after each member.

Threaded through `run_pipeline()` → `process_single_member()`; the download step already
skips GCS when a valid pkl exists in `--input-dir`, so with a pre-staged pkl the run never
touches GCS.

---

## 3. Getting an input pkl (still from GCS, one-off)

The 112-field v2 input pkls live at `gs://aifs-aiquest-us-20251127/<date>_0000/input_v2/`.
A full pkl is **~0.97 GB** (112 fields × 2 timesteps × 542080 N320 cells). Download with the
service account (`coiled-data.json`, in this folder):

```bash
$PY - <<'EOF'
from google.cloud import storage
c=storage.Client.from_service_account_json("coiled-data.json")
b=c.bucket("aifs-aiquest-us-20251127")
b.blob("20260702_0000/input_v2/input_state_member_001.pkl").download_to_filename(
    "/tank/projects/aifs-run/20260702_0000/input_states/input_state_member_001.pkl")
EOF
```

> GCS pull is ~1 MB/s here (~17 min for the full 0.97 GB) — run it in the **background** so a
> 2-minute shell timeout doesn't leave a truncated pkl. (Alternatively, generate pkls locally
> with `ecmwf_opendata_pkl_input_aifsens_v2.py --no-upload --keep-local` — no GCS needed.)

---

## 4. Running the inference (local, no GCS)

```bash
cd /home/ezra/ea-aifs/fp16FahamuAIFSv2
export HF_HOME=/tank/projects/hf_cache
PY=/tank/projects/micromamba/envs/aifs-gpu/bin/python
BASE=/tank/projects/aifs-run/20260702_0000

# 72h validation (fast, ~2-3 min inference after the one-time checkpoint pull)
$PY fp16_automate_aifs_gpu_pipeline_v2.py --date 20260702_0000 --members 1 --lead-time 72 \
    --input-dir $BASE/input_states --output-dir $BASE/fp16_v2_forecasts_local \
    --no-upload --keep-local

# Full 40-day (960h) run
$PY fp16_automate_aifs_gpu_pipeline_v2.py --date 20260702_0000 --members 1 --lead-time 960 \
    --input-dir $BASE/input_states --output-dir $BASE/fp16_v2_forecasts_960h \
    --no-upload --keep-local
```

**72h validation result (member 001):** ✅ 1552 MB GRIB, 1416 messages, 41 params,
`tp`/`msl`/`2t` present, steps h6→h72. 7.9 min wall incl. the first-time checkpoint download.
Benign warnings only (coupled forcings unsupported; NaNs over sea in wave/soil fields from the
LSM mask; legacy data-request).

### Optional: write to a **local Icechunk store** instead of GRIB
```bash
$PY run_local_icechunk_v2.py --date 20260702_0000 --members 1 --lead-time 960 \
    --input-dir $BASE/input_states --store $BASE/icechunk_v2 --float-size f2
```
Native N320, all fields, one commit/member — far smaller than ~20 GB/member of GRIB.

---

## 5. Benchmark — local RTX 5000 Ada vs Cloud Run L4

**Headline:** the full **960h (40-day) member-001 run finished in 394 s (6.6 min)** on the
local RTX 5000 Ada — **~4× faster** than the Cloud-Run-L4 reference scaled to 960h, at a
peak of only **~11.8 GB VRAM** (nvidia-smi, 100% GPU util). No GCS round-trip.

### Reference: Cloud Run (from `serverless_cloudrun/README.md`)
| | |
|---|---|
| GPU | NVIDIA **L4** (24 GB), 8 vCPU, 32 GiB |
| Model / precision | aifs-ens **1.0**, FP16, `NUM_CHUNKS=16` |
| Lead time | **792 h** (132 steps) — *not 960h* |
| Per member | **~22 min** (inference **+ GCS upload**) |
| Full 50-member ensemble | ~45 min wall (3 concurrent L4s), ~$8.33 |

### ⚠️ Why this is not a clean apples-to-apples comparison
1. **Model:** Cloud Run bench is **v1** (aifs-ens-1.0); local is **v2** (aifs-ens-2.0, a
   *larger* model → more compute per step).
2. **Lead time:** 792 h (132 steps) vs 960 h (160 steps). Normalize by **per-step** time or
   scale ×160/132 ≈ ×1.21.
3. **Upload:** Cloud Run's ~22 min **includes GCS upload**; the local run writes to disk only.
4. **GPU:** L4 (24 GB, ~121 FP16 TFLOPS) vs RTX 5000 Ada (30 GB, ~around 175 FP16 TFLOPS).

So the fair metric is **seconds per forecast step** (and a 792→960h-scaled estimate).

### Results (measured 2026-07-06, cycle `20260702_0000`, member 001)

| Metric | Cloud Run L4 (v1, 792h, +upload) | Local RTX 5000 Ada (v2, 960h, no upload) |
|---|---|---|
| Steps | 132 | **160** |
| Wall clock (member 001) | ~22 min | **394 s = 6.6 min** |
| Per step | ~10.0 s/step | **2.46 s/step** |
| 960h-equivalent | ~26.7 min (scaled ×160/132) | **6.6 min (measured)** |
| Peak VRAM | ~fits 24 GB (L4) | **~11.8 GB** (nvidia-smi, 100% util) |
| Output | GRIB → GCS | **14 GRIB files, 20.2 GB** (h6→h960), local disk |

**Speed-up: ~4.0× on 960h-scaled wall clock, ~4.1× per step.** Even discounting that the
Cloud Run figure *includes* GCS upload and runs the *lighter v1* model, the local box is
decisively faster — the RTX 5000 Ada is a stronger GPU than the L4 and there is no upload
leg. Per-step is the honest cross-model metric: **2.46 s/step local vs ~10.0 s/step L4**.

**Notes / fairness:**
- Local number is **pure inference + GRIB write** (checkpoint was already cached in
  `HF_HOME`; the *first-ever* run adds a ~5 min one-time HF checkpoint download — the 72h
  validation took 7.9 min wall for that reason).
- Cloud Run's ~22 min includes per-member GCS upload; its win is **horizontal scale**
  (3 concurrent L4s → 50 members in ~45 min wall). The local box runs members **serially**,
  so 50 × 6.6 min ≈ **5.5 h** for a full ensemble on one GPU — trade wall-clock for $0 cloud
  cost and no upload/egress.
- VRAM headroom (~11.8/30 GB) means `NUM_CHUNKS` could be *lowered* for a bit more speed, or
  2 members could run concurrently on this one GPU if desired.

### Reproduce the benchmark
```bash
cd /home/ezra/ea-aifs/fp16FahamuAIFSv2 && export HF_HOME=/tank/projects/hf_cache
PY=/tank/projects/micromamba/envs/aifs-gpu/bin/python; BASE=/tank/projects/aifs-run/20260702_0000
bash -c 'SECONDS=0; '"$PY"' fp16_automate_aifs_gpu_pipeline_v2.py --date 20260702_0000 \
  --members 1 --lead-time 960 --input-dir '"$BASE"'/input_states \
  --output-dir '"$BASE"'/fp16_v2_forecasts_960h --no-upload --keep-local; echo "WALL=${SECONDS}s"'
```

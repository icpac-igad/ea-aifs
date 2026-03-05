# serverless_cloudrun — Cloud Run GPU for AIFS Ensemble Inference

Direct HTTP invocation of Cloud Run GPU (NVIDIA L4) for parallelized AIFS FP16 ensemble inference. A local Python script sends HTTP POST requests to a Flask app running on Cloud Run GPU instances. No Lithops (see [WHY_NOT_LITHOPS.md](WHY_NOT_LITHOPS.md) for why).

## Architecture

```
run_aifs_ensemble.py (local, via uv run)
  |
  |  HTTP POST /run  {members: [1,2,3], date: "20260301_0000"}
  |  (authenticated with SA ID token)
  |
  +---> Cloud Run GPU instance 1 (L4) --> inference --> GCS upload
  +---> Cloud Run GPU instance 2 (L4) --> inference --> GCS upload
  +---> Cloud Run GPU instance 3 (L4) --> inference --> GCS upload
        (3 concurrent max per quota; Cloud Run auto-queues excess)
        (scales to zero when idle -- no cost between runs)
```

Each Cloud Run instance:
1. Downloads `.pkl` input state from GCS
2. Runs AIFS-ENS FP16 inference (792h lead time, 6h time step)
3. Writes GRIB output files (72h per file)
4. Uploads GRIB files to GCS
5. Cleans up local scratch files

## File Layout

```
serverless_cloudrun/
+-- app.py                    # Flask worker (runs on Cloud Run GPU)
+-- run_aifs_ensemble.py      # Local orchestrator (sends HTTP to Cloud Run)
+-- Dockerfile                # Extends flash-attn base image + CUDA compat
+-- cloudbuild.yaml           # Cloud Build --> Artifact Registry
+-- .dockerignore
+-- .env.example              # Template for environment variables
+-- README.md                 # This file
+-- WHY_NOT_LITHOPS.md        # Why Lithops v3.6.3 doesn't work for GPU
```

---

## Prerequisites

- GCP project with Cloud Run GPU quota (NVIDIA L4) in your chosen region
- Artifact Registry repository for Docker images
- GCS bucket for input/output data
- Service account with `roles/run.invoker`, `roles/storage.admin`
- [uv](https://github.com/astral-sh/uv) for running the orchestrator script

## 1. Configure Environment

```bash
cp .env.example .env
# Edit .env with your project-specific values:
#   - GCP_PROJECT_ID, GCP_REGION
#   - AIFS_SERVICE_URL (after deploying Cloud Run service)
#   - GCS_BUCKET
#   - GOOGLE_APPLICATION_CREDENTIALS (path to SA key)
```

## 2. Build the Docker Image

### 2.1 Update the Dockerfile

Edit `Dockerfile` and replace the `FROM` line with your base image path:

```dockerfile
FROM REGION-docker.pkg.dev/YOUR_PROJECT/YOUR_REPO/flash-attn-notebook:v1.0
```

The base image must contain: CUDA, PyTorch, flash-attn, anemoi-inference, anemoi-models.

### 2.2 Build via Cloud Build

```bash
gcloud builds submit \
  --config=cloudbuild.yaml \
  --project=YOUR_PROJECT_ID \
  --region=YOUR_REGION
```

Build time: ~5 min. No CUDA compilation needed (base image has flash-attn pre-compiled).

### 2.3 Key Dockerfile Details

- **CUDA forward compat:** Installs `cuda-compat-12-9` because Cloud Run GPU provides driver 535.216.03 (CUDA 12.2) but the image uses CUDA 12.9
- **HF model pre-cached:** `ecmwf/aifs-ens-1.0` downloaded at build time to avoid cold-start download
- **Minimal deps:** Only `flask`, `gunicorn`, `google-cloud-storage` added on top of base image

## 3. Deploy Cloud Run GPU Service

```bash
gcloud run deploy aifs-gpu-worker \
  --image=REGION-docker.pkg.dev/YOUR_PROJECT/YOUR_REPO/aifs-gpu-cloudrun:latest \
  --region=YOUR_REGION \
  --project=YOUR_PROJECT_ID \
  --gpu=1 \
  --gpu-type=nvidia-l4 \
  --cpu=8 \
  --memory=32Gi \
  --max-instances=17 \
  --min-instances=0 \
  --timeout=3600 \
  --no-allow-unauthenticated \
  --set-env-vars="GCS_BUCKET=YOUR_BUCKET"
```

Grant invoker access to the service account:

```bash
gcloud run services add-iam-policy-binding aifs-gpu-worker \
  --region=YOUR_REGION --project=YOUR_PROJECT_ID \
  --member="serviceAccount:YOUR_SA@YOUR_PROJECT_ID.iam.gserviceaccount.com" \
  --role="roles/run.invoker"
```

### Verify deployment

```bash
# Health check (authenticated)
TOKEN=$(gcloud auth print-identity-token)
curl -H "Authorization: Bearer $TOKEN" \
  https://YOUR_SERVICE_URL/health
# Expected: {"status":"ok"}
```

## 4. Run AIFS Ensemble Inference

```bash
# Set credentials
export GOOGLE_APPLICATION_CREDENTIALS=path/to/service-account-key.json

# Dry run -- show batches + cost estimate (no GPU cost)
uv run run_aifs_ensemble.py --date 20260301_0000 --members 1-50 --dry-run

# Single member test (~$0.24, ~22 min)
uv run run_aifs_ensemble.py \
  --date 20260301_0000 \
  --members 24 \
  --batch-size 1

# Small test -- 2 batches of 3 (~$1, ~45 min)
uv run run_aifs_ensemble.py \
  --date 20260301_0000 \
  --members 1-6 \
  --batch-size 3

# Full ensemble -- 50 members, 17 batches (~$8.33, ~45 min wall-clock)
uv run run_aifs_ensemble.py \
  --date 20260301_0000 \
  --members 1-50 \
  --batch-size 3 \
  --max-concurrent 3
```

### Safety Features (Avoiding Duplicate GPU Runs)

The orchestrator has three safeguards to prevent unnecessary GPU cost:

1. **Pre-check GCS before dispatch** -- if GRIB files already exist for a member, skip it
2. **On client timeout, poll GCS** -- instead of retrying (which spins up another GPU), wait for the in-flight GPU to finish
3. **GCS rescue after errors** -- check if files landed despite HTTP errors before retrying

**Always use `--dry-run` first** to preview batches and cost before committing GPU resources.

### Verify Output

```bash
# List GRIB files for a date
gsutil ls gs://YOUR_BUCKET/20260301_0000/fp16_forecasts/

# Check specific member
gsutil ls gs://YOUR_BUCKET/20260301_0000/fp16_forecasts/*member024*

# Count total files and size
gsutil du -sh gs://YOUR_BUCKET/20260301_0000/fp16_forecasts/
```

## 5. Cost Estimate

| Component | Details | Cost |
|---|---|---|
| Cloud Run GPU (L4) | ~$0.65/hr per instance (GPU + 8 vCPU + 32 GiB) | - |
| Per member | ~22 min inference + upload | ~$0.24 |
| Per batch (3 members) | ~45 min per instance | ~$0.49 |
| **Full ensemble (17 batches)** | **~45 min wall-clock** | **~$8.33** |

### Monthly cost (daily runs)

| Frequency | Cost/run | Monthly |
|---|---|---|
| 1x daily | $8.33 | ~$250 |
| 2x daily (00Z + 12Z) | $8.33 | ~$500 |

Scales to zero when idle -- no cost between runs.

## 6. Troubleshooting

| Problem | Solution |
|---|---|
| `403 Forbidden` on `/run` | SA doesn't have `roles/run.invoker`. Re-grant IAM binding |
| `CUDA error: device kernel image is invalid` | Missing CUDA forward compat. Dockerfile must have `cuda-compat-12-9` and `LD_LIBRARY_PATH` set |
| `CUDA error: invalid resource handle` | Stale CUDA context after previous error. Service resets runner automatically; retry should work |
| `Neither metadata server or valid service account credentials` | Set `GOOGLE_APPLICATION_CREDENTIALS` to the SA key path |
| Duplicate GPU runs / extra cost | Always `--dry-run` first. Script checks GCS before dispatching and polls GCS on timeout |
| Service not scaling to zero | Check `min-instances=0` in service config. Active requests keep instances alive until timeout |
| Cloud Build fails | Check SA permissions. Ensure disk quota (needs 200 GB for `machineType: E2_HIGHCPU_8`) |

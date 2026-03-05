# Why Not Lithops? — Cloud Run GPU for AIFS Inference

This document explains why Lithops v3.6.3 was **not used** for the serverless Cloud Run GPU deployment, despite an earlier attempt in the `lithops/` folder.

## Background

The initial plan was to use [Lithops](https://github.com/lithops-cloud/lithops) (v3.6.3) to orchestrate AIFS ensemble inference on Cloud Run GPU instances. Lithops provides a `FunctionExecutor` that dispatches Python functions to serverless backends (Cloud Run, Cloud Functions, AWS Lambda, etc.).

However, three fundamental limitations in Lithops v3.6.3's Cloud Run backend made it incompatible with GPU workloads.

## Failure 1: No GPU Resource Support

**Problem:** Lithops' Cloud Run backend has no configuration for GPU resources. There is no `gpu_type`, `gpu_count`, or equivalent parameter in Lithops' Cloud Run config. The `runtime_memory` and `runtime_cpu` options exist, but GPUs cannot be specified.

**Impact:** Cloud Run services deployed by Lithops cannot request GPU accelerators (e.g., NVIDIA L4). Without GPU resources, AIFS inference (which requires CUDA) cannot run.

**Lithops code reference:** `lithops/serverless/backends/gcp_cloudrun/config.py` — the config schema has no GPU fields. The Cloud Run service creation in `gcp_cloudrun.py` uses only CPU/memory resource specs.

## Failure 2: GCR-Only Image Paths

**Problem:** Lithops hardcodes the `gcr.io/` prefix for container image paths. It constructs image URLs as `gcr.io/{project}/{runtime}:{tag}`, which only works with the legacy Google Container Registry.

**Impact:** Our AIFS base image (`flash-attn-notebook:v1.0`) is stored in **Artifact Registry** (`europe-west4-docker.pkg.dev/...`), which is the current GCP standard (GCR is deprecated). Lithops cannot reference Artifact Registry images, so it cannot use our pre-built CUDA/PyTorch/flash-attn image.

**Lithops code reference:** `lithops/serverless/backends/gcp_cloudrun/gcp_cloudrun.py` — image path construction uses `gcr.io/` prefix unconditionally.

## Failure 3: Broken Runtime Caching

**Problem:** When using a custom Docker image (which we must, for CUDA + flash-attn), Lithops' runtime caching mechanism fails. It compares the deployed image digest against its internal cache, but the comparison logic breaks when the image comes from a non-GCR registry or when the image was built externally (not by Lithops).

**Impact:** Every invocation triggers a full image rebuild/redeploy cycle, adding 10-20 minutes of overhead to each function call. This makes Lithops impractical for any workload requiring custom images.

## Solution: Direct HTTP Invocation

Instead of Lithops, we use a simple direct HTTP approach:

1. **`app.py`** — Flask worker running on Cloud Run GPU (L4), deployed as a standard Cloud Run service with GPU resources configured via Terraform/gcloud
2. **`run_aifs_ensemble.py`** — Local Python script that sends HTTP POST requests to the Cloud Run service using `ThreadPoolExecutor` for parallelism

This approach:
- Supports GPU resources (configured in Cloud Run service spec, not in the orchestrator)
- Works with any container registry (Artifact Registry, Docker Hub, etc.)
- Has no caching issues (Cloud Run manages image deployment natively)
- Is simpler — no Lithops dependency, no serverless abstraction layer
- Provides full control over retry logic, timeout handling, and cost management

## Could Lithops Work in the Future?

Possibly, if:
1. Lithops adds GPU resource configuration to its Cloud Run backend
2. Lithops supports Artifact Registry image paths (not just `gcr.io/`)
3. Lithops fixes runtime caching for externally-built custom images

As of Lithops v3.6.3 (tested February 2026), none of these are supported.

## See Also

- `../lithops/` — The earlier Lithops attempt (kept as historical reference)
- `README.md` — Setup and usage for the working Cloud Run GPU approach

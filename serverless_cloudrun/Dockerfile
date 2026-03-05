# AIFS GPU Cloud Run Worker
# ==========================================
#
# NOTE: Update the base image path below to point to YOUR Artifact Registry.
# The base image must contain: CUDA, PyTorch, flash-attn, anemoi-*
# See README.md for how to build the base image.
#
# Thin layer on top of the pre-built flash-attn notebook image.
# Adds Flask/gunicorn for direct HTTP invocation (no Lithops) and
# pre-caches the HuggingFace AIFS model.
#
# Base image already contains: CUDA, PyTorch, flash-attn, anemoi-*
# This layer adds: Flask + gunicorn (~2 min build, no CUDA compilation)
#
# Build with Cloud Build:
#   gcloud builds submit \
#     --config=cloudbuild.yaml \
#     --project=YOUR_PROJECT_ID \
#     --region=YOUR_REGION

# >>> CUSTOMIZE: Replace with your Artifact Registry base image path <<<
FROM europe-west4-docker.pkg.dev/sewaa-416306/coiled-aifs-europe/flash-attn-notebook:v1.0

# ==============================================================================
# CUDA Forward Compatibility
# ==============================================================================
# Cloud Run GPU provides NVIDIA driver 535.216.03 (CUDA 12.2), but the base
# image uses CUDA 12.9.1. Install the CUDA forward compat package so that
# CUDA 12.9 user-space libraries work with the older host driver.
RUN apt-get update && apt-get install -y --no-install-recommends \
        cuda-compat-12-9 \
    && rm -rf /var/lib/apt/lists/*

# Prepend forward compat libs so they take precedence over Cloud Run's
# default /usr/local/nvidia/lib64
ENV LD_LIBRARY_PATH=/usr/local/cuda-12.9/compat:${LD_LIBRARY_PATH}

# Install minimal runtime dependencies (no Lithops)
RUN pip install --no-cache-dir \
        flask \
        gunicorn \
        google-cloud-storage

# ==============================================================================
# Pre-cache HuggingFace model (eliminates ~2 min cold-start download)
# ==============================================================================

ENV HF_HOME=/opt/hf_cache

RUN python -c "from huggingface_hub import snapshot_download; snapshot_download('ecmwf/aifs-ens-1.0')"

# ==============================================================================
# GPU/AIFS environment variables
# ==============================================================================

ENV PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
ENV ANEMOI_INFERENCE_NUM_CHUNKS=16
ENV PYTHONUNBUFFERED=TRUE

# Scratch directories for inference I/O
RUN mkdir -p /scratch/input_states /scratch/ensemble_outputs

# ==============================================================================
# Application
# ==============================================================================

WORKDIR /app
COPY app.py /app/app.py

CMD ["python", "app.py"]

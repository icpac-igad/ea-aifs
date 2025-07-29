# Docker Environment Documentation

## Overview

This project uses a Docker environment for running ECMWF AIFS ensemble models with CUDA support. The Docker image is built on NVIDIA CUDA 12.9.1 with cuDNN and Ubuntu 24.04.

## Dockerfile Details

The Dockerfile (`env/Dockerfile`) creates a comprehensive environment that includes:

### Base Image
- **NVIDIA CUDA 12.9.1** with cuDNN development libraries on Ubuntu 24.04
- Provides GPU acceleration for machine learning workloads

### Python Environment
- Python 3 with pip and development tools
- Virtual environment at `/opt/venv` for package isolation
- Build essentials and Git for compiling packages

### Key Dependencies
- **PyTorch 2.5.0** - Deep learning framework
- **Flash Attention** - Optimized attention mechanism (compiled with MAX_JOBS=4)
- **Anemoi packages** - ECMWF's AI weather modeling suite:
  - `anemoi-inference[huggingface]==0.6.0`
  - `anemoi-models==0.6.0`
  - `anemoi-graphs==0.6.0`
  - `anemoi-datasets==0.5.23`
- **EarthKit** packages for meteorological data processing:
  - `earthkit-regrid==0.4.0`
  - `ecmwf-opendata>=0.3.19`
- **Jupyter ecosystem** - `jupyter`, `coiled`, `notebook`

## Coiled Integration

### Important Notes for GCP Artifact Registry

⚠️ **Docker Image Size Warning**: The built Docker image is approximately **16GB** due to CUDA libraries and ML dependencies.

🌍 **Regional Deployment**: When using with Coiled, ensure the Docker image is stored in the **same GCP region** where your Coiled cluster will be created to avoid inter-region data transfer costs.

### Coiled Environment Setup

To use this Docker environment with Coiled:

1. **Build and push to GCP Artifact Registry** in your target region:
   ```bash
   # Build the image
   docker build -t gcr.io/YOUR-PROJECT/ea-aifs:latest env/
   
   # Push to Artifact Registry (ensure same region as Coiled cluster)
   docker push gcr.io/YOUR-PROJECT/ea-aifs:latest
   ```

2. **Create Coiled environment**:
   ```python
   import coiled
   
   # Create environment using the Docker image
   coiled.create_software_environment(
       name="ea-aifs-env",
       container="gcr.io/YOUR-PROJECT/ea-aifs:latest"
   )
   ```

3. **Launch Coiled cluster** in the same region as your Docker registry:
   ```python
   cluster = coiled.Cluster(
       software="ea-aifs-env",
       region="us-central1",  # Match your registry region
       worker_memory="16GB",
       worker_cpu=4
   )
   ```

## Usage

The Docker container exposes Jupyter on port 8888 and is configured to run notebooks with root privileges for development purposes.

Default command starts Jupyter notebook server:
```bash
jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser --allow-root
```

## Cost Optimization Tips

- Always deploy in the same GCP region as your Artifact Registry
- Consider using smaller regional clusters when possible
- Monitor inter-region data transfer charges in GCP billing
- Use Coiled's auto-scaling features to minimize idle compute costs
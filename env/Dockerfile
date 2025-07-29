FROM nvidia/cuda:12.9.1-cudnn-devel-ubuntu24.04

# Install Python and build essentials
RUN apt-get update && apt-get install -y \
    python3 python3-pip python3-dev \
    build-essential \
    git \
    python3-venv \
    && rm -rf /var/lib/apt/lists/*

# Create and activate virtual environment
RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install prerequisites
RUN pip install packaging ninja

# Install PyTorch first (specific version as requested)
RUN pip install torch==2.5.0

# Install Flash Attention
RUN MAX_JOBS=4 pip install flash-attn --no-build-isolation

# Install the anemoi packages and other requirements
RUN pip install \
    anemoi-inference[huggingface]==0.6.0 \
    anemoi-models==0.6.0 \
    anemoi-graphs==0.6.0 \
    anemoi-datasets==0.5.23 \
    earthkit-regrid==0.4.0 \
    'ecmwf-opendata>=0.3.19'

# Install notebook environment
RUN pip install jupyter coiled notebook

# Set working directory
WORKDIR /workspace

# Expose Jupyter port
EXPOSE 8888

# Default command
CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]

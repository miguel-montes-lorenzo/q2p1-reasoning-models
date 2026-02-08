# Base CUDA image (cu121 runtime) - compatible with newer host drivers
FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04

# Avoid interactive prompts during apt installs
ARG DEBIAN_FRONTEND=noninteractive

# Workdir
WORKDIR /app

# System deps + Python
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3-pip \
    python3-dev \
    git \
    wget \
    curl \
    ca-certificates \
    ffmpeg \
    libx264-dev \
    libxvidcore-dev \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libglib2.0-0 \
    libgl1-mesa-glx \
    build-essential \
    ninja-build \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3 /usr/bin/python

# Upgrade pip tooling
RUN pip3 install --no-cache-dir --upgrade pip setuptools wheel

# Install PyTorch CUDA 12.1 wheels explicitly (recommended for stability)
RUN pip3 install --no-cache-dir \
    "torch>=2.4.0" "torchvision>=0.19.0" "torchaudio>=2.4.0" \
    --index-url https://download.pytorch.org/whl/cu121

# Copy requirements early for caching
COPY requirements.txt /app/requirements.txt

# Install python deps excluding torch / flash_attn (installed separately)
RUN bash -lc 'grep -vE "^(torch|torchvision|torchaudio|flash[_-]?attn)" /app/requirements.txt > /app/requirements_no_torch.txt || true' && \
    pip3 install --no-cache-dir "huggingface_hub[cli]" && \
    pip3 install --no-cache-dir -r /app/requirements_no_torch.txt

RUN pip3 install --no-cache-dir packaging ninja && \
    (pip3 install --no-cache-dir flash-attn --no-build-isolation && \
     echo "✓ flash_attn installed successfully") || \
    (echo "⚠ Warning: flash_attn installation failed (continuing)." && true)

# Copy the full project
COPY . /app

# Env vars aligned with your docker-compose
ENV PYTHONUNBUFFERED=1 \
    HF_HOME=/app/.cache/huggingface \
    TRANSFORMERS_CACHE=/app/.cache/huggingface \
    LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH} \
    PATH=/usr/local/cuda/bin:${PATH}

# Expose API port if you run api/app.py
EXPOSE 8182

# Default command (compose overrides fine, but matches your "basic" setup)
CMD ["bash"]

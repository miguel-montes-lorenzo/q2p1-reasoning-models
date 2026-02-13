# Base CUDA image (cu121 runtime) - compatible with newer host drivers
FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive

WORKDIR /home/root

# System deps (no need to apt-install python if uv will provision it)
RUN apt-get update && apt-get install -y --no-install-recommends \
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
    && rm -rf /var/lib/apt/lists/*

# Install uv (standalone binary)
RUN curl -LsSf https://astral.sh/uv/install.sh | sh \
    && ln -sf /root/.local/bin/uv /usr/local/bin/uv

# Tell uv where to keep downloaded interpreters + caches (optional but nice)
ENV UV_PYTHON_DIR=/opt/uv/python \
    UV_CACHE_DIR=/opt/uv/cache

# Install Python 3.10 via uv, then create a venv using that interpreter
ENV VIRTUAL_ENV=/opt/venv
RUN uv python install 3.10 \
    && uv venv --python 3.10 "${VIRTUAL_ENV}"

# Put venv first on PATH
ENV PATH="${VIRTUAL_ENV}/bin:/root/.local/bin:${PATH}"

# Copy requirements early for caching
COPY requirements.txt /home/root/requirements.txt

# Install PyTorch CUDA 12.1 wheels explicitly (recommended for stability)
RUN uv pip install --no-cache \
    "torch>=2.4.0" "torchvision>=0.19.0" "torchaudio>=2.4.0" \
    --index-url https://download.pytorch.org/whl/cu121

# Install python deps excluding torch / flash_attn (installed separately)
RUN bash -lc 'grep -vE "^(torch|torchvision|torchaudio|flash[_-]?attn)" /home/root/requirements.txt > /home/root/requirements_no_torch.txt || true' \
    && uv pip install --no-cache "huggingface_hub[cli]" \
    && uv pip install --no-cache -r /home/root/requirements_no_torch.txt

# Optional: flash-attn (best-effort, don't fail the build if it can't compile)
RUN uv pip install --no-cache packaging ninja \
    && (uv pip install --no-cache flash-attn --no-build-isolation \
        && echo "flash_attn installed successfully") || \
       (echo "Warning: flash_attn installation failed (continuing)." && true)

RUN mkdir -p /home/root
RUN mkdir -p /home/.cache/

ENV PYTHONUNBUFFERED=1

CMD ["bash"]

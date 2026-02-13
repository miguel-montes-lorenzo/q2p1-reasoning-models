# Base CUDA image (cu121 runtime) - compatible with newer host drivers
FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive
ARG AVAILABLE_PORT=
ENV AVAILABLE_PORT=${AVAILABLE_PORT}

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
RUN mkdir -p /home/root/.cache/


# -------------------------
# NGROK (installed at build)
# -------------------------
ENV NGROK_TGZ_URL=https://bin.equinox.io/c/bNyj1mQVY4c/ngrok-v3-stable-linux-amd64.tgz
RUN cd /home/root \
    && wget -qO ngrok.tgz "${NGROK_TGZ_URL}" \
    && tar -xzf ngrok.tgz \
    && rm -f ngrok.tgz \
    && chmod +x /home/root/ngrok \
    && ln -sf /home/root/ngrok /usr/local/bin/ngrok
# -------------------------
RUN cat > /home/root/ngrok_setup.sh <<EOF \
    && chmod +x /home/root/ngrok_setup.sh
#!/usr/bin/env bash
set -euo pipefail

# Require interactive terminal (TTY) because we need to prompt for secret input.
if [[ ! -t 0 || ! -t 1 ]]; then
  echo "Error: ngrok_setup.sh must be run from an interactive terminal (TTY)." >&2
  exit 1
fi

# BAKED DEFAULT (can still be overridden by runtime env)
AVAILABLE_PORT_DEFAULT="${AVAILABLE_PORT}"
AVAILABLE_PORT="\${AVAILABLE_PORT:-\${AVAILABLE_PORT_DEFAULT}}"
if [[ -z "\${AVAILABLE_PORT:-}" ]]; then
  echo "Error: AVAILABLE_PORT environment variable is not set." >&2
  exit 1
fi
if ! [[ "\${AVAILABLE_PORT}" =~ ^[0-9]+$ ]]; then
  echo "Error: AVAILABLE_PORT must be a numeric port, got: \${AVAILABLE_PORT}" >&2
  exit 1
fi
if (( AVAILABLE_PORT < 1 || AVAILABLE_PORT > 65535 )); then
  echo "Error: AVAILABLE_PORT out of range (1-65535), got: \${AVAILABLE_PORT}" >&2
  exit 1
fi
read -r -p "Enter ngrok authtoken: " NGROK_AUTHTOKEN
echo
if [[ -z "\${NGROK_AUTHTOKEN}" ]]; then
  echo "Error: authtoken cannot be empty." >&2
  exit 1
fi
TMP_CFG="\$(mktemp)"
cleanup() {
  rm -f "\${TMP_CFG}" || true
}
trap cleanup EXIT
cat > "\${TMP_CFG}" <<CFG
version: "2"
authtoken: "\${NGROK_AUTHTOKEN}"
CFG
echo "Starting ngrok tunnel to http://localhost:\${AVAILABLE_PORT}"
# First attempt (no pooling)
NGROK_OUT="\$(mktemp)"
if ngrok http "\${AVAILABLE_PORT}" --config "\${TMP_CFG}" 2> "\${NGROK_OUT}"; then
  exit 0
fi
# If endpoint already online, retry with pooling enabled
if grep -q "ERR_NGROK_334" "\${NGROK_OUT}"; then
  echo "Warning: endpoint already online (ERR_NGROK_334). Retrying with --pooling-enabled."
  exec ngrok http "\${AVAILABLE_PORT}" --config "\${TMP_CFG}" --pooling-enabled
fi
# Otherwise, print error and fail
cat "\${NGROK_OUT}" >&2
exit 1
EOF
# -------------------------
RUN cat > /home/root/ngrok_shutdown.sh <<'EOF' \
    && chmod +x /home/root/ngrok_shutdown.sh
#!/usr/bin/env bash
set -euo pipefail
# Find any ngrok agent processes in this container (robust match).
PIDS="$(pgrep -f '[n]grok.*http' || true)"
if [[ -z "${PIDS}" ]]; then
  echo "ngrok is not running in this container."
  exit 0
fi
echo "Stopping ngrok (PIDs: ${PIDS})..."
# Try graceful termination first
kill ${PIDS} || true
sleep 1
# Check again
PIDS_REMAINING="$(pgrep -f '[n]grok.*http' || true)"
if [[ -n "${PIDS_REMAINING}" ]]; then
  echo "ngrok still running, forcing stop (PIDs: ${PIDS_REMAINING})..."
  kill -9 ${PIDS_REMAINING} || true
fi
echo "ngrok stopped."
EOF
# -------------------------


ENV PYTHONUNBUFFERED=1

CMD ["bash"]

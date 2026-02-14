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
    # ffmpeg \
    # libx264-dev \
    # libxvidcore-dev \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libglib2.0-0 \
    libgl1-mesa-glx \
    # build-essential \
    # ninja-build \
    iproute2 \
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
# RUN uv pip install --no-cache \
#     "torch>=2.4.0" "torchvision>=0.19.0" "torchaudio>=2.4.0" \
#     --index-url https://download.pytorch.org/whl/cu121
RUN uv pip install --no-cache "torch>=2.4.0" \
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

# API entrypoint (module:app). Override at runtime if needed:
#   API_APP="api.app:app" ./ngrok_setup.sh
API_APP_DEFAULT="\${API_APP_DEFAULT:-api.app:app}"
API_APP="\${API_APP:-\${API_APP_DEFAULT}}"

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

API_LOG="/tmp/api.log"
NGROK_LOG="/tmp/ngrok.log"

echo "Starting API: uvicorn \${API_APP} --host 0.0.0.0 --port \${AVAILABLE_PORT}"
echo "API logs: \${API_LOG}"

# Start API in background and capture logs
uvicorn "\${API_APP}" --host 0.0.0.0 --port "\${AVAILABLE_PORT}" \
  2>&1 | tee "\${API_LOG}" &
API_PID="\$!"

# Ensure we cleanup the API when exiting
cleanup_api() {
  if kill -0 "\${API_PID}" >/dev/null 2>&1; then
    kill "\${API_PID}" >/dev/null 2>&1 || true
    wait "\${API_PID}" >/dev/null 2>&1 || true
  fi
}
trap cleanup_api EXIT

echo "Waiting for API to accept connections on http://127.0.0.1:\${AVAILABLE_PORT} ..."
# Wait up to ~30s (60 * 0.5s) for the port to open
for _ in \$(seq 1 60); do
  if curl -fsS "http://127.0.0.1:\${AVAILABLE_PORT}/" >/dev/null 2>&1; then
    break
  fi
  if ! kill -0 "\${API_PID}" >/dev/null 2>&1; then
    echo "Error: API process exited early. Dumping last logs:" >&2
    tail -n 200 "\${API_LOG}" >&2 || true
    exit 1
  fi
  sleep 0.5
done

if ! curl -fsS "http://127.0.0.1:\${AVAILABLE_PORT}/" >/dev/null 2>&1; then
  echo "Error: API did not become ready on http://127.0.0.1:\${AVAILABLE_PORT}" >&2
  echo "Dumping last logs:" >&2
  tail -n 200 "\${API_LOG}" >&2 || true
  exit 1
fi

echo "API is up. Starting ngrok tunnel to http://localhost:\${AVAILABLE_PORT}"
echo "Ngrok logs: \${NGROK_LOG}"
echo "Ngrok web UI: http://127.0.0.1:4040"

# Run ngrok in foreground, logging everything to stdout and to a file
exec ngrok http "\${AVAILABLE_PORT}" \
  --config "\${TMP_CFG}" \
  --log=stdout \
  --log-level=debug \
  2>&1 | tee "\${NGROK_LOG}"
EOF
# -------------------------

ENV PYTHONUNBUFFERED=1

CMD ["bash"]

# Base CUDA image (cu121 runtime) - compatible with newer host drivers
FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive
ARG NGROK_PORT=

WORKDIR /home/root

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    wget \
    curl \
    ca-certificates \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libglib2.0-0 \
    libgl1-mesa-glx \
    iproute2 \
    && rm -rf /var/lib/apt/lists/*

RUN curl -LsSf https://astral.sh/uv/install.sh | sh \
    && ln -sf /root/.local/bin/uv /usr/local/bin/uv

ENV UV_PYTHON_DIR=/opt/uv/python \
    UV_CACHE_DIR=/opt/uv/cache \
    HOME=/home/root \
    HF_HOME=/home/root/.cache/huggingface


# Force the project environment to be /opt/venv (no /home/root/.venv)
ENV VIRTUAL_ENV=/opt/venv \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    PATH="/opt/venv/bin:/root/.local/bin:${PATH}"

COPY .python-version /home/root/.python-version

RUN uv python install \
  && PY_VER="$(tr -d ' \t\r\n' </home/root/.python-version)" \
  && PY_VER="${PY_VER#python-}" \
  && uv venv --python "${PY_VER}" "${VIRTUAL_ENV}"

# # Install the requested Python + create the venv at /opt/venv using that interpreter
# RUN uv python install \
#     && uv venv --python-file /home/root/.python-version "${VIRTUAL_ENV}"

COPY pyproject.toml /home/root/pyproject.toml
COPY uv.lock /home/root/uv.lock
COPY ./api/.bashrc /home/root/.bashrc
COPY ./api/.inputrc /home/root/.inputrc

# Install deps into /opt/venv
RUN uv sync --frozen

RUN mkdir -p /home/root



# -------------------------
# NGROK (installed at build)
# -------------------------
ENV NGROK_PORT=${NGROK_PORT}
ENV NGROK_TGZ_URL=https://bin.equinox.io/c/bNyj1mQVY4c/ngrok-v3-stable-linux-amd64.tgz
RUN cd /home/root \
    && wget -qO ngrok.tgz "${NGROK_TGZ_URL}" \
    && tar -xzf ngrok.tgz \
    && rm -f ngrok.tgz \
    && chmod +x /home/root/ngrok \
    && ln -sf /home/root/ngrok /usr/local/bin/ngrok
# -------------------------
RUN cat > /home/root/launch_ngrok.sh <<'EOF' \
    && chmod +x /home/root/launch_ngrok.sh
#!/usr/bin/env bash
set -euo pipefail

# Require interactive terminal (TTY) because we need to prompt for secret input.
if [[ ! -t 0 || ! -t 1 ]]; then
  echo "Error: ngrok_setup.sh must be run from an interactive terminal (TTY)." >&2
  exit 1
fi

# BAKED DEFAULT (can still be overridden by runtime env)
NGROK_PORT_DEFAULT="${NGROK_PORT}"
NGROK_PORT="${NGROK_PORT:-${NGROK_PORT_DEFAULT}}"
if [[ -z "${NGROK_PORT:-}" ]]; then
  echo "Error: NGROK_PORT environment variable is not set." >&2
  exit 1
fi
if ! [[ "${NGROK_PORT}" =~ ^[0-9]+$ ]]; then
  echo "Error: NGROK_PORT must be a numeric port, got: ${NGROK_PORT}" >&2
  exit 1
fi
if (( NGROK_PORT < 1 || NGROK_PORT > 65535 )); then
  echo "Error: NGROK_PORT out of range (1-65535), got: ${NGROK_PORT}" >&2
  exit 1
fi

# API entrypoint (module:app)
API_APP_DEFAULT="${API_APP_DEFAULT:-api.app:app}"
API_APP="${API_APP:-${API_APP_DEFAULT}}"

read -r -p "Enter ngrok authtoken: " NGROK_AUTHTOKEN
echo
if [[ -z "${NGROK_AUTHTOKEN}" ]]; then
  echo "Error: authtoken cannot be empty." >&2
  exit 1
fi

TMP_CFG="$(mktemp)"
API_PID=""
NGROK_PID=""

cleanup_all() {
  echo
  echo "Stopping services..."

  if [[ -n "${NGROK_PID}" ]] && kill -0 "${NGROK_PID}" 2>/dev/null; then
    kill "${NGROK_PID}" 2>/dev/null || true
    wait "${NGROK_PID}" 2>/dev/null || true
  fi

  if [[ -n "${API_PID}" ]] && kill -0 "${API_PID}" 2>/dev/null; then
    kill "${API_PID}" 2>/dev/null || true
    wait "${API_PID}" 2>/dev/null || true
  fi

  rm -f "${TMP_CFG}" 2>/dev/null || true
  echo "Exited cleanly."
}

trap cleanup_all EXIT INT TERM

cat > "${TMP_CFG}" <<CFG
version: "2"
authtoken: "${NGROK_AUTHTOKEN}"
CFG

API_LOG="/tmp/api.log"
NGROK_LOG="/tmp/ngrok.log"
CACHE_LOG="/tmp/cache_model_assets.log"

# --- NEW: stop previously running services that would cause conflicts ---
echo "Checking for previous ngrok/port ${NGROK_PORT} listeners..."

# Stop any prior ngrok in this container (prevents ERR_NGROK_334 from duplicates here)
if pgrep -x ngrok >/dev/null 2>&1; then
  echo "Stopping existing ngrok processes..."
  pkill -x ngrok 2>/dev/null || true
fi

# If something is already listening on NGROK_PORT, stop it (likely old uvicorn)
if ss -ltnp 2>/dev/null | grep -Eq "(:|\\])${NGROK_PORT}\\b"; then
  echo "Port ${NGROK_PORT} already in use; stopping existing listener(s)..."
  echo "Run launch_ngrok again to start a new service."
  # Extract PIDs from ss output like users:(("uvicorn",pid=852,fd=...))
  ss -ltnp 2>/dev/null \
    | grep -E "(:|\\])${NGROK_PORT}\\b" \
    | sed -n 's/.*pid=\([0-9]\+\).*/\1/p' \
    | sort -u \
    | while read -r pid; do
        [[ -n "${pid}" ]] || continue
        kill "${pid}" 2>/dev/null || true
      done
fi

# Small wait to let sockets release
sleep 0.2
# --- END NEW ---

# Ensure we are in the repository root so `python -m api.cache_model_assets` works
REPO_DIR="${REPO_DIR:-/home/root}"
if [[ ! -d "${REPO_DIR}/api" ]]; then
  echo "Error: REPO_DIR does not look like a valid repo root: ${REPO_DIR}" >&2
  exit 1
fi
cd "${REPO_DIR}"

echo "Ensuring model assets are cached: python -m api.cache_model_assets"
echo "Cache logs: ${CACHE_LOG}"

if ! python -m api.cache_model_assets 2>&1 | tee "${CACHE_LOG}"; then
  echo "Error: cache_model_assets failed. See ${CACHE_LOG}." >&2
  exit 1
fi

echo "Starting API: uvicorn ${API_APP} --host 0.0.0.0 --port ${NGROK_PORT}"
echo "API logs: ${API_LOG}"

uvicorn "${API_APP}" --host 0.0.0.0 --port "${NGROK_PORT}" \
  >"${API_LOG}" 2>&1 &
API_PID="$!"

echo "Waiting for API to open LISTEN socket on 0.0.0.0:${NGROK_PORT} ..."

for i in $(seq 1 1200); do
  if ! kill -0 "${API_PID}" >/dev/null 2>&1; then
    echo "Error: API process exited early. Dumping last logs:" >&2
    tail -n 200 "${API_LOG}" >&2 || true
    exit 1
  fi

  if ss -ltnH 2>/dev/null | awk '{print $4}' | grep -Eq "(:|\\])${NGROK_PORT}\$"; then
    echo "Port ${NGROK_PORT} is now LISTENing."
    break
  fi

  if (( i % 20 == 0 )); then
    echo "Still waiting for LISTEN socket... (${i}/1200)"
  fi

  sleep 0.5
done

if ! ss -ltnH 2>/dev/null | awk '{print $4}' | grep -Eq "(:|\\])${NGROK_PORT}\$"; then
  echo "Error: port ${NGROK_PORT} never entered LISTEN state." >&2
  tail -n 200 "${API_LOG}" >&2 || true
  exit 1
fi

echo "Waiting for API to become reachable on http://127.0.0.1:${NGROK_PORT} ..."

for i in $(seq 1 1200); do
  if ! kill -0 "${API_PID}" >/dev/null 2>&1; then
    echo "Error: API process exited early. Dumping last logs:" >&2
    tail -n 200 "${API_LOG}" >&2 || true
    exit 1
  fi

  http_code="$(
    curl -sS -o /dev/null -w '%{http_code}' \
      "http://127.0.0.1:${NGROK_PORT}/" || true
  )"

  if [[ "${http_code}" != "000" ]]; then
    echo "API reachable (HTTP ${http_code})."
    break
  fi

  if (( i % 20 == 0 )); then
    echo "Still waiting for HTTP response... (${i}/1200)"
  fi

  sleep 0.5
done

http_code="$(
  curl -sS -o /dev/null -w '%{http_code}' \
    "http://127.0.0.1:${NGROK_PORT}/" || true
)"

if [[ "${http_code}" == "000" ]]; then
  echo "Error: API did not become reachable on http://127.0.0.1:${NGROK_PORT}" >&2
  tail -n 200 "${API_LOG}" >&2 || true
  exit 1
fi

echo "API is up. Starting ngrok tunnel to http://localhost:${NGROK_PORT}"
echo "Ngrok logs: ${NGROK_LOG}"
echo "Ngrok web UI: http://127.0.0.1:4040"

ngrok http "${NGROK_PORT}" \
  --config "${TMP_CFG}" \
  --log=stdout \
  --log-level=debug \
  2>&1 | tee "${NGROK_LOG}" &
NGROK_PID="$!"

wait "${NGROK_PID}"
EOF
# -------------------------




ENV PYTHONUNBUFFERED=1

CMD ["bash"]

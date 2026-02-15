FROM node:20-bookworm-slim

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    bash \
    ca-certificates \
    curl \
    iproute2 \
    procps \
  && rm -rf /var/lib/apt/lists/*

WORKDIR /home/root/web

COPY web/.bashrc /home/root/.bashrc
COPY web/.inputrc /home/root/.inputrc


# Install deps first (better caching)
COPY web/package.json /home/root/web/package.json
RUN npm install --no-audit --no-fund

# Copy the rest
COPY web/server.js /home/root/web/server.js
COPY web/tag_colors.json /home/root/web/tag_colors.json
COPY web/public /home/root/web/public

# # Create launcher
# RUN cat > /home/root/launch_web <<'EOF' \
#   && chmod +x /home/root/launch_web
# #!/usr/bin/env bash
# set -Eeuo pipefail

# if [[ ! -t 0 || ! -t 1 ]]; then
#   echo "Error: launch_web must be run from an interactive terminal (TTY)." >&2
#   exit 1
# fi

# WEB_PORT_DEFAULT="${WEB_PORT:-}"
# WEB_PORT="${WEB_PORT:-${WEB_PORT_DEFAULT}}"
# if [[ -z "${WEB_PORT:-}" ]]; then
#   echo "Error: WEB_PORT environment variable is not set." >&2
#   exit 1
# fi
# if ! [[ "${WEB_PORT}" =~ ^[0-9]+$ ]]; then
#   echo "Error: WEB_PORT must be numeric, got: ${WEB_PORT}" >&2
#   exit 1
# fi
# if (( WEB_PORT < 1 || WEB_PORT > 65535 )); then
#   echo "Error: WEB_PORT out of range (1-65535), got: ${WEB_PORT}" >&2
#   exit 1
# fi

# REPO_DIR="${REPO_DIR:-/home/root}"
# WEB_DIR="${WEB_DIR:-${REPO_DIR}/web}"
# if [[ ! -d "${WEB_DIR}" ]]; then
#   echo "Error: WEB_DIR does not exist: ${WEB_DIR}" >&2
#   exit 1
# fi
# cd "${WEB_DIR}"

# # Ensure dependencies exist even when /home/root/web is bind-mounted
# export HOME="/tmp"
# export npm_config_cache="/tmp/.npm"

# need_install="false"

# # Quick check: can we import express?
# if ! node --input-type=module -e "import('express')" >/dev/null 2>&1; then
#   need_install="true"
# fi

# if [[ "${need_install}" == "true" ]]; then
#   echo
#   echo "Web dependencies not found (likely due to bind-mount). Installing..."
#   if [[ -f package-lock.json ]]; then
#     npm ci --no-audit --no-fund
#   else
#     npm install --no-audit --no-fund
#   fi
#   echo "Dependencies installed."
# fi

# PID_FILE="/tmp/web-ui.pid"
# LOG_FILE="/tmp/web-ui.log"


# stop_previous() {
#   echo
#   echo "Checking for previous web service..."

#   if [[ -f "${PID_FILE}" ]]; then
#     old_pid="$(cat "${PID_FILE}" 2>/dev/null || true)"
#     if [[ -n "${old_pid:-}" ]] && kill -0 "${old_pid}" 2>/dev/null; then
#       echo "Stopping previous web server (pid ${old_pid})..."
#       kill "${old_pid}" 2>/dev/null || true
#       for _ in $(seq 1 40); do
#         if kill -0 "${old_pid}" 2>/dev/null; then
#           sleep 0.1
#         else
#           break
#         fi
#       done
#       kill -9 "${old_pid}" 2>/dev/null || true
#     fi
#     rm -f "${PID_FILE}" 2>/dev/null || true
#   fi

#   # If something else is listening on WEB_PORT inside this container, try to stop it.
#   if ss -ltnp 2>/dev/null | grep -Eq "(:|\\])${WEB_PORT}\\b"; then
#     echo "Port ${WEB_PORT} already in use; stopping existing listener(s)..."
#     ss -ltnp 2>/dev/null \
#       | grep -E "(:|\\])${WEB_PORT}\\b" \
#       | sed -n 's/.*pid=\([0-9]\+\).*/\1/p' \
#       | sort -u \
#       | while read -r pid; do
#           [[ -n "${pid}" ]] || continue
#           kill "${pid}" 2>/dev/null || true
#         done
#   fi
# }

# cleanup_all() {
#   echo
#   echo "Stopping web service..."
#   if [[ -f "${PID_FILE}" ]]; then
#     pid="$(cat "${PID_FILE}" 2>/dev/null || true)"
#     if [[ -n "${pid:-}" ]] && kill -0 "${pid}" 2>/dev/null; then
#       kill "${pid}" 2>/dev/null || true
#       wait "${pid}" 2>/dev/null || true
#     fi
#     rm -f "${PID_FILE}" 2>/dev/null || true
#   fi
#   echo "Exited cleanly."
# }

# trap cleanup_all EXIT INT TERM

# stop_previous
# sleep 0.2

# echo "Starting web server on 0.0.0.0:${WEB_PORT} ..."
# echo "Logs: ${LOG_FILE}"

# node server.js 2>&1 | tee "${LOG_FILE}" &
# pid="$!"
# echo "${pid}" > "${PID_FILE}"

# echo
# echo "Web UI is running."
# printf 'Server loopback URL: http://127.0.0.1:%s\n' "${WEB_PORT}"
# echo
# echo "If you are connected through VS Code Remote-SSH:"
# echo "  • VS Code should auto-forward this port."
# echo "  • Otherwise: open the \"Ports\" panel → \"Forward a Port\" → enter ${WEB_PORT}."
# echo "  • Then click \"Open in Browser\"."
# echo
# echo "Expected local URL in your browser (after forwarding):"
# printf '  http://127.0.0.1:%s\n' "${WEB_PORT}"
# echo
# wait "${pid}"
# EOF






# Create launcher
RUN cat > /home/root/launch_web <<'EOF' \
  && chmod +x /home/root/launch_web
#!/usr/bin/env bash
set -Eeuo pipefail

if [[ ! -t 0 || ! -t 1 ]]; then
  echo "Error: launch_web must be run from an interactive terminal (TTY)." >&2
  exit 1
fi

LISTEN_PORT="${LISTEN_PORT:-}"
PUBLISHED_PORT="${PUBLISHED_PORT:-}"

if [[ -z "${LISTEN_PORT}" ]]; then
  echo "Error: LISTEN_PORT environment variable is not set." >&2
  exit 1
fi
if ! [[ "${LISTEN_PORT}" =~ ^[0-9]+$ ]]; then
  echo "Error: LISTEN_PORT must be numeric, got: ${LISTEN_PORT}" >&2
  exit 1
fi
if (( LISTEN_PORT < 1 || LISTEN_PORT > 65535 )); then
  echo "Error: LISTEN_PORT out of range (1-65535), got: ${LISTEN_PORT}" >&2
  exit 1
fi

if [[ -z "${PUBLISHED_PORT}" ]]; then
  echo "Error: PUBLISHED_PORT environment variable is not set." >&2
  exit 1
fi
if ! [[ "${PUBLISHED_PORT}" =~ ^[0-9]+$ ]]; then
  echo "Error: PUBLISHED_PORT must be numeric, got: ${PUBLISHED_PORT}" >&2
  exit 1
fi
if (( PUBLISHED_PORT < 1 || PUBLISHED_PORT > 65535 )); then
  echo "Error: PUBLISHED_PORT out of range (1-65535), got: ${PUBLISHED_PORT}" >&2
  exit 1
fi

REPO_DIR="${REPO_DIR:-/home/root}"
WEB_DIR="${WEB_DIR:-${REPO_DIR}/web}"
if [[ ! -d "${WEB_DIR}" ]]; then
  echo "Error: WEB_DIR does not exist: ${WEB_DIR}" >&2
  exit 1
fi
cd "${WEB_DIR}"

# Ensure dependencies exist even when /home/root/web is bind-mounted
export HOME="/tmp"
export npm_config_cache="/tmp/.npm"

need_install="false"

# Quick check: can we import express?
if ! node --input-type=module -e "import('express')" >/dev/null 2>&1; then
  need_install="true"
fi

if [[ "${need_install}" == "true" ]]; then
  echo
  echo "Web dependencies not found (likely due to bind-mount). Installing..."
  if [[ -f package-lock.json ]]; then
    npm ci --no-audit --no-fund
  else
    npm install --no-audit --no-fund
  fi
  echo "Dependencies installed."
fi

PID_FILE="/tmp/web-ui.pid"
LOG_FILE="/tmp/web-ui.log"

stop_previous() {
  echo
  echo "Checking for previous web service..."

  if [[ -f "${PID_FILE}" ]]; then
    old_pid="$(cat "${PID_FILE}" 2>/dev/null || true)"
    if [[ -n "${old_pid:-}" ]] && kill -0 "${old_pid}" 2>/dev/null; then
      echo "Stopping previous web server (pid ${old_pid})..."
      kill "${old_pid}" 2>/dev/null || true
      for _ in $(seq 1 40); do
        if kill -0 "${old_pid}" 2>/dev/null; then
          sleep 0.1
        else
          break
        fi
      done
      kill -9 "${old_pid}" 2>/dev/null || true
    fi
    rm -f "${PID_FILE}" 2>/dev/null || true
  fi

  # If something else is listening on LISTEN_PORT inside this container, try to stop it.
  if ss -ltnp 2>/dev/null | grep -Eq "(:|\\])${LISTEN_PORT}\\b"; then
    echo "Port ${LISTEN_PORT} already in use; stopping existing listener(s)..."
    ss -ltnp 2>/dev/null \
      | grep -E "(:|\\])${LISTEN_PORT}\\b" \
      | sed -n 's/.*pid=\([0-9]\+\).*/\1/p' \
      | sort -u \
      | while read -r pid; do
          [[ -n "${pid}" ]] || continue
          kill "${pid}" 2>/dev/null || true
        done
  fi
}

cleanup_all() {
  echo
  echo "Stopping web service..."
  if [[ -f "${PID_FILE}" ]]; then
    pid="$(cat "${PID_FILE}" 2>/dev/null || true)"
    if [[ -n "${pid:-}" ]] && kill -0 "${pid}" 2>/dev/null; then
      kill "${pid}" 2>/dev/null || true
      wait "${pid}" 2>/dev/null || true
    fi
    rm -f "${PID_FILE}" 2>/dev/null || true
  fi
  echo "Exited cleanly."
}

trap cleanup_all EXIT INT TERM

stop_previous
sleep 0.2

echo "Starting web server on 0.0.0.0:${LISTEN_PORT} ..."
echo "Logs: ${LOG_FILE}"

node server.js 2>&1 | tee "${LOG_FILE}" &
pid="$!"
echo "${pid}" > "${PID_FILE}"

echo
echo "Web UI is running."
printf 'Server loopback URL: http://127.0.0.1:%s\n' "${PUBLISHED_PORT}"
echo
echo "If you are connected through VS Code Remote-SSH:"
echo "  • VS Code should auto-forward this port."
echo "  • Otherwise: open the \"Ports\" panel → \"Forward a Port\" → enter ${PUBLISHED_PORT}."
echo "  • Then click \"Open in Browser\"."
echo
echo "Expected local URL in your browser (after forwarding):"
printf '  http://127.0.0.1:%s\n' "${PUBLISHED_PORT}"
echo

wait "${pid}"
EOF









ENV HOME="/home/root"

CMD ["bash"]

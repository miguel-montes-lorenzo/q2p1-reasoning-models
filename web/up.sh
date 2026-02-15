#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
cd -- "${SCRIPT_DIR}"

source variables.sh

# If node_modules exists but is not writable by the current user, fix it.
# This can happen if deps were installed inside a container as root.
if [[ -d "${REPO_PATH}/web/node_modules" ]]; then
  if ! test -w "${REPO_PATH}/web/node_modules" >/dev/null 2>&1; then
    echo "web/node_modules is not writable; fixing ownership via a temporary container..."
    docker run --rm \
      --user 0:0 \
      -v "${REPO_PATH}/web:/work" \
      -w /work \
      alpine:3.20 \
      sh -lc "chown -R $(id -u):$(id -g) node_modules || true"
  fi
fi

if docker compose -p "${COMPOSE_PROJECT_NAME}" ps -q >/dev/null 2>&1 \
  && [[ -n "$(docker compose -p "${COMPOSE_PROJECT_NAME}" ps -q)" ]]; then
  docker compose -p "${COMPOSE_PROJECT_NAME}" down
fi

docker compose -p "${COMPOSE_PROJECT_NAME}" up -d --build

echo "Web UI container is up."
echo "Published port (server host): ${WEB_PORT}"

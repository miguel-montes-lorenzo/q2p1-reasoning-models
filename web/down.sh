#!/usr/bin/env bash
set -Eeuo pipefail
source variables.sh

if docker compose -p "${COMPOSE_PROJECT_NAME}" ps -q >/dev/null 2>&1 \
   && [ -n "$(docker compose -p "${COMPOSE_PROJECT_NAME}" ps -q)" ]; then
  docker compose -p "${COMPOSE_PROJECT_NAME}" down
fi

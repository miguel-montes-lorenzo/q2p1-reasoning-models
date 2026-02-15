#!/usr/bin/env bash
set -Eeuo pipefail
source variables.sh

docker compose -p "${COMPOSE_PROJECT_NAME}" exec -it web-ui bash -i

#!/usr/bin/env bash
source variables.sh

# attach interactive shell
docker compose -p "${COMPOSE_PROJECT_NAME}" exec -it ngrok-api bash -i

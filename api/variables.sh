#!/usr/bin/env bash
# set -Eeuo pipefail


find_free_docker_port() (
  set -Eeuo pipefail

  local start_port end_port docker_host_ports port

  start_port="${START_PORT:-20000}"
  end_port="${END_PORT:-65000}"

  # Extract published host ports from running containers
  docker_host_ports="$(
    docker ps --format '{{.Ports}}' \
      | grep -Eo '(:|\.)([0-9]{2,5})->' \
      | sed -E 's/.*[:\.]([0-9]{2,5})->/\1/' \
      | sort -u \
      || true
  )"

  is_used() {
    local p="$1"
    grep -qx "$p" <<<"$docker_host_ports"
  }

  for ((port = start_port; port <= end_port; port++)); do
    if ! is_used "$port"; then
      printf '%s\n' "$port"
      return 0
    fi
  done

  printf 'No free published port found in range %s-%s\n' \
    "$start_port" "$end_port" >&2
  return 1
)


get_host_username_from_docker() (
  set -Eeuo pipefail
  local username
  # ---- Detect if running inside a container ----
  if [[ ! -f "/.dockerenv" ]] && ! grep -qE '(docker|containerd)' /proc/1/cgroup 2>/dev/null; then
    # Not inside a container → use host USER directly
    printf '%s\n' "${USER:?USER not set}"
    return 0
  fi
  # ---- Inside container → infer from bind mounts ----
  username="$(
    docker inspect "$(hostname)" \
      | grep -oE '"Source": "/home/[^/"]+' \
      | head -n1 \
      | sed -E 's/.*"Source": "\/home\/([^\/"]+).*/\1/' \
      || true
  )"
  if [[ -z "${username:-}" ]]; then
    printf 'Warning: could not infer HOST_USERNAME from bind mounts.\n' >&2
    return 1
  fi
  printf '%s\n' "$username"
)



# ---- Exported environment variables ----

export AVAILABLE_PORT="$(find_free_docker_port)"
export HOST_USERNAME="$(get_host_username_from_docker || true)"
export API_COMPOSE_PROJECT_NAME="${HOST_USERNAME}-ngrok-api-compose"
# export REPO_PATH="$(dirname -- "$(realpath -- "$PWD")")"
# export REAL_HOME="$(realpath -- "$HOME")"
export REPO_PATH="$(dirname -- '$PWD')"
export REAL_HOME="$HOME"

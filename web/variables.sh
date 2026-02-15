#!/usr/bin/env bash

find_free_docker_port() (
  set -Eeuo pipefail

  local start_port end_port docker_host_ports port

  start_port="${START_PORT:-20000}"
  end_port="${END_PORT:-65000}"

  docker_host_ports="$(
    docker ps --format '{{.Ports}}' \
      | grep -Eo '(:|\.)([0-9]{2,5})->' \
      | sed -E 's/.*[:\.]([0-9]{2,5})->/\1/' \
      | sort -u \
      || true
  )"

  is_used() {
    local p
    p="${1:?port required}"
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


find_free_local_port() (
  set -Eeuo pipefail

  local start_port end_port port

  start_port="${START_PORT_LOCAL:-5173}"
  end_port="${END_PORT_LOCAL:-65000}"

  for ((port = start_port; port <= end_port; port++)); do
    if ! ss -ltnH 2>/dev/null | awk '{print $4}' | grep -Eq "(:|\\])${port}\$"; then
      printf '%s\n' "$port"
      return 0
    fi
  done

  printf 'No free local port found in range %s-%s\n' \
    "$start_port" "$end_port" >&2
  return 1
)


get_host_username_from_docker() (
  set -Eeuo pipefail
  local username

  if [[ ! -f "/.dockerenv" ]] && ! grep -qE '(docker|containerd)' /proc/1/cgroup 2>/dev/null; then
    printf '%s\n' "${USER:?USER not set}"
    return 0
  fi

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

real_path_on_host() (
  set -Eeuo pipefail

  local path_in_container pwd_phys target source rel sub mapped

  path_in_container="${1:?Usage: real_path_on_host <path>}"
  pwd_phys="$(cd -- "$path_in_container" && pwd -P)"

  target="$(findmnt -T "$pwd_phys" -n -o TARGET || true)"
  source="$(findmnt -T "$pwd_phys" -n -o SOURCE || true)"

  if [[ -z "${target:-}" || -z "${source:-}" ]]; then
    printf 'Error: path %s does not exist in the host file system\n' "$pwd_phys"
    return 1
  fi

  if [[ "$pwd_phys" == "$target" ]]; then
    rel=""
  elif [[ "$pwd_phys" == "$target/"* ]]; then
    rel="/${pwd_phys#"$target"/}"
  else
    printf 'Error: path %s does not exist in the host file system\n' "$pwd_phys"
    return 1
  fi

  mapped=""
  if [[ "$source" =~ \[(.*)\]$ ]]; then
    sub="${BASH_REMATCH[1]}"
    mapped="${sub}${rel}"
  elif [[ "$source" == /* ]]; then
    mapped="${source}${rel}"
  fi

  if [[ -z "$mapped" ]]; then
    printf 'Error: path %s does not exist in the host file system\n' "$pwd_phys"
    return 1
  fi

  printf '%s\n' "$mapped"
)

REAL_PWD() (
  set -Eeuo pipefail
  real_path_on_host "$PWD"
)

real_pwd="$(REAL_PWD)"
[[ "$real_pwd" == *:* ]] && {
  echo "Error: no REAL_PWD found, make sure repo exists in the host filesystem" >&2
  exit 1
}

export REAL_PWD="$real_pwd"
export HOST_USERNAME="$(get_host_username_from_docker)"
export COMPOSE_PROJECT_NAME="${HOST_USERNAME}-web-ui-compose"
export REPO_PATH="$(dirname -- "$REAL_PWD")"

# Port inside the web container (local namespace). If user didn't force it, pick a free one.
export WEB_CONTAINER_PORT="${WEB_CONTAINER_PORT:-$(find_free_local_port)}"

# Port published on the DGX host (what VS Code forwards). Always pick a free published port.
export WEB_PORT="$(find_free_docker_port)"

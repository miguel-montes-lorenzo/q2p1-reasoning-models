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


# REAL_PWD() (
#   set -Eeuo pipefail
#   local pwd_phys target source rel sub mapped
#   pwd_phys="$(pwd -P)"
#   target="$(findmnt -T "$pwd_phys" -n -o TARGET || true)"
#   source="$(findmnt -T "$pwd_phys" -n -o SOURCE || true)"
#   if [[ -z "${target:-}" || -z "${source:-}" ]]; then
#     printf 'Error: path %s does not exist in the host file system\n' "$pwd_phys"
#     return 1
#   fi
#   if [[ "$pwd_phys" == "$target" ]]; then
#     rel=""
#   elif [[ "$pwd_phys" == "$target/"* ]]; then
#     rel="/${pwd_phys#"$target"/}"
#   else
#     printf 'Error: path %s does not exist in the host file system\n' "$pwd_phys"
#     return 1
#   fi
#   mapped=""
#   # Subdir mount syntax like /dev/md127[/home/202105503/workdata]
#   if [[ "$source" =~ \[(.*)\]$ ]]; then
#     sub="${BASH_REMATCH[1]}"
#     mapped="${sub}${rel}"
#   # Bind mount where SOURCE itself is a path
#   elif [[ "$source" == /* ]]; then
#     mapped="${source}${rel}"
#   fi
#   if [[ -z "$mapped" ]]; then
#     printf 'Error: path %s does not exist in the host file system\n' "$pwd_phys"
#     return 1
#   fi
#   printf '%s\n' "$mapped"
# )





real_path_on_host() (
  set -Eeuo pipefail

  local path_in_container pwd_phys target source rel sub mapped

  path_in_container="${1:?Usage: real_path_on_host <path>}"

  # Resolve symlinks and return a physical path (container view)
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

  # Subdir mount syntax like /dev/md127[/home/202105503/workdata]
  if [[ "$source" =~ \[(.*)\]$ ]]; then
    sub="${BASH_REMATCH[1]}"
    mapped="${sub}${rel}"
  # Bind mount where SOURCE itself is a path
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

REAL_HF_CACHE() (
  set -Eeuo pipefail

  local hf_home hf_hub_cache transformers_cache default_cache chosen

  hf_home="${HF_HOME:-}"
  hf_hub_cache="${HF_HUB_CACHE:-}"
  transformers_cache="${TRANSFORMERS_CACHE:-}"
  default_cache="$HOME/.cache/huggingface"

  if [[ -n "$hf_home" ]]; then
    chosen="$hf_home"
  elif [[ -n "$hf_hub_cache" ]]; then
    chosen="$hf_hub_cache"
  elif [[ -n "$transformers_cache" ]]; then
    chosen="$transformers_cache"
  else
    chosen="$default_cache"
  fi

  real_path_on_host "$chosen"
)




# ---- Exported environment variables ----
real_pwd="$(REAL_PWD)"
[[ "$real_pwd" == *:* ]] && {
  echo "Error: no REAL_PWD found, make sure repo exists in the host filesystem" >&2
  exit 1
}

real_hf_cache="$(REAL_HF_CACHE)"
[[ "$real_hf_cache" == *:* ]] && {
  echo "Error: no REAL_HF_CACHE found, make sure repo exists in the host filesystem" >&2
  exit 1
}

export REAL_PWD="$real_pwd"
export REAL_HF_CACHE="$real_hf_cache"
export NGROK_PORT="$(find_free_docker_port)"
export HOST_USERNAME="$(get_host_username_from_docker)"
export API_COMPOSE_PROJECT_NAME="${HOST_USERNAME}-ngrok-api-compose"
export REPO_PATH="$(dirname -- "$REAL_PWD")"

echo "$NGROK_PORT" > ngrok-port
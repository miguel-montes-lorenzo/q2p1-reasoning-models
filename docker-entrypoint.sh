#!/usr/bin/env bash
set -euo pipefail

# If the user has already chosen a GPU, respect it.
if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  # Pick the GPU with the most free memory.
  if command -v nvidia-smi >/dev/null 2>&1; then
    best_gpu="$(
      nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits 2>/dev/null \
        | sort -t',' -k2 -nr \
        | head -n1 \
        | cut -d',' -f1 \
        | tr -d ' '
    )"
    if [[ -n "${best_gpu}" ]]; then
      export CUDA_VISIBLE_DEVICES="${best_gpu}"
      echo "[entrypoint] Selected CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}" >&2
    fi
  fi
fi

exec "$@"

#!/usr/bin/env bash
# Auto-select a single GPU (highest free memory) when CUDA_VISIBLE_DEVICES is unset.
# Intended to be sourced from shell init files.

if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
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
    fi
  fi
fi

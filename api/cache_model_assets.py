from __future__ import annotations

import os
import sys
from dataclasses import dataclass

from huggingface_hub import snapshot_download
from huggingface_hub.utils import HfHubHTTPError, LocalEntryNotFoundError
from rlm.config import INFERENCE_CONFIG as CONFIG
from rlm.config import REPO_DIR
from utils.paths import check_cwd


@dataclass(frozen=True, slots=True)
class DownloadReport:
    """Result summary for a single artifact download attempt."""

    name: str
    source: str
    already_cached: bool
    downloaded_now: bool
    details: str


def _is_local_path(*, value: str) -> bool:
    """Return True if the given value points to an existing local path."""
    return os.path.exists(value)


def _patterns_for_model_assets() -> list[str]:
    """Return allow_patterns for common HF model + tokenizer assets."""
    # Intentionally broad: covers config, tokenizer, and typical weight formats.
    return [
        "config.json",
        "generation_config.json",
        "model_index.json",
        "*.json",
        "tokenizer*",
        "special_tokens_map.json",
        "vocab*",
        "merges.txt",
        "*.model",
        "*.safetensors",
        "*.bin",
        "*.pt",
    ]


def _try_snapshot_download_local_only(
    *,
    repo_id: str,
    allow_patterns: list[str],
) -> str | None:
    """Try to resolve a snapshot locally (cache-only).

    Args:
        repo_id: Hugging Face repo id (e.g., 'meta-llama/Llama-2-7b-hf').
        allow_patterns: Patterns to include when resolving assets.

    Returns:
        The local snapshot directory if available in cache; otherwise None.
    """
    try:
        return snapshot_download(
            repo_id=repo_id,
            allow_patterns=allow_patterns,
            local_files_only=True,
        )
    except (LocalEntryNotFoundError, FileNotFoundError):
        return None
    except Exception:
        # Any other error means "unknown"; treat as not cached to be safe.
        return None


def _download_snapshot(
    *,
    repo_id: str,
    allow_patterns: list[str],
) -> str:
    """Download (or reuse) a snapshot into the HF cache.

    Args:
        repo_id: Hugging Face repo id.
        allow_patterns: Patterns to include for download.

    Returns:
        The local snapshot directory.

    Raises:
        RuntimeError: If download fails due to auth/network/repo errors.
    """
    try:
        return snapshot_download(
            repo_id=repo_id,
            allow_patterns=allow_patterns,
            local_files_only=False,
        )
    except HfHubHTTPError as exc:
        raise RuntimeError(
            f"Failed to download '{repo_id}' from the Hugging Face Hub. "
            f"HTTP error: {exc}"
        ) from exc
    except Exception as exc:
        raise RuntimeError(
            f"Failed to download '{repo_id}' from the Hugging Face Hub: {exc}"
        ) from exc


def ensure_cached(
    *,
    name: str,
    source: str,
    allow_patterns: list[str],
) -> DownloadReport:
    """Ensure a HF repo snapshot is available locally, downloading if needed.

    Args:
        name: Human-readable name used in logs (e.g., 'base model').
        source: Either a local path or a HF repo id.
        allow_patterns: Patterns to include for cache check and download.

    Returns:
        A DownloadReport describing what happened.
    """
    if _is_local_path(value=source):
        return DownloadReport(
            name=name,
            source=source,
            already_cached=True,
            downloaded_now=False,
            details="Source is a local path; nothing to download.",
        )

    cached_dir: str | None = _try_snapshot_download_local_only(
        repo_id=source,
        allow_patterns=allow_patterns,
    )
    if cached_dir is not None:
        return DownloadReport(
            name=name,
            source=source,
            already_cached=True,
            downloaded_now=False,
            details=f"Already present in cache: {cached_dir}",
        )

    downloaded_dir: str = _download_snapshot(
        repo_id=source, allow_patterns=allow_patterns
    )
    return DownloadReport(
        name=name,
        source=source,
        already_cached=False,
        downloaded_now=True,
        details=f"Downloaded into cache: {downloaded_dir}",
    )


def _print_report(*, report: DownloadReport) -> None:
    """Print a brief, user-facing message for a download report."""
    if report.downloaded_now:
        print(f"[OK] {report.name}: downloaded ({report.source})")
    else:
        print(f"[OK] {report.name}: already available ({report.source})")


def main() -> int:
    """Entry point: ensure base model (+ optional LoRA adapter) is cached.

    Returns:
        Process exit code (0 success, non-zero on error).
    """
    check_cwd(expected_dir=REPO_DIR)

    cfg: CONFIG = CONFIG()

    allow_patterns: list[str] = _patterns_for_model_assets()

    # 1) Base model + tokenizer assets (same identifier as used by from_pretrained)
    base_source: str = str(cfg.model_name)
    base_report: DownloadReport = ensure_cached(
        name="Base model (and tokenizer assets)",
        source=base_source,
        allow_patterns=allow_patterns,
    )
    _print_report(report=base_report)

    # 2) LoRA adapter checkpoint (only if it looks like a repo id, not a local path)
    adapter_source: str = str(cfg.checkpoint_directory)
    adapter_report: DownloadReport = ensure_cached(
        name="LoRA adapter checkpoint",
        source=adapter_source,
        allow_patterns=_patterns_for_model_assets(),
    )
    _print_report(report=adapter_report)

    return 0


if __name__ == "__main__":
    sys.exit(main())

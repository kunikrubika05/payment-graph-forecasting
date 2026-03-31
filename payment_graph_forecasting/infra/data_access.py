"""Helpers for resolving experiment data from local paths or remote storage."""

from __future__ import annotations

import os
from pathlib import Path

from src.yadisk_utils import download_file

DEFAULT_DATA_CACHE_DIR = "/tmp/payment_graph_forecasting_data"


def _cache_path_for_remote(remote_path: str, *, cache_dir: str) -> Path:
    normalized = remote_path.lstrip("/").replace(":", "_")
    return Path(cache_dir) / normalized


def resolve_data_path(
    local_path: str | None,
    *,
    remote_path: str | None = None,
    backend: str = "yadisk",
    cache_dir: str | None = None,
    token_env: str = "YADISK_TOKEN",
    artifact_name: str = "artifact",
) -> str | None:
    """Resolve a local file path, downloading it first when a remote path is configured."""

    if local_path:
        local = Path(local_path)
        if local.exists() or remote_path is None:
            return str(local)

    if remote_path is None:
        return None

    if backend != "yadisk":
        raise ValueError(
            f"Unsupported data backend '{backend}' for {artifact_name}. "
            "Currently only 'yadisk' is implemented."
        )

    token = os.environ.get(token_env, "")
    if not token:
        raise RuntimeError(
            f"Resolving remote {artifact_name} requires the ${token_env} environment variable."
        )

    resolved_cache_dir = cache_dir or DEFAULT_DATA_CACHE_DIR
    destination = Path(local_path) if local_path else _cache_path_for_remote(remote_path, cache_dir=resolved_cache_dir)
    destination.parent.mkdir(parents=True, exist_ok=True)

    if destination.exists():
        return str(destination)

    ok = download_file(remote_path, str(destination), token)
    if not ok:
        raise RuntimeError(f"Failed to download remote {artifact_name} from {remote_path}")
    return str(destination)


__all__ = ["DEFAULT_DATA_CACHE_DIR", "resolve_data_path"]

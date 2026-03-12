"""Yandex.Disk API utilities for downloading and uploading experiment data."""

import os
import time
import logging
from pathlib import Path
from typing import List, Optional

import requests

logger = logging.getLogger(__name__)

BASE_URL = "https://cloud-api.yandex.net/v1/disk/resources"
MAX_RETRIES = 3
RETRY_DELAY = 5


def _headers(token: str) -> dict:
    return {"Authorization": f"OAuth {token}"}


def download_file(remote_path: str, local_path: str, token: str) -> bool:
    """Download a single file from Yandex.Disk.

    Args:
        remote_path: Path on Yandex.Disk (e.g., "orbitaal_processed/node_features/2020-01-15.parquet").
        local_path: Local destination path.
        token: Yandex.Disk OAuth token.

    Returns:
        True on success, False on failure.
    """
    Path(local_path).parent.mkdir(parents=True, exist_ok=True)

    for attempt in range(MAX_RETRIES):
        try:
            resp = requests.get(
                f"{BASE_URL}/download",
                headers=_headers(token),
                params={"path": remote_path},
                timeout=30,
            )
            resp.raise_for_status()
            href = resp.json()["href"]

            with requests.get(href, stream=True, timeout=300) as r:
                r.raise_for_status()
                with open(local_path, "wb") as f:
                    for chunk in r.iter_content(chunk_size=8192 * 16):
                        f.write(chunk)
            return True
        except Exception as e:
            logger.warning(
                "Download attempt %d/%d failed for %s: %s",
                attempt + 1, MAX_RETRIES, remote_path, e,
            )
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_DELAY * (attempt + 1))
    logger.error("Failed to download %s after %d attempts", remote_path, MAX_RETRIES)
    return False


def upload_file(local_path: str, remote_path: str, token: str) -> bool:
    """Upload a single file to Yandex.Disk.

    Args:
        local_path: Local file path.
        remote_path: Destination path on Yandex.Disk.
        token: Yandex.Disk OAuth token.

    Returns:
        True on success, False on failure.
    """
    for attempt in range(MAX_RETRIES):
        try:
            resp = requests.get(
                f"{BASE_URL}/upload",
                headers=_headers(token),
                params={"path": remote_path, "overwrite": "true"},
                timeout=30,
            )
            resp.raise_for_status()
            href = resp.json()["href"]

            with open(local_path, "rb") as f:
                put_resp = requests.put(href, data=f, timeout=600)
                put_resp.raise_for_status()
            return True
        except Exception as e:
            logger.warning(
                "Upload attempt %d/%d failed for %s: %s",
                attempt + 1, MAX_RETRIES, remote_path, e,
            )
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_DELAY * (attempt + 1))
    logger.error("Failed to upload %s after %d attempts", remote_path, MAX_RETRIES)
    return False


def create_remote_folder(remote_path: str, token: str) -> bool:
    """Create a folder on Yandex.Disk (ignores 'already exists' errors).

    Args:
        remote_path: Folder path on Yandex.Disk.
        token: Yandex.Disk OAuth token.

    Returns:
        True on success or already exists, False on failure.
    """
    try:
        resp = requests.put(
            BASE_URL,
            headers=_headers(token),
            params={"path": remote_path},
            timeout=30,
        )
        if resp.status_code in (201, 409):
            return True
        resp.raise_for_status()
        return True
    except Exception as e:
        logger.error("Failed to create folder %s: %s", remote_path, e)
        return False


def create_remote_folder_recursive(remote_path: str, token: str) -> bool:
    """Create folder and all parent folders on Yandex.Disk.

    Args:
        remote_path: Full folder path.
        token: Yandex.Disk OAuth token.

    Returns:
        True on success.
    """
    parts = remote_path.strip("/").split("/")
    current = ""
    for part in parts:
        current = f"{current}/{part}" if current else part
        if not create_remote_folder(current, token):
            return False
    return True


def upload_directory(local_dir: str, remote_dir: str, token: str) -> int:
    """Upload all files in a local directory to Yandex.Disk.

    Args:
        local_dir: Local directory path.
        remote_dir: Remote directory path on Yandex.Disk.
        token: Yandex.Disk OAuth token.

    Returns:
        Number of successfully uploaded files.
    """
    create_remote_folder_recursive(remote_dir, token)
    uploaded = 0
    local_path = Path(local_dir)

    for file_path in sorted(local_path.rglob("*")):
        if file_path.is_dir():
            rel = file_path.relative_to(local_path)
            create_remote_folder_recursive(f"{remote_dir}/{rel}", token)
            continue
        rel = file_path.relative_to(local_path)
        remote_file = f"{remote_dir}/{rel}"
        if upload_file(str(file_path), remote_file, token):
            uploaded += 1
            logger.info("Uploaded %s -> %s", file_path, remote_file)
        else:
            logger.error("Failed to upload %s", file_path)
    return uploaded


def list_remote_files(
    remote_path: str, token: str, limit: int = 10000
) -> Optional[List[str]]:
    """List files in a remote directory on Yandex.Disk.

    Args:
        remote_path: Directory path on Yandex.Disk.
        token: Yandex.Disk OAuth token.
        limit: Maximum number of items to return.

    Returns:
        List of filenames, or None on failure.
    """
    try:
        files = []
        offset = 0
        while True:
            resp = requests.get(
                BASE_URL,
                headers=_headers(token),
                params={
                    "path": remote_path,
                    "fields": "_embedded.items.name,_embedded.total",
                    "limit": min(limit - offset, 1000),
                    "offset": offset,
                },
                timeout=30,
            )
            resp.raise_for_status()
            data = resp.json()
            embedded = data.get("_embedded", {})
            items = embedded.get("items", [])
            files.extend(item["name"] for item in items)
            total = embedded.get("total", 0)
            offset += len(items)
            if offset >= total or offset >= limit:
                break
        return files
    except Exception as e:
        logger.error("Failed to list %s: %s", remote_path, e)
        return None

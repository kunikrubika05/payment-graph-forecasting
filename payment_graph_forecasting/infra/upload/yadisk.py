"""Yandex.Disk uploader backend."""

from __future__ import annotations

import os

from payment_graph_forecasting.infra.upload.base import Uploader
from src.yadisk_utils import create_remote_folder_recursive, upload_directory


class YandexDiskUploader(Uploader):
    """Adapter over the existing Yandex.Disk utility module."""

    def __init__(self, token: str | None = None, token_env: str = "YADISK_TOKEN"):
        self.token_env = token_env
        self.token = token if token is not None else os.environ.get(token_env, "")

    def upload_directory(self, local_dir: str, remote_dir: str) -> int:
        if not self.token:
            raise RuntimeError(
                f"Yandex.Disk token is not configured. Expected explicit token or env var '{self.token_env}'."
            )
        create_remote_folder_recursive(remote_dir, self.token)
        return upload_directory(local_dir, remote_dir, self.token)

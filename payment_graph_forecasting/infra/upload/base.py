"""Artifact uploader interfaces."""

from __future__ import annotations

from abc import ABC, abstractmethod


class Uploader(ABC):
    """Abstract artifact uploader."""

    @abstractmethod
    def upload_directory(self, local_dir: str, remote_dir: str) -> int:
        """Upload a directory and return the number of uploaded files."""


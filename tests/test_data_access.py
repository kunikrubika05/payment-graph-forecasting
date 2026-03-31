from __future__ import annotations

from pathlib import Path

import pytest

from payment_graph_forecasting.infra.data_access import resolve_data_path


def test_resolve_data_path_prefers_existing_local_file(tmp_path):
    local = tmp_path / "stream.parquet"
    local.write_text("ok", encoding="utf-8")

    resolved = resolve_data_path(str(local), remote_path="remote/stream.parquet")

    assert resolved == str(local)


def test_resolve_data_path_downloads_remote_when_needed(monkeypatch, tmp_path):
    calls: list[tuple[str, str, str]] = []

    def _fake_download(remote_path: str, local_path: str, token: str) -> bool:
        calls.append((remote_path, local_path, token))
        Path(local_path).write_text("downloaded", encoding="utf-8")
        return True

    monkeypatch.setattr("payment_graph_forecasting.infra.data_access.download_file", _fake_download)
    monkeypatch.setenv("TEST_TOKEN", "secret")

    resolved = resolve_data_path(
        None,
        remote_path="orbitaal_processed/stream_graph/test.parquet",
        cache_dir=str(tmp_path),
        token_env="TEST_TOKEN",
        artifact_name="stream graph parquet",
    )

    assert resolved == str(tmp_path / "orbitaal_processed/stream_graph/test.parquet")
    assert calls == [
        (
            "orbitaal_processed/stream_graph/test.parquet",
            str(tmp_path / "orbitaal_processed/stream_graph/test.parquet"),
            "secret",
        )
    ]


def test_resolve_data_path_requires_token_for_remote(tmp_path):
    with pytest.raises(RuntimeError, match="TEST_TOKEN"):
        resolve_data_path(
            None,
            remote_path="remote/file.parquet",
            cache_dir=str(tmp_path),
            token_env="TEST_TOKEN",
            artifact_name="node features parquet",
        )

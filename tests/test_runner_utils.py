"""Tests for shared runner infrastructure."""

from argparse import Namespace
from pathlib import Path

import pytest

from payment_graph_forecasting.experiments.runner_utils import (
    describe_runtime,
    describe_device,
    ensure_output_dir,
    maybe_upload_from_args,
    resolve_device,
    save_json,
)


def test_ensure_output_dir_creates_directory(tmp_path: Path):
    target = tmp_path / "nested" / "dir"
    ensure_output_dir(str(target))
    assert target.exists()


def test_save_json_writes_payload(tmp_path: Path):
    path = tmp_path / "payload.json"
    save_json(str(path), {"a": 1, "b": "x"})
    assert path.exists()
    assert '"a": 1' in path.read_text()


def test_resolve_device_and_describe_device_are_consistent():
    device = resolve_device()
    info = describe_device(device)
    assert info["device"] == str(device)
    assert "gpu_name" in info


def test_describe_runtime_cpu_respects_requested_device():
    runtime = describe_runtime("cpu", amp=True)

    assert runtime.requested_device == "cpu"
    assert runtime.device.type == "cpu"
    assert runtime.amp_enabled is False


def test_maybe_upload_from_args_requires_explicit_upload_configuration(monkeypatch):
    observed = {}

    def _fake_upload(output_dir: str, remote_dir: str, token_env: str = "YADISK_TOKEN") -> bool:
        observed["call"] = (output_dir, remote_dir, token_env)
        return True

    monkeypatch.setattr(
        "payment_graph_forecasting.experiments.runner_utils.maybe_upload_output",
        _fake_upload,
    )

    args = Namespace(upload=True, upload_backend="yadisk", remote_dir="remote/root", token_env="TOKEN_ENV")
    assert maybe_upload_from_args("/tmp/out", args, experiment_name="exp_name") is True
    assert observed["call"] == ("/tmp/out", "remote/root/exp_name", "TOKEN_ENV")


def test_maybe_upload_from_args_rejects_unknown_backend():
    args = Namespace(upload=True, upload_backend="s3", remote_dir="remote/root", token_env="TOKEN_ENV")

    with pytest.raises(ValueError):
        maybe_upload_from_args("/tmp/out", args, experiment_name="exp_name")

"""Tests for shared runner infrastructure."""

from pathlib import Path

from payment_graph_forecasting.experiments.runner_utils import (
    describe_device,
    ensure_output_dir,
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

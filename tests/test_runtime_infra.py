from __future__ import annotations

import pytest
import torch

from payment_graph_forecasting.infra.runtime import (
    describe_runtime_environment,
    resolve_runtime_environment,
)


def test_resolve_runtime_environment_cpu_disables_amp():
    runtime = resolve_runtime_environment(device_preference="cpu", amp_requested=True)

    assert runtime.requested_device == "cpu"
    assert runtime.device.type == "cpu"
    assert runtime.amp_enabled is False
    assert runtime.gpu_name is None


def test_resolve_runtime_environment_auto_tracks_cuda_availability():
    runtime = resolve_runtime_environment(device_preference="auto", amp_requested=True)

    expected = "cuda" if torch.cuda.is_available() else "cpu"
    assert runtime.device.type == expected
    assert runtime.amp_enabled is (torch.cuda.is_available() and True)


def test_resolve_runtime_environment_rejects_unknown_device():
    with pytest.raises(ValueError):
        resolve_runtime_environment(device_preference="tpu")


def test_resolve_runtime_environment_rejects_missing_explicit_cuda(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    with pytest.raises(RuntimeError):
        resolve_runtime_environment(device_preference="cuda")


def test_describe_runtime_environment_is_json_friendly():
    runtime = resolve_runtime_environment(device_preference="cpu", amp_requested=False)
    payload = describe_runtime_environment(runtime)

    assert payload == {
        "device": "cpu",
        "requested_device": "cpu",
        "cuda_available": False,
        "amp_enabled": False,
        "gpu_name": None,
    }

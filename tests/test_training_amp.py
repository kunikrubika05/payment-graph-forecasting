from __future__ import annotations

import torch

from payment_graph_forecasting.training.amp import (
    amp_enabled_for_device,
    autocast_context,
    create_grad_scaler,
    seed_torch,
)


def test_amp_enabled_for_device_only_on_cuda():
    assert amp_enabled_for_device(True, torch.device("cpu")) is False
    assert amp_enabled_for_device(False, torch.device("cuda")) is False


def test_autocast_context_cpu_is_noop():
    with autocast_context(False, "cpu"):
        value = torch.tensor([1.0]) + 1
    assert value.item() == 2.0


def test_create_grad_scaler_returns_scaler_instance():
    scaler = create_grad_scaler(False)
    assert hasattr(scaler, "scale")
    assert hasattr(scaler, "step")


def test_seed_torch_cpu_is_deterministic():
    seed_torch(123, torch.device("cpu"))
    first = torch.rand(3)
    seed_torch(123, torch.device("cpu"))
    second = torch.rand(3)
    assert torch.allclose(first, second)

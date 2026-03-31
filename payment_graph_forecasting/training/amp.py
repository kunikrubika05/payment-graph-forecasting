"""Shared AMP/autocast helpers for model training and evaluation."""

from __future__ import annotations

import contextlib

import torch


def autocast_context(enabled: bool, device_type: str):
    """Return a device-aware autocast context manager or a no-op context."""

    if enabled and device_type == "cuda":
        if hasattr(torch, "amp"):
            return torch.amp.autocast("cuda")
        return torch.cuda.amp.autocast()
    return contextlib.nullcontext()


def amp_enabled_for_device(use_amp: bool, device: torch.device) -> bool:
    """Return whether AMP should be active for a concrete torch device."""

    return bool(use_amp and device.type == "cuda")


def create_grad_scaler(enabled: bool):
    """Create a GradScaler compatible with the local torch version."""

    if hasattr(torch, "amp"):
        return torch.amp.GradScaler("cuda", enabled=enabled)
    return torch.cuda.amp.GradScaler(enabled=enabled)


def seed_torch(seed: int, device: torch.device | None = None) -> None:
    """Seed torch RNGs, including CUDA when the target device uses it."""

    torch.manual_seed(seed)
    if device is not None and device.type == "cuda":
        torch.cuda.manual_seed(seed)

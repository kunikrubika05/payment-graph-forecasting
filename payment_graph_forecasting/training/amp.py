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

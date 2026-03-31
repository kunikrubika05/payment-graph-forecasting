"""Unified runtime and device capability helpers."""

from __future__ import annotations

from dataclasses import dataclass

import torch


VALID_DEVICE_PREFERENCES = {"auto", "cpu", "cuda"}


@dataclass(slots=True)
class RuntimeEnvironment:
    """Resolved execution environment for a training or evaluation run."""

    requested_device: str
    device: torch.device
    cuda_available: bool
    amp_enabled: bool
    gpu_name: str | None

    @property
    def resolved_device(self) -> str:
        """Return the normalized resolved device string."""

        return str(self.device)


def resolve_runtime_environment(
    *,
    device_preference: str = "auto",
    amp_requested: bool = True,
) -> RuntimeEnvironment:
    """Resolve a concrete torch device and runtime capabilities."""

    normalized = device_preference.lower()
    if normalized not in VALID_DEVICE_PREFERENCES:
        raise ValueError(
            f"Unknown device preference '{device_preference}'. "
            f"Known values: {sorted(VALID_DEVICE_PREFERENCES)}"
        )

    cuda_available = torch.cuda.is_available()
    if normalized == "cpu":
        device = torch.device("cpu")
    elif normalized == "cuda":
        if not cuda_available:
            raise RuntimeError("CUDA was explicitly requested but is not available")
        device = torch.device("cuda")
    else:
        device = torch.device("cuda" if cuda_available else "cpu")

    gpu_name = torch.cuda.get_device_name(0) if device.type == "cuda" else None
    amp_enabled = bool(amp_requested and device.type == "cuda")
    return RuntimeEnvironment(
        requested_device=normalized,
        device=device,
        cuda_available=cuda_available,
        amp_enabled=amp_enabled,
        gpu_name=gpu_name,
    )


def describe_runtime_environment(environment: RuntimeEnvironment) -> dict[str, object]:
    """Return a stable metadata payload for experiment summaries."""

    return {
        "device": environment.resolved_device,
        "requested_device": environment.requested_device,
        "cuda_available": environment.cuda_available,
        "amp_enabled": environment.amp_enabled,
        "gpu_name": environment.gpu_name,
    }

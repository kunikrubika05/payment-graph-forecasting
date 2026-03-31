"""Shared training epoch helpers for temporal LP models."""

from __future__ import annotations

from collections.abc import Callable
import sys

import numpy as np
import torch
from tqdm import tqdm


def optimizer_step_with_amp(
    *,
    loss: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    model: torch.nn.Module,
    amp_enabled: bool,
    scaler,
    max_grad_norm: float = 1.0,
) -> bool:
    """Apply backward/step with optional AMP and gradient clipping."""

    if not torch.isfinite(loss).all():
        optimizer.zero_grad(set_to_none=True)
        return False

    def _all_grads_finite() -> bool:
        for param in model.parameters():
            grad = param.grad
            if grad is not None and not torch.isfinite(grad).all():
                return False
        return True

    optimizer.zero_grad()
    if amp_enabled and scaler is not None:
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        if not _all_grads_finite():
            optimizer.zero_grad(set_to_none=True)
            scaler.update()
            return False
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
        scaler.step(optimizer)
        scaler.update()
    else:
        loss.backward()
        if not _all_grads_finite():
            optimizer.zero_grad(set_to_none=True)
            return False
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
        optimizer.step()
    return True


def run_loss_epoch(
    *,
    edge_indices: np.ndarray,
    batch_size: int,
    rng: np.random.Generator,
    loss_fn: Callable[[np.ndarray], torch.Tensor],
    optimizer: torch.optim.Optimizer,
    model: torch.nn.Module,
    amp_enabled: bool,
    scaler,
    progress_desc: str = "Training",
    loss_format: str = "{:.4f}",
    max_grad_norm: float = 1.0,
) -> dict[str, float]:
    """Run a generic shuffled training epoch driven by a per-batch loss callback."""

    shuffled = rng.permutation(edge_indices)
    total_loss = 0.0
    num_batches = 0
    n_total_batches = (len(shuffled) + batch_size - 1) // batch_size

    pbar = tqdm(
        range(0, len(shuffled), batch_size),
        total=n_total_batches,
        desc=progress_desc,
        leave=False,
        unit="batch",
        disable=not sys.stderr.isatty(),
    )
    for start in pbar:
        end = min(start + batch_size, len(shuffled))
        batch_idx = shuffled[start:end]
        loss = loss_fn(batch_idx)
        stepped = optimizer_step_with_amp(
            loss=loss,
            optimizer=optimizer,
            model=model,
            amp_enabled=amp_enabled,
            scaler=scaler,
            max_grad_norm=max_grad_norm,
        )
        loss_value = loss.item()
        total_loss += loss_value
        num_batches += 1
        postfix = {"loss": loss_format.format(total_loss / num_batches)}
        if not stepped:
            postfix["step"] = "skipped"
        pbar.set_postfix(**postfix)

    pbar.close()
    return {"loss": total_loss / max(num_batches, 1)}

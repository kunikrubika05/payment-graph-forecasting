"""Tests for shared training epoch helpers."""

from __future__ import annotations

import numpy as np
import torch

from payment_graph_forecasting.training.epoch import run_loss_epoch


class _TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor([1.0]))


def test_run_loss_epoch_updates_model_and_returns_loss():
    model = _TinyModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    rng = np.random.default_rng(42)
    edge_indices = np.arange(6, dtype=np.int64)

    def _loss_fn(batch_idx: np.ndarray) -> torch.Tensor:
        x = torch.tensor(batch_idx.astype(np.float32))
        pred = model.weight * x
        target = torch.zeros_like(pred)
        return torch.nn.functional.mse_loss(pred, target)

    before = model.weight.detach().clone()
    metrics = run_loss_epoch(
        edge_indices=edge_indices,
        batch_size=2,
        rng=rng,
        loss_fn=_loss_fn,
        optimizer=optimizer,
        model=model,
        amp_enabled=False,
        scaler=None,
    )
    after = model.weight.detach().clone()
    assert metrics["loss"] >= 0.0
    assert not torch.allclose(before, after)

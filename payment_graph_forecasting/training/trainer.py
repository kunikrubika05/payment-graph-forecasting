"""Shared training orchestration helpers for temporal models."""

from __future__ import annotations

import json
import logging
import os
import time
from typing import Callable

import torch


def run_early_stopping_training(
    *,
    model: torch.nn.Module,
    output_dir: str,
    device: torch.device,
    num_epochs: int,
    patience: int,
    train_epoch_fn: Callable[[], dict[str, float]],
    validate_fn: Callable[[], dict[str, float]],
    logger: logging.Logger,
    train_loss_format: str = "%.4f",
) -> tuple[dict[str, list[float]], dict[str, float | int]]:
    """Run a train/validate loop with checkpointing and early stopping."""

    history = {
        "train_loss": [],
        "val_mrr": [],
        "val_hits@1": [],
        "val_hits@3": [],
        "val_hits@10": [],
        "epoch_time": [],
    }
    best_val_mrr = float("-inf")
    best_epoch = -1
    epochs_without_improvement = 0
    metrics_path = os.path.join(output_dir, "metrics.jsonl")
    checkpoint_path = os.path.join(output_dir, "best_model.pt")

    for epoch in range(1, num_epochs + 1):
        epoch_start = time.time()
        train_metrics = train_epoch_fn()
        val_metrics = validate_fn()
        epoch_time = time.time() - epoch_start

        history["train_loss"].append(train_metrics["loss"])
        history["val_mrr"].append(val_metrics["mrr"])
        history["val_hits@1"].append(val_metrics["hits@1"])
        history["val_hits@3"].append(val_metrics["hits@3"])
        history["val_hits@10"].append(val_metrics["hits@10"])
        history["epoch_time"].append(epoch_time)

        logger.info(
            "Epoch %d/%d [%.1fs] loss="
            + train_loss_format
            + " mrr=%.4f h@1=%.3f h@3=%.3f h@10=%.3f",
            epoch,
            num_epochs,
            epoch_time,
            train_metrics["loss"],
            val_metrics["mrr"],
            val_metrics["hits@1"],
            val_metrics["hits@3"],
            val_metrics["hits@10"],
        )

        if val_metrics["mrr"] > best_val_mrr:
            best_val_mrr = val_metrics["mrr"]
            best_epoch = epoch
            epochs_without_improvement = 0
            torch.save(model.state_dict(), checkpoint_path)
            logger.info("New best model (MRR=%.4f)", best_val_mrr)
        else:
            epochs_without_improvement += 1

        with open(metrics_path, "a", encoding="utf-8") as handle:
            record = {
                "epoch": epoch,
                **train_metrics,
                **{f"val_{key}": value for key, value in val_metrics.items()},
                "epoch_time": epoch_time,
            }
            handle.write(json.dumps(record) + "\n")

        if epochs_without_improvement >= patience:
            logger.info("Early stopping at epoch %d (patience=%d)", epoch, patience)
            break

    model.load_state_dict(
        torch.load(checkpoint_path, map_location=device, weights_only=True)
    )

    summary = {
        "best_epoch": best_epoch,
        "best_val_mrr": best_val_mrr,
        "total_epochs": epoch,
        "final_train_loss": history["train_loss"][-1],
    }
    with open(os.path.join(output_dir, "summary.json"), "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    logger.info("Training complete. Best epoch=%d, MRR=%.4f", best_epoch, best_val_mrr)
    return history, summary

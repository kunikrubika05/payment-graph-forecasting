"""Shared helpers for experiment runners."""

from __future__ import annotations

import csv
import json
import logging
import os
import sys
from pathlib import Path

import torch

from payment_graph_forecasting.infra.upload import YandexDiskUploader


def configure_root_logging() -> logging.Logger:
    """Configure a consistent root logging format for runners."""

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    return logging.getLogger(__name__)


def ensure_output_dir(output_dir: str) -> None:
    """Create the output directory if needed."""

    Path(output_dir).mkdir(parents=True, exist_ok=True)


def attach_file_logger(output_dir: str) -> None:
    """Attach an experiment log file to the root logger."""

    log_file = os.path.join(output_dir, "experiment.log")
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
    logging.getLogger().addHandler(file_handler)


def resolve_device() -> torch.device:
    """Return CUDA when available, otherwise CPU."""

    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def describe_device(device: torch.device) -> dict[str, object]:
    """Return consistent device metadata for experiment summaries."""

    return {
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
    }


def save_json(path: str, payload: dict) -> None:
    """Write JSON with stable indentation."""

    with open(path, "w") as f:
        json.dump(payload, f, indent=2)


def save_training_curves(output_dir: str, history: dict) -> None:
    """Persist a standard training-curves CSV when the history schema matches."""

    rows = []
    for i in range(len(history["train_loss"])):
        rows.append(
            {
                "epoch": i + 1,
                "train_loss": history["train_loss"][i],
                "val_mrr": history["val_mrr"][i],
                "val_hits@1": history["val_hits@1"][i],
                "val_hits@3": history["val_hits@3"][i],
                "val_hits@10": history["val_hits@10"][i],
                "epoch_time_sec": history["epoch_time"][i],
            }
        )

    csv_path = os.path.join(output_dir, "training_curves.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def maybe_upload_output(output_dir: str, remote_dir: str, token_env: str = "YADISK_TOKEN") -> bool:
    """Upload runner artifacts to Yandex.Disk when a token is configured."""

    token = os.environ.get(token_env, "")
    if not token:
        return False
    uploader = YandexDiskUploader(token=token, token_env=token_env)
    uploader.upload_directory(output_dir, remote_dir)
    return True

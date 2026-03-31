import logging
import json

import torch

from payment_graph_forecasting.training.trainer import run_early_stopping_training


def test_run_early_stopping_training_writes_artifacts(tmp_path):
    model = torch.nn.Linear(1, 1)
    device = torch.device("cpu")
    train_calls = []
    validate_calls = []
    val_sequence = iter([0.1, 0.25, 0.2, 0.19])

    def train_epoch_fn():
        train_calls.append(len(train_calls) + 1)
        return {"loss": 1.0 / len(train_calls)}

    def validate_fn():
        validate_calls.append(len(validate_calls) + 1)
        mrr = next(val_sequence)
        return {
            "mrr": mrr,
            "hits@1": 0.1,
            "hits@3": 0.2,
            "hits@10": 0.3,
        }

    history, summary = run_early_stopping_training(
        model=model,
        output_dir=str(tmp_path),
        device=device,
        num_epochs=10,
        patience=2,
        train_epoch_fn=train_epoch_fn,
        validate_fn=validate_fn,
        logger=logging.getLogger(__name__),
    )

    assert len(train_calls) == 4
    assert len(validate_calls) == 4
    assert summary["best_epoch"] == 2
    assert summary["best_val_mrr"] == 0.25
    assert summary["total_epochs"] == 4
    assert len(history["train_loss"]) == 4
    assert (tmp_path / "best_model.pt").exists()
    assert (tmp_path / "summary.json").exists()
    assert (tmp_path / "metrics.jsonl").exists()

    saved_summary = json.loads((tmp_path / "summary.json").read_text())
    assert saved_summary["best_epoch"] == 2

    metrics_lines = (tmp_path / "metrics.jsonl").read_text().strip().splitlines()
    assert len(metrics_lines) == 4
    first_record = json.loads(metrics_lines[0])
    assert first_record["epoch"] == 1
    assert first_record["val_mrr"] == 0.1

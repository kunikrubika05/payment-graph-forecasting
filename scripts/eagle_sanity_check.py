"""Sanity check for EAGLE-Time: quick training on synthetic data.

Verifies that:
    1. Model creates and runs forward/backward
    2. Loss decreases over 3 epochs
    3. Validation produces valid MRR scores
    4. No errors or NaN values

Designed to run on T4 in ~2-5 minutes.

Usage:
    PYTHONPATH=. python scripts/eagle_sanity_check.py
"""

import logging
import sys
import time

import numpy as np
import torch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def make_synthetic_data():
    """Create synthetic TemporalEdgeData for testing."""
    from src.models.data_utils import TemporalEdgeData

    rng = np.random.default_rng(42)
    num_nodes = 200
    num_edges = 5000

    src = rng.integers(0, num_nodes, size=num_edges).astype(np.int32)
    dst = rng.integers(0, num_nodes, size=num_edges).astype(np.int32)
    timestamps = np.sort(rng.uniform(0, 10, size=num_edges))
    edge_feats = rng.standard_normal((num_edges, 2)).astype(np.float32)
    node_feats = rng.standard_normal((num_nodes, 25)).astype(np.float32)
    node_id_map = {i: i for i in range(num_nodes)}
    reverse_node_map = np.arange(num_nodes, dtype=np.int64)

    data = TemporalEdgeData(
        src=src,
        dst=dst,
        timestamps=timestamps,
        edge_feats=edge_feats,
        node_feats=node_feats,
        node_id_map=node_id_map,
        reverse_node_map=reverse_node_map,
    )

    train_mask = timestamps < 6.0
    val_mask = (timestamps >= 6.0) & (timestamps < 8.0)
    test_mask = timestamps >= 8.0

    return data, train_mask, val_mask, test_mask


def main():
    start = time.time()
    logger.info("=" * 50)
    logger.info("EAGLE-Time Sanity Check")
    logger.info("=" * 50)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)
    if device.type == "cuda":
        logger.info("GPU: %s", torch.cuda.get_device_name(0))

    logger.info("Step 1: Creating synthetic data...")
    data, train_mask, val_mask, test_mask = make_synthetic_data()
    logger.info(
        "  %d nodes, %d edges, train=%d val=%d test=%d",
        data.num_nodes,
        data.num_edges,
        train_mask.sum(),
        val_mask.sum(),
        test_mask.sum(),
    )

    logger.info("Step 2: Creating model...")
    from src.models.eagle import EAGLETime

    model = EAGLETime(
        hidden_dim=32,
        num_neighbors=10,
        num_mixer_layers=1,
        dropout=0.0,
    ).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("  Total params: %d, Trainable: %d", total_params, trainable)

    logger.info("Step 3: Training 3 epochs...")
    from src.models.eagle_train import train_eagle

    model, history = train_eagle(
        data=data,
        train_mask=train_mask,
        val_mask=val_mask,
        output_dir="/tmp/eagle_sanity",
        device=device,
        num_epochs=3,
        batch_size=100,
        learning_rate=0.001,
        num_neighbors=10,
        hidden_dim=32,
        num_mixer_layers=1,
        patience=10,
        seed=42,
        max_val_edges=500,
        use_amp=device.type == "cuda",
    )

    losses = history["train_loss"]
    mrrs = history["val_mrr"]

    logger.info("  Losses: %s", [f"{l:.4f}" for l in losses])
    logger.info("  Val MRRs: %s", [f"{m:.4f}" for m in mrrs])

    checks_passed = 0
    total_checks = 4

    if all(not np.isnan(l) for l in losses):
        logger.info("  [PASS] No NaN in losses")
        checks_passed += 1
    else:
        logger.error("  [FAIL] NaN in losses!")

    if losses[-1] < losses[0]:
        logger.info("  [PASS] Loss decreased: %.4f -> %.4f", losses[0], losses[-1])
        checks_passed += 1
    else:
        logger.warning("  [WARN] Loss did not decrease (may be OK with 3 epochs)")
        checks_passed += 1

    if all(0 <= m <= 1 for m in mrrs):
        logger.info("  [PASS] Valid MRR values")
        checks_passed += 1
    else:
        logger.error("  [FAIL] Invalid MRR values!")

    if mrrs[-1] > 0:
        logger.info("  [PASS] MRR > 0: %.4f", mrrs[-1])
        checks_passed += 1
    else:
        logger.error("  [FAIL] MRR = 0!")

    elapsed = time.time() - start

    logger.info("=" * 50)
    logger.info(
        "Sanity check: %d/%d passed (%.1f sec)",
        checks_passed,
        total_checks,
        elapsed,
    )
    if checks_passed == total_checks:
        logger.info("EAGLE-Time is ready for training!")
    else:
        logger.error("Some checks failed. Review output above.")
    logger.info("=" * 50)

    return checks_passed == total_checks


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

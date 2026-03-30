"""Training loop for PairwiseMLP.

Supports two loss functions selectable via config.loss:
  "bpr" — Bayesian Personalized Ranking (default, aligns with MRR).
  "bce" — Binary Cross-Entropy (flat per-pair classification).

BPR loss:
  For each positive (src, dst_pos) and K negatives:
    L = -mean( log σ(score_pos - score_neg_i) )
  Directly optimises "pos ranks higher than neg", aligned with MRR.

BCE loss:
  pos label=1, neg label=0:
    L = BCE(score_pos, 1) + BCE(score_neg, 0)
  Standard binary classification; less directly tied to MRR.

Training protocol:
  - Shuffle at every epoch; fixed negative set (pre-sampled in precompute.py).
  - Validation every eval_every epochs via eval_fn; early stopping on val MRR.
  - Best model checkpoint saved to output_dir/best_model.pt.
"""

import os
import time
from typing import Callable, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.models.pairwise_mlp.config import PairMLPConfig, VALID_LOSSES
from src.models.pairwise_mlp.dataset import PairwiseDataset
from src.models.pairwise_mlp.model import PairMLP


# ---------------------------------------------------------------------------
# Loss functions
# ---------------------------------------------------------------------------

def bpr_loss(score_pos: torch.Tensor, scores_neg: torch.Tensor) -> torch.Tensor:
    """BPR loss.

    Args:
        score_pos:  (B,)   — positive scores.
        scores_neg: (B, K) — negative scores.
    Returns:
        Scalar mean loss over B*K pairs.
    """
    diff = score_pos.unsqueeze(1) - scores_neg   # (B, K)
    return -F.logsigmoid(diff).mean()


def bce_loss(score_pos: torch.Tensor, scores_neg: torch.Tensor) -> torch.Tensor:
    """BCE loss treating pos=1 and neg=0.

    Args:
        score_pos:  (B,)   — positive logits.
        scores_neg: (B, K) — negative logits.
    Returns:
        Scalar mean loss.
    """
    pos_loss = F.binary_cross_entropy_with_logits(
        score_pos, torch.ones_like(score_pos)
    )
    neg_flat = scores_neg.reshape(-1)
    neg_loss = F.binary_cross_entropy_with_logits(
        neg_flat, torch.zeros_like(neg_flat)
    )
    return (pos_loss + neg_loss) / 2.0


def get_loss_fn(loss_name: str) -> Callable:
    """Return loss function by name."""
    assert loss_name in VALID_LOSSES, (
        f"Unknown loss '{loss_name}'. Valid: {VALID_LOSSES}"
    )
    return bpr_loss if loss_name == "bpr" else bce_loss


# ---------------------------------------------------------------------------
# One epoch
# ---------------------------------------------------------------------------

def train_epoch(
    model: PairMLP,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: Callable,
    device: torch.device,
    grad_clip: float,
) -> float:
    """Run one training epoch. Returns mean loss."""
    model.train()
    total_loss = 0.0
    n_batches  = 0

    for pos_feat, neg_feat in loader:
        pos_feat = pos_feat.to(device)           # (B, n_feat)
        neg_feat = neg_feat.to(device)           # (B, K, n_feat)
        B, K, F = neg_feat.shape

        score_pos  = model(pos_feat)             # (B,)
        scores_neg = model(neg_feat.view(B * K, F)).view(B, K)  # (B, K)

        loss = loss_fn(score_pos, scores_neg)
        optimizer.zero_grad()
        loss.backward()
        if grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        total_loss += loss.item()
        n_batches  += 1

    return total_loss / max(n_batches, 1)


# ---------------------------------------------------------------------------
# Full training loop with early stopping
# ---------------------------------------------------------------------------

def train(
    model: PairMLP,
    dataset: PairwiseDataset,
    cfg: PairMLPConfig,
    device: torch.device,
    eval_fn: Callable[[PairMLP, str], Dict],
    output_dir: str,
    eval_every: int = 2,
) -> Dict:
    """Training loop with early stopping on val MRR.

    Args:
        model:       PairMLP (already on device).
        dataset:     PairwiseDataset with precomputed features.
        cfg:         PairMLPConfig.
        device:      Training device.
        eval_fn:     eval_fn(model, split) → dict with 'mrr'.
        output_dir:  Directory for best_model.pt checkpoint.
        eval_every:  Epochs between evaluations.

    Returns:
        History dict with per-epoch losses, val MRRs, best checkpoint info.
    """
    assert cfg.loss in VALID_LOSSES, f"Invalid loss: {cfg.loss}"
    os.makedirs(output_dir, exist_ok=True)
    ckpt_path = os.path.join(output_dir, "best_model.pt")

    loss_fn = get_loss_fn(cfg.loss)

    loader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )

    optimizer = torch.optim.Adam(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
    )

    history: Dict = {
        "train_loss":     [],
        "val_mrr":        [],
        "val_mrr_epoch":  [],
    }
    best_val_mrr    = -1.0
    patience_counter = 0
    t_train_start   = time.time()

    print(f"\nTraining PairMLP ({cfg.loss.upper()} loss) for up to {cfg.n_epochs} epochs")
    print(f"  Features ({cfg.n_input_features}): {cfg.selected_feature_names}")
    print(f"  Dataset: {len(dataset):,} positive edges, K={dataset.neg.shape[1]} neg")
    print(f"  Batches/epoch: {len(loader):,}, batch_size={cfg.batch_size}")
    print(f"  patience={cfg.patience} (eval every {eval_every} epochs)")

    for epoch in range(1, cfg.n_epochs + 1):
        t0   = time.time()
        loss = train_epoch(model, loader, optimizer, loss_fn, device, cfg.grad_clip)
        history["train_loss"].append(loss)
        ep_time = time.time() - t0

        if epoch % eval_every == 0 or epoch == cfg.n_epochs:
            val_metrics = eval_fn(model, "val")
            val_mrr     = val_metrics["mrr"]
            history["val_mrr"].append(val_mrr)
            history["val_mrr_epoch"].append(epoch)

            improved = val_mrr > best_val_mrr
            marker   = " ← best" if improved else ""
            print(
                f"  Epoch {epoch:3d}/{cfg.n_epochs} | loss={loss:.4f} | "
                f"val_MRR={val_mrr:.4f}{marker} | {ep_time:.1f}s",
                flush=True,
            )

            if improved:
                best_val_mrr     = val_mrr
                patience_counter = 0
                torch.save({
                    "epoch":       epoch,
                    "model_state": model.state_dict(),
                    "val_mrr":     val_mrr,
                    "config":      cfg.to_dict(),
                }, ckpt_path)
            else:
                patience_counter += 1
                if patience_counter >= cfg.patience:
                    print(f"  Early stopping at epoch {epoch} "
                          f"(no val improvement for {cfg.patience} eval checks)")
                    break
        else:
            print(
                f"  Epoch {epoch:3d}/{cfg.n_epochs} | loss={loss:.4f} | {ep_time:.1f}s",
                flush=True,
            )

    elapsed = time.time() - t_train_start
    print(f"\nTraining done: {elapsed / 60:.1f} min | best val MRR={best_val_mrr:.4f}")

    history["best_val_mrr"] = best_val_mrr
    history["ckpt_path"]    = ckpt_path
    return history

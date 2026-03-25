"""Training and validation for EAGLE-Time model.

Reuses data_utils infrastructure (TemporalCSR, neighbor sampling).
Supports three feature modes controlled by model.edge_feat_dim and
model.node_feat_dim:
    - 0/0: delta times only (original EAGLE-Time, fastest)
    - >0/0: delta times + neighbor edge features
    - 0/>0: delta times + query node features
    - >0/>0: all three signals

Supports mixed precision (AMP) for faster training on GPU.
TGB-style evaluation is in eagle_evaluate.py.
"""

import contextlib
import os
import time
import json
import logging
from pathlib import Path
from typing import Optional, Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from src.models.EAGLE.eagle import EAGLETime
from src.models.EAGLE.data_utils import (
    TemporalEdgeData,
    TemporalCSR,
    build_temporal_csr,
    sample_neighbors_batch,
)
from src.models.data_utils import featurize_neighbors

logger = logging.getLogger(__name__)


def _amp_autocast(enabled: bool, device_type: str):
    """Return AMP autocast context or no-op."""
    if enabled and device_type == "cuda":
        return torch.cuda.amp.autocast()
    return contextlib.nullcontext()


def prepare_eagle_batch(
    csr: TemporalCSR,
    data: TemporalEdgeData,
    src_nodes: np.ndarray,
    dst_nodes: np.ndarray,
    timestamps: np.ndarray,
    neg_dst_nodes: np.ndarray,
    num_neighbors: int,
    device: torch.device,
    use_edge_feats: bool = False,
    use_node_feats: bool = False,
) -> Dict[str, torch.Tensor]:
    """Prepare a training batch for EAGLE-Time.

    When use_edge_feats=True, extracts features of neighboring edges via
    featurize_neighbors and adds them to the batch. When use_node_feats=True,
    looks up each query node's own feature vector from data.node_feats.

    Args:
        csr: Temporal CSR for neighbor lookups.
        data: Full temporal edge data.
        src_nodes: [batch] source node indices.
        dst_nodes: [batch] positive destination indices.
        timestamps: [batch] query timestamps.
        neg_dst_nodes: [batch, num_neg] negative destination indices.
        num_neighbors: K neighbors to sample per node.
        device: Target torch device.
        use_edge_feats: Extract per-neighbor edge feature vectors.
        use_node_feats: Extract query-node feature vectors.

    Returns:
        Dictionary of tensors for model forward pass.
        Optional keys (present only when the corresponding flag is True):
            src_edge_feats, pos_dst_edge_feats, neg_dst_edge_feats,
            src_node_feats, pos_dst_node_feats, neg_dst_node_feats.
    """
    batch_size = len(src_nodes)
    num_neg = neg_dst_nodes.shape[1] if neg_dst_nodes.ndim > 1 else 1

    def _get_feats(nodes, ts_arr):
        n_nodes, neighbor_ts, neighbor_eids, lengths = sample_neighbors_batch(
            csr, nodes, ts_arr, num_neighbors
        )
        delta_times = np.maximum(
            ts_arr[:, None] - neighbor_ts, 0.0
        ).astype(np.float32)

        edge_feat_batch = None
        node_feat_batch = None

        if use_edge_feats or use_node_feats:
            _, edge_feat_raw, _ = featurize_neighbors(
                n_nodes, neighbor_eids, lengths,
                neighbor_ts, ts_arr,
                data.node_feats, data.edge_feats,
            )
            if use_edge_feats:
                edge_feat_batch = edge_feat_raw.astype(np.float32)
            if use_node_feats:
                node_feat_batch = data.node_feats[nodes].astype(np.float32)

        return delta_times, edge_feat_batch, lengths, node_feat_batch

    src_dt, src_ef, src_len, src_nf = _get_feats(src_nodes, timestamps)
    dst_dt, dst_ef, dst_len, dst_nf = _get_feats(dst_nodes, timestamps)

    neg_flat = neg_dst_nodes.reshape(-1)
    neg_ts = np.repeat(timestamps, num_neg)
    neg_dt, neg_ef, neg_len, neg_nf = _get_feats(neg_flat, neg_ts)

    def _t(arr, dtype=torch.float32):
        return torch.tensor(arr, dtype=dtype, device=device)

    batch = {
        "src_delta_times": _t(src_dt),
        "src_lengths": _t(src_len, torch.int64),
        "pos_dst_delta_times": _t(dst_dt),
        "pos_dst_lengths": _t(dst_len, torch.int64),
        "neg_dst_delta_times": _t(
            neg_dt.reshape(batch_size, num_neg, num_neighbors)
        ),
        "neg_dst_lengths": _t(
            neg_len.reshape(batch_size, num_neg), torch.int64
        ),
    }

    if use_edge_feats:
        ef_dim = src_ef.shape[-1]
        batch["src_edge_feats"] = _t(src_ef)
        batch["pos_dst_edge_feats"] = _t(dst_ef)
        batch["neg_dst_edge_feats"] = _t(
            neg_ef.reshape(batch_size, num_neg, num_neighbors, ef_dim)
        )

    if use_node_feats:
        batch["src_node_feats"] = _t(src_nf)
        batch["pos_dst_node_feats"] = _t(dst_nf)
        batch["neg_dst_node_feats"] = _t(
            neg_nf.reshape(batch_size, num_neg, -1)
        )

    return batch


def train_epoch(
    model: EAGLETime,
    data: TemporalEdgeData,
    csr: TemporalCSR,
    edge_indices: np.ndarray,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    batch_size: int = 200,
    num_neighbors: int = 20,
    neg_per_positive: int = 1,
    use_amp: bool = True,
    scaler: Optional[torch.cuda.amp.GradScaler] = None,
    rng: np.random.Generator = None,
) -> Dict[str, float]:
    """Run one training epoch.

    Args:
        model: EAGLE-Time model.
        data: Temporal edge data.
        csr: Temporal CSR (built from training edges only).
        edge_indices: Indices of training edges.
        optimizer: Torch optimizer.
        device: Torch device.
        batch_size: Edges per batch.
        num_neighbors: Neighbors to sample per node.
        neg_per_positive: Negative samples per positive edge.
        use_amp: Enable mixed precision.
        scaler: GradScaler for AMP.
        rng: Random number generator.

    Returns:
        Dict with 'loss' (average batch loss).
    """
    if rng is None:
        rng = np.random.default_rng()

    model.train()
    criterion = nn.BCEWithLogitsLoss()
    amp_enabled = use_amp and device.type == "cuda"

    shuffled = rng.permutation(edge_indices)
    total_loss = 0.0
    num_batches = 0
    n_total_batches = (len(shuffled) + batch_size - 1) // batch_size

    use_edge_feats = model.edge_feat_dim > 0
    use_node_feats = model.node_feat_dim > 0

    pbar = tqdm(
        range(0, len(shuffled), batch_size),
        total=n_total_batches,
        desc="Training",
        leave=False,
        unit="batch",
    )
    for start in pbar:
        end = min(start + batch_size, len(shuffled))
        batch_idx = shuffled[start:end]
        actual_batch = len(batch_idx)

        src = data.src[batch_idx]
        dst = data.dst[batch_idx]
        ts = data.timestamps[batch_idx]
        neg_dst = rng.integers(
            0, data.num_nodes, size=(actual_batch, neg_per_positive)
        )

        batch = prepare_eagle_batch(
            csr, data, src, dst, ts, neg_dst, num_neighbors, device,
            use_edge_feats=use_edge_feats,
            use_node_feats=use_node_feats,
        )

        with _amp_autocast(amp_enabled, device.type):
            pos_logits = model(
                batch["src_delta_times"],
                batch["src_lengths"],
                batch["pos_dst_delta_times"],
                batch["pos_dst_lengths"],
                src_edge_feats=batch.get("src_edge_feats"),
                dst_edge_feats=batch.get("pos_dst_edge_feats"),
                src_node_feats=batch.get("src_node_feats"),
                dst_node_feats=batch.get("pos_dst_node_feats"),
            )

            neg_logits_list = []
            for neg_i in range(neg_per_positive):
                neg_ef_i = (
                    batch["neg_dst_edge_feats"][:, neg_i, :, :]
                    if use_edge_feats else None
                )
                neg_nf_i = (
                    batch["neg_dst_node_feats"][:, neg_i, :]
                    if use_node_feats else None
                )
                neg_logits_i = model(
                    batch["src_delta_times"],
                    batch["src_lengths"],
                    batch["neg_dst_delta_times"][:, neg_i, :],
                    batch["neg_dst_lengths"][:, neg_i],
                    src_edge_feats=batch.get("src_edge_feats"),
                    dst_edge_feats=neg_ef_i,
                    src_node_feats=batch.get("src_node_feats"),
                    dst_node_feats=neg_nf_i,
                )
                neg_logits_list.append(neg_logits_i)

            all_logits = torch.cat([pos_logits] + neg_logits_list)
            all_labels = torch.cat([
                torch.ones(actual_batch, device=device),
                torch.zeros(actual_batch * neg_per_positive, device=device),
            ])
            loss = criterion(all_logits, all_labels)

        optimizer.zero_grad()
        if amp_enabled and scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        total_loss += loss.item()
        num_batches += 1
        pbar.set_postfix(loss=f"{total_loss / num_batches:.4f}")

    pbar.close()
    return {"loss": total_loss / max(num_batches, 1)}


@torch.no_grad()
def validate(
    model: EAGLETime,
    data: TemporalEdgeData,
    csr: TemporalCSR,
    edge_indices: np.ndarray,
    device: torch.device,
    num_neighbors: int = 20,
    n_eval_negatives: int = 100,
    max_eval_edges: int = 5000,
    use_amp: bool = True,
    rng: np.random.Generator = None,
) -> Dict[str, float]:
    """Run validation with ranking metrics.

    Args:
        model: EAGLE-Time model.
        data: Temporal edge data.
        csr: Temporal CSR.
        edge_indices: Indices of val/test edges.
        device: Torch device.
        num_neighbors: Neighbors to sample.
        n_eval_negatives: Negatives per positive for ranking.
        max_eval_edges: Max edges to evaluate (subsample for speed).
        use_amp: Enable mixed precision.
        rng: Random number generator.

    Returns:
        Dict with 'mrr', 'hits@1', 'hits@3', 'hits@10'.
    """
    if rng is None:
        rng = np.random.default_rng(42)

    model.eval()
    amp_enabled = use_amp and device.type == "cuda"
    K = num_neighbors
    use_edge_feats = model.edge_feat_dim > 0
    use_node_feats = model.node_feat_dim > 0

    if len(edge_indices) > max_eval_edges:
        eval_idx = rng.choice(edge_indices, size=max_eval_edges, replace=False)
    else:
        eval_idx = edge_indices

    ranks = []

    for idx in eval_idx:
        src_node = data.src[idx]
        true_dst = data.dst[idx]
        ts = data.timestamps[idx]

        neg_nodes = rng.integers(
            0, data.num_nodes, size=n_eval_negatives
        ).astype(np.int32)
        all_dst = np.concatenate([[true_dst], neg_nodes])
        num_candidates = len(all_dst)

        src_arr = np.array([src_node], dtype=np.int32)
        ts_arr = np.array([ts], dtype=np.float64)
        src_nids, src_nts, src_neids, src_lens = sample_neighbors_batch(
            csr, src_arr, ts_arr, K
        )
        src_dt = np.maximum(
            (ts_arr[:, None] - src_nts), 0.0
        ).astype(np.float32)

        dst_ts_arr = np.full(num_candidates, ts, dtype=np.float64)
        dst_nids, dst_nts, dst_neids, dst_lens = sample_neighbors_batch(
            csr, all_dst, dst_ts_arr, K
        )
        dst_dt = np.maximum(
            (dst_ts_arr[:, None] - dst_nts), 0.0
        ).astype(np.float32)

        src_ef = dst_ef = src_nf = dst_nf = None
        if use_edge_feats or use_node_feats:
            _, src_ef_raw, _ = featurize_neighbors(
                src_nids, src_neids, src_lens, src_nts, ts_arr,
                data.node_feats, data.edge_feats,
            )
            _, dst_ef_raw, _ = featurize_neighbors(
                dst_nids, dst_neids, dst_lens, dst_nts, dst_ts_arr,
                data.node_feats, data.edge_feats,
            )
            if use_edge_feats:
                src_ef = src_ef_raw.astype(np.float32)
                dst_ef = dst_ef_raw.astype(np.float32)
            if use_node_feats:
                src_nf = data.node_feats[[src_node]].astype(np.float32)
                dst_nf = data.node_feats[all_dst].astype(np.float32)

        def _t(arr, dtype=torch.float32):
            return torch.tensor(arr, dtype=dtype, device=device)

        with _amp_autocast(amp_enabled, device.type):
            h_src = model.encode_nodes(
                _t(src_dt), _t(src_lens, torch.int64),
                edge_feats=_t(src_ef) if src_ef is not None else None,
                node_feats=_t(src_nf) if src_nf is not None else None,
            )
            h_dst = model.encode_nodes(
                _t(dst_dt), _t(dst_lens, torch.int64),
                edge_feats=_t(dst_ef) if dst_ef is not None else None,
                node_feats=_t(dst_nf) if dst_nf is not None else None,
            )
            h_src_exp = h_src.expand(num_candidates, -1)
            scores = model.edge_predictor(
                h_src_exp, h_dst
            ).cpu().float().numpy()

        true_score = scores[0]
        rank = (
            1
            + (scores[1:] > true_score).sum()
            + 0.5 * (scores[1:] == true_score).sum()
        )
        ranks.append(rank)

    ranks = np.array(ranks, dtype=np.float64)
    return {
        "mrr": float(np.mean(1.0 / ranks)),
        "hits@1": float(np.mean(ranks <= 1)),
        "hits@3": float(np.mean(ranks <= 3)),
        "hits@10": float(np.mean(ranks <= 10)),
        "mean_rank": float(np.mean(ranks)),
        "n_queries": len(ranks),
    }


def train_eagle(
    data: TemporalEdgeData,
    train_mask: np.ndarray,
    val_mask: np.ndarray,
    output_dir: str,
    device: torch.device = None,
    num_epochs: int = 100,
    batch_size: int = 200,
    learning_rate: float = 0.001,
    weight_decay: float = 5e-5,
    num_neighbors: int = 20,
    hidden_dim: int = 100,
    num_mixer_layers: int = 1,
    token_expansion: float = 0.5,
    channel_expansion: float = 4.0,
    dropout: float = 0.1,
    patience: int = 10,
    seed: int = 42,
    max_val_edges: int = 5000,
    use_amp: bool = True,
    edge_feat_dim: int = 0,
    node_feat_dim: int = 0,
) -> Tuple[EAGLETime, Dict]:
    """Full training pipeline for EAGLE-Time.

    Args:
        data: TemporalEdgeData.
        train_mask: Boolean mask for training edges.
        val_mask: Boolean mask for validation edges.
        output_dir: Directory for checkpoints and logs.
        device: Torch device. Auto-detected if None.
        num_epochs: Maximum training epochs.
        batch_size: Training batch size.
        learning_rate: Adam learning rate.
        weight_decay: Adam weight decay.
        num_neighbors: K neighbors per node.
        hidden_dim: Hidden dimension for all modules.
        num_mixer_layers: Number of MLP-Mixer layers.
        token_expansion: Token-mixing expansion factor.
        channel_expansion: Channel-mixing expansion factor.
        dropout: Dropout rate.
        patience: Early stopping patience.
        seed: Random seed.
        max_val_edges: Max validation edges per epoch.
        use_amp: Enable mixed precision training.

    Returns:
        Tuple of (trained model, training history dict).
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed(seed)

    train_csr = build_temporal_csr(data, train_mask)
    full_mask = train_mask | val_mask
    full_csr = build_temporal_csr(data, full_mask)

    model = EAGLETime(
        hidden_dim=hidden_dim,
        num_neighbors=num_neighbors,
        num_mixer_layers=num_mixer_layers,
        token_expansion=token_expansion,
        channel_expansion=channel_expansion,
        dropout=dropout,
        edge_feat_dim=edge_feat_dim,
        node_feat_dim=node_feat_dim,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    logger.info(
        "EAGLE-Time: %d total params, %d trainable",
        total_params,
        trainable_params,
    )

    optimizer = torch.optim.Adam(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    amp_enabled = use_amp and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)

    train_indices = np.where(train_mask)[0]
    val_indices = np.where(val_mask)[0]

    history = {
        "train_loss": [],
        "val_mrr": [],
        "val_hits@1": [],
        "val_hits@3": [],
        "val_hits@10": [],
        "epoch_time": [],
    }
    best_val_mrr = -1.0
    best_epoch = -1
    epochs_without_improvement = 0

    config = {
        "model": "EAGLETime",
        "hidden_dim": hidden_dim,
        "num_neighbors": num_neighbors,
        "num_mixer_layers": num_mixer_layers,
        "token_expansion": token_expansion,
        "channel_expansion": channel_expansion,
        "dropout": dropout,
        "edge_feat_dim": edge_feat_dim,
        "node_feat_dim": node_feat_dim,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "num_epochs": num_epochs,
        "patience": patience,
        "seed": seed,
        "use_amp": use_amp,
        "num_nodes": data.num_nodes,
        "num_edges": data.num_edges,
        "total_params": total_params,
        "trainable_params": trainable_params,
        "device": str(device),
    }
    with open(os.path.join(output_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    logger.info(
        "Training EAGLE-Time: %d epochs, %d train, %d val edges",
        num_epochs,
        len(train_indices),
        len(val_indices),
    )

    for epoch in range(1, num_epochs + 1):
        epoch_start = time.time()

        train_metrics = train_epoch(
            model,
            data,
            train_csr,
            train_indices,
            optimizer,
            device,
            batch_size=batch_size,
            num_neighbors=num_neighbors,
            use_amp=use_amp,
            scaler=scaler,
            rng=rng,
        )

        val_metrics = validate(
            model,
            data,
            full_csr,
            val_indices,
            device,
            num_neighbors=num_neighbors,
            max_eval_edges=max_val_edges,
            use_amp=use_amp,
            rng=rng,
        )

        epoch_time = time.time() - epoch_start

        history["train_loss"].append(train_metrics["loss"])
        history["val_mrr"].append(val_metrics["mrr"])
        history["val_hits@1"].append(val_metrics["hits@1"])
        history["val_hits@3"].append(val_metrics["hits@3"])
        history["val_hits@10"].append(val_metrics["hits@10"])
        history["epoch_time"].append(epoch_time)

        logger.info(
            "Epoch %d/%d [%.1fs] loss=%.4f mrr=%.4f h@1=%.3f h@3=%.3f h@10=%.3f",
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
            torch.save(
                model.state_dict(),
                os.path.join(output_dir, "best_model.pt"),
            )
            logger.info("New best model (MRR=%.4f)", best_val_mrr)
        else:
            epochs_without_improvement += 1

        with open(os.path.join(output_dir, "metrics.jsonl"), "a") as f:
            record = {
                "epoch": epoch,
                **train_metrics,
                **{f"val_{k}": v for k, v in val_metrics.items()},
                "epoch_time": epoch_time,
            }
            f.write(json.dumps(record) + "\n")

        if epochs_without_improvement >= patience:
            logger.info(
                "Early stopping at epoch %d (patience=%d)", epoch, patience
            )
            break

    model.load_state_dict(
        torch.load(
            os.path.join(output_dir, "best_model.pt"),
            map_location=device,
            weights_only=True,
        )
    )

    summary = {
        "best_epoch": best_epoch,
        "best_val_mrr": best_val_mrr,
        "total_epochs": epoch,
        "final_train_loss": history["train_loss"][-1],
    }
    with open(os.path.join(output_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    logger.info(
        "Training complete. Best epoch=%d, MRR=%.4f", best_epoch, best_val_mrr
    )
    return model, history

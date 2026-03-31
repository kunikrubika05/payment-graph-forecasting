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

from payment_graph_forecasting.training.amp import autocast_context
from payment_graph_forecasting.training.amp import (
    amp_enabled_for_device,
    create_grad_scaler,
    seed_torch,
)
from payment_graph_forecasting.evaluation.temporal_ranking import (
    conservative_rank_from_scores,
    score_candidate_contexts,
)
from payment_graph_forecasting.training.epoch import run_loss_epoch
from payment_graph_forecasting.training.temporal_context import (
    sample_node_contexts,
    to_device_tensor,
)
from payment_graph_forecasting.training.trainer import run_early_stopping_training
from src.models.EAGLE.eagle import EAGLETime
from src.models.EAGLE.data_utils import (
    TemporalEdgeData,
    TemporalCSR,
    build_temporal_csr,
    sample_neighbors_batch,
)

logger = logging.getLogger(__name__)


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
        context = sample_node_contexts(
            csr=csr,
            data=data,
            sample_neighbors_fn=sample_neighbors_batch,
            nodes=nodes,
            query_timestamps=ts_arr,
            num_neighbors=num_neighbors,
            use_edge_feats=use_edge_feats,
            use_node_feats=use_node_feats,
        )
        return (
            context.delta_times,
            context.edge_features,
            context.lengths,
            context.node_features,
        )

    src_dt, src_ef, src_len, src_nf = _get_feats(src_nodes, timestamps)
    dst_dt, dst_ef, dst_len, dst_nf = _get_feats(dst_nodes, timestamps)

    neg_flat = neg_dst_nodes.reshape(-1)
    neg_ts = np.repeat(timestamps, num_neg)
    neg_dt, neg_ef, neg_len, neg_nf = _get_feats(neg_flat, neg_ts)

    batch = {
        "src_delta_times": to_device_tensor(src_dt, device),
        "src_lengths": to_device_tensor(src_len, device, torch.int64),
        "pos_dst_delta_times": to_device_tensor(dst_dt, device),
        "pos_dst_lengths": to_device_tensor(dst_len, device, torch.int64),
        "neg_dst_delta_times": to_device_tensor(
            neg_dt.reshape(batch_size, num_neg, num_neighbors), device
        ),
        "neg_dst_lengths": to_device_tensor(
            neg_len.reshape(batch_size, num_neg), device, torch.int64
        ),
    }

    if use_edge_feats:
        ef_dim = src_ef.shape[-1]
        batch["src_edge_feats"] = to_device_tensor(src_ef, device)
        batch["pos_dst_edge_feats"] = to_device_tensor(dst_ef, device)
        batch["neg_dst_edge_feats"] = to_device_tensor(
            neg_ef.reshape(batch_size, num_neg, num_neighbors, ef_dim), device
        )

    if use_node_feats:
        batch["src_node_feats"] = to_device_tensor(src_nf, device)
        batch["pos_dst_node_feats"] = to_device_tensor(dst_nf, device)
        batch["neg_dst_node_feats"] = to_device_tensor(
            neg_nf.reshape(batch_size, num_neg, -1), device
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
    active_nodes: Optional[np.ndarray] = None,
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
        active_nodes: Train node indices for negative sampling. If None,
                      falls back to uniform sampling from all nodes.

    Returns:
        Dict with 'loss' (average batch loss).
    """
    if rng is None:
        rng = np.random.default_rng()

    model.train()
    criterion = nn.BCEWithLogitsLoss()
    amp_enabled = use_amp and device.type == "cuda"

    use_edge_feats = model.edge_feat_dim > 0
    use_node_feats = model.node_feat_dim > 0

    def _loss_for_batch(batch_idx: np.ndarray) -> torch.Tensor:
        actual_batch = len(batch_idx)

        src = data.src[batch_idx]
        dst = data.dst[batch_idx]
        ts = data.timestamps[batch_idx]
        if active_nodes is not None:
            neg_idx = rng.integers(
                0, len(active_nodes), size=(actual_batch, neg_per_positive)
            )
            neg_dst = active_nodes[neg_idx].astype(np.int32)
        else:
            neg_dst = rng.integers(
                0, data.num_nodes, size=(actual_batch, neg_per_positive)
            )

        batch = prepare_eagle_batch(
            csr, data, src, dst, ts, neg_dst, num_neighbors, device,
            use_edge_feats=use_edge_feats,
            use_node_feats=use_node_feats,
        )

        with autocast_context(amp_enabled, device.type):
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
            return criterion(all_logits, all_labels)

    return run_loss_epoch(
        edge_indices=edge_indices,
        batch_size=batch_size,
        rng=rng,
        loss_fn=_loss_for_batch,
        optimizer=optimizer,
        model=model,
        amp_enabled=amp_enabled,
        scaler=scaler,
        progress_desc="Training",
        loss_format="{:.4f}",
    )


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
    active_nodes: Optional[np.ndarray] = None,
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
        active_nodes: Train node indices for negative sampling.

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

        if active_nodes is not None:
            neg_idx = rng.integers(0, len(active_nodes), size=n_eval_negatives)
            neg_nodes = active_nodes[neg_idx].astype(np.int32)
        else:
            neg_nodes = rng.integers(
                0, data.num_nodes, size=n_eval_negatives
            ).astype(np.int32)
        all_dst = np.concatenate([[true_dst], neg_nodes])
        num_candidates = len(all_dst)

        src_arr = np.array([src_node], dtype=np.int32)
        ts_arr = np.array([ts], dtype=np.float64)
        src_context = sample_node_contexts(
            csr=csr,
            data=data,
            sample_neighbors_fn=sample_neighbors_batch,
            nodes=src_arr,
            query_timestamps=ts_arr,
            num_neighbors=K,
            use_edge_feats=use_edge_feats,
            use_node_feats=use_node_feats,
        )

        dst_ts_arr = np.full(num_candidates, ts, dtype=np.float64)
        dst_context = sample_node_contexts(
            csr=csr,
            data=data,
            sample_neighbors_fn=sample_neighbors_batch,
            nodes=all_dst,
            query_timestamps=dst_ts_arr,
            num_neighbors=K,
            use_edge_feats=use_edge_feats,
            use_node_feats=use_node_feats,
        )

        scores = score_candidate_contexts(
            model=model,
            device=device,
            src_context=src_context,
            dst_context=dst_context,
            amp_enabled=amp_enabled,
        )
        rank = conservative_rank_from_scores(scores)
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
    active_nodes: Optional[np.ndarray] = None,
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
        active_nodes: Sorted array of train node indices for negative
                      sampling. If None, samples from all nodes.

    Returns:
        Tuple of (trained model, training history dict).
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(seed)
    seed_torch(seed, device)

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
    amp_enabled = amp_enabled_for_device(use_amp, device)
    scaler = create_grad_scaler(amp_enabled)

    train_indices = np.where(train_mask)[0]
    val_indices = np.where(val_mask)[0]

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
    history, _summary = run_early_stopping_training(
        model=model,
        output_dir=output_dir,
        device=device,
        num_epochs=num_epochs,
        patience=patience,
        train_epoch_fn=lambda: train_epoch(
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
            active_nodes=active_nodes,
        ),
        validate_fn=lambda: validate(
            model,
            data,
            full_csr,
            val_indices,
            device,
            num_neighbors=num_neighbors,
            max_eval_edges=max_val_edges,
            use_amp=use_amp,
            rng=rng,
            active_nodes=active_nodes,
        ),
        logger=logger,
        train_loss_format="%.4f",
    )
    return model, history

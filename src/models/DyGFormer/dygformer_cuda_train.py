"""DyGFormer training with TemporalGraphSampler-backed sampling."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn

from payment_graph_forecasting.training.amp import (
    amp_enabled_for_device,
    autocast_context,
    create_grad_scaler,
    seed_torch,
)
from payment_graph_forecasting.training.epoch import run_loss_epoch
from payment_graph_forecasting.training.trainer import run_early_stopping_training
from src.models.data_utils import build_temporal_csr, build_unified_sampler
from src.models.DyGFormer.dygformer import DyGFormerTime
from src.models.DyGFormer.dygformer_train import validate

logger = logging.getLogger(__name__)


def _neighbor_cooccurrence_torch(
    src_nids: torch.Tensor,
    src_lens: torch.Tensor,
    dst_nids: torch.Tensor,
    dst_lens: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute DyGFormer co-occurrence counts fully in torch."""

    num_neighbors = src_nids.shape[1]
    idx = torch.arange(num_neighbors, device=src_nids.device)
    src_mask = idx.unsqueeze(0) < src_lens.unsqueeze(1)
    dst_mask = idx.unsqueeze(0) < dst_lens.unsqueeze(1)

    src_self = (src_nids[:, :, None] == src_nids[:, None, :]) & src_mask[:, None, :]
    src_cross = (src_nids[:, :, None] == dst_nids[:, None, :]) & dst_mask[:, None, :]
    dst_self = (dst_nids[:, :, None] == dst_nids[:, None, :]) & dst_mask[:, None, :]
    dst_cross = (dst_nids[:, :, None] == src_nids[:, None, :]) & src_mask[:, None, :]

    src_cooc = torch.zeros(
        (src_nids.shape[0], num_neighbors, 2), device=src_nids.device, dtype=torch.float32
    )
    dst_cooc = torch.zeros_like(src_cooc)
    src_cooc[:, :, 0] = src_self.sum(dim=2) * src_mask
    src_cooc[:, :, 1] = src_cross.sum(dim=2) * src_mask
    dst_cooc[:, :, 0] = dst_self.sum(dim=2) * dst_mask
    dst_cooc[:, :, 1] = dst_cross.sum(dim=2) * dst_mask
    return src_cooc, dst_cooc


def _sample_context_cuda(
    *,
    sampler,
    data,
    nodes: np.ndarray,
    timestamps: np.ndarray,
    num_neighbors: int,
    use_edge_feats: bool,
    use_node_feats: bool,
    device: torch.device,
) -> dict[str, torch.Tensor | None]:
    """Sample neighbors/features through TemporalGraphSampler and keep tensors on device."""

    nbr = sampler.sample_neighbors(nodes, timestamps, num_neighbors=num_neighbors)
    q_ts = torch.as_tensor(timestamps, dtype=torch.float64, device=device)

    if nbr.on_gpu:
        neighbor_ids = nbr.neighbor_ids.to(device=device, dtype=torch.int32, non_blocking=True)
        neighbor_ts = nbr.timestamps.to(device=device, dtype=torch.float64, non_blocking=True)
        lengths = nbr.lengths.to(device=device, dtype=torch.int64, non_blocking=True)
    else:
        neighbor_ids = torch.as_tensor(nbr.neighbor_ids, dtype=torch.int32, device=device)
        neighbor_ts = torch.as_tensor(nbr.timestamps, dtype=torch.float64, device=device)
        lengths = torch.as_tensor(nbr.lengths, dtype=torch.int64, device=device)

    delta_times = torch.clamp(q_ts.unsqueeze(1) - neighbor_ts, min=0.0).to(torch.float32)
    idx = torch.arange(num_neighbors, device=device)
    valid_mask = idx.unsqueeze(0) < lengths.unsqueeze(1)
    delta_times = delta_times.masked_fill(~valid_mask, 0.0)

    edge_features = None
    node_features = None
    if use_edge_feats or use_node_feats:
        feat = sampler.featurize(nbr, query_timestamps=timestamps)
        if use_edge_feats:
            if feat.on_gpu:
                edge_features = feat.edge_features.to(device=device, dtype=torch.float32, non_blocking=True)
            else:
                edge_features = torch.as_tensor(feat.edge_features, dtype=torch.float32, device=device)
        if use_node_feats:
            node_features = torch.as_tensor(
                data.node_feats[nodes], dtype=torch.float32, device=device
            )

    return {
        "neighbor_ids": neighbor_ids,
        "delta_times": delta_times,
        "lengths": lengths,
        "edge_features": edge_features,
        "node_features": node_features,
    }


def prepare_dygformer_batch_cuda(
    *,
    sampler,
    data,
    src_nodes: np.ndarray,
    dst_nodes: np.ndarray,
    timestamps: np.ndarray,
    neg_dst_nodes: np.ndarray,
    num_neighbors: int,
    device: torch.device,
    use_edge_feats: bool = True,
    use_node_feats: bool = False,
) -> dict[str, torch.Tensor]:
    """Prepare one DyGFormer batch using TemporalGraphSampler-backed sampling."""

    batch_size = len(src_nodes)
    num_neg = neg_dst_nodes.shape[1] if neg_dst_nodes.ndim > 1 else 1
    neg_flat = neg_dst_nodes.reshape(-1).astype(np.int32)
    neg_timestamps = np.repeat(timestamps, num_neg)

    all_nodes = np.concatenate([src_nodes, dst_nodes, neg_flat]).astype(np.int32, copy=False)
    all_timestamps = np.concatenate([timestamps, timestamps, neg_timestamps]).astype(np.float64, copy=False)
    all_context = _sample_context_cuda(
        sampler=sampler,
        data=data,
        nodes=all_nodes,
        timestamps=all_timestamps,
        num_neighbors=num_neighbors,
        use_edge_feats=use_edge_feats,
        use_node_feats=use_node_feats,
        device=device,
    )

    def _slice(start: int, end: int) -> dict[str, torch.Tensor | None]:
        return {
            "neighbor_ids": all_context["neighbor_ids"][start:end],
            "delta_times": all_context["delta_times"][start:end],
            "lengths": all_context["lengths"][start:end],
            "edge_features": (
                all_context["edge_features"][start:end]
                if all_context["edge_features"] is not None
                else None
            ),
            "node_features": (
                all_context["node_features"][start:end]
                if all_context["node_features"] is not None
                else None
            ),
        }

    src_context = _slice(0, batch_size)
    pos_context = _slice(batch_size, 2 * batch_size)
    neg_context = _slice(2 * batch_size, 2 * batch_size + batch_size * num_neg)

    pos_src_cooc, pos_dst_cooc = _neighbor_cooccurrence_torch(
        src_context["neighbor_ids"],
        src_context["lengths"],
        pos_context["neighbor_ids"],
        pos_context["lengths"],
    )
    neg_src_cooc, neg_dst_cooc = _neighbor_cooccurrence_torch(
        src_context["neighbor_ids"].repeat_interleave(num_neg, dim=0),
        src_context["lengths"].repeat_interleave(num_neg),
        neg_context["neighbor_ids"],
        neg_context["lengths"],
    )

    batch = {
        "src_delta_times": src_context["delta_times"],
        "src_lengths": src_context["lengths"],
        "pos_dst_delta_times": pos_context["delta_times"],
        "pos_dst_lengths": pos_context["lengths"],
        "pos_src_cooc": pos_src_cooc,
        "pos_dst_cooc": pos_dst_cooc,
        "neg_dst_delta_times": neg_context["delta_times"].reshape(batch_size, num_neg, num_neighbors),
        "neg_dst_lengths": neg_context["lengths"].reshape(batch_size, num_neg),
        "neg_src_cooc": neg_src_cooc.reshape(batch_size, num_neg, num_neighbors, 2),
        "neg_dst_cooc": neg_dst_cooc.reshape(batch_size, num_neg, num_neighbors, 2),
    }

    if use_edge_feats:
        edge_dim = int(src_context["edge_features"].shape[-1])
        batch["src_edge_feats"] = src_context["edge_features"]
        batch["pos_dst_edge_feats"] = pos_context["edge_features"]
        batch["neg_dst_edge_feats"] = neg_context["edge_features"].reshape(
            batch_size, num_neg, num_neighbors, edge_dim
        )

    if use_node_feats:
        node_dim = int(src_context["node_features"].shape[-1])
        batch["src_node_feats"] = src_context["node_features"]
        batch["pos_dst_node_feats"] = pos_context["node_features"]
        batch["neg_dst_node_feats"] = neg_context["node_features"].reshape(
            batch_size, num_neg, num_neighbors, node_dim
        )

    return batch


def train_epoch_cuda(
    model: DyGFormerTime,
    data,
    sampler,
    edge_indices: np.ndarray,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    batch_size: int = 200,
    num_neighbors: int = 32,
    neg_per_positive: int = 5,
    use_amp: bool = True,
    scaler: Optional[torch.cuda.amp.GradScaler] = None,
    rng: Optional[np.random.Generator] = None,
    neg_node_pool: Optional[np.ndarray] = None,
) -> dict[str, float]:
    """Run one DyGFormer epoch with sampler-backed batch preparation."""

    if rng is None:
        rng = np.random.default_rng()

    model.train()
    criterion = nn.BCEWithLogitsLoss()
    amp_enabled = amp_enabled_for_device(use_amp, device)
    use_edge_feats = model.edge_feat_dim > 0
    use_node_feats = model.node_feat_dim > 0

    def _loss_for_batch(batch_idx: np.ndarray) -> torch.Tensor:
        actual_batch = len(batch_idx)
        src_nodes = data.src[batch_idx]
        dst_nodes = data.dst[batch_idx]
        timestamps = data.timestamps[batch_idx]

        if neg_node_pool is not None:
            neg_indices = rng.integers(0, len(neg_node_pool), size=(actual_batch, neg_per_positive))
            neg_dst_nodes = neg_node_pool[neg_indices].astype(np.int32)
        else:
            neg_dst_nodes = rng.integers(0, data.num_nodes, size=(actual_batch, neg_per_positive)).astype(np.int32)

        batch = prepare_dygformer_batch_cuda(
            sampler=sampler,
            data=data,
            src_nodes=src_nodes,
            dst_nodes=dst_nodes,
            timestamps=timestamps,
            neg_dst_nodes=neg_dst_nodes,
            num_neighbors=num_neighbors,
            device=device,
            use_edge_feats=use_edge_feats,
            use_node_feats=use_node_feats,
        )

        with autocast_context(amp_enabled, device.type):
            pos_logits = model(
                src_delta_times=batch["src_delta_times"],
                src_lengths=batch["src_lengths"],
                dst_delta_times=batch["pos_dst_delta_times"],
                dst_lengths=batch["pos_dst_lengths"],
                src_cooc_counts=batch["pos_src_cooc"],
                dst_cooc_counts=batch["pos_dst_cooc"],
                src_edge_feats=batch.get("src_edge_feats"),
                dst_edge_feats=batch.get("pos_dst_edge_feats"),
                src_node_feats=batch.get("src_node_feats"),
                dst_node_feats=batch.get("pos_dst_node_feats"),
            )

            num_neighbors_local = model.num_neighbors
            src_dt_rep = batch["src_delta_times"].repeat_interleave(neg_per_positive, dim=0)
            src_len_rep = batch["src_lengths"].repeat_interleave(neg_per_positive)
            neg_logits = model(
                src_delta_times=src_dt_rep,
                src_lengths=src_len_rep,
                dst_delta_times=batch["neg_dst_delta_times"].reshape(actual_batch * neg_per_positive, num_neighbors_local),
                dst_lengths=batch["neg_dst_lengths"].reshape(actual_batch * neg_per_positive),
                src_cooc_counts=batch["neg_src_cooc"].reshape(actual_batch * neg_per_positive, num_neighbors_local, 2),
                dst_cooc_counts=batch["neg_dst_cooc"].reshape(actual_batch * neg_per_positive, num_neighbors_local, 2),
                src_edge_feats=batch["src_edge_feats"].repeat_interleave(neg_per_positive, dim=0)
                if use_edge_feats else None,
                dst_edge_feats=batch["neg_dst_edge_feats"].reshape(
                    actual_batch * neg_per_positive,
                    num_neighbors_local,
                    batch["neg_dst_edge_feats"].shape[-1],
                ) if use_edge_feats else None,
                src_node_feats=batch["src_node_feats"].repeat_interleave(neg_per_positive, dim=0)
                if use_node_feats else None,
                dst_node_feats=batch["neg_dst_node_feats"].reshape(
                    actual_batch * neg_per_positive,
                    num_neighbors_local,
                    batch["neg_dst_node_feats"].shape[-1],
                ) if use_node_feats else None,
            )

            all_logits = torch.cat([pos_logits, neg_logits], dim=0)
            all_labels = torch.cat(
                [
                    torch.ones(actual_batch, device=device),
                    torch.zeros(actual_batch * neg_per_positive, device=device),
                ],
                dim=0,
            )
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
        loss_format="{:.8f}",
    )


def train_dygformer_cuda(
    data,
    train_mask: np.ndarray,
    val_mask: np.ndarray,
    output_dir: str,
    device: Optional[torch.device] = None,
    num_epochs: int = 100,
    batch_size: int = 4000,
    learning_rate: float = 0.0001,
    weight_decay: float = 1e-5,
    num_neighbors: int = 32,
    patch_size: int = 1,
    time_dim: int = 100,
    aligned_dim: int = 50,
    num_transformer_layers: int = 2,
    num_attention_heads: int = 2,
    cooc_dim: int = 50,
    output_dim: int = 172,
    dropout: float = 0.1,
    patience: int = 20,
    seed: int = 42,
    max_val_edges: int = 2000,
    use_amp: bool = True,
    edge_feat_dim: int = 2,
    node_feat_dim: int = 0,
    neg_per_positive: int = 5,
    sampling_backend: str = "auto",
) -> tuple[DyGFormerTime, dict[str, list[float]]]:
    """Full DyGFormer training pipeline with sampler-backed batch preparation."""

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    seed_torch(seed, device=device)
    rng = np.random.default_rng(seed)

    train_sampler = build_unified_sampler(data, train_mask, backend=sampling_backend)
    full_csr = build_temporal_csr(data, train_mask | val_mask)

    model = DyGFormerTime(
        time_dim=time_dim,
        aligned_dim=aligned_dim,
        num_neighbors=num_neighbors,
        patch_size=patch_size,
        num_transformer_layers=num_transformer_layers,
        num_attention_heads=num_attention_heads,
        dropout=dropout,
        edge_feat_dim=edge_feat_dim,
        node_feat_dim=node_feat_dim,
        cooc_dim=cooc_dim,
        output_dim=output_dim,
    ).to(device)

    total_params = sum(parameter.numel() for parameter in model.parameters())
    trainable_params = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    with open(os.path.join(output_dir, "config.json"), "w", encoding="utf-8") as handle:
        json.dump(
            {
                "model": "DyGFormerTime",
                "sampling_backend": train_sampler.backend,
                "time_dim": time_dim,
                "aligned_dim": aligned_dim,
                "num_neighbors": num_neighbors,
                "patch_size": patch_size,
                "num_transformer_layers": num_transformer_layers,
                "num_attention_heads": num_attention_heads,
                "cooc_dim": cooc_dim,
                "output_dim": output_dim,
                "dropout": dropout,
                "edge_feat_dim": edge_feat_dim,
                "node_feat_dim": node_feat_dim,
                "neg_per_positive": neg_per_positive,
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
            },
            handle,
            indent=2,
        )

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scaler = create_grad_scaler(enabled=amp_enabled_for_device(use_amp, device))

    train_indices = np.where(train_mask)[0]
    val_indices = np.where(val_mask)[0]
    train_active_nodes = np.unique(
        np.concatenate([data.src[train_indices], data.dst[train_indices]])
    ).astype(np.int32)

    logger.info(
        "DyGFormer CUDA sampler path: backend=%s, %d trainable params, %d active train nodes",
        train_sampler.backend,
        trainable_params,
        len(train_active_nodes),
    )

    history, _summary = run_early_stopping_training(
        model=model,
        output_dir=output_dir,
        device=device,
        num_epochs=num_epochs,
        patience=patience,
        train_epoch_fn=lambda: train_epoch_cuda(
            model=model,
            data=data,
            sampler=train_sampler,
            edge_indices=train_indices,
            optimizer=optimizer,
            device=device,
            batch_size=batch_size,
            num_neighbors=num_neighbors,
            neg_per_positive=neg_per_positive,
            use_amp=use_amp,
            scaler=scaler,
            rng=rng,
            neg_node_pool=train_active_nodes,
        ),
        validate_fn=lambda: validate(
            model=model,
            data=data,
            csr=full_csr,
            edge_indices=val_indices,
            device=device,
            num_neighbors=num_neighbors,
            max_eval_edges=max_val_edges,
            use_amp=use_amp,
            rng=rng,
            neg_node_pool=train_active_nodes,
        ),
        logger=logger,
        train_loss_format="%.8f",
    )
    return model, history


__all__ = [
    "prepare_dygformer_batch_cuda",
    "train_dygformer_cuda",
    "train_epoch_cuda",
]

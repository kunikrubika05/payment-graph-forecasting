"""Training and validation for DyGFormer temporal link prediction."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn

from payment_graph_forecasting.evaluation.temporal_ranking import conservative_rank_from_scores
from payment_graph_forecasting.training.amp import (
    amp_enabled_for_device,
    autocast_context,
    create_grad_scaler,
    seed_torch,
)
from payment_graph_forecasting.training.epoch import run_loss_epoch
from payment_graph_forecasting.training.temporal_context import (
    NodeTemporalContext,
    sample_node_contexts,
    to_device_tensor,
)
from payment_graph_forecasting.training.trainer import run_early_stopping_training
from src.models.DyGFormer.dygformer import DyGFormerTime
from src.models.DyGFormer.data_utils import (
    TemporalCSR,
    TemporalEdgeData,
    build_temporal_csr,
    sample_neighbors_batch,
)

logger = logging.getLogger(__name__)


def compute_neighbor_cooccurrence(
    src_nids: np.ndarray,
    src_lens: np.ndarray,
    dst_nids: np.ndarray,
    dst_lens: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute DyGFormer neighbor co-occurrence features for aligned batches."""

    _, num_neighbors = src_nids.shape
    src_mask = np.arange(num_neighbors)[None, :] < src_lens[:, None]
    dst_mask = np.arange(num_neighbors)[None, :] < dst_lens[:, None]

    src_self = (src_nids[:, :, None] == src_nids[:, None, :]) & src_mask[:, None, :]
    src_cross = (src_nids[:, :, None] == dst_nids[:, None, :]) & dst_mask[:, None, :]
    dst_self = (dst_nids[:, :, None] == dst_nids[:, None, :]) & dst_mask[:, None, :]
    dst_cross = (dst_nids[:, :, None] == src_nids[:, None, :]) & src_mask[:, None, :]

    src_cooc = np.zeros((len(src_nids), num_neighbors, 2), dtype=np.float32)
    dst_cooc = np.zeros((len(dst_nids), num_neighbors, 2), dtype=np.float32)
    src_cooc[:, :, 0] = src_self.sum(axis=2) * src_mask
    src_cooc[:, :, 1] = src_cross.sum(axis=2) * src_mask
    dst_cooc[:, :, 0] = dst_self.sum(axis=2) * dst_mask
    dst_cooc[:, :, 1] = dst_cross.sum(axis=2) * dst_mask
    return src_cooc, dst_cooc


def _sample_contexts(
    *,
    csr: TemporalCSR,
    data: TemporalEdgeData,
    nodes: np.ndarray,
    timestamps: np.ndarray,
    num_neighbors: int,
    use_edge_feats: bool,
    use_node_feats: bool,
) -> NodeTemporalContext:
    return sample_node_contexts(
        csr=csr,
        data=data,
        sample_neighbors_fn=sample_neighbors_batch,
        nodes=nodes,
        query_timestamps=timestamps,
        num_neighbors=num_neighbors,
        use_edge_feats=use_edge_feats,
        use_node_feats=use_node_feats,
        zero_pad_delta=True,
    )


def prepare_dygformer_batch(
    *,
    csr: TemporalCSR,
    data: TemporalEdgeData,
    src_nodes: np.ndarray,
    dst_nodes: np.ndarray,
    timestamps: np.ndarray,
    neg_dst_nodes: np.ndarray,
    num_neighbors: int,
    device: torch.device,
    use_edge_feats: bool = True,
    use_node_feats: bool = False,
) -> dict[str, torch.Tensor]:
    """Prepare one DyGFormer training batch with vectorized negative handling."""

    batch_size = len(src_nodes)
    num_neg = neg_dst_nodes.shape[1] if neg_dst_nodes.ndim > 1 else 1

    src_context = _sample_contexts(
        csr=csr,
        data=data,
        nodes=src_nodes,
        timestamps=timestamps,
        num_neighbors=num_neighbors,
        use_edge_feats=use_edge_feats,
        use_node_feats=use_node_feats,
    )
    pos_context = _sample_contexts(
        csr=csr,
        data=data,
        nodes=dst_nodes,
        timestamps=timestamps,
        num_neighbors=num_neighbors,
        use_edge_feats=use_edge_feats,
        use_node_feats=use_node_feats,
    )

    neg_flat = neg_dst_nodes.reshape(-1).astype(np.int32)
    neg_timestamps = np.repeat(timestamps, num_neg)
    neg_context = _sample_contexts(
        csr=csr,
        data=data,
        nodes=neg_flat,
        timestamps=neg_timestamps,
        num_neighbors=num_neighbors,
        use_edge_feats=use_edge_feats,
        use_node_feats=use_node_feats,
    )

    pos_src_cooc, pos_dst_cooc = compute_neighbor_cooccurrence(
        src_context.neighbor_ids,
        src_context.lengths,
        pos_context.neighbor_ids,
        pos_context.lengths,
    )

    neg_src_cooc, neg_dst_cooc = compute_neighbor_cooccurrence(
        np.repeat(src_context.neighbor_ids, num_neg, axis=0),
        np.repeat(src_context.lengths, num_neg),
        neg_context.neighbor_ids,
        neg_context.lengths,
    )

    batch = {
        "src_delta_times": to_device_tensor(src_context.delta_times, device),
        "src_lengths": to_device_tensor(src_context.lengths, device, torch.int64),
        "pos_dst_delta_times": to_device_tensor(pos_context.delta_times, device),
        "pos_dst_lengths": to_device_tensor(pos_context.lengths, device, torch.int64),
        "pos_src_cooc": to_device_tensor(pos_src_cooc, device),
        "pos_dst_cooc": to_device_tensor(pos_dst_cooc, device),
        "neg_dst_delta_times": to_device_tensor(
            neg_context.delta_times.reshape(batch_size, num_neg, num_neighbors), device
        ),
        "neg_dst_lengths": to_device_tensor(
            neg_context.lengths.reshape(batch_size, num_neg), device, torch.int64
        ),
        "neg_src_cooc": to_device_tensor(
            neg_src_cooc.reshape(batch_size, num_neg, num_neighbors, 2), device
        ),
        "neg_dst_cooc": to_device_tensor(
            neg_dst_cooc.reshape(batch_size, num_neg, num_neighbors, 2), device
        ),
    }

    if use_edge_feats:
        edge_dim = int(src_context.edge_features.shape[-1])
        batch["src_edge_feats"] = to_device_tensor(src_context.edge_features, device)
        batch["pos_dst_edge_feats"] = to_device_tensor(pos_context.edge_features, device)
        batch["neg_dst_edge_feats"] = to_device_tensor(
            neg_context.edge_features.reshape(batch_size, num_neg, num_neighbors, edge_dim), device
        )

    if use_node_feats:
        node_dim = int(src_context.node_features.shape[-1])
        batch["src_node_feats"] = to_device_tensor(src_context.node_features, device)
        batch["pos_dst_node_feats"] = to_device_tensor(pos_context.node_features, device)
        batch["neg_dst_node_feats"] = to_device_tensor(
            neg_context.node_features.reshape(batch_size, num_neg, num_neighbors, node_dim), device
        )

    return batch


def score_dygformer_candidates(
    *,
    model: DyGFormerTime,
    src_context: NodeTemporalContext,
    dst_context: NodeTemporalContext,
    device: torch.device,
    amp_enabled: bool,
) -> np.ndarray:
    """Score all destination candidates for one source in a single model pass."""

    num_candidates = int(dst_context.delta_times.shape[0])
    src_cooc, dst_cooc = compute_neighbor_cooccurrence(
        np.repeat(src_context.neighbor_ids, num_candidates, axis=0),
        np.repeat(src_context.lengths, num_candidates),
        dst_context.neighbor_ids,
        dst_context.lengths,
    )

    with autocast_context(amp_enabled, device.type):
        logits = model(
            src_delta_times=to_device_tensor(np.repeat(src_context.delta_times, num_candidates, axis=0), device),
            src_lengths=to_device_tensor(np.repeat(src_context.lengths, num_candidates), device, torch.int64),
            dst_delta_times=to_device_tensor(dst_context.delta_times, device),
            dst_lengths=to_device_tensor(dst_context.lengths, device, torch.int64),
            src_cooc_counts=to_device_tensor(src_cooc, device),
            dst_cooc_counts=to_device_tensor(dst_cooc, device),
            src_edge_feats=to_device_tensor(
                np.repeat(src_context.edge_features, num_candidates, axis=0), device
            ) if src_context.edge_features is not None else None,
            dst_edge_feats=to_device_tensor(dst_context.edge_features, device)
            if dst_context.edge_features is not None else None,
            src_node_feats=to_device_tensor(
                np.repeat(src_context.node_features, num_candidates, axis=0), device
            ) if src_context.node_features is not None else None,
            dst_node_feats=to_device_tensor(dst_context.node_features, device)
            if dst_context.node_features is not None else None,
        )
    return logits.detach().cpu().float().numpy()


def train_epoch(
    model: DyGFormerTime,
    data: TemporalEdgeData,
    csr: TemporalCSR,
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
    """Run one DyGFormer epoch."""

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

        batch = prepare_dygformer_batch(
            csr=csr,
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


@torch.no_grad()
def validate(
    model: DyGFormerTime,
    data: TemporalEdgeData,
    csr: TemporalCSR,
    edge_indices: np.ndarray,
    device: torch.device,
    num_neighbors: int = 32,
    n_eval_negatives: int = 100,
    max_eval_edges: int = 5000,
    use_amp: bool = True,
    rng: Optional[np.random.Generator] = None,
    neg_node_pool: Optional[np.ndarray] = None,
) -> dict[str, float]:
    """Validate DyGFormer with batched candidate scoring."""

    if rng is None:
        rng = np.random.default_rng(42)

    model.eval()
    amp_enabled = amp_enabled_for_device(use_amp, device)
    use_edge_feats = model.edge_feat_dim > 0
    use_node_feats = model.node_feat_dim > 0

    if len(edge_indices) > max_eval_edges:
        eval_indices = rng.choice(edge_indices, size=max_eval_edges, replace=False)
    else:
        eval_indices = edge_indices

    ranks: list[float] = []
    for edge_idx in eval_indices:
        src_node = int(data.src[edge_idx])
        true_dst = int(data.dst[edge_idx])
        timestamp = float(data.timestamps[edge_idx])

        if neg_node_pool is not None:
            neg_indices = rng.integers(0, len(neg_node_pool), size=n_eval_negatives)
            neg_nodes = neg_node_pool[neg_indices].astype(np.int32)
        else:
            neg_nodes = rng.integers(0, data.num_nodes, size=n_eval_negatives).astype(np.int32)

        all_dst = np.concatenate(([true_dst], neg_nodes)).astype(np.int32)
        src_context = _sample_contexts(
            csr=csr,
            data=data,
            nodes=np.array([src_node], dtype=np.int32),
            timestamps=np.array([timestamp], dtype=np.float64),
            num_neighbors=num_neighbors,
            use_edge_feats=use_edge_feats,
            use_node_feats=use_node_feats,
        )
        dst_context = _sample_contexts(
            csr=csr,
            data=data,
            nodes=all_dst,
            timestamps=np.full(len(all_dst), timestamp, dtype=np.float64),
            num_neighbors=num_neighbors,
            use_edge_feats=use_edge_feats,
            use_node_feats=use_node_feats,
        )
        scores = score_dygformer_candidates(
            model=model,
            src_context=src_context,
            dst_context=dst_context,
            device=device,
            amp_enabled=amp_enabled,
        )
        ranks.append(conservative_rank_from_scores(scores))

    ranks_arr = np.asarray(ranks, dtype=np.float64)
    return {
        "mrr": float(np.mean(1.0 / ranks_arr)),
        "hits@1": float(np.mean(ranks_arr <= 1)),
        "hits@3": float(np.mean(ranks_arr <= 3)),
        "hits@10": float(np.mean(ranks_arr <= 10)),
        "mean_rank": float(np.mean(ranks_arr)),
        "n_queries": int(len(ranks_arr)),
    }


def train_dygformer(
    data: TemporalEdgeData,
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
) -> tuple[DyGFormerTime, dict[str, list[float]]]:
    """Full DyGFormer training pipeline with early stopping."""

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    seed_torch(seed, device=device)
    rng = np.random.default_rng(seed)

    train_csr = build_temporal_csr(data, train_mask)
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
        "DyGFormer: %d trainable params, %d active train nodes",
        trainable_params,
        len(train_active_nodes),
    )

    history, _summary = run_early_stopping_training(
        model=model,
        output_dir=output_dir,
        device=device,
        num_epochs=num_epochs,
        patience=patience,
        train_epoch_fn=lambda: train_epoch(
            model=model,
            data=data,
            csr=train_csr,
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
    "compute_neighbor_cooccurrence",
    "prepare_dygformer_batch",
    "score_dygformer_candidates",
    "train_epoch",
    "train_dygformer",
    "validate",
]

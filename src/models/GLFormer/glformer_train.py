"""Training and validation for GLFormer temporal link prediction.

Reuses the same data infrastructure as EAGLE (TemporalCSR, neighbor
sampling, featurize_neighbors). Key differences from eagle_train.py:

    1. Co-occurrence computation: when use_cooccurrence=True, intersection
       sizes between src and dst neighbor sets are pre-computed before the
       forward pass and passed as cooc_counts to the model.

    2. Neighbor ordering: sequences are kept in most-recent-first order
       (matching sample_neighbors_batch output) — the AdaptiveTokenMixer
       handles the causal aggregation direction internally.

    3. Edge feature mode: GLFormer is designed for use with edge features
       (btc/usd per neighbor interaction). The time-only mode is supported
       but is not the primary intended configuration.
"""

import json
import logging
import os
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

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
from src.models.GLFormer.glformer import GLFormerTime
from src.models.GLFormer.data_utils import (
    TemporalEdgeData,
    TemporalCSR,
    build_temporal_csr,
    sample_neighbors_batch,
)

logger = logging.getLogger(__name__)


def _compute_cooccurrence(
    src_nids: np.ndarray,
    src_lens: np.ndarray,
    dst_nids: np.ndarray,
    dst_lens: np.ndarray,
) -> np.ndarray:
    """Compute intersection size between sampled neighbor sets (fallback).

    Uses only the K most-recent sampled neighbors — approximate and noisy
    for nodes with many neighbors. Prefer _compute_cn_from_adj when a
    pre-built adjacency matrix is available.

    Args:
        src_nids: [B, K] source neighbor node IDs.
        src_lens: [B] number of valid source neighbors.
        dst_nids: [B, K] destination neighbor node IDs.
        dst_lens: [B] number of valid destination neighbors.

    Returns:
        [B] float32 array of intersection sizes.
    """
    B = len(src_lens)
    cooc = np.zeros(B, dtype=np.float32)
    for i in range(B):
        s_set = set(src_nids[i, :src_lens[i]].tolist())
        d_set = set(dst_nids[i, :dst_lens[i]].tolist())
        cooc[i] = float(len(s_set & d_set))
    return cooc


def _compute_cn_from_adj(
    adj,
    node_mapping: np.ndarray,
    src_global: np.ndarray,
    dst_global: np.ndarray,
) -> np.ndarray:
    """Compute Common Neighbors using the full train adjacency matrix.

    Looks up each (src, dst) pair in the pre-built sparse adjacency matrix
    via adj[src].multiply(adj[dst]).sum(axis=1). Nodes absent from the
    training graph (not in node_mapping) get CN=0.

    Args:
        adj: scipy CSR binary adjacency matrix (local indices, n_active × n_active).
        node_mapping: int64 array of length n_active mapping local_idx → global_idx.
            Must be sorted (output of np.unique).
        src_global: [B] source global node IDs.
        dst_global: [B] destination global node IDs.

    Returns:
        [B] float32 array of Common Neighbor counts.
    """
    src_g = np.asarray(src_global, dtype=np.int64)
    dst_g = np.asarray(dst_global, dtype=np.int64)
    n = adj.shape[0]

    src_local = np.searchsorted(node_mapping, src_g)
    dst_local = np.searchsorted(node_mapping, dst_g)

    src_in = (src_local < n) & (node_mapping[np.minimum(src_local, n - 1)] == src_g)
    dst_in = (dst_local < n) & (node_mapping[np.minimum(dst_local, n - 1)] == dst_g)
    valid = src_in & dst_in

    result = np.zeros(len(src_g), dtype=np.float32)
    if valid.any():
        sv, dv = src_local[valid], dst_local[valid]
        result[valid] = np.array(
            adj[sv].multiply(adj[dv]).sum(axis=1), dtype=np.float32
        ).ravel()
    return result


def prepare_glformer_batch(
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
    use_cooccurrence: bool = False,
    adj=None,
    node_mapping: Optional[np.ndarray] = None,
) -> Dict[str, torch.Tensor]:
    """Prepare a training batch for GLFormer.

    Samples K most-recent neighbors for each node (src, pos_dst, neg_dst),
    computes delta times and optional edge/node/co-occurrence features,
    and returns everything as tensors on the target device.

    Neighbor sequences are in most-recent-first order (index 0 = most recent),
    matching the output of sample_neighbors_batch and the convention expected
    by AdaptiveTokenMixer.

    When use_cooccurrence=True, intersection sizes between src and dst neighbor
    sets are pre-computed here (before the model forward pass) and included
    in the returned batch dict as 'pos_cooc_counts' and 'neg_cooc_counts'.

    Args:
        csr: Temporal CSR for neighbor lookups.
        data: Full temporal edge data.
        src_nodes: [B] source node indices.
        dst_nodes: [B] positive destination indices.
        timestamps: [B] query timestamps.
        neg_dst_nodes: [B, num_neg] negative destination indices.
        num_neighbors: K neighbors to sample per node.
        device: Target torch device.
        use_edge_feats: Include per-neighbor edge feature vectors (btc/usd).
        use_node_feats: Include query-node own feature vectors.
        use_cooccurrence: Compute and include shared-neighbor counts.
        adj: scipy CSR binary adjacency matrix (local indices) for full-graph
            CN computation. When provided (together with node_mapping), CN is
            computed from the full train adjacency instead of the K-sampled
            neighbor intersection. Nodes absent from the training graph get
            CN=0.
        node_mapping: int64 sorted array mapping local_idx → global_idx.
            Required when adj is not None.

    Returns:
        Dictionary of tensors for the GLFormerTime forward pass.
        Keys always present: src_delta_times, src_lengths,
            pos_dst_delta_times, pos_dst_lengths,
            neg_dst_delta_times, neg_dst_lengths.
        Optional keys: src_edge_feats, pos_dst_edge_feats, neg_dst_edge_feats,
            src_node_feats, pos_dst_node_feats, neg_dst_node_feats,
            pos_cooc_counts, neg_cooc_counts.
    """
    batch_size = len(src_nodes)
    num_neg = neg_dst_nodes.shape[1] if neg_dst_nodes.ndim > 1 else 1

    def _get_neighbors(nodes, ts_arr):
        """Sample neighbors and extract features for a set of nodes."""
        return sample_node_contexts(
            csr=csr,
            data=data,
            sample_neighbors_fn=sample_neighbors_batch,
            nodes=nodes,
            query_timestamps=ts_arr,
            num_neighbors=num_neighbors,
            use_edge_feats=use_edge_feats,
            use_node_feats=use_node_feats,
            zero_pad_delta=True,
        )

    src_context = _get_neighbors(src_nodes, timestamps)
    pos_context = _get_neighbors(dst_nodes, timestamps)

    neg_flat = neg_dst_nodes.reshape(-1)
    neg_ts = np.repeat(timestamps, num_neg)
    neg_context = _get_neighbors(neg_flat, neg_ts)

    batch = {
        "src_delta_times":      to_device_tensor(src_context.delta_times, device),
        "src_lengths":          to_device_tensor(src_context.lengths, device, torch.int64),
        "pos_dst_delta_times":  to_device_tensor(pos_context.delta_times, device),
        "pos_dst_lengths":      to_device_tensor(pos_context.lengths, device, torch.int64),
        "neg_dst_delta_times":  to_device_tensor(neg_context.delta_times.reshape(batch_size, num_neg, num_neighbors), device),
        "neg_dst_lengths":      to_device_tensor(neg_context.lengths.reshape(batch_size, num_neg), device, torch.int64),
    }

    if use_edge_feats:
        ef_dim = src_context.edge_features.shape[-1]
        batch["src_edge_feats"]     = to_device_tensor(src_context.edge_features, device)
        batch["pos_dst_edge_feats"] = to_device_tensor(pos_context.edge_features, device)
        batch["neg_dst_edge_feats"] = to_device_tensor(
            neg_context.edge_features.reshape(batch_size, num_neg, num_neighbors, ef_dim), device
        )

    if use_node_feats:
        batch["src_node_feats"]     = to_device_tensor(src_context.node_features, device)
        batch["pos_dst_node_feats"] = to_device_tensor(pos_context.node_features, device)
        batch["neg_dst_node_feats"] = to_device_tensor(neg_context.node_features.reshape(batch_size, num_neg, -1), device)

    if use_cooccurrence:
        if adj is not None and node_mapping is not None:
            pos_cooc = _compute_cn_from_adj(
                adj, node_mapping,
                src_nodes.astype(np.int64),
                dst_nodes.astype(np.int64),
            )
            neg_cooc_flat = _compute_cn_from_adj(
                adj, node_mapping,
                np.repeat(src_nodes.astype(np.int64), num_neg),
                neg_flat.astype(np.int64),
            )
        else:
            pos_cooc = _compute_cooccurrence(
                src_context.neighbor_ids, src_context.lengths,
                pos_context.neighbor_ids, pos_context.lengths,
            )
            src_nids_rep = np.repeat(src_context.neighbor_ids, num_neg, axis=0)
            src_len_rep = np.repeat(src_context.lengths, num_neg)
            neg_cooc_flat = _compute_cooccurrence(
                src_nids_rep, src_len_rep, neg_context.neighbor_ids, neg_context.lengths
            )
        batch["pos_cooc_counts"] = to_device_tensor(pos_cooc, device)
        batch["neg_cooc_counts"] = to_device_tensor(neg_cooc_flat.reshape(batch_size, num_neg), device)

    return batch


def train_epoch(
    model: GLFormerTime,
    data: TemporalEdgeData,
    csr: TemporalCSR,
    edge_indices: np.ndarray,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    batch_size: int = 200,
    num_neighbors: int = 20,
    neg_per_positive: int = 5,
    use_amp: bool = True,
    scaler: Optional[torch.cuda.amp.GradScaler] = None,
    rng: Optional[np.random.Generator] = None,
    neg_node_pool: Optional[np.ndarray] = None,
    adj=None,
    node_mapping: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """Run one training epoch for GLFormer.

    Shuffles edges, iterates in batches, computes BCE loss on positive and
    negative edge predictions, and updates model parameters.

    Args:
        model: GLFormerTime model.
        data: Temporal edge data.
        csr: Temporal CSR built from training edges only.
        edge_indices: Indices of training edges to iterate over.
        optimizer: Torch optimizer.
        device: Torch device.
        batch_size: Number of edges per batch.
        num_neighbors: K neighbors sampled per node.
        neg_per_positive: Number of random negatives per positive edge.
        use_amp: Enable mixed-precision training (CUDA only).
        scaler: GradScaler instance for AMP.
        rng: Random number generator.
        neg_node_pool: Node index pool for negative sampling. When provided,
            negatives are drawn from this array instead of the full node range.
            Pass unique active training nodes to avoid trivially-easy inactive
            negatives in datasets with sparse global node spaces.

    Returns:
        Dict with 'loss' (mean batch loss over the epoch).
    """
    if rng is None:
        rng = np.random.default_rng()

    model.train()
    criterion = nn.BCEWithLogitsLoss()
    amp_enabled = use_amp and device.type == "cuda"

    use_edge_feats = model.edge_feat_dim > 0
    use_node_feats = model.node_feat_dim > 0
    use_cooc = model.use_cooccurrence

    def _loss_for_batch(batch_idx: np.ndarray) -> torch.Tensor:
        B = len(batch_idx)

        src = data.src[batch_idx]
        dst = data.dst[batch_idx]
        ts = data.timestamps[batch_idx]
        if neg_node_pool is not None:
            neg_idx = rng.integers(0, len(neg_node_pool), size=(B, neg_per_positive))
            neg_dst = neg_node_pool[neg_idx]
        else:
            neg_dst = rng.integers(0, data.num_nodes, size=(B, neg_per_positive))

        batch = prepare_glformer_batch(
            csr, data, src, dst, ts, neg_dst, num_neighbors, device,
            use_edge_feats=use_edge_feats,
            use_node_feats=use_node_feats,
            use_cooccurrence=use_cooc,
            adj=adj,
            node_mapping=node_mapping,
        )

        with autocast_context(amp_enabled, device.type):
            pos_logits = model(
                src_delta_times=batch["src_delta_times"],
                src_lengths=batch["src_lengths"],
                dst_delta_times=batch["pos_dst_delta_times"],
                dst_lengths=batch["pos_dst_lengths"],
                src_edge_feats=batch.get("src_edge_feats"),
                dst_edge_feats=batch.get("pos_dst_edge_feats"),
                src_node_feats=batch.get("src_node_feats"),
                dst_node_feats=batch.get("pos_dst_node_feats"),
                cooc_counts=batch.get("pos_cooc_counts"),
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
                neg_cooc_i = (
                    batch["neg_cooc_counts"][:, neg_i]
                    if use_cooc else None
                )
                neg_logits_list.append(model(
                    src_delta_times=batch["src_delta_times"],
                    src_lengths=batch["src_lengths"],
                    dst_delta_times=batch["neg_dst_delta_times"][:, neg_i, :],
                    dst_lengths=batch["neg_dst_lengths"][:, neg_i],
                    src_edge_feats=batch.get("src_edge_feats"),
                    dst_edge_feats=neg_ef_i,
                    src_node_feats=batch.get("src_node_feats"),
                    dst_node_feats=neg_nf_i,
                    cooc_counts=neg_cooc_i,
                ))

            all_logits = torch.cat([pos_logits] + neg_logits_list)
            all_labels = torch.cat([
                torch.ones(B, device=device),
                torch.zeros(B * neg_per_positive, device=device),
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
        loss_format="{:.8f}",
    )


@torch.no_grad()
def validate(
    model: GLFormerTime,
    data: TemporalEdgeData,
    csr: TemporalCSR,
    edge_indices: np.ndarray,
    device: torch.device,
    num_neighbors: int = 20,
    n_eval_negatives: int = 100,
    max_eval_edges: int = 5000,
    use_amp: bool = True,
    rng: Optional[np.random.Generator] = None,
    neg_node_pool: Optional[np.ndarray] = None,
    adj=None,
    node_mapping: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """Validate GLFormer with ranking metrics.

    For each validation edge, generates n_eval_negatives negatives,
    scores all candidates, and computes the rank of the true destination.

    Args:
        model: GLFormerTime model.
        data: Temporal edge data.
        csr: Temporal CSR (train+val edges for final eval, train-only for
            mid-training validation).
        edge_indices: Indices of edges to evaluate.
        device: Torch device.
        num_neighbors: K neighbors sampled per node.
        n_eval_negatives: Number of negatives per query.
        max_eval_edges: Maximum edges to evaluate (subsampled for speed).
        use_amp: Enable mixed precision.
        rng: Random number generator.
        neg_node_pool: Node index pool for negative sampling. When provided,
            negatives are drawn from this array instead of the full node range.

    Returns:
        Dict with 'mrr', 'hits@1', 'hits@3', 'hits@10', 'mean_rank',
        'n_queries'.
    """
    if rng is None:
        rng = np.random.default_rng(42)

    model.eval()
    amp_enabled = use_amp and device.type == "cuda"
    K = num_neighbors
    use_edge_feats = model.edge_feat_dim > 0
    use_node_feats = model.node_feat_dim > 0
    use_cooc = model.use_cooccurrence

    if len(edge_indices) > max_eval_edges:
        eval_idx = rng.choice(edge_indices, size=max_eval_edges, replace=False)
    else:
        eval_idx = edge_indices

    ranks = []

    for idx in eval_idx:
        src_node = data.src[idx]
        true_dst = data.dst[idx]
        ts = data.timestamps[idx]

        if neg_node_pool is not None:
            idx = rng.integers(0, len(neg_node_pool), size=n_eval_negatives)
            neg_nodes = neg_node_pool[idx].astype(np.int32)
        else:
            neg_nodes = rng.integers(
                0, data.num_nodes, size=n_eval_negatives
            ).astype(np.int32)
        all_dst = np.concatenate([[true_dst], neg_nodes])
        C = len(all_dst)

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
            zero_pad_delta=True,
        )

        dst_ts_arr = np.full(C, ts, dtype=np.float64)
        dst_context = sample_node_contexts(
            csr=csr,
            data=data,
            sample_neighbors_fn=sample_neighbors_batch,
            nodes=all_dst,
            query_timestamps=dst_ts_arr,
            num_neighbors=K,
            use_edge_feats=use_edge_feats,
            use_node_feats=use_node_feats,
            zero_pad_delta=True,
        )

        cooc_counts = None
        if use_cooc:
            if adj is not None and node_mapping is not None:
                cooc_np = _compute_cn_from_adj(
                    adj, node_mapping,
                    np.full(C, src_node, dtype=np.int64),
                    all_dst.astype(np.int64),
                )
            else:
                src_nids_rep = np.repeat(src_context.neighbor_ids, C, axis=0)
                src_lens_rep = np.repeat(src_context.lengths, C)
                cooc_np = _compute_cooccurrence(
                    src_nids_rep, src_lens_rep, dst_context.neighbor_ids, dst_context.lengths
                )
            cooc_counts = torch.tensor(cooc_np, dtype=torch.float32, device=device)

        scores = score_candidate_contexts(
            model=model,
            device=device,
            src_context=src_context,
            dst_context=dst_context,
            amp_enabled=amp_enabled,
            cooc_counts=cooc_counts,
        )
        rank = conservative_rank_from_scores(scores)
        ranks.append(rank)

    ranks = np.array(ranks, dtype=np.float64)
    return {
        "mrr":       float(np.mean(1.0 / ranks)),
        "hits@1":    float(np.mean(ranks <= 1)),
        "hits@3":    float(np.mean(ranks <= 3)),
        "hits@10":   float(np.mean(ranks <= 10)),
        "mean_rank": float(np.mean(ranks)),
        "n_queries": len(ranks),
    }


def train_glformer(
    data: TemporalEdgeData,
    train_mask: np.ndarray,
    val_mask: np.ndarray,
    output_dir: str,
    device: Optional[torch.device] = None,
    num_epochs: int = 100,
    batch_size: int = 200,
    learning_rate: float = 0.0001,
    weight_decay: float = 1e-5,
    num_neighbors: int = 20,
    hidden_dim: int = 100,
    num_glformer_layers: int = 2,
    channel_expansion: float = 4.0,
    dropout: float = 0.1,
    patience: int = 20,
    seed: int = 42,
    max_val_edges: int = 5000,
    use_amp: bool = True,
    edge_feat_dim: int = 2,
    node_feat_dim: int = 0,
    use_cooccurrence: bool = False,
    cooc_dim: int = 16,
    adj=None,
    node_mapping: Optional[np.ndarray] = None,
) -> Tuple[GLFormerTime, Dict]:
    """Full GLFormer training pipeline with early stopping.

    Builds train/val CSR structures, creates the model, trains for up to
    num_epochs epochs, saves the best checkpoint by val MRR, and restores
    best weights at the end.

    Args:
        data: TemporalEdgeData loaded from a stream graph parquet file.
        train_mask: Boolean mask selecting training edges.
        val_mask: Boolean mask selecting validation edges.
        output_dir: Directory for checkpoints, config, and metrics.
        device: Torch device. Auto-detected (CUDA > CPU) if None.
        num_epochs: Maximum training epochs.
        batch_size: Training batch size.
        learning_rate: Adam learning rate.
        weight_decay: Adam L2 regularization.
        num_neighbors: K most-recent neighbors sampled per node.
        hidden_dim: Hidden dimension for all model components.
        num_glformer_layers: Number of stacked GLFormerBlocks (1-3).
        channel_expansion: FFN expansion factor in channel mixer.
        dropout: Dropout rate.
        patience: Early stopping patience (epochs without val MRR improvement).
        seed: Random seed for reproducibility.
        max_val_edges: Maximum edges evaluated per validation pass.
        use_amp: Enable AMP mixed precision (CUDA only).
        edge_feat_dim: Per-neighbor edge feature dimension (2 for btc+usd).
        node_feat_dim: Query-node feature dimension (0 = disabled).
        use_cooccurrence: Enable shared-neighbor co-occurrence features.
        cooc_dim: Co-occurrence encoding dimension.
        adj: scipy CSR binary adjacency matrix for full-graph CN computation.
            Built from train edges only. When None, falls back to K-sampled
            neighbor intersection.
        node_mapping: int64 sorted array mapping local_idx → global_idx.
            Required when adj is not None.

    Returns:
        Tuple of (best model with loaded checkpoint, training history dict).
        History keys: train_loss, val_mrr, val_hits@1, val_hits@3,
            val_hits@10, epoch_time.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(seed)
    seed_torch(seed, device)

    train_csr = build_temporal_csr(data, train_mask)
    full_csr = build_temporal_csr(data, train_mask | val_mask)


    model = GLFormerTime(
        hidden_dim=hidden_dim,
        num_neighbors=num_neighbors,
        num_glformer_layers=num_glformer_layers,
        channel_expansion=channel_expansion,
        dropout=dropout,
        edge_feat_dim=edge_feat_dim,
        node_feat_dim=node_feat_dim,
        use_cooccurrence=use_cooccurrence,
        cooc_dim=cooc_dim,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    logger.info(
        "GLFormer: %d total params, %d trainable",
        total_params, trainable_params,
    )

    optimizer = torch.optim.Adam(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    amp_enabled = amp_enabled_for_device(use_amp, device)
    scaler = create_grad_scaler(amp_enabled)

    train_indices = np.where(train_mask)[0]
    val_indices = np.where(val_mask)[0]

    train_active_nodes = np.unique(
        np.concatenate([data.src[train_indices], data.dst[train_indices]])
    ).astype(np.int32)
    logger.info(
        "Active training nodes: %d (out of %d total)",
        len(train_active_nodes), data.num_nodes,
    )

    config = {
        "model": "GLFormerTime",
        "hidden_dim": hidden_dim,
        "num_neighbors": num_neighbors,
        "num_glformer_layers": num_glformer_layers,
        "channel_expansion": channel_expansion,
        "dropout": dropout,
        "edge_feat_dim": edge_feat_dim,
        "node_feat_dim": node_feat_dim,
        "use_cooccurrence": use_cooccurrence,
        "cooc_dim": cooc_dim,
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
        "Training GLFormer: %d epochs, %d train, %d val edges",
        num_epochs, len(train_indices), len(val_indices),
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
            neg_node_pool=train_active_nodes,
            adj=adj,
            node_mapping=node_mapping,
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
            neg_node_pool=train_active_nodes,
            adj=adj,
            node_mapping=node_mapping,
        ),
        logger=logger,
        train_loss_format="%.8f",
    )
    return model, history

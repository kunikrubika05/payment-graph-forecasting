"""Training and validation for DyGFormer temporal link prediction.

Key differences from EAGLE/GLFormer training:

    1. Per-neighbor co-occurrence encoding (NCoE): for each (src, dst) pair,
       we count how many times each neighbor in src's sequence also appears
       in dst's sequence, and vice versa. This produces a [S, 2] count
       matrix per sequence (not a single scalar per pair).

    2. Joint encoding: DyGFormer processes src and dst together through
       a shared Transformer, so each negative destination requires a
       separate encoder forward pass (cannot reuse src encoding).

    3. Patching: sequences are divided into patches of size P before
       being fed to the Transformer. This is handled inside the model.
"""

import contextlib
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

from src.models.DyGFormer.dygformer import DyGFormerTime
from src.models.DyGFormer.data_utils import (
    TemporalEdgeData,
    TemporalCSR,
    build_temporal_csr,
    sample_neighbors_batch,
)
from src.models.data_utils import featurize_neighbors

logger = logging.getLogger(__name__)


def _amp_autocast(enabled: bool, device_type: str):
    """Return AMP autocast context or a no-op context manager."""
    if enabled and device_type == "cuda":
        return torch.amp.autocast("cuda")
    return contextlib.nullcontext()


def compute_neighbor_cooccurrence(
    src_nids: np.ndarray,
    src_lens: np.ndarray,
    dst_nids: np.ndarray,
    dst_lens: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute per-neighbor co-occurrence counts for NCoE.

    For each neighbor in src's sequence, counts:
        [count_in_src_seq, count_in_dst_seq]
    and symmetrically for each neighbor in dst's sequence.

    This is the core of DyGFormer's Neighbor Co-occurrence Encoding
    scheme (Section 4.1, Eq. 1 in the paper).

    Args:
        src_nids: [B, K] source neighbor node IDs (padded with 0).
        src_lens: [B] number of valid source neighbors.
        dst_nids: [B, K] destination neighbor node IDs (padded with 0).
        dst_lens: [B] number of valid destination neighbors.

    Returns:
        Tuple of:
            src_cooc: [B, K, 2] float32 — for each src neighbor position,
                [count_in_src_seq, count_in_dst_seq].
            dst_cooc: [B, K, 2] float32 — for each dst neighbor position,
                [count_in_dst_seq, count_in_src_seq].
    """
    B, K = src_nids.shape
    src_cooc = np.zeros((B, K, 2), dtype=np.float32)
    dst_cooc = np.zeros((B, K, 2), dtype=np.float32)

    for b in range(B):
        sl = int(src_lens[b])
        dl = int(dst_lens[b])
        if sl == 0 and dl == 0:
            continue

        src_seq = src_nids[b, :sl]
        dst_seq = dst_nids[b, :dl]

        src_unique, src_counts_in_src = np.unique(src_seq, return_counts=True)
        dst_unique, dst_counts_in_dst = np.unique(dst_seq, return_counts=True)

        src_count_map = dict(zip(src_unique.tolist(), src_counts_in_src.tolist()))
        dst_count_map = dict(zip(dst_unique.tolist(), dst_counts_in_dst.tolist()))

        for j in range(sl):
            nid = int(src_seq[j])
            src_cooc[b, j, 0] = src_count_map.get(nid, 0)
            src_cooc[b, j, 1] = dst_count_map.get(nid, 0)

        for j in range(dl):
            nid = int(dst_seq[j])
            dst_cooc[b, j, 0] = dst_count_map.get(nid, 0)
            dst_cooc[b, j, 1] = src_count_map.get(nid, 0)

    return src_cooc, dst_cooc


def _get_neighbors(csr, data, nodes, ts_arr, num_neighbors,
                   use_edge_feats, use_node_feats):
    """Sample neighbors and extract features for a set of nodes."""
    n_nids, neighbor_ts, neighbor_eids, lengths = sample_neighbors_batch(
        csr, nodes, ts_arr, num_neighbors
    )
    delta_times = np.maximum(
        ts_arr[:, None] - neighbor_ts, 0.0
    ).astype(np.float32)

    for b in range(len(nodes)):
        delta_times[b, lengths[b]:] = 0.0

    edge_feats_out = None
    node_feats_out = None

    if use_edge_feats or use_node_feats:
        nf_raw, ef_raw, _ = featurize_neighbors(
            n_nids, neighbor_eids, lengths,
            neighbor_ts, ts_arr,
            data.node_feats, data.edge_feats,
        )
        if use_edge_feats:
            edge_feats_out = ef_raw.astype(np.float32)
        if use_node_feats:
            node_feats_out = nf_raw.astype(np.float32)

    return n_nids, delta_times, edge_feats_out, node_feats_out, lengths


def prepare_dygformer_batch(
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
) -> Dict[str, torch.Tensor]:
    """Prepare a training batch for DyGFormer.

    Samples K most-recent neighbors for each node (src, pos_dst, neg_dst),
    computes delta times, edge/node features, and per-neighbor co-occurrence
    counts (NCoE), returning everything as tensors on the target device.

    Because DyGFormer encodes src and dst jointly, we prepare separate
    co-occurrence counts for each (src, pos_dst) and (src, neg_dst) pair.

    Args:
        csr: Temporal CSR for neighbor lookups.
        data: Full temporal edge data.
        src_nodes: [B] source node indices.
        dst_nodes: [B] positive destination indices.
        timestamps: [B] query timestamps.
        neg_dst_nodes: [B, num_neg] negative destination indices.
        num_neighbors: K neighbors to sample per node.
        device: Target torch device.
        use_edge_feats: Include per-neighbor edge feature vectors.
        use_node_feats: Include per-neighbor node feature vectors.

    Returns:
        Dictionary of tensors for the DyGFormerTime forward pass.
    """
    batch_size = len(src_nodes)
    num_neg = neg_dst_nodes.shape[1] if neg_dst_nodes.ndim > 1 else 1

    src_nids, src_dt, src_ef, src_nf, src_len = _get_neighbors(
        csr, data, src_nodes, timestamps, num_neighbors,
        use_edge_feats, use_node_feats,
    )
    pos_nids, pos_dt, pos_ef, pos_nf, pos_len = _get_neighbors(
        csr, data, dst_nodes, timestamps, num_neighbors,
        use_edge_feats, use_node_feats,
    )

    pos_src_cooc, pos_dst_cooc = compute_neighbor_cooccurrence(
        src_nids, src_len, pos_nids, pos_len
    )

    neg_flat = neg_dst_nodes.reshape(-1)
    neg_ts = np.repeat(timestamps, num_neg)
    neg_nids, neg_dt, neg_ef, neg_nf, neg_len = _get_neighbors(
        csr, data, neg_flat, neg_ts, num_neighbors,
        use_edge_feats, use_node_feats,
    )

    src_nids_rep = np.repeat(src_nids, num_neg, axis=0)
    src_len_rep = np.repeat(src_len, num_neg)
    neg_src_cooc, neg_dst_cooc = compute_neighbor_cooccurrence(
        src_nids_rep, src_len_rep, neg_nids, neg_len
    )

    def _t(arr, dtype=torch.float32):
        return torch.tensor(arr, dtype=dtype, device=device)

    batch = {
        "src_delta_times":      _t(src_dt),
        "src_lengths":          _t(src_len, torch.int64),
        "pos_dst_delta_times":  _t(pos_dt),
        "pos_dst_lengths":      _t(pos_len, torch.int64),
        "pos_src_cooc":         _t(pos_src_cooc),
        "pos_dst_cooc":         _t(pos_dst_cooc),
        "neg_dst_delta_times":  _t(neg_dt.reshape(batch_size, num_neg, num_neighbors)),
        "neg_dst_lengths":      _t(neg_len.reshape(batch_size, num_neg), torch.int64),
        "neg_src_cooc":         _t(neg_src_cooc.reshape(batch_size, num_neg, num_neighbors, 2)),
        "neg_dst_cooc":         _t(neg_dst_cooc.reshape(batch_size, num_neg, num_neighbors, 2)),
    }

    if use_edge_feats:
        ef_dim = src_ef.shape[-1]
        batch["src_edge_feats"]     = _t(src_ef)
        batch["pos_dst_edge_feats"] = _t(pos_ef)
        batch["neg_dst_edge_feats"] = _t(
            neg_ef.reshape(batch_size, num_neg, num_neighbors, ef_dim)
        )

    if use_node_feats:
        nf_dim = src_nf.shape[-1]
        batch["src_node_feats"]     = _t(src_nf)
        batch["pos_dst_node_feats"] = _t(pos_nf)
        batch["neg_dst_node_feats"] = _t(
            neg_nf.reshape(batch_size, num_neg, num_neighbors, nf_dim)
        )

    return batch


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
) -> Dict[str, float]:
    """Run one training epoch for DyGFormer.

    For each positive edge, samples neg_per_positive negatives. Because
    DyGFormer encodes src and dst jointly, each negative requires a
    separate forward pass through the encoder.

    Args:
        model: DyGFormerTime model.
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
        neg_node_pool: Node index pool for negative sampling.

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

    shuffled = rng.permutation(edge_indices)
    total_loss = 0.0
    num_batches = 0
    n_total = (len(shuffled) + batch_size - 1) // batch_size

    pbar = tqdm(
        range(0, len(shuffled), batch_size),
        total=n_total,
        desc="Training",
        leave=False,
        unit="batch",
    )
    for start in pbar:
        end = min(start + batch_size, len(shuffled))
        idx = shuffled[start:end]
        B = len(idx)

        src = data.src[idx]
        dst = data.dst[idx]
        ts = data.timestamps[idx]
        if neg_node_pool is not None:
            neg_idx = rng.integers(0, len(neg_node_pool), size=(B, neg_per_positive))
            neg_dst = neg_node_pool[neg_idx]
        else:
            neg_dst = rng.integers(0, data.num_nodes, size=(B, neg_per_positive))

        batch = prepare_dygformer_batch(
            csr, data, src, dst, ts, neg_dst, model.num_neighbors, device,
            use_edge_feats=use_edge_feats,
            use_node_feats=use_node_feats,
        )

        with _amp_autocast(amp_enabled, device.type):
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

            neg_logits_list = []
            for neg_i in range(neg_per_positive):
                neg_ef_i = (
                    batch["neg_dst_edge_feats"][:, neg_i, :, :]
                    if use_edge_feats else None
                )
                neg_nf_i = (
                    batch["neg_dst_node_feats"][:, neg_i, :, :]
                    if use_node_feats else None
                )
                neg_logits_list.append(model(
                    src_delta_times=batch["src_delta_times"],
                    src_lengths=batch["src_lengths"],
                    dst_delta_times=batch["neg_dst_delta_times"][:, neg_i, :],
                    dst_lengths=batch["neg_dst_lengths"][:, neg_i],
                    src_cooc_counts=batch["neg_src_cooc"][:, neg_i, :, :],
                    dst_cooc_counts=batch["neg_dst_cooc"][:, neg_i, :, :],
                    src_edge_feats=batch.get("src_edge_feats"),
                    dst_edge_feats=neg_ef_i,
                    src_node_feats=batch.get("src_node_feats"),
                    dst_node_feats=neg_nf_i,
                ))

            all_logits = torch.cat([pos_logits] + neg_logits_list)
            all_labels = torch.cat([
                torch.ones(B, device=device),
                torch.zeros(B * neg_per_positive, device=device),
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
        pbar.set_postfix(loss=f"{total_loss / num_batches:.8f}")

    pbar.close()
    return {"loss": total_loss / max(num_batches, 1)}


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
) -> Dict[str, float]:
    """Validate DyGFormer with ranking metrics.

    For each validation edge, generates n_eval_negatives negatives,
    scores all candidates jointly with the source, and computes
    the rank of the true destination.

    Note: because DyGFormer encodes src and dst jointly, each candidate
    requires a separate encoder pass. This makes validation slower than
    independent-encoder models (EAGLE, GLFormer).

    Args:
        model: DyGFormerTime model.
        data: Temporal edge data.
        csr: Temporal CSR.
        edge_indices: Indices of edges to evaluate.
        device: Torch device.
        num_neighbors: K neighbors sampled per node.
        n_eval_negatives: Number of negatives per query.
        max_eval_edges: Maximum edges to evaluate.
        use_amp: Enable mixed precision.
        rng: Random number generator.
        neg_node_pool: Node index pool for negative sampling.

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
            ni = rng.integers(0, len(neg_node_pool), size=n_eval_negatives)
            neg_nodes = neg_node_pool[ni].astype(np.int32)
        else:
            neg_nodes = rng.integers(
                0, data.num_nodes, size=n_eval_negatives
            ).astype(np.int32)
        all_dst = np.concatenate([[true_dst], neg_nodes])
        C = len(all_dst)

        src_arr = np.array([src_node], dtype=np.int32)
        ts_arr = np.array([ts], dtype=np.float64)
        src_nids, src_nts, src_neids, src_lens = sample_neighbors_batch(
            csr, src_arr, ts_arr, K
        )
        src_dt = np.maximum(ts_arr[:, None] - src_nts, 0.0).astype(np.float32)
        src_dt[0, src_lens[0]:] = 0.0

        src_ef = src_nf = None
        if use_edge_feats or use_node_feats:
            nf_raw, ef_raw, _ = featurize_neighbors(
                src_nids, src_neids, src_lens, src_nts, ts_arr,
                data.node_feats, data.edge_feats,
            )
            if use_edge_feats:
                src_ef = ef_raw.astype(np.float32)
            if use_node_feats:
                src_nf = nf_raw.astype(np.float32)

        scores = np.zeros(C, dtype=np.float32)

        for c_idx in range(C):
            dst_node = all_dst[c_idx]
            dst_arr = np.array([dst_node], dtype=np.int32)
            dst_ts_arr = np.array([ts], dtype=np.float64)
            dst_nids_c, dst_nts_c, dst_neids_c, dst_lens_c = sample_neighbors_batch(
                csr, dst_arr, dst_ts_arr, K
            )
            dst_dt_c = np.maximum(dst_ts_arr[:, None] - dst_nts_c, 0.0).astype(np.float32)
            dst_dt_c[0, dst_lens_c[0]:] = 0.0

            dst_ef_c = dst_nf_c = None
            if use_edge_feats or use_node_feats:
                nf_raw_c, ef_raw_c, _ = featurize_neighbors(
                    dst_nids_c, dst_neids_c, dst_lens_c, dst_nts_c, dst_ts_arr,
                    data.node_feats, data.edge_feats,
                )
                if use_edge_feats:
                    dst_ef_c = ef_raw_c.astype(np.float32)
                if use_node_feats:
                    dst_nf_c = nf_raw_c.astype(np.float32)

            src_cooc_c, dst_cooc_c = compute_neighbor_cooccurrence(
                src_nids, src_lens, dst_nids_c, dst_lens_c
            )

            def _t(arr, dtype=torch.float32):
                if arr is None:
                    return None
                return torch.tensor(arr, dtype=dtype, device=device)

            with _amp_autocast(amp_enabled, device.type):
                logit = model(
                    src_delta_times=_t(src_dt),
                    src_lengths=_t(src_lens, torch.int64),
                    dst_delta_times=_t(dst_dt_c),
                    dst_lengths=_t(dst_lens_c, torch.int64),
                    src_cooc_counts=_t(src_cooc_c),
                    dst_cooc_counts=_t(dst_cooc_c),
                    src_edge_feats=_t(src_ef),
                    dst_edge_feats=_t(dst_ef_c),
                    src_node_feats=_t(src_nf),
                    dst_node_feats=_t(dst_nf_c),
                )
                scores[c_idx] = logit.cpu().float().item()

        true_score = scores[0]
        rank = 1.0 + (scores[1:] > true_score).sum()
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


def train_dygformer(
    data: TemporalEdgeData,
    train_mask: np.ndarray,
    val_mask: np.ndarray,
    output_dir: str,
    device: Optional[torch.device] = None,
    num_epochs: int = 100,
    batch_size: int = 200,
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
) -> Tuple[DyGFormerTime, Dict]:
    """Full DyGFormer training pipeline with early stopping.

    Builds train/val CSR structures, creates the model, trains for up to
    num_epochs epochs, saves the best checkpoint by val MRR, and restores
    best weights at the end.

    Note: max_val_edges defaults to 2000 (lower than GLFormer's 5000)
    because DyGFormer's joint encoding makes validation significantly
    slower (each candidate requires a full encoder forward pass).

    Args:
        data: TemporalEdgeData loaded from a stream graph parquet file.
        train_mask: Boolean mask selecting training edges.
        val_mask: Boolean mask selecting validation edges.
        output_dir: Directory for checkpoints, config, and metrics.
        device: Torch device. Auto-detected (CUDA > CPU) if None.
        num_epochs: Maximum training epochs.
        batch_size: Training batch size.
        learning_rate: Adam learning rate (paper: 0.0001).
        weight_decay: Adam L2 regularization.
        num_neighbors: K most-recent neighbors sampled (paper: 32-4096).
        patch_size: Patch size P (paper: 1-128, scales with num_neighbors).
        time_dim: Time encoding dimension d_T (paper: 100).
        aligned_dim: Per-channel aligned dimension d (paper: 50).
        num_transformer_layers: Number of Transformer layers L (paper: 2).
        num_attention_heads: Number of attention heads I (paper: 2).
        cooc_dim: Co-occurrence encoding dimension d_C (paper: 50).
        output_dim: Output embedding dimension d_out (paper: 172).
        dropout: Dropout rate.
        patience: Early stopping patience.
        seed: Random seed.
        max_val_edges: Maximum edges evaluated per validation pass.
        use_amp: Enable AMP mixed precision (CUDA only).
        edge_feat_dim: Per-neighbor edge feature dimension.
        node_feat_dim: Per-neighbor node feature dimension.
        neg_per_positive: Random negatives per positive edge during training.

    Returns:
        Tuple of (best model, training history dict).
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed(seed)

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

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    logger.info(
        "DyGFormer: %d total params, %d trainable",
        total_params, trainable_params,
    )

    optimizer = torch.optim.Adam(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    amp_enabled = use_amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

    train_indices = np.where(train_mask)[0]
    val_indices = np.where(val_mask)[0]

    train_active_nodes = np.unique(
        np.concatenate([data.src[train_indices], data.dst[train_indices]])
    ).astype(np.int32)
    logger.info(
        "Active training nodes: %d (out of %d total)",
        len(train_active_nodes), data.num_nodes,
    )

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
    epochs_no_improve = 0

    config = {
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
    }
    with open(os.path.join(output_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    logger.info(
        "Training DyGFormer: %d epochs, %d train, %d val edges",
        num_epochs, len(train_indices), len(val_indices),
    )

    for epoch in range(1, num_epochs + 1):
        epoch_start = time.time()

        train_metrics = train_epoch(
            model, data, train_csr, train_indices, optimizer, device,
            batch_size=batch_size,
            num_neighbors=num_neighbors,
            neg_per_positive=neg_per_positive,
            use_amp=use_amp,
            scaler=scaler,
            rng=rng,
            neg_node_pool=train_active_nodes,
        )

        val_metrics = validate(
            model, data, full_csr, val_indices, device,
            num_neighbors=num_neighbors,
            max_eval_edges=max_val_edges,
            use_amp=use_amp,
            rng=rng,
            neg_node_pool=train_active_nodes,
        )

        epoch_time = time.time() - epoch_start

        history["train_loss"].append(train_metrics["loss"])
        history["val_mrr"].append(val_metrics["mrr"])
        history["val_hits@1"].append(val_metrics["hits@1"])
        history["val_hits@3"].append(val_metrics["hits@3"])
        history["val_hits@10"].append(val_metrics["hits@10"])
        history["epoch_time"].append(epoch_time)

        logger.info(
            "Epoch %d/%d [%.1fs] loss=%.8f mrr=%.4f h@1=%.3f h@3=%.3f h@10=%.3f",
            epoch, num_epochs, epoch_time,
            train_metrics["loss"],
            val_metrics["mrr"], val_metrics["hits@1"],
            val_metrics["hits@3"], val_metrics["hits@10"],
        )

        if val_metrics["mrr"] > best_val_mrr:
            best_val_mrr = val_metrics["mrr"]
            best_epoch = epoch
            epochs_no_improve = 0
            torch.save(
                model.state_dict(),
                os.path.join(output_dir, "best_model.pt"),
            )
            logger.info("New best model (MRR=%.4f)", best_val_mrr)
        else:
            epochs_no_improve += 1

        with open(os.path.join(output_dir, "metrics.jsonl"), "a") as f:
            record = {
                "epoch": epoch,
                **train_metrics,
                **{f"val_{k}": v for k, v in val_metrics.items()},
                "epoch_time": epoch_time,
            }
            f.write(json.dumps(record) + "\n")

        if epochs_no_improve >= patience:
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

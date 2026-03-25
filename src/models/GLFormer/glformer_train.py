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

from src.models.GLFormer.glformer import GLFormerTime
from src.models.GLFormer.data_utils import (
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
        return torch.cuda.amp.autocast()
    return contextlib.nullcontext()


def _compute_cooccurrence(
    src_nids: np.ndarray,
    src_lens: np.ndarray,
    dst_nids: np.ndarray,
    dst_lens: np.ndarray,
) -> np.ndarray:
    """Compute intersection size between neighbor sets for each batch element.

    For batch element b, returns |neighbors(src_b) ∩ neighbors(dst_b)|
    where neighbors are the actual (non-padded) neighbor node IDs.

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
        n_nids, neighbor_ts, neighbor_eids, lengths = sample_neighbors_batch(
            csr, nodes, ts_arr, num_neighbors
        )
        delta_times = np.maximum(
            ts_arr[:, None] - neighbor_ts, 0.0
        ).astype(np.float32)

        # Zero out delta_times for padded positions
        for b in range(len(nodes)):
            delta_times[b, lengths[b]:] = 0.0

        edge_feats_out = None
        node_feats_out = None

        if use_edge_feats or use_node_feats:
            _, ef_raw, _ = featurize_neighbors(
                n_nids, neighbor_eids, lengths,
                neighbor_ts, ts_arr,
                data.node_feats, data.edge_feats,
            )
            if use_edge_feats:
                edge_feats_out = ef_raw.astype(np.float32)
            if use_node_feats:
                node_feats_out = data.node_feats[nodes].astype(np.float32)

        return n_nids, delta_times, edge_feats_out, node_feats_out, lengths

    src_nids, src_dt, src_ef, src_nf, src_len = _get_neighbors(
        src_nodes, timestamps
    )
    pos_nids, pos_dt, pos_ef, pos_nf, pos_len = _get_neighbors(
        dst_nodes, timestamps
    )

    neg_flat = neg_dst_nodes.reshape(-1)
    neg_ts = np.repeat(timestamps, num_neg)
    neg_nids, neg_dt, neg_ef, neg_nf, neg_len = _get_neighbors(neg_flat, neg_ts)

    def _t(arr, dtype=torch.float32):
        return torch.tensor(arr, dtype=dtype, device=device)

    batch = {
        "src_delta_times":      _t(src_dt),
        "src_lengths":          _t(src_len, torch.int64),
        "pos_dst_delta_times":  _t(pos_dt),
        "pos_dst_lengths":      _t(pos_len, torch.int64),
        "neg_dst_delta_times":  _t(neg_dt.reshape(batch_size, num_neg, num_neighbors)),
        "neg_dst_lengths":      _t(neg_len.reshape(batch_size, num_neg), torch.int64),
    }

    if use_edge_feats:
        ef_dim = src_ef.shape[-1]
        batch["src_edge_feats"]     = _t(src_ef)
        batch["pos_dst_edge_feats"] = _t(pos_ef)
        batch["neg_dst_edge_feats"] = _t(
            neg_ef.reshape(batch_size, num_neg, num_neighbors, ef_dim)
        )

    if use_node_feats:
        batch["src_node_feats"]     = _t(src_nf)
        batch["pos_dst_node_feats"] = _t(pos_nf)
        batch["neg_dst_node_feats"] = _t(neg_nf.reshape(batch_size, num_neg, -1))

    if use_cooccurrence:
        pos_cooc = _compute_cooccurrence(src_nids, src_len, pos_nids, pos_len)

        src_nids_rep = np.repeat(src_nids, num_neg, axis=0)
        src_len_rep = np.repeat(src_len, num_neg)
        neg_cooc_flat = _compute_cooccurrence(
            src_nids_rep, src_len_rep, neg_nids, neg_len
        )
        batch["pos_cooc_counts"] = _t(pos_cooc)
        batch["neg_cooc_counts"] = _t(neg_cooc_flat.reshape(batch_size, num_neg))

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
    neg_per_positive: int = 1,
    use_amp: bool = True,
    scaler: Optional[torch.cuda.amp.GradScaler] = None,
    rng: Optional[np.random.Generator] = None,
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
        neg_dst = rng.integers(0, data.num_nodes, size=(B, neg_per_positive))

        batch = prepare_glformer_batch(
            csr, data, src, dst, ts, neg_dst, num_neighbors, device,
            use_edge_feats=use_edge_feats,
            use_node_feats=use_node_feats,
            use_cooccurrence=use_cooc,
        )

        with _amp_autocast(amp_enabled, device.type):
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
) -> Dict[str, float]:
    """Validate GLFormer with ranking metrics.

    For each validation edge, generates n_eval_negatives random negatives,
    scores all candidates, and computes the rank of the true destination.

    Args:
        model: GLFormerTime model.
        data: Temporal edge data.
        csr: Temporal CSR (train+val edges for final eval, train-only for
            mid-training validation).
        edge_indices: Indices of edges to evaluate.
        device: Torch device.
        num_neighbors: K neighbors sampled per node.
        n_eval_negatives: Number of random negatives per query.
        max_eval_edges: Maximum edges to evaluate (subsampled for speed).
        use_amp: Enable mixed precision.
        rng: Random number generator.

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

        dst_ts_arr = np.full(C, ts, dtype=np.float64)
        dst_nids, dst_nts, dst_neids, dst_lens = sample_neighbors_batch(
            csr, all_dst, dst_ts_arr, K
        )
        dst_dt = np.maximum(dst_ts_arr[:, None] - dst_nts, 0.0).astype(np.float32)
        for b in range(C):
            dst_dt[b, dst_lens[b]:] = 0.0

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

        cooc_counts = None
        if use_cooc:
            src_nids_rep = np.repeat(src_nids, C, axis=0)
            src_lens_rep = np.repeat(src_lens, C)
            cooc_np = _compute_cooccurrence(
                src_nids_rep, src_lens_rep, dst_nids, dst_lens
            )
            cooc_counts = torch.tensor(cooc_np, dtype=torch.float32, device=device)

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
            h_src_exp = h_src.expand(C, -1)

            cooc_feat = None
            if model.cooc_encoder is not None and cooc_counts is not None:
                cooc_feat = model.cooc_encoder(cooc_counts)

            scores = model.edge_predictor(
                h_src_exp, h_dst, cooc_feat
            ).cpu().float().numpy()

        true_score = scores[0]
        rank = (
            1.0
            + (scores[1:] > true_score).sum()
            + 0.5 * (scores[1:] == true_score).sum()
        )
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

    Returns:
        Tuple of (best model with loaded checkpoint, training history dict).
        History keys: train_loss, val_mrr, val_hits@1, val_hits@3,
            val_hits@10, epoch_time.
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
    epochs_no_improve = 0

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

    for epoch in range(1, num_epochs + 1):
        epoch_start = time.time()

        train_metrics = train_epoch(
            model, data, train_csr, train_indices, optimizer, device,
            batch_size=batch_size,
            num_neighbors=num_neighbors,
            use_amp=use_amp,
            scaler=scaler,
            rng=rng,
        )

        val_metrics = validate(
            model, data, full_csr, val_indices, device,
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

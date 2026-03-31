"""Training and validation for GraphMixer temporal link prediction on stream graphs.

Mirrors glformer_train.py in structure. Key differences from the original
src/models/train.py:
    1. Uses stream graph parquet format (load_stream_graph_data) instead of
       the sliding-window daily snapshot format.
    2. Supports AMP mixed precision (use_amp flag).
    3. Supports optional node features (node_feat_dim > 0) by fetching
       neighbor_node_feats from featurize_neighbors.
    4. Uses weight_decay (Adam) as a hyperparameter.
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

from payment_graph_forecasting.training.amp import (
    amp_enabled_for_device,
    autocast_context,
    create_grad_scaler,
    seed_torch,
)
from src.models.GraphMixer.graphmixer import GraphMixerTime
from src.models.GraphMixer.data_utils import (
    TemporalEdgeData,
    TemporalCSR,
    build_temporal_csr,
    sample_neighbors_batch,
)
from src.models.data_utils import featurize_neighbors

logger = logging.getLogger(__name__)


def prepare_graphmixer_batch(
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
    """Prepare a training batch for GraphMixerTime.

    Samples K most-recent neighbors for src, pos_dst, and neg_dst nodes,
    computes delta times and optional features, and returns tensors on device.

    Args:
        csr: Temporal CSR for neighbor lookups.
        data: Full temporal edge data.
        src_nodes: [B] source node indices.
        dst_nodes: [B] positive destination indices.
        timestamps: [B] query timestamps.
        neg_dst_nodes: [B, num_neg] negative destination indices.
        num_neighbors: K neighbors sampled per node.
        device: Target torch device.
        use_edge_feats: Include per-neighbor edge features (btc/usd).
        use_node_feats: Include own and neighbor node features.

    Returns:
        Dictionary of tensors. Always present: src_delta_times, src_lengths,
        pos_dst_delta_times, pos_dst_lengths, neg_dst_delta_times,
        neg_dst_lengths. Optional: *_edge_feats, *_node_feats,
        *_neighbor_node_feats.
    """
    batch_size = len(src_nodes)
    num_neg = neg_dst_nodes.shape[1] if neg_dst_nodes.ndim > 1 else 1

    def _get_neighbors(nodes, ts_arr):
        """Sample neighbors and compute features for a set of nodes."""
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
        neighbor_nf_out = None

        if use_edge_feats or use_node_feats:
            nnf_raw, ef_raw, _ = featurize_neighbors(
                n_nids, neighbor_eids, lengths,
                neighbor_ts, ts_arr,
                data.node_feats, data.edge_feats,
            )
            if use_edge_feats:
                edge_feats_out = ef_raw.astype(np.float32)
            if use_node_feats:
                node_feats_out = data.node_feats[nodes].astype(np.float32)
                neighbor_nf_out = nnf_raw.astype(np.float32)

        return delta_times, edge_feats_out, node_feats_out, neighbor_nf_out, lengths

    src_dt, src_ef, src_nf, src_nnf, src_len = _get_neighbors(
        src_nodes, timestamps
    )
    pos_dt, pos_ef, pos_nf, pos_nnf, pos_len = _get_neighbors(
        dst_nodes, timestamps
    )

    neg_flat = neg_dst_nodes.reshape(-1)
    neg_ts = np.repeat(timestamps, num_neg)
    neg_dt, neg_ef, neg_nf, neg_nnf, neg_len = _get_neighbors(neg_flat, neg_ts)

    def _t(arr, dtype=torch.float32):
        return torch.tensor(arr, dtype=dtype, device=device)

    batch = {
        "src_delta_times":     _t(src_dt),
        "src_lengths":         _t(src_len, torch.int64),
        "pos_dst_delta_times": _t(pos_dt),
        "pos_dst_lengths":     _t(pos_len, torch.int64),
        "neg_dst_delta_times": _t(neg_dt.reshape(batch_size, num_neg, num_neighbors)),
        "neg_dst_lengths":     _t(neg_len.reshape(batch_size, num_neg), torch.int64),
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
        batch["src_node_feats"]              = _t(src_nf)
        batch["pos_dst_node_feats"]          = _t(pos_nf)
        batch["neg_dst_node_feats"]          = _t(neg_nf.reshape(batch_size, num_neg, nf_dim))
        batch["src_neighbor_node_feats"]     = _t(src_nnf)
        batch["pos_dst_neighbor_node_feats"] = _t(pos_nnf)
        batch["neg_dst_neighbor_node_feats"] = _t(
            neg_nnf.reshape(batch_size, num_neg, num_neighbors, nf_dim)
        )

    return batch


def train_epoch(
    model: GraphMixerTime,
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
    """Run one training epoch for GraphMixer.

    Args:
        model: GraphMixerTime model.
        data: Temporal edge data.
        csr: Temporal CSR built from training edges only.
        edge_indices: Indices of training edges to iterate over.
        optimizer: Torch optimizer.
        device: Torch device.
        batch_size: Number of edges per batch.
        num_neighbors: K neighbors sampled per node.
        neg_per_positive: Number of random negatives per positive edge.
        use_amp: Enable mixed-precision training (CUDA only).
        scaler: GradScaler for AMP.
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

        batch = prepare_graphmixer_batch(
            csr, data, src, dst, ts, neg_dst, num_neighbors, device,
            use_edge_feats=use_edge_feats,
            use_node_feats=use_node_feats,
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
                src_neighbor_node_feats=batch.get("src_neighbor_node_feats"),
                dst_neighbor_node_feats=batch.get("pos_dst_neighbor_node_feats"),
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
                neg_nnf_i = (
                    batch["neg_dst_neighbor_node_feats"][:, neg_i, :, :]
                    if use_node_feats else None
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
                    src_neighbor_node_feats=batch.get("src_neighbor_node_feats"),
                    dst_neighbor_node_feats=neg_nnf_i,
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
    model: GraphMixerTime,
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
    """Validate GraphMixer with ranking metrics.

    For each validation edge, generates n_eval_negatives random negatives,
    scores all candidates, and computes the rank of the true destination.

    Args:
        model: GraphMixerTime model.
        data: Temporal edge data.
        csr: Temporal CSR (train-only for mid-training, train+val for test).
        edge_indices: Indices of edges to evaluate.
        device: Torch device.
        num_neighbors: K neighbors sampled per node.
        n_eval_negatives: Random negatives per query.
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

        src_ef = dst_ef = src_nf = dst_nf = src_nnf = dst_nnf = None
        if use_edge_feats or use_node_feats:
            src_nnf_raw, src_ef_raw, _ = featurize_neighbors(
                src_nids, src_neids, src_lens, src_nts, ts_arr,
                data.node_feats, data.edge_feats,
            )
            dst_nnf_raw, dst_ef_raw, _ = featurize_neighbors(
                dst_nids, dst_neids, dst_lens, dst_nts, dst_ts_arr,
                data.node_feats, data.edge_feats,
            )
            if use_edge_feats:
                src_ef = src_ef_raw.astype(np.float32)
                dst_ef = dst_ef_raw.astype(np.float32)
            if use_node_feats:
                src_nf = data.node_feats[[src_node]].astype(np.float32)
                dst_nf = data.node_feats[all_dst].astype(np.float32)
                src_nnf = src_nnf_raw.astype(np.float32)
                dst_nnf = dst_nnf_raw.astype(np.float32)

        def _t(arr, dtype=torch.float32):
            return torch.tensor(arr, dtype=dtype, device=device)

        with autocast_context(amp_enabled, device.type):
            h_src = model.encode_nodes(
                _t(src_dt), _t(src_lens, torch.int64),
                edge_feats=_t(src_ef) if src_ef is not None else None,
                node_feats=_t(src_nf) if src_nf is not None else None,
                neighbor_node_feats=_t(src_nnf) if src_nnf is not None else None,
            )
            h_dst = model.encode_nodes(
                _t(dst_dt), _t(dst_lens, torch.int64),
                edge_feats=_t(dst_ef) if dst_ef is not None else None,
                node_feats=_t(dst_nf) if dst_nf is not None else None,
                neighbor_node_feats=_t(dst_nnf) if dst_nnf is not None else None,
            )
            h_src_exp = h_src.expand(C, -1)
            scores = model.edge_predictor(h_src_exp, h_dst).cpu().float().numpy()

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


def train_graphmixer(
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
    num_mixer_layers: int = 2,
    dropout: float = 0.1,
    patience: int = 20,
    seed: int = 42,
    max_val_edges: int = 5000,
    use_amp: bool = True,
    edge_feat_dim: int = 2,
    node_feat_dim: int = 0,
) -> Tuple[GraphMixerTime, Dict]:
    """Full GraphMixer training pipeline with early stopping.

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
        num_mixer_layers: Number of MLP-Mixer blocks in LinkEncoder.
        dropout: Dropout rate.
        patience: Early stopping patience (epochs without val MRR improvement).
        seed: Random seed.
        max_val_edges: Maximum edges evaluated per validation pass.
        use_amp: Enable AMP mixed precision (CUDA only).
        edge_feat_dim: Per-neighbor edge feature dimension (2 for btc+usd).
        node_feat_dim: Query-node feature dimension (0 = NodeEncoder disabled).

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

    model = GraphMixerTime(
        hidden_dim=hidden_dim,
        num_neighbors=num_neighbors,
        num_mixer_layers=num_mixer_layers,
        dropout=dropout,
        edge_feat_dim=edge_feat_dim,
        node_feat_dim=node_feat_dim,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(
        "GraphMixerTime: %d total params, %d trainable",
        total_params, trainable_params,
    )

    optimizer = torch.optim.Adam(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    amp_enabled = amp_enabled_for_device(use_amp, device)
    scaler = create_grad_scaler(amp_enabled)

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
        "model": "GraphMixerTime",
        "hidden_dim": hidden_dim,
        "num_neighbors": num_neighbors,
        "num_mixer_layers": num_mixer_layers,
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
        "Training GraphMixerTime: %d epochs, %d train, %d val edges",
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

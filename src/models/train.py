"""Training loop for GraphMixer with logging, checkpointing, and early stopping."""

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

from src.models.graphmixer import GraphMixer
from src.models.data_utils import (
    TemporalEdgeData,
    TemporalCSR,
    sample_neighbors_batch,
    featurize_neighbors,
)

logger = logging.getLogger(__name__)


def prepare_batch(
    data: TemporalEdgeData,
    csr: TemporalCSR,
    src_nodes: np.ndarray,
    dst_nodes: np.ndarray,
    timestamps: np.ndarray,
    neg_dst_nodes: np.ndarray,
    num_neighbors: int,
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    """Prepare a training batch with neighbor sampling.

    Args:
        data: Full temporal edge data.
        csr: Temporal CSR for neighbor lookups.
        src_nodes: [batch] source node indices.
        dst_nodes: [batch] positive destination indices.
        timestamps: [batch] query timestamps.
        neg_dst_nodes: [batch, num_neg] negative destination indices.
        num_neighbors: K neighbors to sample per node.
        device: Target torch device.

    Returns:
        Dictionary of tensors ready for model forward pass.
    """
    batch_size = len(src_nodes)
    num_neg = neg_dst_nodes.shape[1] if neg_dst_nodes.ndim > 1 else 1

    def _sample_and_featurize(nodes, ts_arr):
        n_nodes, n_ts, n_eids, lengths = sample_neighbors_batch(
            csr, nodes, ts_arr, num_neighbors
        )
        node_feats_batch = data.node_feats[nodes]
        nnf, nef, nrt = featurize_neighbors(
            n_nodes, n_eids, lengths, n_ts, ts_arr,
            data.node_feats, data.edge_feats,
        )
        return node_feats_batch, nnf, nef, nrt, lengths

    src_nf, src_nnf, src_nef, src_nrt, src_len = _sample_and_featurize(
        src_nodes, timestamps
    )

    dst_nf, dst_nnf, dst_nef, dst_nrt, dst_len = _sample_and_featurize(
        dst_nodes, timestamps
    )

    neg_flat = neg_dst_nodes.reshape(-1)
    neg_ts = np.repeat(timestamps, num_neg)
    neg_nf, neg_nnf, neg_nef, neg_nrt, neg_len = _sample_and_featurize(
        neg_flat, neg_ts
    )

    def _to_tensor(arr, dtype=torch.float32):
        return torch.tensor(arr, dtype=dtype, device=device)

    return {
        "src_feats": _to_tensor(src_nf),
        "src_neighbor_node_feats": _to_tensor(src_nnf),
        "src_neighbor_edge_feats": _to_tensor(src_nef),
        "src_neighbor_rel_ts": _to_tensor(src_nrt),
        "src_neighbor_lengths": _to_tensor(src_len, torch.int64),
        "pos_dst_feats": _to_tensor(dst_nf),
        "pos_dst_neighbor_node_feats": _to_tensor(dst_nnf),
        "pos_dst_neighbor_edge_feats": _to_tensor(dst_nef),
        "pos_dst_neighbor_rel_ts": _to_tensor(dst_nrt),
        "pos_dst_neighbor_lengths": _to_tensor(dst_len, torch.int64),
        "neg_dst_feats": _to_tensor(neg_nf.reshape(batch_size, num_neg, -1)),
        "neg_dst_neighbor_node_feats": _to_tensor(
            neg_nnf.reshape(batch_size, num_neg, num_neighbors, -1)
        ),
        "neg_dst_neighbor_edge_feats": _to_tensor(
            neg_nef.reshape(batch_size, num_neg, num_neighbors, -1)
        ),
        "neg_dst_neighbor_rel_ts": _to_tensor(
            neg_nrt.reshape(batch_size, num_neg, num_neighbors)
        ),
        "neg_dst_neighbor_lengths": _to_tensor(
            neg_len.reshape(batch_size, num_neg), torch.int64
        ),
    }


def train_epoch(
    model: GraphMixer,
    data: TemporalEdgeData,
    csr: TemporalCSR,
    edge_indices: np.ndarray,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    batch_size: int = 600,
    num_neighbors: int = 20,
    neg_per_positive: int = 1,
    rng: np.random.Generator = None,
) -> Dict[str, float]:
    """Run one training epoch.

    Args:
        model: GraphMixer model.
        data: Temporal edge data.
        csr: Temporal CSR (built from training edges only).
        edge_indices: Indices of training edges.
        optimizer: Torch optimizer.
        device: Torch device.
        batch_size: Edges per batch.
        num_neighbors: Neighbors to sample per node.
        neg_per_positive: Number of negative samples per positive edge.
        rng: Random number generator.

    Returns:
        Dict with 'loss' (average batch loss).
    """
    if rng is None:
        rng = np.random.default_rng()

    model.train()
    criterion = nn.BCEWithLogitsLoss()

    shuffled = rng.permutation(edge_indices)
    total_loss = 0.0
    num_batches = 0

    for start in range(0, len(shuffled), batch_size):
        end = min(start + batch_size, len(shuffled))
        batch_idx = shuffled[start:end]
        actual_batch = len(batch_idx)

        src = data.src[batch_idx]
        dst = data.dst[batch_idx]
        ts = data.timestamps[batch_idx]
        neg_dst = rng.integers(0, data.num_nodes, size=(actual_batch, neg_per_positive))

        batch = prepare_batch(
            data, csr, src, dst, ts, neg_dst, num_neighbors, device
        )

        pos_logits = model(
            batch["src_feats"],
            batch["src_neighbor_node_feats"],
            batch["src_neighbor_edge_feats"],
            batch["src_neighbor_rel_ts"],
            batch["src_neighbor_lengths"],
            batch["pos_dst_feats"],
            batch["pos_dst_neighbor_node_feats"],
            batch["pos_dst_neighbor_edge_feats"],
            batch["pos_dst_neighbor_rel_ts"],
            batch["pos_dst_neighbor_lengths"],
        )

        neg_logits_list = []
        for neg_i in range(neg_per_positive):
            neg_logits_i = model(
                batch["src_feats"],
                batch["src_neighbor_node_feats"],
                batch["src_neighbor_edge_feats"],
                batch["src_neighbor_rel_ts"],
                batch["src_neighbor_lengths"],
                batch["neg_dst_feats"][:, neg_i, :],
                batch["neg_dst_neighbor_node_feats"][:, neg_i, :, :],
                batch["neg_dst_neighbor_edge_feats"][:, neg_i, :, :],
                batch["neg_dst_neighbor_rel_ts"][:, neg_i, :],
                batch["neg_dst_neighbor_lengths"][:, neg_i],
            )
            neg_logits_list.append(neg_logits_i)

        all_logits = torch.cat([pos_logits] + neg_logits_list)
        all_labels = torch.cat([
            torch.ones(actual_batch, device=device),
            torch.zeros(actual_batch * neg_per_positive, device=device),
        ])

        loss = criterion(all_logits, all_labels)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    return {"loss": total_loss / max(num_batches, 1)}


@torch.no_grad()
def validate(
    model: GraphMixer,
    data: TemporalEdgeData,
    csr: TemporalCSR,
    edge_indices: np.ndarray,
    device: torch.device,
    num_neighbors: int = 20,
    n_eval_negatives: int = 100,
    max_eval_edges: int = 5000,
    rng: np.random.Generator = None,
) -> Dict[str, float]:
    """Run validation with ranking metrics.

    Args:
        model: GraphMixer model.
        data: Temporal edge data.
        csr: Temporal CSR.
        edge_indices: Indices of val/test edges.
        device: Torch device.
        num_neighbors: Neighbors to sample.
        n_eval_negatives: Negatives per positive for ranking.
        max_eval_edges: Max edges to evaluate (subsample for speed).
        rng: Random number generator.

    Returns:
        Dict with 'mrr', 'hits@1', 'hits@3', 'hits@10'.
    """
    if rng is None:
        rng = np.random.default_rng(42)

    model.eval()

    if len(edge_indices) > max_eval_edges:
        eval_idx = rng.choice(edge_indices, size=max_eval_edges, replace=False)
    else:
        eval_idx = edge_indices

    ranks = []

    for idx in eval_idx:
        src_node = data.src[idx]
        true_dst = data.dst[idx]
        ts = data.timestamps[idx]

        neg_nodes = rng.integers(0, data.num_nodes, size=n_eval_negatives).astype(np.int32)
        all_dst = np.concatenate([[true_dst], neg_nodes])

        src_arr = np.array([src_node], dtype=np.int32)
        ts_arr = np.array([ts], dtype=np.float64)

        src_nn, src_nts, src_neids, src_lens = sample_neighbors_batch(
            csr, src_arr, ts_arr, num_neighbors
        )
        src_nf = data.node_feats[src_arr]
        src_nnf, src_nef, src_nrt = featurize_neighbors(
            src_nn, src_neids, src_lens, src_nts, ts_arr,
            data.node_feats, data.edge_feats,
        )

        num_candidates = len(all_dst)
        dst_nf = data.node_feats[all_dst]
        dst_ts_arr = np.full(num_candidates, ts, dtype=np.float64)
        dst_nn, dst_nts, dst_neids, dst_lens = sample_neighbors_batch(
            csr, all_dst, dst_ts_arr, num_neighbors
        )
        dst_nnf, dst_nef, dst_nrt = featurize_neighbors(
            dst_nn, dst_neids, dst_lens, dst_nts, dst_ts_arr,
            data.node_feats, data.edge_feats,
        )

        def _t(arr, dtype=torch.float32):
            return torch.tensor(arr, dtype=dtype, device=device)

        h_src = model.encode_node(
            _t(src_nf), _t(src_nnf), _t(src_nef), _t(src_nrt), _t(src_lens, torch.int64)
        )

        h_dst = model.encode_node(
            _t(dst_nf), _t(dst_nnf), _t(dst_nef), _t(dst_nrt), _t(dst_lens, torch.int64)
        )

        h_src_expanded = h_src.expand(num_candidates, -1)
        scores = model.link_classifier(h_src_expanded, h_dst).cpu().numpy()

        true_score = scores[0]
        rank = 1 + (scores[1:] > true_score).sum() + 0.5 * (scores[1:] == true_score).sum()
        ranks.append(rank)

    ranks = np.array(ranks, dtype=np.float64)
    mrr = float(np.mean(1.0 / ranks))
    hits1 = float(np.mean(ranks <= 1))
    hits3 = float(np.mean(ranks <= 3))
    hits10 = float(np.mean(ranks <= 10))

    return {
        "mrr": mrr,
        "hits@1": hits1,
        "hits@3": hits3,
        "hits@10": hits10,
        "mean_rank": float(np.mean(ranks)),
        "n_queries": len(ranks),
    }


def train_graphmixer(
    data: TemporalEdgeData,
    train_mask: np.ndarray,
    val_mask: np.ndarray,
    output_dir: str,
    device: torch.device = None,
    num_epochs: int = 100,
    batch_size: int = 600,
    learning_rate: float = 0.0001,
    num_neighbors: int = 20,
    hidden_dim: int = 100,
    num_mixer_layers: int = 2,
    dropout: float = 0.1,
    patience: int = 20,
    seed: int = 42,
    max_val_edges: int = 5000,
) -> Tuple[GraphMixer, Dict]:
    """Full training pipeline for GraphMixer.

    Args:
        data: TemporalEdgeData.
        train_mask: Boolean mask for training edges.
        val_mask: Boolean mask for validation edges.
        output_dir: Directory for checkpoints and logs.
        device: Torch device. Auto-detected if None.
        num_epochs: Maximum training epochs.
        batch_size: Training batch size.
        learning_rate: Adam learning rate.
        num_neighbors: K neighbors per node.
        hidden_dim: Hidden dimension for all modules.
        num_mixer_layers: Number of MLP-Mixer layers.
        dropout: Dropout rate.
        patience: Early stopping patience.
        seed: Random seed.
        max_val_edges: Max validation edges per epoch.

    Returns:
        Tuple of (trained model, training history dict).
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)

    train_csr = TemporalCSR(
        data.num_nodes,
        data.src[train_mask],
        data.dst[train_mask],
        data.timestamps[train_mask],
        np.where(train_mask)[0].astype(np.int64),
    )

    full_mask = train_mask | val_mask
    full_csr = TemporalCSR(
        data.num_nodes,
        data.src[full_mask],
        data.dst[full_mask],
        data.timestamps[full_mask],
        np.where(full_mask)[0].astype(np.int64),
    )

    model = GraphMixer(
        edge_feat_dim=data.edge_feats.shape[1],
        node_feat_dim=data.node_feats.shape[1],
        hidden_dim=hidden_dim,
        num_neighbors=num_neighbors,
        num_mixer_layers=num_mixer_layers,
        dropout=dropout,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("Model: %d total params, %d trainable", total_params, trainable_params)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

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
        "edge_feat_dim": data.edge_feats.shape[1],
        "node_feat_dim": data.node_feats.shape[1],
        "hidden_dim": hidden_dim,
        "num_neighbors": num_neighbors,
        "num_mixer_layers": num_mixer_layers,
        "dropout": dropout,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "num_epochs": num_epochs,
        "patience": patience,
        "seed": seed,
        "num_nodes": data.num_nodes,
        "num_edges": data.num_edges,
        "total_params": total_params,
        "trainable_params": trainable_params,
        "device": str(device),
    }
    with open(os.path.join(output_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    logger.info("Starting training: %d epochs, %d train edges, %d val edges",
                num_epochs, len(train_indices), len(val_indices))

    for epoch in range(1, num_epochs + 1):
        epoch_start = time.time()

        train_metrics = train_epoch(
            model, data, train_csr, train_indices, optimizer, device,
            batch_size=batch_size, num_neighbors=num_neighbors, rng=rng,
        )

        val_metrics = validate(
            model, data, full_csr, val_indices, device,
            num_neighbors=num_neighbors, max_eval_edges=max_val_edges, rng=rng,
        )

        epoch_time = time.time() - epoch_start

        history["train_loss"].append(train_metrics["loss"])
        history["val_mrr"].append(val_metrics["mrr"])
        history["val_hits@1"].append(val_metrics["hits@1"])
        history["val_hits@3"].append(val_metrics["hits@3"])
        history["val_hits@10"].append(val_metrics["hits@10"])
        history["epoch_time"].append(epoch_time)

        logger.info(
            "Epoch %d/%d [%.1fs] loss=%.4f val_mrr=%.4f hits@1=%.3f hits@3=%.3f hits@10=%.3f",
            epoch, num_epochs, epoch_time,
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
            torch.save(model.state_dict(), os.path.join(output_dir, "best_model.pt"))
            logger.info("New best model saved (MRR=%.4f)", best_val_mrr)
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
            logger.info("Early stopping at epoch %d (patience=%d)", epoch, patience)
            break

    model.load_state_dict(
        torch.load(os.path.join(output_dir, "best_model.pt"), map_location=device, weights_only=True)
    )

    summary = {
        "best_epoch": best_epoch,
        "best_val_mrr": best_val_mrr,
        "total_epochs": epoch,
        "final_train_loss": history["train_loss"][-1],
    }
    with open(os.path.join(output_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    logger.info("Training complete. Best epoch=%d, best MRR=%.4f", best_epoch, best_val_mrr)
    return model, history

"""Training loop for GraphMixer on stream graph data.

Key differences from src/models/train.py:
- Hard negative sampling: 50% historical + 50% random (matches eval distribution)
- Negatives sampled from TRAIN NODES only (not all nodes)
- Split is 70/15/15 from stream graph period
- Early stopping uses TGB-style eval from evaluate.py
"""

import os
import time
import json
import logging
from pathlib import Path
from typing import Dict, Tuple

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


def build_train_neighbors_dense(
    train_neighbors: dict[int, set[int]],
    global_to_dense: dict[int, int],
) -> dict[int, np.ndarray]:
    """Convert train_neighbors from global to dense indices.

    Returns dict mapping dense src -> numpy array of dense dst neighbors.
    Pre-converts to arrays for fast random indexing during training.
    """
    result = {}
    for src_global, dst_set in train_neighbors.items():
        src_dense = global_to_dense.get(int(src_global))
        if src_dense is None:
            continue
        dst_dense = []
        for d in dst_set:
            dd = global_to_dense.get(int(d))
            if dd is not None:
                dst_dense.append(dd)
        if dst_dense:
            result[src_dense] = np.array(dst_dense, dtype=np.int32)
    return result


def _sample_hard_negatives_batch(
    src_dense: np.ndarray,
    dst_dense: np.ndarray,
    train_neighbors_dense: dict[int, np.ndarray],
    active_nodes_dense: np.ndarray,
    neg_per_positive: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample negatives per edge: ~50% historical neighbors + ~50% random.

    For each positive (src, dst), tries to fill half the negatives from
    src's historical train neighbors (excluding dst), rest from random
    train nodes. This matches eval distribution where negatives include
    actual past contacts of src.
    """
    batch_size = len(src_dense)
    neg_dst = np.zeros((batch_size, neg_per_positive), dtype=np.int32)
    n_hist_target = neg_per_positive // 2
    n_active = len(active_nodes_dense)

    for i in range(batch_size):
        s = int(src_dense[i])
        d = int(dst_dense[i])

        hist_arr = train_neighbors_dense.get(s)
        chosen = []

        if hist_arr is not None and len(hist_arr) > 0:
            mask = hist_arr != d
            if s < len(hist_arr):
                mask = mask & (hist_arr != s)
            filtered = hist_arr[mask]
            n_hist = min(n_hist_target, len(filtered))
            if n_hist > 0:
                idx = rng.choice(len(filtered), size=n_hist, replace=False)
                chosen.extend(filtered[idx].tolist())

        n_rand = neg_per_positive - len(chosen)
        exclude = {s, d}
        exclude.update(chosen)

        attempts = 0
        while len(chosen) < neg_per_positive and attempts < n_rand * 10:
            batch_rand = active_nodes_dense[
                rng.integers(0, n_active, size=min(n_rand * 3, n_active))
            ]
            for c in batch_rand:
                c_int = int(c)
                if c_int not in exclude:
                    chosen.append(c_int)
                    exclude.add(c_int)
                    if len(chosen) >= neg_per_positive:
                        break
            attempts += len(batch_rand)

        neg_dst[i, :len(chosen)] = chosen[:neg_per_positive]

    return neg_dst


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
        csr: Temporal CSR for neighbor lookups (train edges only, undirected).
        src_nodes: [batch] source node dense indices.
        dst_nodes: [batch] positive destination dense indices.
        timestamps: [batch] query timestamps.
        neg_dst_nodes: [batch, num_neg] negative destination dense indices.
        num_neighbors: K neighbors to sample per node.
        device: Target torch device.

    Returns:
        Dictionary of tensors for model forward pass.
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
        return torch.tensor(np.ascontiguousarray(arr), dtype=dtype, device=device)

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
    active_nodes_dense: np.ndarray,
    train_neighbors_dense: dict[int, np.ndarray],
    batch_size: int = 4000,
    num_neighbors: int = 30,
    neg_per_positive: int = 5,
    rng: np.random.Generator = None,
) -> Dict[str, float]:
    """Run one training epoch with hard negative sampling.

    Uses 50% historical + 50% random negatives per positive edge,
    matching the eval distribution for better generalization.

    Args:
        model: GraphMixer model.
        data: Temporal edge data.
        csr: Temporal CSR (train edges only, undirected).
        edge_indices: Indices of training edges.
        optimizer: Torch optimizer.
        device: Torch device.
        active_nodes_dense: Dense indices of train nodes.
        train_neighbors_dense: Per-source neighbor arrays in dense indices.
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
    n_total_batches = (len(shuffled) + batch_size - 1) // batch_size

    pbar = tqdm(range(0, len(shuffled), batch_size), total=n_total_batches,
                desc="Training", leave=False, unit="batch")
    for start in pbar:
        end = min(start + batch_size, len(shuffled))
        batch_idx = shuffled[start:end]
        actual_batch = len(batch_idx)

        src = data.src[batch_idx]
        dst = data.dst[batch_idx]
        ts = data.timestamps[batch_idx]

        neg_dst = _sample_hard_negatives_batch(
            src, dst, train_neighbors_dense, active_nodes_dense,
            neg_per_positive, rng,
        )

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
        pbar.set_postfix(loss=f"{total_loss / num_batches:.4f}")

    pbar.close()
    return {"loss": total_loss / max(num_batches, 1)}


def train_graphmixer(
    data: TemporalEdgeData,
    train_mask: np.ndarray,
    val_mask: np.ndarray,
    train_neighbors: dict[int, set[int]],
    active_nodes: np.ndarray,
    node_mapping: np.ndarray,
    output_dir: str,
    device: torch.device = None,
    num_epochs: int = 100,
    batch_size: int = 4000,
    learning_rate: float = 0.001,
    weight_decay: float = 3e-6,
    num_neighbors: int = 30,
    hidden_dim: int = 200,
    num_mixer_layers: int = 1,
    dropout: float = 0.2,
    patience: int = 20,
    seed: int = 42,
    max_val_queries: int = 10_000,
    n_negatives: int = 100,
    neg_per_positive: int = 5,
) -> Tuple[GraphMixer, Dict]:
    """Full training pipeline for GraphMixer on stream graph.

    Args:
        data: TemporalEdgeData (with undirected edges, dense indices).
        train_mask: Boolean mask for training edges.
        val_mask: Boolean mask for validation edges.
        train_neighbors: Per-source neighbor sets (global indices) from train.
        active_nodes: Sorted global indices of train nodes.
        node_mapping: Same as active_nodes (local->global mapping).
        output_dir: Directory for checkpoints and logs.
        device: Torch device. Auto-detected if None.
        num_epochs: Maximum training epochs.
        batch_size: Training batch size.
        learning_rate: Adam learning rate.
        weight_decay: Adam weight decay.
        num_neighbors: K neighbors per node.
        hidden_dim: Hidden dimension for all modules.
        num_mixer_layers: Number of MLP-Mixer layers.
        dropout: Dropout rate.
        patience: Early stopping patience.
        seed: Random seed.
        max_val_queries: Max val queries per epoch (10K for speed).
        n_negatives: Number of negatives for eval (100 per TGB).
        neg_per_positive: Training negatives per positive (default 5).

    Returns:
        Tuple of (trained model, training history dict).
    """
    from src.models.sg_graphmixer.evaluate import evaluate_tgb_style

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed(seed)

    print(f"  Building TemporalCSR from train edges...", flush=True)
    train_csr = TemporalCSR(
        data.num_nodes,
        data.src[train_mask],
        data.dst[train_mask],
        data.timestamps[train_mask],
        np.where(train_mask)[0].astype(np.int64),
    )

    global_to_dense = {int(g): i for i, g in enumerate(node_mapping)}
    active_nodes_dense = np.arange(len(node_mapping), dtype=np.int32)

    print(f"  Building dense train neighbor sets...", flush=True)
    t0 = time.time()
    train_neighbors_dense = build_train_neighbors_dense(
        train_neighbors, global_to_dense
    )
    print(f"  {len(train_neighbors_dense):,} sources with neighbors ({time.time() - t0:.1f}s)",
          flush=True)

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
    print(f"  Model: {trainable_params:,} trainable params, device={device}", flush=True)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )

    train_indices = np.where(train_mask)[0]

    history = {
        "train_loss": [], "val_mrr": [], "val_hits@1": [],
        "val_hits@3": [], "val_hits@10": [], "epoch_time": [],
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
        "weight_decay": weight_decay,
        "num_epochs": num_epochs,
        "patience": patience,
        "seed": seed,
        "neg_per_positive": neg_per_positive,
        "num_nodes": data.num_nodes,
        "num_edges": data.num_edges,
        "total_params": total_params,
        "trainable_params": trainable_params,
        "device": str(device),
        "n_negatives": n_negatives,
        "max_val_queries": max_val_queries,
    }
    with open(os.path.join(output_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    print(f"\n  Starting training: {num_epochs} epochs max, "
          f"{train_mask.sum():,} train edges, patience={patience}", flush=True)
    print(f"  Training negatives: {neg_per_positive} per positive (50% hist + 50% rand)",
          flush=True)
    print(f"  Val eval: {max_val_queries:,} queries, "
          f"{n_negatives} negatives each\n", flush=True)

    for epoch in range(1, num_epochs + 1):
        epoch_start = time.time()

        train_metrics = train_epoch(
            model, data, train_csr, train_indices, optimizer, device,
            active_nodes_dense=active_nodes_dense,
            train_neighbors_dense=train_neighbors_dense,
            batch_size=batch_size, num_neighbors=num_neighbors,
            neg_per_positive=neg_per_positive, rng=rng,
        )

        val_metrics = evaluate_tgb_style(
            model=model,
            data=data,
            csr=train_csr,
            eval_mask=val_mask,
            device=device,
            num_neighbors=num_neighbors,
            train_neighbors=train_neighbors,
            active_nodes=active_nodes,
            node_mapping=node_mapping,
            n_negatives=n_negatives,
            max_queries=max_val_queries,
            seed=seed + 200,
        )

        epoch_time = time.time() - epoch_start

        history["train_loss"].append(train_metrics["loss"])
        history["val_mrr"].append(val_metrics["mrr"])
        history["val_hits@1"].append(val_metrics["hits@1"])
        history["val_hits@3"].append(val_metrics["hits@3"])
        history["val_hits@10"].append(val_metrics["hits@10"])
        history["epoch_time"].append(epoch_time)

        marker = ""
        if val_metrics["mrr"] > best_val_mrr:
            best_val_mrr = val_metrics["mrr"]
            best_epoch = epoch
            epochs_without_improvement = 0
            torch.save(model.state_dict(), os.path.join(output_dir, "best_model.pt"))
            marker = " *BEST*"
        else:
            epochs_without_improvement += 1

        print(
            f"  Epoch {epoch:3d}/{num_epochs} [{epoch_time:.0f}s] "
            f"loss={train_metrics['loss']:.4f} "
            f"val_mrr={val_metrics['mrr']:.4f} "
            f"h@1={val_metrics['hits@1']:.3f} h@10={val_metrics['hits@10']:.3f}"
            f"{marker}",
            flush=True,
        )

        with open(os.path.join(output_dir, "metrics.jsonl"), "a") as f:
            record = {
                "epoch": epoch,
                **train_metrics,
                **{f"val_{k}": v for k, v in val_metrics.items()},
                "epoch_time": epoch_time,
            }
            f.write(json.dumps(record) + "\n")

        if epochs_without_improvement >= patience:
            print(f"\n  Early stopping at epoch {epoch} (patience={patience})", flush=True)
            break

    model.load_state_dict(
        torch.load(os.path.join(output_dir, "best_model.pt"),
                    map_location=device, weights_only=True)
    )

    summary = {
        "best_epoch": best_epoch,
        "best_val_mrr": best_val_mrr,
        "total_epochs": epoch,
        "final_train_loss": history["train_loss"][-1],
    }
    with open(os.path.join(output_dir, "train_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n  Training complete. Best epoch={best_epoch}, best val MRR={best_val_mrr:.4f}",
          flush=True)
    return model, history

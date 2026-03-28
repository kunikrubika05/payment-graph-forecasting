"""Training loop for the GraphMixer CUDA sampler benchmark.

Trains GraphMixerTime with TemporalGraphSampler (python/cpp/cuda backend)
and records sampling_time_sec and forward_time_sec separately per epoch.

Key design choice: default batch_size=2000.
At B=2000, C++ sampling costs ~8.6ms/batch while GPU forward costs ~2ms/batch
(sampling = 81% of batch time). CUDA reduces sampling to 0.8ms → 3-4x speedup
over C++. At B=200 the ratio would be lower (~1.7x) because GPU kernel launch
overhead dominates forward time.
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

from src.models.GraphMixer.graphmixer import GraphMixerTime
from src.models.data_utils import TemporalEdgeData
from src.models.temporal_graph_sampler import TemporalGraphSampler

logger = logging.getLogger(__name__)


def _amp_ctx(enabled: bool, device_type: str):
    """Return AMP autocast context or no-op."""
    if enabled and device_type == "cuda":
        return torch.amp.autocast("cuda")
    return contextlib.nullcontext()


def build_sampler(
    data: TemporalEdgeData,
    mask: np.ndarray,
    backend: str,
) -> TemporalGraphSampler:
    """Build TemporalGraphSampler from the edges selected by mask.

    Args:
        data: Full temporal edge dataset.
        mask: Boolean mask selecting which edges to include.
        backend: 'python', 'cpp', or 'cuda'.

    Returns:
        Initialised TemporalGraphSampler.
    """
    edge_ids = np.where(mask)[0].astype(np.int64)
    n_edges = int(mask.sum())
    edge_feats = (
        data.edge_feats[mask].astype(np.float32)
        if data.edge_feats is not None
        else np.zeros((n_edges, 0), dtype=np.float32)
    )
    node_feats = np.zeros((data.num_nodes, 0), dtype=np.float32)
    return TemporalGraphSampler(
        num_nodes=data.num_nodes,
        src=data.src[mask].astype(np.int32),
        dst=data.dst[mask].astype(np.int32),
        timestamps=data.timestamps[mask].astype(np.float64),
        edge_ids=edge_ids,
        node_feats=node_feats,
        edge_feats=edge_feats,
        backend=backend,
    )


def _sample(
    sampler: TemporalGraphSampler,
    nodes: np.ndarray,
    ts: np.ndarray,
    K: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sample K neighbors and return (edge_feats, rel_timestamps, lengths).

    Args:
        sampler: TemporalGraphSampler with any backend.
        nodes: [B] int32 node indices.
        ts: [B] float64 query timestamps.
        K: Number of neighbors to retrieve.

    Returns:
        edge_feats [B, K, D], rel_timestamps [B, K], lengths [B] — all float32/int64.
    """
    nbr = sampler.sample_neighbors(nodes, ts, K)
    nbr_np = sampler.to_numpy(nbr)
    feat = sampler.featurize(nbr, query_timestamps=ts)
    feat_np = sampler.to_numpy_features(feat)
    return (
        feat_np.edge_features.astype(np.float32),
        feat_np.rel_timestamps.astype(np.float32),
        nbr_np.lengths.astype(np.int64),
    )


def train_epoch(
    model: GraphMixerTime,
    data: TemporalEdgeData,
    sampler: TemporalGraphSampler,
    train_indices: np.ndarray,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    batch_size: int,
    K: int,
    amp_enabled: bool,
    rng: np.random.Generator,
) -> Dict[str, float]:
    """Run one training epoch and return loss + timing breakdown.

    Args:
        model: GraphMixerTime instance.
        data: Temporal edge dataset.
        sampler: TemporalGraphSampler (any backend).
        train_indices: Edge indices to train on.
        optimizer: Adam optimiser.
        scaler: GradScaler for AMP.
        device: Torch device.
        batch_size: Edges per batch.
        K: Neighbours per node.
        amp_enabled: Whether AMP is active.
        rng: NumPy random generator.

    Returns:
        Dict with 'loss', 'sampling_time_sec', 'forward_time_sec'.
    """
    model.train()
    criterion = nn.BCEWithLogitsLoss()
    shuffled = rng.permutation(train_indices)

    total_loss = 0.0
    total_sampling = 0.0
    total_forward = 0.0
    n_batches = 0
    use_ef = model.edge_feat_dim > 0

    n_total = (len(shuffled) + batch_size - 1) // batch_size
    pbar = tqdm(range(0, len(shuffled), batch_size), total=n_total,
                desc="Train", leave=False, unit="batch")

    for start in pbar:
        idx = shuffled[start: start + batch_size]
        B = len(idx)

        src = data.src[idx].astype(np.int32)
        pos_dst = data.dst[idx].astype(np.int32)
        ts = data.timestamps[idx].astype(np.float64)
        neg_dst = rng.integers(0, data.num_nodes, size=B, dtype=np.int32)

        # -------- sampling (measured separately) --------
        t0 = time.perf_counter()
        src_ef, src_rt, src_len = _sample(sampler, src, ts, K)
        pos_ef, pos_rt, pos_len = _sample(sampler, pos_dst, ts, K)
        neg_ef, neg_rt, neg_len = _sample(sampler, neg_dst, ts, K)
        total_sampling += time.perf_counter() - t0

        # -------- forward + backward (measured separately) --------
        def _t(a, dtype=torch.float32):
            return torch.tensor(a, dtype=dtype, device=device)

        t1 = time.perf_counter()
        with _amp_ctx(amp_enabled, device.type):
            h_src = model.encode_nodes(
                _t(src_rt), _t(src_len, torch.int64),
                edge_feats=_t(src_ef) if use_ef else None,
            )
            h_pos = model.encode_nodes(
                _t(pos_rt), _t(pos_len, torch.int64),
                edge_feats=_t(pos_ef) if use_ef else None,
            )
            h_neg = model.encode_nodes(
                _t(neg_rt), _t(neg_len, torch.int64),
                edge_feats=_t(neg_ef) if use_ef else None,
            )
            pos_logits = model.edge_predictor(h_src, h_pos)
            neg_logits = model.edge_predictor(h_src, h_neg)
            logits = torch.cat([pos_logits, neg_logits])
            labels = torch.cat([
                torch.ones(B, device=device),
                torch.zeros(B, device=device),
            ])
            loss = criterion(logits, labels)

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        total_forward += time.perf_counter() - t1

        total_loss += loss.item()
        n_batches += 1
        pbar.set_postfix(
            loss=f"{total_loss / n_batches:.4f}",
            samp=f"{total_sampling:.1f}s",
            fwd=f"{total_forward:.1f}s",
        )

    pbar.close()
    return {
        "loss": total_loss / max(n_batches, 1),
        "sampling_time_sec": total_sampling,
        "forward_time_sec": total_forward,
    }


@torch.no_grad()
def validate(
    model: GraphMixerTime,
    data: TemporalEdgeData,
    val_sampler: TemporalGraphSampler,
    val_indices: np.ndarray,
    device: torch.device,
    K: int,
    max_edges: int,
    amp_enabled: bool,
    rng: np.random.Generator,
) -> Dict[str, float]:
    """Quick MRR validation on a random subset of val edges.

    Uses 50 random negatives per edge for speed.

    Args:
        model: GraphMixerTime instance (eval mode).
        data: Temporal edge dataset.
        val_sampler: TemporalGraphSampler (C++ backend recommended).
        val_indices: Val edge indices.
        device: Torch device.
        K: Neighbours per node.
        max_edges: Max edges to evaluate.
        amp_enabled: Whether AMP is active.
        rng: NumPy random generator.

    Returns:
        Dict with 'mrr', 'hits@1', 'hits@3', 'hits@10'.
    """
    from src.baselines.evaluation import compute_ranking_metrics

    model.eval()
    use_ef = model.edge_feat_dim > 0

    if len(val_indices) > max_edges:
        eval_idx = rng.choice(val_indices, size=max_edges, replace=False)
        eval_idx.sort()
    else:
        eval_idx = val_indices

    all_ranks = []
    for idx in eval_idx:
        src_node = np.array([data.src[idx]], dtype=np.int32)
        ts_q = np.array([data.timestamps[idx]], dtype=np.float64)
        true_dst = data.dst[idx]

        neg_nodes = rng.integers(0, data.num_nodes, size=50, dtype=np.int32)
        all_dst = np.concatenate([[true_dst], neg_nodes]).astype(np.int32)
        C = len(all_dst)

        src_ef, src_rt, src_len = _sample(val_sampler, src_node, ts_q, K)
        dst_ef, dst_rt, dst_len = _sample(
            val_sampler, all_dst,
            np.full(C, ts_q[0], dtype=np.float64), K,
        )

        def _t(a, dtype=torch.float32):
            return torch.tensor(a, dtype=dtype, device=device)

        with _amp_ctx(amp_enabled, device.type):
            h_src = model.encode_nodes(
                _t(src_rt), _t(src_len, torch.int64),
                edge_feats=_t(src_ef) if use_ef else None,
            )
            h_dst = model.encode_nodes(
                _t(dst_rt), _t(dst_len, torch.int64),
                edge_feats=_t(dst_ef) if use_ef else None,
            )
            scores = model.edge_predictor(
                h_src.expand(C, -1), h_dst
            ).cpu().float().numpy()

        true_score = scores[0]
        rank = (1.0
                + (scores[1:] > true_score).sum()
                + 0.5 * (scores[1:] == true_score).sum())
        all_ranks.append(float(rank))

    return compute_ranking_metrics(np.array(all_ranks, dtype=np.float64))


def train_graphmixer(
    data: TemporalEdgeData,
    train_mask: np.ndarray,
    val_mask: np.ndarray,
    output_dir: str,
    sampling_backend: str,
    device: torch.device,
    num_epochs: int = 3,
    batch_size: int = 2000,
    learning_rate: float = 1e-4,
    weight_decay: float = 1e-5,
    num_neighbors: int = 20,
    hidden_dim: int = 100,
    num_mixer_layers: int = 2,
    dropout: float = 0.1,
    patience: int = 5,
    seed: int = 42,
    max_val_edges: int = 5000,
    use_amp: bool = True,
    edge_feat_dim: int = 2,
    node_feat_dim: int = 0,
    test_mask: Optional[np.ndarray] = None,
) -> Tuple[GraphMixerTime, Dict]:
    """Train GraphMixerTime with a chosen backend, tracking timing per epoch.

    Args:
        data: Temporal edge dataset.
        train_mask: Boolean mask for training edges.
        val_mask: Boolean mask for validation edges.
        output_dir: Directory for checkpoints and logs.
        sampling_backend: 'python', 'cpp', or 'cuda'.
        device: Torch device.
        num_epochs: Number of training epochs.
        batch_size: Edges per batch (2000 recommended for C++→CUDA demo).
        learning_rate: Adam learning rate.
        weight_decay: Adam weight decay.
        num_neighbors: K neighbours per node (20 appropriate for dataset).
        hidden_dim: GraphMixer hidden dimension.
        num_mixer_layers: Number of MLP-Mixer layers.
        dropout: Dropout rate.
        patience: Early stopping patience.
        seed: Random seed.
        max_val_edges: Max edges per validation run.
        use_amp: Use AMP mixed precision.
        edge_feat_dim: Edge feature dimension (2 = log1p btc+usd).
        node_feat_dim: Node feature dimension (0 = disabled).
        test_mask: Optional boolean mask for test edges. If provided,
            evaluates the best model on test set after training.

    Returns:
        (best model, history dict with per-epoch timing breakdown).
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    amp_enabled = use_amp and device.type == "cuda"

    logger.info("Building train sampler (backend=%s)...", sampling_backend)
    train_sampler = build_sampler(data, train_mask, backend=sampling_backend)

    logger.info("Building val sampler (backend=cpp)...")
    val_sampler = build_sampler(data, train_mask | val_mask, backend="cpp")

    model = GraphMixerTime(
        hidden_dim=hidden_dim,
        num_neighbors=num_neighbors,
        num_mixer_layers=num_mixer_layers,
        dropout=dropout,
        edge_feat_dim=edge_feat_dim,
        node_feat_dim=node_feat_dim,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    logger.info("GraphMixerTime: %d params, backend=%s, batch=%d, K=%d",
                n_params, sampling_backend, batch_size, num_neighbors)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

    train_indices = np.where(train_mask)[0]
    val_indices = np.where(val_mask)[0]

    config = {
        "sampling_backend": sampling_backend,
        "batch_size": batch_size,
        "num_neighbors": num_neighbors,
        "num_epochs": num_epochs,
        "hidden_dim": hidden_dim,
        "edge_feat_dim": edge_feat_dim,
        "use_amp": use_amp,
        "num_nodes": int(data.num_nodes),
        "train_edges": int(train_mask.sum()),
        "val_edges": int(val_mask.sum()),
        "n_params": n_params,
        "batches_per_epoch": int(np.ceil(len(train_indices) / batch_size)),
    }
    with open(os.path.join(output_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    history: Dict = {
        "train_loss": [],
        "sampling_time_sec": [],
        "forward_time_sec": [],
        "val_mrr": [],
        "val_hits@1": [],
        "val_hits@3": [],
        "val_hits@10": [],
        "epoch_time_sec": [],
    }

    best_val_mrr = -1.0
    best_epoch = 0
    no_improve = 0

    for epoch in range(1, num_epochs + 1):
        t_epoch = time.time()

        epoch_metrics = train_epoch(
            model, data, train_sampler, train_indices,
            optimizer, scaler, device, batch_size,
            num_neighbors, amp_enabled, rng,
        )

        val_metrics = validate(
            model, data, val_sampler, val_indices,
            device, num_neighbors, max_val_edges, amp_enabled, rng,
        )

        epoch_time = time.time() - t_epoch
        s = epoch_metrics["sampling_time_sec"]
        f = epoch_metrics["forward_time_sec"]
        s_pct = 100 * s / (s + f + 1e-9)

        history["train_loss"].append(epoch_metrics["loss"])
        history["sampling_time_sec"].append(s)
        history["forward_time_sec"].append(f)
        history["val_mrr"].append(val_metrics["mrr"])
        history["val_hits@1"].append(val_metrics["hits@1"])
        history["val_hits@3"].append(val_metrics["hits@3"])
        history["val_hits@10"].append(val_metrics["hits@10"])
        history["epoch_time_sec"].append(epoch_time)

        logger.info(
            "Epoch %d/%d [%.0fs] loss=%.4f  "
            "sampling=%.1fs (%.0f%%)  forward=%.1fs  val_mrr=%.4f",
            epoch, num_epochs, epoch_time,
            epoch_metrics["loss"], s, s_pct, f, val_metrics["mrr"],
        )

        with open(os.path.join(output_dir, "metrics.jsonl"), "a") as mf:
            mf.write(json.dumps({
                "epoch": epoch,
                "backend": sampling_backend,
                **epoch_metrics,
                **{f"val_{k}": v for k, v in val_metrics.items()},
                "epoch_time_sec": epoch_time,
            }) + "\n")

        if val_metrics["mrr"] > best_val_mrr:
            best_val_mrr = val_metrics["mrr"]
            best_epoch = epoch
            no_improve = 0
            torch.save(model.state_dict(),
                       os.path.join(output_dir, "best_model.pt"))
        else:
            no_improve += 1
            if no_improve >= patience:
                logger.info("Early stopping at epoch %d", epoch)
                break

    model.load_state_dict(
        torch.load(os.path.join(output_dir, "best_model.pt"),
                   map_location=device, weights_only=True)
    )

    avg_s = float(np.mean(history["sampling_time_sec"]))
    avg_f = float(np.mean(history["forward_time_sec"]))
    summary = {
        "sampling_backend": sampling_backend,
        "best_epoch": best_epoch,
        "best_val_mrr": best_val_mrr,
        "total_epochs_ran": len(history["train_loss"]),
        "avg_epoch_sec": float(np.mean(history["epoch_time_sec"])),
        "avg_sampling_sec": avg_s,
        "avg_forward_sec": avg_f,
        "sampling_fraction_pct": 100 * avg_s / (avg_s + avg_f + 1e-9),
    }
    with open(os.path.join(output_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    logger.info(
        "Finished backend=%s | epoch=%.0fs | sampling=%.1fs (%.0f%%) | "
        "forward=%.1fs | best_val_mrr=%.4f",
        sampling_backend,
        summary["avg_epoch_sec"],
        avg_s, summary["sampling_fraction_pct"], avg_f,
        best_val_mrr,
    )

    if test_mask is not None and test_mask.sum() > 0:
        logger.info("Evaluating on test set (full evaluation, no subsampling)...")
        all_mask = train_mask | val_mask | test_mask
        test_sampler = build_sampler(data, all_mask, backend="cpp")
        test_indices = np.where(test_mask)[0]
        test_metrics = validate(
            model, data, test_sampler, test_indices,
            device, num_neighbors,
            max_edges=len(test_indices),
            amp_enabled=amp_enabled,
            rng=np.random.default_rng(seed),
        )
        logger.info(
            "TEST RESULTS | MRR=%.4f  Hits@1=%.4f  Hits@3=%.4f  Hits@10=%.4f",
            test_metrics["mrr"], test_metrics["hits@1"],
            test_metrics["hits@3"], test_metrics["hits@10"],
        )
        summary["test_mrr"] = test_metrics["mrr"]
        summary["test_hits@1"] = test_metrics["hits@1"]
        summary["test_hits@3"] = test_metrics["hits@3"]
        summary["test_hits@10"] = test_metrics["hits@10"]
        with open(os.path.join(output_dir, "summary.json"), "w") as f:
            json.dump(summary, f, indent=2)
        history["test_metrics"] = test_metrics

    return model, history

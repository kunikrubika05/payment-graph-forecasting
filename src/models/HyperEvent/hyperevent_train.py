"""Training pipeline for HyperEvent temporal link prediction.

Key differences from EAGLE/GLFormer:
  - No TemporalCSR for model features; uses AdjacencyTable instead.
  - Training processes edges in chronological order (NOT shuffled) so that
    the adjacency table reflects the correct historical state for each event.
  - Relational vectors are 12-dimensional handcrafted features; no learned
    node/edge embeddings are required.
  - The adjacency table is reset and rebuilt from scratch at each epoch start.

AdjacencyTable:
    Maintains the n_neighbor most-recent interaction partners for each node
    using a memory-efficient NumPy circular buffer:
        adj_data[node, :] — partner node IDs, -1 for empty slots
        adj_ptr[node]     — circular write pointer (index of next write mod n_neighbor)
        adj_cnt[node]     — number of valid entries (≤ n_neighbor)

Relational vector (12-dim) for context event e = (eu, ev) w.r.t. query (u*, v*):
    d0(a,b) = count(a in adj[b]) / max(cnt[b], 1)
    d1(a,b) = |set(adj[a]) ∩ set(adj[b])| / max(cnt[a]*cnt[b], 1)
    d2(a,b) = |2hop(a) ∩ 2hop(b)| / max(|2hop(a)|*|2hop(b)|, 1)
              where 2hop(x) = ∪_{y in adj[x]} adj[y][-k2:], k2=floor(√n_neighbor)
    r = [d0(u*,eu), d0(u*,ev), d0(v*,eu), d0(v*,ev),
         d1(u*,eu), d1(u*,ev), d1(v*,eu), d1(v*,ev),
         d2(u*,eu), d2(u*,ev), d2(v*,eu), d2(v*,ev)]
"""

import contextlib
import json
import logging
import math
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from payment_graph_forecasting.training.amp import create_grad_scaler, seed_torch
from src.models.HyperEvent.hyperevent import HyperEventModel
from src.models.HyperEvent.data_utils import TemporalEdgeData

logger = logging.getLogger(__name__)

RELATIONAL_DIM = 12


# ---------------------------------------------------------------------------
# AdjacencyTable
# ---------------------------------------------------------------------------

class AdjacencyTable:
    """Per-node circular buffer of the n_neighbor most-recent interaction partners.

    Memory layout: one NumPy array of shape [num_nodes, n_neighbor] (int32).
    Each row is a circular buffer; entries with value -1 are empty.

    Args:
        num_nodes: Total number of nodes in the graph.
        n_neighbor: Maximum number of stored neighbors per node.
    """

    def __init__(self, num_nodes: int, n_neighbor: int):
        self.num_nodes = num_nodes
        self.n_neighbor = n_neighbor
        self.k2 = max(1, int(math.floor(math.sqrt(n_neighbor))))
        self.adj_data = np.full((num_nodes, n_neighbor), -1, dtype=np.int32)
        self.adj_ptr = np.zeros(num_nodes, dtype=np.int32)
        self.adj_cnt = np.zeros(num_nodes, dtype=np.int32)

    def reset(self) -> None:
        """Clear all entries (used at the start of each training epoch)."""
        self.adj_data.fill(-1)
        self.adj_ptr.fill(0)
        self.adj_cnt.fill(0)

    def add_edge(self, u: int, v: int) -> None:
        """Record directed interaction u → v (call for both directions if undirected)."""
        pos = int(self.adj_ptr[u]) % self.n_neighbor
        self.adj_data[u, pos] = v
        self.adj_ptr[u] += 1
        if self.adj_cnt[u] < self.n_neighbor:
            self.adj_cnt[u] += 1

    def update_batch(self, srcs: np.ndarray, dsts: np.ndarray) -> None:
        """Update adjacency table with a batch of directed edges.

        Args:
            srcs: [N] source node indices (int).
            dsts: [N] destination node indices (int).
        """
        for u, v in zip(srcs.tolist(), dsts.tolist()):
            self.add_edge(u, v)

    def get_neighbors(self, node: int) -> np.ndarray:
        """Return valid neighbor IDs for node (oldest → newest).

        Args:
            node: Node index.

        Returns:
            1-D int32 array of length ≤ n_neighbor; empty if no neighbors.
        """
        cnt = int(self.adj_cnt[node])
        if cnt == 0:
            return np.empty(0, dtype=np.int32)
        n = self.n_neighbor
        ptr = int(self.adj_ptr[node])
        if cnt < n:
            return self.adj_data[node, :cnt].copy()
        # Full circular buffer: oldest entry is at ptr % n
        start = ptr % n
        if start == 0:
            return self.adj_data[node].copy()
        return np.concatenate([self.adj_data[node, start:], self.adj_data[node, :start]])


# ---------------------------------------------------------------------------
# Relational vector computation
# ---------------------------------------------------------------------------

def _d0(query_node: int, context_adj: np.ndarray, context_cnt: int) -> float:
    """0-hop correlation: fraction of adj[context_node] equal to query_node."""
    if context_cnt == 0:
        return 0.0
    return float(np.sum(context_adj[:context_cnt] == query_node)) / context_cnt


def _d1_sets(set_a: set, len_a: int, set_b: set, len_b: int) -> float:
    """1-hop correlation: |adj[a] ∩ adj[b]| / (|adj[a]| * |adj[b]|)."""
    denom = len_a * len_b
    if denom == 0:
        return 0.0
    return len(set_a & set_b) / denom


def _make_2hop(
    adj_data: np.ndarray,
    adj_cnt: np.ndarray,
    neighbors: np.ndarray,
    k2: int,
) -> set:
    """Compute 2-hop neighborhood as a set.

    For each neighbor x in neighbors, take the k2 most-recent partners of x.
    Returns the union as a set (duplicates are collapsed).

    Args:
        adj_data: [num_nodes, n_neighbor] adjacency buffer.
        adj_cnt: [num_nodes] valid entry counts.
        neighbors: 1-D array of neighbor node IDs (no -1 values).
        k2: Number of 2-hop partners to take per neighbor (= floor(sqrt(n_neighbor))).

    Returns:
        Set of node IDs reachable in 2 hops.
    """
    result: set = set()
    for nb in neighbors.tolist():
        cnt = int(adj_cnt[nb])
        if cnt == 0:
            continue
        row = adj_data[nb, :cnt]
        tail = row[-k2:] if cnt > k2 else row
        for x in tail.tolist():
            if x >= 0:
                result.add(x)
    return result


def _d2_sets(hop2_a: set, hop2_b: set) -> float:
    """2-hop correlation: |2hop(a) ∩ 2hop(b)| / (|2hop(a)| * |2hop(b)|)."""
    la, lb = len(hop2_a), len(hop2_b)
    denom = la * lb
    if denom == 0:
        return 0.0
    return len(hop2_a & hop2_b) / denom


def compute_relational_vectors(
    adj: AdjacencyTable,
    u_star: int,
    v_star: int,
    n_latest: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute the relational vector sequence for one query event (u*, v*).

    Extracts context H = S_{u*} ∪ S_{v*}:
        S_{u*}: last n_latest neighbors of u* → events (u*, neighbor)
        S_{v*}: last n_latest neighbors of v* → events (v*, neighbor)

    For each context event e_i = (eu_i, ev_i) computes 12-dim r_i.

    Args:
        adj: Current AdjacencyTable.
        u_star: Query source node.
        v_star: Query destination node.
        n_latest: Number of context events to take per query node.

    Returns:
        Tuple (rel_vecs, pad_mask):
            rel_vecs:  [max_seq, 12] float32, max_seq = 2 * n_latest.
            pad_mask:  [max_seq] bool, True = padding position.
    """
    max_seq = 2 * n_latest
    rel_vecs = np.zeros((max_seq, RELATIONAL_DIM), dtype=np.float32)
    pad_mask = np.ones(max_seq, dtype=bool)

    # --- neighbourhoods of query nodes ---
    nb_u = adj.get_neighbors(u_star)  # all valid, oldest→newest
    nb_v = adj.get_neighbors(v_star)
    ctx_u = nb_u[-n_latest:] if len(nb_u) >= n_latest else nb_u
    ctx_v = nb_v[-n_latest:] if len(nb_v) >= n_latest else nb_v

    # Build context event list: (eu, ev) pairs
    ctx_eu: List[int] = [u_star] * len(ctx_u) + [v_star] * len(ctx_v)
    ctx_ev: List[int] = ctx_u.tolist() + ctx_v.tolist()
    seq_len = len(ctx_eu)
    if seq_len == 0:
        return rel_vecs, pad_mask

    # Pre-compute query node adjacency info (shared across all context events)
    nb_u_valid = nb_u[nb_u >= 0] if len(nb_u) else np.empty(0, dtype=np.int32)
    nb_v_valid = nb_v[nb_v >= 0] if len(nb_v) else np.empty(0, dtype=np.int32)
    set_u = set(nb_u_valid.tolist())
    set_v = set(nb_v_valid.tolist())
    len_u = len(set_u)
    len_v = len(set_v)
    hop2_u = _make_2hop(adj.adj_data, adj.adj_cnt, nb_u_valid, adj.k2)
    hop2_v = _make_2hop(adj.adj_data, adj.adj_cnt, nb_v_valid, adj.k2)

    for i in range(seq_len):
        eu = ctx_eu[i]
        ev = ctx_ev[i]

        # Neighbours of context event endpoints
        cnt_eu = int(adj.adj_cnt[eu])
        cnt_ev = int(adj.adj_cnt[ev])
        row_eu = adj.adj_data[eu, :cnt_eu] if cnt_eu > 0 else np.empty(0, dtype=np.int32)
        row_ev = adj.adj_data[ev, :cnt_ev] if cnt_ev > 0 else np.empty(0, dtype=np.int32)
        set_eu = set(row_eu[row_eu >= 0].tolist())
        set_ev = set(row_ev[row_ev >= 0].tolist())

        # d0
        d0_u_eu = _d0(u_star, row_eu, cnt_eu)
        d0_u_ev = _d0(u_star, row_ev, cnt_ev)
        d0_v_eu = _d0(v_star, row_eu, cnt_eu)
        d0_v_ev = _d0(v_star, row_ev, cnt_ev)

        # d1
        len_eu = len(set_eu)
        len_ev = len(set_ev)
        d1_u_eu = _d1_sets(set_u, len_u, set_eu, len_eu)
        d1_u_ev = _d1_sets(set_u, len_u, set_ev, len_ev)
        d1_v_eu = _d1_sets(set_v, len_v, set_eu, len_eu)
        d1_v_ev = _d1_sets(set_v, len_v, set_ev, len_ev)

        # d2
        nb_eu_valid = row_eu[row_eu >= 0] if cnt_eu > 0 else np.empty(0, dtype=np.int32)
        nb_ev_valid = row_ev[row_ev >= 0] if cnt_ev > 0 else np.empty(0, dtype=np.int32)
        hop2_eu = _make_2hop(adj.adj_data, adj.adj_cnt, nb_eu_valid, adj.k2)
        hop2_ev = _make_2hop(adj.adj_data, adj.adj_cnt, nb_ev_valid, adj.k2)
        d2_u_eu = _d2_sets(hop2_u, hop2_eu)
        d2_u_ev = _d2_sets(hop2_u, hop2_ev)
        d2_v_eu = _d2_sets(hop2_v, hop2_eu)
        d2_v_ev = _d2_sets(hop2_v, hop2_ev)

        rel_vecs[i] = [
            d0_u_eu, d0_u_ev, d0_v_eu, d0_v_ev,
            d1_u_eu, d1_u_ev, d1_v_eu, d1_v_ev,
            d2_u_eu, d2_u_ev, d2_v_eu, d2_v_ev,
        ]
        pad_mask[i] = False

    return rel_vecs, pad_mask


def compute_batch_relational_vectors(
    adj: AdjacencyTable,
    u_stars: np.ndarray,
    v_stars: np.ndarray,
    n_latest: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute relational vectors for a batch of (u*, v*) query pairs.

    Args:
        adj: Current AdjacencyTable.
        u_stars: [B] source node indices.
        v_stars: [B] destination node indices.
        n_latest: Context events per query node.

    Returns:
        Tuple (rel_vecs, pad_mask):
            rel_vecs:  [B, max_seq, 12] float32.
            pad_mask:  [B, max_seq] bool.
    """
    B = len(u_stars)
    max_seq = 2 * n_latest
    all_vecs = np.zeros((B, max_seq, RELATIONAL_DIM), dtype=np.float32)
    all_masks = np.ones((B, max_seq), dtype=bool)
    for i in range(B):
        v, m = compute_relational_vectors(adj, int(u_stars[i]), int(v_stars[i]), n_latest)
        all_vecs[i] = v
        all_masks[i] = m
    return all_vecs, all_masks


def ensure_non_empty_relational_sequences(
    rel_vecs: np.ndarray,
    pad_masks: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Ensure every sequence contains at least one unmasked token.

    Tiny validation slices can produce queries with no historical context at all.
    PyTorch's Transformer nested-tensor path raises when an entire sequence is
    padding, so we expose a single zero-vector token for those rows and let the
    classifier fall back to its learned prior for "no context" cases.
    """

    empty_rows = np.all(pad_masks, axis=1)
    if not np.any(empty_rows):
        return rel_vecs, pad_masks

    safe_masks = pad_masks.copy()
    safe_masks[empty_rows, 0] = False
    return rel_vecs, safe_masks


# ---------------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------------

def _amp_autocast(enabled: bool, device_type: str):
    if enabled and device_type == "cuda":
        return torch.cuda.amp.autocast()
    return contextlib.nullcontext()


def train_epoch(
    model: HyperEventModel,
    data: TemporalEdgeData,
    adj: AdjacencyTable,
    edge_indices: np.ndarray,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    batch_size: int = 200,
    n_latest: int = 10,
    neg_per_positive: int = 1,
    use_amp: bool = True,
    scaler: Optional[torch.cuda.amp.GradScaler] = None,
    rng: Optional[np.random.Generator] = None,
) -> Dict[str, float]:
    """Run one training epoch in chronological edge order.

    Processes edges in chronological order (no shuffle) so the adjacency
    table reflects the correct historical state for each event.  After each
    batch, the adjacency table is updated with the batch's positive edges.

    Args:
        model: HyperEventModel.
        data: TemporalEdgeData.
        adj: AdjacencyTable (reset before calling; updated in-place here).
        edge_indices: Indices of training edges in chronological order.
        optimizer: Torch optimizer.
        device: Torch device.
        batch_size: Edges per batch.
        n_latest: Context events per query node.
        neg_per_positive: Negative samples per positive edge.
        use_amp: Enable mixed precision.
        scaler: GradScaler for AMP.
        rng: Random number generator.

    Returns:
        Dict with 'loss' (mean batch loss).
    """
    if rng is None:
        rng = np.random.default_rng()

    model.train()
    criterion = nn.BCEWithLogitsLoss()
    amp_enabled = use_amp and device.type == "cuda"

    total_loss = 0.0
    num_batches = 0
    n_batches = math.ceil(len(edge_indices) / batch_size)

    pbar = tqdm(
        range(0, len(edge_indices), batch_size),
        total=n_batches,
        desc="Training",
        leave=False,
        unit="batch",
    )

    for start in pbar:
        end = min(start + batch_size, len(edge_indices))
        batch_idx = edge_indices[start:end]
        B = len(batch_idx)

        src = data.src[batch_idx]
        dst = data.dst[batch_idx]

        neg_dst = rng.integers(0, data.num_nodes, size=(B, neg_per_positive)).astype(np.int32)

        # Positive pairs
        pos_vecs, pos_masks = compute_batch_relational_vectors(adj, src, dst, n_latest)

        # Negative pairs (reuse src adj table lookup)
        neg_vecs_list = []
        neg_masks_list = []
        for ni in range(neg_per_positive):
            nv, nm = compute_batch_relational_vectors(adj, src, neg_dst[:, ni], n_latest)
            neg_vecs_list.append(nv)
            neg_masks_list.append(nm)

        def _t(arr, dtype=torch.float32):
            return torch.tensor(arr, dtype=dtype, device=device)

        with _amp_autocast(amp_enabled, device.type):
            pos_logits = model(_t(pos_vecs), _t(pos_masks, torch.bool))

            neg_logits_list = []
            for ni in range(neg_per_positive):
                neg_logits_list.append(
                    model(_t(neg_vecs_list[ni]), _t(neg_masks_list[ni], torch.bool))
                )

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

        # Update adjacency table AFTER gradients (uses batch edge state)
        adj.update_batch(src, dst)

    pbar.close()
    return {"loss": total_loss / max(num_batches, 1)}


@torch.no_grad()
def validate(
    model: HyperEventModel,
    data: TemporalEdgeData,
    adj: AdjacencyTable,
    edge_indices: np.ndarray,
    device: torch.device,
    n_latest: int = 10,
    n_eval_negatives: int = 100,
    max_eval_edges: int = 5000,
    use_amp: bool = True,
    rng: Optional[np.random.Generator] = None,
) -> Dict[str, float]:
    """Validate with ranking metrics (random negatives, fast version).

    Uses the adjacency table as-is (built from training edges).  Evaluates
    on a subsample of edge_indices for speed.

    Args:
        model: HyperEventModel.
        data: TemporalEdgeData.
        adj: AdjacencyTable (already populated with training history).
        edge_indices: Validation edge indices.
        device: Torch device.
        n_latest: Context events per query node.
        n_eval_negatives: Random negatives per positive edge.
        max_eval_edges: Maximum edges to evaluate.
        use_amp: Enable mixed precision.
        rng: Random number generator.

    Returns:
        Dict with 'mrr', 'hits@1', 'hits@3', 'hits@10'.
    """
    if rng is None:
        rng = np.random.default_rng(42)

    model.eval()
    amp_enabled = use_amp and device.type == "cuda"

    if len(edge_indices) > max_eval_edges:
        eval_idx = rng.choice(edge_indices, size=max_eval_edges, replace=False)
    else:
        eval_idx = edge_indices

    ranks = []

    for idx in eval_idx:
        u_star = int(data.src[idx])
        true_dst = int(data.dst[idx])

        neg_nodes = rng.integers(0, data.num_nodes, size=n_eval_negatives).astype(np.int32)
        all_dst = np.concatenate([[true_dst], neg_nodes])
        C = len(all_dst)

        u_stars = np.full(C, u_star, dtype=np.int32)
        vecs, masks = compute_batch_relational_vectors(adj, u_stars, all_dst, n_latest)

        def _t(arr, dtype=torch.float32):
            return torch.tensor(arr, dtype=dtype, device=device)

        vecs, masks = ensure_non_empty_relational_sequences(vecs, masks)
        with _amp_autocast(amp_enabled, device.type):
            scores = model(_t(vecs), _t(masks, torch.bool)).cpu().float().numpy()

        true_score = scores[0]
        rank = (
            1.0
            + float((scores[1:] > true_score).sum())
            + 0.5 * float((scores[1:] == true_score).sum())
        )
        ranks.append(rank)

    ranks = np.array(ranks, dtype=np.float64)
    return {
        "mrr": float(np.mean(1.0 / ranks)),
        "hits@1": float(np.mean(ranks <= 1)),
        "hits@3": float(np.mean(ranks <= 3)),
        "hits@10": float(np.mean(ranks <= 10)),
        "n_queries": len(ranks),
    }


def build_adj_from_mask(
    data: TemporalEdgeData,
    mask: np.ndarray,
    n_neighbor: int,
) -> AdjacencyTable:
    """Build an adjacency table from all edges selected by mask (chronological).

    Args:
        data: TemporalEdgeData.
        mask: Boolean mask of shape [num_edges].
        n_neighbor: Adjacency table capacity per node.

    Returns:
        Populated AdjacencyTable.
    """
    adj = AdjacencyTable(data.num_nodes, n_neighbor)
    indices = np.where(mask)[0]
    srcs = data.src[indices]
    dsts = data.dst[indices]
    adj.update_batch(srcs, dsts)
    return adj


# ---------------------------------------------------------------------------
# Full training pipeline
# ---------------------------------------------------------------------------

def train_hyperevent(
    data: TemporalEdgeData,
    train_mask: np.ndarray,
    val_mask: np.ndarray,
    output_dir: str,
    device: Optional[torch.device] = None,
    num_epochs: int = 50,
    batch_size: int = 200,
    learning_rate: float = 0.0001,
    weight_decay: float = 1e-5,
    n_neighbor: int = 20,
    n_latest: int = 10,
    d_model: int = 64,
    n_heads: int = 4,
    n_layers: int = 3,
    dropout: float = 0.1,
    patience: int = 20,
    seed: int = 42,
    max_val_edges: int = 5000,
    use_amp: bool = True,
) -> Tuple[HyperEventModel, Dict]:
    """Full HyperEvent training pipeline.

    Args:
        data: TemporalEdgeData loaded from stream graph.
        train_mask: Boolean mask for training edges.
        val_mask: Boolean mask for validation edges.
        output_dir: Directory for checkpoints and logs.
        device: Torch device (auto-detected if None).
        num_epochs: Maximum training epochs.
        batch_size: Edges per training batch.
        learning_rate: Adam learning rate (paper: 0.0001).
        weight_decay: Adam weight decay.
        n_neighbor: Adjacency table capacity per node (paper: 10-50).
        n_latest: Context events taken per query node (paper: 10).
        d_model: Transformer hidden dimension (paper: 64).
        n_heads: Transformer attention heads (paper: 4).
        n_layers: Transformer encoder layers (paper: 3).
        dropout: Dropout rate (paper: 0.1).
        patience: Early stopping patience epochs.
        seed: Random seed.
        max_val_edges: Maximum validation edges per epoch (for speed).
        use_amp: Enable AMP mixed precision on GPU.

    Returns:
        Tuple of (trained model, training history dict).
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(seed)
    seed_torch(seed, device)

    model = HyperEventModel(
        feat_dim=RELATIONAL_DIM,
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        dropout=dropout,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(
        "HyperEvent: %d total params, %d trainable",
        total_params, trainable_params,
    )

    optimizer = torch.optim.Adam(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    amp_enabled = use_amp and device.type == "cuda"
    scaler = create_grad_scaler(enabled=amp_enabled)

    train_indices = np.where(train_mask)[0]  # already sorted chronologically
    val_indices = np.where(val_mask)[0]

    config = {
        "model": "HyperEvent",
        "n_neighbor": n_neighbor,
        "n_latest": n_latest,
        "d_model": d_model,
        "n_heads": n_heads,
        "n_layers": n_layers,
        "dropout": dropout,
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

    history: Dict = {
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

    logger.info(
        "Training HyperEvent: %d epochs, %d train, %d val edges",
        num_epochs, len(train_indices), len(val_indices),
    )

    for epoch in range(1, num_epochs + 1):
        epoch_start = time.time()

        # Rebuild adjacency table from scratch each epoch
        adj = AdjacencyTable(data.num_nodes, n_neighbor)

        train_metrics = train_epoch(
            model, data, adj, train_indices, optimizer, device,
            batch_size=batch_size,
            n_latest=n_latest,
            use_amp=use_amp,
            scaler=scaler,
            rng=rng,
        )

        # adj now contains full training history; rebuild for val (same state)
        val_adj = build_adj_from_mask(data, train_mask, n_neighbor)

        val_metrics = validate(
            model, data, val_adj, val_indices, device,
            n_latest=n_latest,
            max_eval_edges=max_val_edges,
            use_amp=use_amp,
            rng=np.random.default_rng(42),
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
            val_metrics["mrr"],
            val_metrics["hits@1"],
            val_metrics["hits@3"],
            val_metrics["hits@10"],
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
            logger.info("Early stopping at epoch %d (patience=%d)", epoch, patience)
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

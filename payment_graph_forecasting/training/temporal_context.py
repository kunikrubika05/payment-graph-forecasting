"""Shared temporal context extraction helpers for temporal LP models."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from src.models.data_utils import featurize_neighbors


@dataclass(slots=True)
class NodeTemporalContext:
    """Sampled temporal neighborhood plus optional attached features."""

    neighbor_ids: np.ndarray
    delta_times: np.ndarray
    lengths: np.ndarray
    edge_features: np.ndarray | None = None
    node_features: np.ndarray | None = None


def to_device_tensor(arr, device: torch.device, dtype=torch.float32) -> torch.Tensor:
    """Convert an array-like object to a torch tensor on the target device."""

    return torch.tensor(arr, dtype=dtype, device=device)


def sample_node_contexts(
    *,
    csr,
    data,
    sample_neighbors_fn,
    nodes: np.ndarray,
    query_timestamps: np.ndarray,
    num_neighbors: int,
    use_edge_feats: bool = False,
    use_node_feats: bool = False,
    zero_pad_delta: bool = False,
) -> NodeTemporalContext:
    """Sample temporal neighbors and build model-ready context arrays."""

    neighbor_ids, neighbor_ts, neighbor_eids, lengths = sample_neighbors_fn(
        csr, nodes, query_timestamps, num_neighbors
    )
    delta_times = np.maximum(
        query_timestamps[:, None] - neighbor_ts, 0.0
    ).astype(np.float32)

    if zero_pad_delta:
        for row_idx in range(len(nodes)):
            delta_times[row_idx, lengths[row_idx]:] = 0.0

    edge_features = None
    node_features = None
    if use_edge_feats or use_node_feats:
        _, edge_feat_raw, _ = featurize_neighbors(
            neighbor_ids,
            neighbor_eids,
            lengths,
            neighbor_ts,
            query_timestamps,
            data.node_feats,
            data.edge_feats,
        )
        if use_edge_feats:
            edge_features = edge_feat_raw.astype(np.float32)
        if use_node_feats:
            node_features = data.node_feats[nodes].astype(np.float32)

    return NodeTemporalContext(
        neighbor_ids=neighbor_ids,
        delta_times=delta_times,
        lengths=lengths,
        edge_features=edge_features,
        node_features=node_features,
    )

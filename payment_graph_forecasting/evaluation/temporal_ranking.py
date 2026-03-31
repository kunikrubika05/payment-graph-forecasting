"""Shared ranking/scoring helpers for temporal link prediction models."""

from __future__ import annotations

import numpy as np
import torch

from payment_graph_forecasting.training.amp import autocast_context
from payment_graph_forecasting.training.temporal_context import (
    NodeTemporalContext,
    to_device_tensor,
)


def score_candidate_contexts(
    *,
    model,
    device: torch.device,
    src_context: NodeTemporalContext,
    dst_context: NodeTemporalContext,
    amp_enabled: bool,
    cooc_counts: torch.Tensor | None = None,
) -> np.ndarray:
    """Score destination candidates for a single source context.

    This helper assumes a model API compatible with EAGLE/GLFormer:
    - ``encode_nodes(...)``
    - ``edge_predictor(...)``
    - optional ``cooc_encoder`` attribute for GLFormer-like models
    """

    with autocast_context(amp_enabled, device.type):
        h_src = model.encode_nodes(
            to_device_tensor(src_context.delta_times, device),
            to_device_tensor(src_context.lengths, device, torch.int64),
            edge_feats=to_device_tensor(src_context.edge_features, device) if src_context.edge_features is not None else None,
            node_feats=to_device_tensor(src_context.node_features, device) if src_context.node_features is not None else None,
        )
        h_dst = model.encode_nodes(
            to_device_tensor(dst_context.delta_times, device),
            to_device_tensor(dst_context.lengths, device, torch.int64),
            edge_feats=to_device_tensor(dst_context.edge_features, device) if dst_context.edge_features is not None else None,
            node_feats=to_device_tensor(dst_context.node_features, device) if dst_context.node_features is not None else None,
        )
        h_src_exp = h_src.expand(dst_context.delta_times.shape[0], -1)

        cooc_feat = None
        if getattr(model, "cooc_encoder", None) is not None and cooc_counts is not None:
            cooc_feat = model.cooc_encoder(cooc_counts)

        if cooc_feat is not None:
            scores = model.edge_predictor(h_src_exp, h_dst, cooc_feat)
        else:
            scores = model.edge_predictor(h_src_exp, h_dst)

    return scores.detach().cpu().float().numpy()


def conservative_rank_from_scores(scores: np.ndarray) -> float:
    """Return the conservative rank of the first score among candidates."""

    true_score = scores[0]
    return float(1.0 + np.sum(scores[1:] > true_score))

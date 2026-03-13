"""Feature engineering: aggregation over time windows, pair feature construction."""

import logging
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd

from src.baselines.config import NODE_FEATURE_COLUMNS

logger = logging.getLogger(__name__)


def aggregate_features_mean(
    features_by_date: Dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """Aggregate node features across multiple days using simple mean.

    Args:
        features_by_date: Dict mapping date strings to DataFrames
            (node_idx as index, feature columns).

    Returns:
        DataFrame with node_idx as index, 25 aggregated feature columns.
        Only includes nodes that appeared in at least one day.
    """
    if not features_by_date:
        return pd.DataFrame(columns=NODE_FEATURE_COLUMNS)

    all_frames = []
    for date, df in features_by_date.items():
        if df is not None and len(df) > 0:
            cols = [c for c in NODE_FEATURE_COLUMNS if c in df.columns]
            all_frames.append(df[cols])

    if not all_frames:
        return pd.DataFrame(columns=NODE_FEATURE_COLUMNS)

    combined = pd.concat(all_frames)
    aggregated = combined.groupby(level=0).mean()

    for col in NODE_FEATURE_COLUMNS:
        if col not in aggregated.columns:
            aggregated[col] = 0.0

    return aggregated[NODE_FEATURE_COLUMNS]


def aggregate_features_time_weighted(
    features_by_date: Dict[str, pd.DataFrame],
    dates: List[str],
    decay_lambda: float = 0.3,
) -> pd.DataFrame:
    """Aggregate node features with exponential time decay.

    Args:
        features_by_date: Dict mapping date strings to DataFrames.
        dates: Ordered list of dates in the window (earliest to latest).
        decay_lambda: Decay parameter. Higher = more weight on recent days.

    Returns:
        DataFrame with node_idx as index, 25 aggregated feature columns.
    """
    if not features_by_date or not dates:
        return pd.DataFrame(columns=NODE_FEATURE_COLUMNS)

    T = len(dates) - 1
    weights = {}
    for i, date in enumerate(dates):
        weights[date] = np.exp(-decay_lambda * (T - i))

    all_nodes = set()
    for df in features_by_date.values():
        if df is not None and len(df) > 0:
            all_nodes.update(df.index)

    if not all_nodes:
        return pd.DataFrame(columns=NODE_FEATURE_COLUMNS)

    all_nodes_arr = np.array(sorted(all_nodes))
    n_nodes = len(all_nodes_arr)
    n_features = len(NODE_FEATURE_COLUMNS)
    node_to_pos = {node: i for i, node in enumerate(all_nodes_arr)}

    weighted_sum = np.zeros((n_nodes, n_features), dtype=np.float64)
    weight_sum = np.zeros((n_nodes, 1), dtype=np.float64)

    for date in dates:
        df = features_by_date.get(date)
        if df is None or len(df) == 0:
            continue
        w = weights.get(date, 0.0)
        cols = [c for c in NODE_FEATURE_COLUMNS if c in df.columns]
        node_indices = df.index.values
        positions = np.array([node_to_pos[n] for n in node_indices if n in node_to_pos])
        valid_mask = np.array([n in node_to_pos for n in node_indices])
        if len(positions) == 0:
            continue
        values = df.loc[node_indices[valid_mask], cols].values
        if values.shape[1] < n_features:
            padded = np.zeros((values.shape[0], n_features))
            for fi, col in enumerate(NODE_FEATURE_COLUMNS):
                if col in cols:
                    padded[:, fi] = values[:, cols.index(col)]
            values = padded
        weighted_sum[positions] += values * w
        weight_sum[positions] += w

    nonzero = weight_sum.ravel() > 0
    weighted_sum[nonzero] /= weight_sum[nonzero]

    result = pd.DataFrame(
        weighted_sum, index=all_nodes_arr, columns=NODE_FEATURE_COLUMNS
    )
    result = result.loc[nonzero]
    result.index.name = "node_idx"
    return result


def build_pair_features(
    node_features: pd.DataFrame,
    src_indices: np.ndarray,
    dst_indices: np.ndarray,
    mode: str = "extended",
) -> Tuple[np.ndarray, List[str]]:
    """Build feature vectors for pairs of nodes.

    Args:
        node_features: DataFrame with node_idx as index, feature columns.
        src_indices: Source node indices array.
        dst_indices: Destination node indices array.
        mode: 'base' for [src, dst] (50 features) or
              'extended' for [src, dst, src-dst, src*dst] (100 features).

    Returns:
        Tuple of (feature_matrix, feature_names).
        feature_matrix has shape (n_pairs, n_features).
    """
    feature_cols = [c for c in NODE_FEATURE_COLUMNS if c in node_features.columns]
    n_feat = len(feature_cols)

    feat_matrix = node_features[feature_cols].values.astype(np.float32)
    idx_to_row = pd.Series(
        np.arange(len(node_features)), index=node_features.index
    )

    src_rows = idx_to_row.reindex(src_indices).values
    dst_rows = idx_to_row.reindex(dst_indices).values

    valid_mask = ~(np.isnan(src_rows) | np.isnan(dst_rows))
    src_rows = src_rows.astype(int)
    dst_rows = dst_rows.astype(int)

    src_feats = feat_matrix[src_rows]
    dst_feats = feat_matrix[dst_rows]

    src_feats[~valid_mask] = 0.0
    dst_feats[~valid_mask] = 0.0

    names_src = [f"src_{c}" for c in feature_cols]
    names_dst = [f"dst_{c}" for c in feature_cols]

    if mode == "base":
        X = np.hstack([src_feats, dst_feats])
        feature_names = names_src + names_dst
    elif mode == "extended":
        diff_feats = src_feats - dst_feats
        prod_feats = src_feats * dst_feats
        X = np.hstack([src_feats, dst_feats, diff_feats, prod_feats])
        names_diff = [f"diff_{c}" for c in feature_cols]
        names_prod = [f"prod_{c}" for c in feature_cols]
        feature_names = names_src + names_dst + names_diff + names_prod
    else:
        raise ValueError(f"Unknown feature mode: {mode}")

    nan_mask = np.isnan(X)
    if nan_mask.any():
        X = np.nan_to_num(X, nan=0.0)

    return X.astype(np.float32), feature_names


def get_feature_names(mode: str = "extended") -> List[str]:
    """Get list of feature names for pair features.

    Args:
        mode: 'base' or 'extended'.

    Returns:
        List of feature name strings.
    """
    names_src = [f"src_{c}" for c in NODE_FEATURE_COLUMNS]
    names_dst = [f"dst_{c}" for c in NODE_FEATURE_COLUMNS]
    if mode == "base":
        return names_src + names_dst
    names_diff = [f"diff_{c}" for c in NODE_FEATURE_COLUMNS]
    names_prod = [f"prod_{c}" for c in NODE_FEATURE_COLUMNS]
    return names_src + names_dst + names_diff + names_prod


def compute_feature_correlations(
    X: np.ndarray, feature_names: List[str]
) -> pd.DataFrame:
    """Compute Pearson correlation matrix.

    Args:
        X: Feature matrix (n_samples, n_features).
        feature_names: List of feature names.

    Returns:
        Correlation matrix as DataFrame.
    """
    df = pd.DataFrame(X, columns=feature_names)
    return df.corr()

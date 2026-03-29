"""Negative sampling for training and evaluation.

Protocol:
- Training: for each positive edge, sample negative_ratio negatives
  (50% historical + 50% random).
- Evaluation (TGB-style): for each positive edge, sample n_negatives
  candidates (50% historical + 50% random), rank true destination.

Historical negatives = train neighbors of source that are NOT positive
destinations in the current split. This ensures no target leakage.
Random negatives = sampled from all nodes active in train.
"""

import numpy as np


def sample_negatives_for_training(
    src: np.ndarray,
    dst: np.ndarray,
    train_neighbors: dict[int, set[int]],
    active_nodes: np.ndarray,
    negative_ratio: int,
    rng: np.random.RandomState,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sample negative pairs for binary classification training.

    For each positive (src_i, dst_i), samples negative_ratio negative
    destinations. 50% from historical neighbors of src_i (excluding dst_i),
    50% random from active_nodes.

    Args:
        src: Positive source indices (global).
        dst: Positive destination indices (global).
        train_neighbors: Per-source neighbor sets from train.
        active_nodes: Sorted array of all active node indices (from train).
        negative_ratio: Number of negatives per positive.
        rng: Random state.

    Returns:
        (all_src, all_dst, all_labels) where labels are 1 for positive, 0 for negative.
    """
    n_pos = len(src)
    active_set = set(active_nodes.tolist())

    all_src_list = list(src)
    all_dst_list = list(dst)
    all_labels = [1] * n_pos

    positive_set_per_src: dict[int, set[int]] = {}
    for s, d in zip(src, dst):
        positive_set_per_src.setdefault(s, set()).add(d)

    for i in range(n_pos):
        s = int(src[i])
        d = int(dst[i])
        positives_of_s = positive_set_per_src.get(s, set())
        hist_candidates = train_neighbors.get(s, set()) - positives_of_s - {s}

        n_total = negative_ratio
        n_hist = min(n_total // 2, len(hist_candidates))
        n_rand = n_total - n_hist

        if n_hist > 0:
            hist_arr = np.array(list(hist_candidates), dtype=np.int64)
            chosen_hist = rng.choice(hist_arr, size=n_hist, replace=False)
            for h in chosen_hist:
                all_src_list.append(s)
                all_dst_list.append(int(h))
                all_labels.append(0)

        exclude = positives_of_s | hist_candidates | {s}
        rand_negs = _sample_random(active_nodes, exclude, n_rand, rng)
        for r in rand_negs:
            all_src_list.append(s)
            all_dst_list.append(int(r))
            all_labels.append(0)

    return (
        np.array(all_src_list, dtype=np.int64),
        np.array(all_dst_list, dtype=np.int64),
        np.array(all_labels, dtype=np.int32),
    )


def sample_negatives_for_eval(
    src: int,
    dst_true: int,
    train_neighbors: dict[int, set[int]],
    eval_positives_of_src: set[int],
    active_nodes: np.ndarray,
    n_negatives: int,
    rng: np.random.RandomState,
) -> np.ndarray:
    """Sample negative candidates for one query in TGB-style evaluation.

    Args:
        src: Source node (global index).
        dst_true: True destination (global index).
        train_neighbors: Per-source neighbor sets from train.
        eval_positives_of_src: All positive destinations of src in eval split.
        active_nodes: Sorted array of active nodes from train.
        n_negatives: Number of negative candidates.
        rng: Random state.

    Returns:
        Array of n_negatives negative destination indices (global).
    """
    exclude = eval_positives_of_src | {src, dst_true}
    hist_candidates = train_neighbors.get(src, set()) - exclude

    n_hist = min(n_negatives // 2, len(hist_candidates))
    n_rand = n_negatives - n_hist

    negatives = []

    if n_hist > 0:
        hist_arr = np.array(list(hist_candidates), dtype=np.int64)
        chosen = rng.choice(hist_arr, size=n_hist, replace=False)
        negatives.extend(chosen.tolist())

    exclude_for_rand = exclude | set(negatives)
    rand_negs = _sample_random(active_nodes, exclude_for_rand, n_rand, rng)
    negatives.extend(rand_negs)

    return np.array(negatives[:n_negatives], dtype=np.int64)


def _sample_random(
    active_nodes: np.ndarray,
    exclude: set[int],
    n_needed: int,
    rng: np.random.RandomState,
) -> list[int]:
    """Sample random nodes from active_nodes, excluding a set."""
    if n_needed <= 0:
        return []

    result = []
    max_attempts = n_needed * 20
    attempts = 0

    while len(result) < n_needed and attempts < max_attempts:
        batch_size = min((n_needed - len(result)) * 3, len(active_nodes))
        candidates = active_nodes[rng.randint(0, len(active_nodes), size=batch_size)]
        for c in candidates:
            c_int = int(c)
            if c_int not in exclude:
                result.append(c_int)
                exclude.add(c_int)
                if len(result) >= n_needed:
                    break
        attempts += batch_size

    return result[:n_needed]

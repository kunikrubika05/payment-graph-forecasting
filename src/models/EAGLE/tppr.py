"""Temporal Personalized PageRank (TPPR) for EAGLE structure component.

TPPR computes structural similarity between nodes by maintaining
per-node PPR dictionaries that are updated incrementally as edges
are processed in temporal order.

Usage (standalone):
    PYTHONPATH=. python src/models/EAGLE/tppr.py \
        --parquet-path stream_graph/2020-06-01_2020-08-31.parquet \
        --topk 100 --alpha 0.9 --beta 0.8 \
        --output /tmp/eagle_tppr 2>&1 | tee /tmp/eagle_tppr.log
"""

import argparse
import heapq
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from tqdm import tqdm

from src.models.EAGLE.data_utils import (
    TemporalEdgeData,
    TemporalCSR,
    build_temporal_csr,
    generate_negatives_for_eval,
    load_stream_graph_data,
)
from src.baselines.evaluation import compute_ranking_metrics

logger = logging.getLogger(__name__)


def get_forward_edge_mask(data: TemporalEdgeData) -> np.ndarray:
    """Get mask for forward-only edges in undirected TemporalEdgeData.

    In undirected data built by build_event_stream(undirected=True),
    each timestamp has forward edges first then reverse edges (stable sort).
    This returns a mask selecting only the first half per timestamp.

    Args:
        data: Undirected TemporalEdgeData.

    Returns:
        Boolean mask of shape [num_edges].
    """
    mask = np.zeros(data.num_edges, dtype=bool)
    unique_ts, ts_starts, ts_counts = np.unique(
        data.timestamps, return_index=True, return_counts=True
    )
    for start, count in zip(ts_starts, ts_counts):
        half = count // 2
        if half == 0:
            half = count
        mask[start:start + half] = True
    return mask


class TPPR:
    """Temporal Personalized PageRank.

    Maintains a PPR dictionary per node, updated incrementally as
    edges are observed. The PPR captures structural proximity with
    temporal decay — recent connections have stronger influence.

    Args:
        num_nodes: Total number of nodes.
        topk: Maximum PPR entries per node.
        alpha: Restart probability (higher = more weight on direct neighbor).
        beta: Temporal decay factor (higher = slower decay of old connections).
    """

    def __init__(self, num_nodes: int, topk: int = 100,
                 alpha: float = 0.9, beta: float = 0.8):
        self.num_nodes = num_nodes
        self.topk = topk
        self.alpha = alpha
        self.beta = beta
        self.ppr: List[Dict[int, float]] = [dict() for _ in range(num_nodes)]
        self.norms = np.zeros(num_nodes, dtype=np.float64)

    def get_similarity(self, src: int, dst: int) -> float:
        """Compute TPPR similarity between two nodes.

        Args:
            src: Source node index.
            dst: Destination node index.

        Returns:
            Dot-product similarity of PPR vectors (unnormalized).
        """
        src_ppr = self.ppr[src]
        dst_ppr = self.ppr[dst]
        if len(src_ppr) == 0 or len(dst_ppr) == 0:
            return 0.0

        sim = 0.0
        smaller, larger = (src_ppr, dst_ppr) if len(src_ppr) <= len(dst_ppr) else (dst_ppr, src_ppr)
        for key, weight in smaller.items():
            if key in larger:
                sim += weight * larger[key]

        # Account for direct mutual affinity as well. Immediately after seeing a
        # single edge (u, v), the two sparse PPR dictionaries often point to
        # each other but do not yet share overlapping support, so a pure
        # dot-product stays at zero.
        direct_affinity = 0.5 * (src_ppr.get(dst, 0.0) + dst_ppr.get(src, 0.0))
        return sim + direct_affinity

    def _update_single(self, s1: int, s2: int) -> Dict[int, float]:
        """Compute updated PPR for s1 after observing edge to s2.

        Uses PRE-UPDATE PPR of both s1 and s2 (important: call before writing).

        Args:
            s1: Node whose PPR is being updated.
            s2: The neighbor node.

        Returns:
            New PPR dictionary for s1.
        """
        alpha = self.alpha
        beta = self.beta

        if self.norms[s1] == 0:
            new_ppr: Dict[int, float] = {}
            scale_s2 = 1 - alpha
        else:
            new_ppr = dict(self.ppr[s1])
            last_norm = self.norms[s1]
            new_norm = last_norm * beta + beta
            scale_s1 = last_norm / new_norm * beta
            scale_s2 = beta / new_norm * (1 - alpha)
            for key in new_ppr:
                new_ppr[key] *= scale_s1

        if self.norms[s2] == 0:
            restart = scale_s2 * alpha if alpha != 0 else scale_s2
            new_ppr[s2] = new_ppr.get(s2, 0.0) + restart
        else:
            s2_ppr = self.ppr[s2]
            for key, value in s2_ppr.items():
                new_ppr[key] = new_ppr.get(key, 0.0) + value * scale_s2
            restart = scale_s2 * alpha if alpha != 0 else scale_s2
            new_ppr[s2] = new_ppr.get(s2, 0.0) + restart

        if len(new_ppr) > self.topk:
            items = sorted(new_ppr.items(), key=lambda x: x[1], reverse=True)
            new_ppr = dict(items[:self.topk])

        return new_ppr

    def update_edge(self, src: int, dst: int) -> None:
        """Update PPR dictionaries after observing edge (src, dst).

        Both src and dst PPRs are updated symmetrically.
        Updates are computed from pre-update state (no ordering dependency).

        Args:
            src: Source node.
            dst: Destination node.
        """
        new_src_ppr = self._update_single(src, dst)
        if src != dst:
            new_dst_ppr = self._update_single(dst, src)
            self.ppr[dst] = new_dst_ppr
            self.norms[dst] = self.norms[dst] * self.beta + self.beta
        self.ppr[src] = new_src_ppr
        self.norms[src] = self.norms[src] * self.beta + self.beta

    def process_edges(self, src_arr: np.ndarray, dst_arr: np.ndarray,
                      desc: str = "TPPR building") -> None:
        """Process directed edges in order, building PPR.

        Args:
            src_arr: Source node array.
            dst_arr: Destination node array.
            desc: Progress bar description.
        """
        for src, dst in tqdm(
            zip(src_arr, dst_arr), total=len(src_arr), desc=desc
        ):
            self.update_edge(int(src), int(dst))

    def evaluate_edges(
        self,
        data: TemporalEdgeData,
        eval_indices: np.ndarray,
        neg_csr: TemporalCSR,
        n_hist_neg: int = 50,
        n_random_neg: int = 50,
        seed: int = 42,
        update_after_score: bool = True,
    ) -> Dict[str, float]:
        """Evaluate edges with TGB-style ranking protocol.

        For each evaluation edge:
            1. Generate negatives
            2. Score positive + negatives using current TPPR
            3. Compute rank
            4. Optionally update PPR with true edge

        Args:
            data: Temporal edge data.
            eval_indices: Indices of edges to evaluate.
            neg_csr: CSR for generating historical negatives.
            n_hist_neg: Historical negatives per query.
            n_random_neg: Random negatives per query.
            seed: Random seed.
            update_after_score: Update PPR after scoring each edge.

        Returns:
            Dict with MRR, Hits@K metrics.
        """
        rng = np.random.default_rng(seed)
        all_ranks = []

        for idx in tqdm(eval_indices, desc="TPPR evaluating"):
            src = int(data.src[idx])
            true_dst = int(data.dst[idx])
            ts = data.timestamps[idx]

            neg_nodes = generate_negatives_for_eval(
                src, true_dst, ts, neg_csr, data.num_nodes,
                n_hist=n_hist_neg, n_random=n_random_neg, rng=rng,
            )
            all_dst = np.concatenate([[true_dst], neg_nodes]).astype(np.int32)

            scores = np.array([
                self.get_similarity(src, int(d)) for d in all_dst
            ])

            noise = rng.uniform(0, 1e-10, size=len(scores))
            scores[scores == 0.0] += noise[scores == 0.0]

            true_score = scores[0]
            rank = (
                1
                + (scores[1:] > true_score).sum()
                + 0.5 * (scores[1:] == true_score).sum()
            )
            all_ranks.append(float(rank))

            if update_after_score:
                self.update_edge(src, true_dst)

        ranks_arr = np.array(all_ranks, dtype=np.float64)
        return compute_ranking_metrics(ranks_arr)


def run_tppr_experiment(args):
    """Run TPPR structure scoring experiment on a stream graph."""
    total_start = time.time()

    parquet_name = Path(args.parquet_path).stem
    exp_name = f"tppr_{parquet_name}_k{args.topk}_a{args.alpha}_b{args.beta}"
    output_dir = os.path.join(args.output, exp_name)
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("TPPR Experiment: %s", exp_name)
    logger.info("=" * 60)

    logger.info("Loading stream graph: %s", args.parquet_path)
    data, train_mask, val_mask, test_mask = load_stream_graph_data(
        args.parquet_path,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        undirected=True,
    )
    logger.info("Data: %s", data)

    forward_mask = get_forward_edge_mask(data)
    logger.info(
        "Forward edges: %d / %d total",
        forward_mask.sum(), data.num_edges,
    )

    tppr = TPPR(data.num_nodes, topk=args.topk, alpha=args.alpha, beta=args.beta)

    train_forward = np.where(train_mask & forward_mask)[0]
    logger.info("Building TPPR from %d training edges...", len(train_forward))
    build_start = time.time()
    tppr.process_edges(
        data.src[train_forward], data.dst[train_forward], desc="TPPR train"
    )
    build_time = time.time() - build_start
    logger.info("TPPR built in %.1f sec", build_time)

    train_csr = build_temporal_csr(data, train_mask)

    val_forward = np.where(val_mask & forward_mask)[0]
    logger.info("Evaluating on %d val edges...", len(val_forward))
    val_metrics = tppr.evaluate_edges(
        data, val_forward, train_csr,
        n_hist_neg=50, n_random_neg=50, seed=42,
    )
    logger.info("Val MRR=%.4f Hits@1=%.3f Hits@10=%.3f",
                val_metrics["mrr"], val_metrics["hits@1"], val_metrics["hits@10"])

    full_csr = build_temporal_csr(data, train_mask | val_mask)

    test_forward = np.where(test_mask & forward_mask)[0]
    logger.info("Evaluating on %d test edges...", len(test_forward))
    test_metrics = tppr.evaluate_edges(
        data, test_forward, full_csr,
        n_hist_neg=50, n_random_neg=50, seed=42,
    )

    total_time = time.time() - total_start

    logger.info("=" * 60)
    logger.info("TPPR RESULTS: %s", exp_name)
    logger.info("  Test MRR:     %.4f", test_metrics["mrr"])
    logger.info("  Test Hits@1:  %.4f", test_metrics["hits@1"])
    logger.info("  Test Hits@3:  %.4f", test_metrics["hits@3"])
    logger.info("  Test Hits@10: %.4f", test_metrics["hits@10"])
    logger.info("  Build time:   %.1f sec", build_time)
    logger.info("  Total time:   %.1f min", total_time / 60)
    logger.info("=" * 60)

    results = {
        "experiment": exp_name,
        "model": "TPPR",
        "topk": args.topk,
        "alpha": args.alpha,
        "beta": args.beta,
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "build_time_sec": build_time,
        "total_time_sec": total_time,
    }
    with open(os.path.join(output_dir, "tppr_results.json"), "w") as f:
        json.dump(results, f, indent=2)


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    parser = argparse.ArgumentParser(
        description="EAGLE TPPR structure scoring on stream graphs"
    )
    parser.add_argument(
        "--parquet-path",
        type=str,
        required=True,
        help="Path to stream graph parquet file",
    )
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--topk", type=int, default=100)
    parser.add_argument("--alpha", type=float, default=0.9)
    parser.add_argument("--beta", type=float, default=0.8)
    parser.add_argument("--output", type=str, default="/tmp/eagle_tppr")

    args = parser.parse_args()
    run_tppr_experiment(args)


if __name__ == "__main__":
    main()

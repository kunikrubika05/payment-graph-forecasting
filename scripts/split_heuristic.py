"""Run heuristic baselines for a subset of evaluation days.

Usage:
    PYTHONPATH=. python scripts/split_heuristic.py \
        <period_start> <period_end> <window_size> \
        <day_from> <day_to> <output_file>

Example (3-way split of 85 eval days):
    PYTHONPATH=. python scripts/split_heuristic.py 2015-07-01 2015-09-30 7 0 28 /tmp/heur_a.json
    PYTHONPATH=. python scripts/split_heuristic.py 2015-07-01 2015-09-30 7 28 57 /tmp/heur_b.json
    PYTHONPATH=. python scripts/split_heuristic.py 2015-07-01 2015-09-30 7 57 85 /tmp/heur_c.json
"""

import gc
import json
import logging
import os
import sys
import time

import numpy as np

from src.baselines.heuristic_baselines import (
    _build_adjacency,
    _evaluate_all_heuristics_ranking,
)
from src.baselines.data_loader import (
    get_available_dates,
    download_period_data,
    load_daily_snapshot,
)
from src.baselines.evaluation import compute_ranking_metrics

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def main():
    """Run heuristic evaluation for a day range."""
    if len(sys.argv) != 7:
        print(__doc__)
        sys.exit(1)

    period_start = sys.argv[1]
    period_end = sys.argv[2]
    window_size = int(sys.argv[3])
    day_from = int(sys.argv[4])
    day_to = int(sys.argv[5])
    output_file = sys.argv[6]

    token = os.environ.get("YADISK_TOKEN")
    if not token:
        logger.error("YADISK_TOKEN not set")
        sys.exit(1)

    n_negatives = 100
    seed_base = 42
    local_data_dir = "/tmp/baseline_data"

    all_dates = get_available_dates(period_start, period_end)
    prediction_dates = all_dates[window_size:]
    day_to = min(day_to, len(prediction_dates))

    logger.info(
        "Processing days %d-%d of %d (dates %s to %s)",
        day_from, day_to, len(prediction_dates),
        prediction_dates[day_from], prediction_dates[day_to - 1],
    )

    download_period_data(
        all_dates, local_data_dir, token,
        need_node_features=False, need_snapshots=True,
    )

    results = []
    t_total = time.time()
    n_days = day_to - day_from

    for i, day_i in enumerate(range(day_from, day_to)):
        target_date = prediction_dates[day_i]
        t_day = time.time()

        target_idx = all_dates.index(target_date)
        window_start = max(0, target_idx - window_size)
        window_dates = all_dates[window_start:target_idx]

        window_snapshots = {}
        for d in window_dates:
            snap = load_daily_snapshot(d, local_data_dir)
            if snap is not None:
                window_snapshots[d] = snap

        if not window_snapshots:
            continue

        adj, nodes = _build_adjacency(window_snapshots)
        if len(nodes) < 2:
            continue
        node_to_idx = {n: idx for idx, n in enumerate(nodes)}

        target_snap = load_daily_snapshot(target_date, local_data_dir)
        if target_snap is None or len(target_snap) == 0:
            continue

        target_edges = set(
            zip(target_snap["src_idx"].values, target_snap["dst_idx"].values)
        )

        historical_neighbors = {}
        for snap in window_snapshots.values():
            if snap is not None and len(snap) > 0:
                for s, d in zip(snap["src_idx"].values, snap["dst_idx"].values):
                    historical_neighbors.setdefault(s, set()).add(d)

        seed = seed_base + hash(target_date) % 10000
        ranks_by_heuristic = _evaluate_all_heuristics_ranking(
            adj, node_to_idx, nodes, target_edges,
            historical_neighbors, nodes, n_negatives, seed,
        )

        day_results = {}
        for hname, ranks in ranks_by_heuristic.items():
            if len(ranks) == 0:
                continue
            metrics = compute_ranking_metrics(ranks)
            metrics["date"] = target_date
            metrics["heuristic"] = hname
            results.append(metrics)
            day_results[hname] = f"MRR={metrics.get('mrr', 0):.4f}"

        del adj, window_snapshots
        gc.collect()

        elapsed = time.time() - t_day
        elapsed_total = time.time() - t_total
        avg = elapsed_total / (i + 1)
        eta = avg * (n_days - i - 1)

        logger.info(
            "  [%d/%d] %s (%.1fs): %s | ETA: %.0fm",
            i + 1, n_days, target_date, elapsed, day_results, eta / 60,
        )

    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)

    logger.info("Saved %d records to %s", len(results), output_file)


if __name__ == "__main__":
    main()

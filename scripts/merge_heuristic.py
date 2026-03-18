"""Merge split heuristic results into experiment format and upload to YaDisk.

Usage:
    PYTHONPATH=. python scripts/merge_heuristic.py \
        <output_dir> <yadisk_remote_dir> <part1.json> <part2.json> [part3.json ...]

Example:
    PYTHONPATH=. python scripts/merge_heuristic.py \
        /tmp/baseline_results/exp_003_heuristic_baselines/period_mid_2015q3_w7_mean_modeA \
        orbitaal_processed/experiments/exp_003_heuristic_baselines/period_mid_2015q3_w7_mean_modeA \
        /tmp/heur_a.json /tmp/heur_b.json /tmp/heur_c.json
"""

import json
import logging
import os
import sys

import numpy as np

from src.baselines.experiment_logger import ExperimentLogger

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    """Merge partial heuristic results and write summary."""
    if len(sys.argv) < 4:
        print(__doc__)
        sys.exit(1)

    output_dir = sys.argv[1]
    yadisk_remote_dir = sys.argv[2]
    part_files = sys.argv[3:]

    all_results = []
    for pf in part_files:
        with open(pf) as f:
            data = json.load(f)
            all_results.extend(data)
            logger.info("Loaded %d records from %s", len(data), pf)

    logger.info("Total: %d records", len(all_results))

    os.makedirs(output_dir, exist_ok=True)

    metrics_file = os.path.join(output_dir, "metrics.jsonl")
    with open(metrics_file, "w") as f:
        for r in all_results:
            f.write(json.dumps(r) + "\n")

    heuristic_metrics = {}
    for r in all_results:
        hname = r["heuristic"]
        heuristic_metrics.setdefault(hname, []).append(r)

    summary = {"heuristics": {}}
    for hname, metrics_list in heuristic_metrics.items():
        numeric_keys = [
            k for k in metrics_list[0]
            if isinstance(metrics_list[0][k], (int, float))
        ]
        h_summary = {}
        for key in numeric_keys:
            values = [
                m[key] for m in metrics_list
                if key in m and not np.isnan(m.get(key, float("nan")))
            ]
            if values:
                h_summary[f"mean_{key}"] = float(np.mean(values))
                h_summary[f"std_{key}"] = float(np.std(values))
        h_summary["n_days"] = len(metrics_list)
        summary["heuristics"][hname] = h_summary

    summary_file = os.path.join(output_dir, "summary.json")
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)

    logger.info("Summary written to %s", summary_file)

    for hname, hs in summary["heuristics"].items():
        mrr = hs.get("mean_mrr", 0)
        logger.info("  %s: mean_MRR=%.4f (n_days=%d)", hname, mrr, hs["n_days"])

    token = os.environ.get("YADISK_TOKEN")
    if token and yadisk_remote_dir:
        exp_logger = ExperimentLogger(output_dir)
        exp_logger.upload_to_yadisk(yadisk_remote_dir, token)
        exp_logger.close()
        logger.info("Uploaded to YaDisk: %s", yadisk_remote_dir)
    else:
        logger.warning("YADISK_TOKEN not set, skipping upload")


if __name__ == "__main__":
    main()

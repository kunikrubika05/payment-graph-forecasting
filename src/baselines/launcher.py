"""Launcher: generates experiment configs, distributes across tmux sessions."""

import argparse
import json
import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, List

from src.baselines.config import PERIODS, WINDOW_SIZES, ExperimentConfig

logger = logging.getLogger(__name__)

QUEUE_DIR = "/tmp/baseline_queues"
PROJECT_DIR = os.path.expanduser("~/payment-graph-forecasting")


def generate_link_prediction_configs() -> List[dict]:
    """Generate link prediction experiment configs.

    Returns:
        List of config dictionaries covering the full experiment matrix.
    """
    configs = []

    base_periods = [
        "early_2012q1", "mid_2015q3", "growth_2017q1",
        "peak_2018q2", "mature_2020q2",
    ]
    extra_periods = [
        "early_2013q1", "mid_2014q3", "growth_2016q3",
        "post_peak_2019q1", "late_2020q4",
    ]

    for period in base_periods:
        configs.append(ExperimentConfig(
            experiment_name="exp_001_link_pred_baselines",
            task="link_prediction",
            period_name=period,
            window_size=7,
            aggregation="mean",
            negative_strategy="random",
            mode="A",
            models=["logreg", "catboost", "rf"],
            feature_mode="extended",
            negative_ratio=5,
        ).to_dict())

    for period in base_periods:
        configs.append(ExperimentConfig(
            experiment_name="exp_001_link_pred_baselines",
            task="link_prediction",
            period_name=period,
            window_size=7,
            aggregation="mean",
            negative_strategy="historical",
            mode="A",
            models=["logreg", "catboost", "rf"],
            feature_mode="extended",
            negative_ratio=5,
        ).to_dict())

    for period in base_periods:
        configs.append(ExperimentConfig(
            experiment_name="exp_001_link_pred_baselines",
            task="link_prediction",
            period_name=period,
            window_size=7,
            aggregation="time_weighted",
            decay_lambda=0.3,
            negative_strategy="random",
            mode="A",
            models=["logreg", "catboost", "rf"],
            feature_mode="extended",
            negative_ratio=5,
        ).to_dict())

    for period in ["mid_2015q3", "mature_2020q2"]:
        for w in [3, 14, 30]:
            configs.append(ExperimentConfig(
                experiment_name="exp_001_link_pred_baselines",
                task="link_prediction",
                period_name=period,
                window_size=w,
                aggregation="mean",
                negative_strategy="random",
                mode="A",
                models=["logreg", "catboost", "rf"],
                feature_mode="extended",
                negative_ratio=5,
            ).to_dict())

    for period in ["mid_2015q3", "peak_2018q2", "mature_2020q2"]:
        configs.append(ExperimentConfig(
            experiment_name="exp_001_link_pred_baselines",
            task="link_prediction",
            period_name=period,
            window_size=7,
            aggregation="mean",
            negative_strategy="random",
            mode="B",
            models=["logreg", "catboost", "rf"],
            feature_mode="extended",
            negative_ratio=5,
        ).to_dict())

    for period in ["mid_2015q3", "mature_2020q2"]:
        configs.append(ExperimentConfig(
            experiment_name="exp_001_link_pred_baselines",
            task="link_prediction",
            period_name=period,
            window_size=7,
            aggregation="mean",
            negative_strategy="both",
            mode="A",
            models=["logreg", "catboost", "rf"],
            feature_mode="base",
            negative_ratio=5,
        ).to_dict())

    for period in extra_periods:
        configs.append(ExperimentConfig(
            experiment_name="exp_001_link_pred_baselines",
            task="link_prediction",
            period_name=period,
            window_size=7,
            aggregation="mean",
            negative_strategy="random",
            mode="A",
            models=["logreg", "catboost", "rf"],
            feature_mode="extended",
            negative_ratio=5,
        ).to_dict())

    return configs


def generate_graph_forecasting_configs() -> List[dict]:
    """Generate graph-level forecasting experiment configs."""
    configs = []

    periods = [
        ("full_2012_2020", "2012-01-01", "2020-12-31"),
        ("mid_2015_2018", "2015-01-01", "2018-12-31"),
        ("recent_2018_2020", "2018-01-01", "2020-12-31"),
    ]

    for name, start, end in periods:
        configs.append(ExperimentConfig(
            experiment_name="exp_002_graph_level_baselines",
            sub_experiment=f"period_{name}",
            task="graph_forecasting",
            period_name=name,
            period_start=start,
            period_end=end,
            train_ratio=0.6,
            val_ratio=0.2,
        ).to_dict())

    return configs


def generate_heuristic_configs() -> List[dict]:
    """Generate heuristic baseline experiment configs."""
    configs = []

    for period in ["early_2012q1", "mid_2015q3", "growth_2017q1", "peak_2018q2", "mature_2020q2"]:
        for w in [3, 7]:
            configs.append(ExperimentConfig(
                experiment_name="exp_003_heuristic_baselines",
                task="heuristic",
                period_name=period,
                window_size=w,
                negative_ratio=5,
            ).to_dict())

    return configs


def estimate_difficulty(config: dict) -> float:
    """Estimate relative runtime for load balancing.

    Args:
        config: Config dictionary.

    Returns:
        Estimated difficulty score (higher = longer runtime).
    """
    period = config.get("period_name", "")

    period_weights = {
        "early_2012q1": 1.0, "early_2013q1": 2.0,
        "mid_2014q3": 5.0, "mid_2015q3": 10.0,
        "growth_2016q3": 15.0, "growth_2017q1": 20.0,
        "peak_2018q2": 30.0, "post_peak_2019q1": 25.0,
        "mature_2020q2": 35.0, "late_2020q4": 35.0,
    }
    base = period_weights.get(period, 10.0)

    if config.get("task") == "graph_forecasting":
        return 2.0
    if config.get("task") == "heuristic":
        return base * 0.5
    if config.get("mode") == "B":
        base *= 1.5
    if config.get("window_size", 7) > 14:
        base *= 1.3

    return base


def distribute_configs(
    configs: List[dict], n_sessions: int
) -> List[List[dict]]:
    """Distribute configs across sessions, balancing estimated load.

    Args:
        configs: List of config dictionaries.
        n_sessions: Number of tmux sessions.

    Returns:
        List of config lists, one per session.
    """
    scored = [(estimate_difficulty(c), c) for c in configs]
    scored.sort(key=lambda x: -x[0])

    queues = [[] for _ in range(n_sessions)]
    loads = [0.0] * n_sessions

    for difficulty, config in scored:
        min_idx = loads.index(min(loads))
        queues[min_idx].append(config)
        loads[min_idx] += difficulty

    for i, (queue, load) in enumerate(zip(queues, loads)):
        logger.info("Session %d: %d experiments, estimated load=%.1f", i, len(queue), load)

    return queues


def launch(n_sessions: int = 8, dry_run: bool = False) -> None:
    """Generate configs, create queues, launch tmux sessions.

    Args:
        n_sessions: Number of parallel tmux sessions.
        dry_run: If True, generate configs and show plan without launching.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    lp_configs = generate_link_prediction_configs()
    gf_configs = generate_graph_forecasting_configs()
    heur_configs = generate_heuristic_configs()

    all_configs = gf_configs + heur_configs + lp_configs

    logger.info(
        "Total experiments: %d (link_pred=%d, graph_forecast=%d, heuristic=%d)",
        len(all_configs), len(lp_configs), len(gf_configs), len(heur_configs),
    )

    queues = distribute_configs(all_configs, n_sessions)

    Path(QUEUE_DIR).mkdir(parents=True, exist_ok=True)
    for i, queue in enumerate(queues):
        queue_path = os.path.join(QUEUE_DIR, f"queue_{i}.json")
        with open(queue_path, "w") as f:
            json.dump(queue, f, indent=2, ensure_ascii=False)

    if dry_run:
        logger.info("DRY RUN — configs written to %s, no sessions launched", QUEUE_DIR)
        for i, queue in enumerate(queues):
            logger.info("  Session %d: %d experiments", i, len(queue))
            for c in queue:
                logger.info(
                    "    %s / %s (task=%s)",
                    c.get("experiment_name", "?"),
                    c.get("sub_experiment", "?"),
                    c.get("task", "?"),
                )
        return

    for i in range(n_sessions):
        if not queues[i]:
            continue

        session_name = f"baseline_{i}"
        queue_path = os.path.join(QUEUE_DIR, f"queue_{i}.json")
        log_path = f"/tmp/baseline_logs/session_{i}.log"

        cmd = (
            f"cd {PROJECT_DIR} && "
            f"PYTHONPATH={PROJECT_DIR} "
            f"python -u src/baselines/runner.py "
            f"--queue {queue_path} "
            f"--session {session_name} "
            f"2>&1 | tee -a {log_path}"
        )

        subprocess.run(
            ["tmux", "kill-session", "-t", session_name],
            capture_output=True,
        )

        subprocess.run(
            ["tmux", "new-session", "-d", "-s", session_name, cmd],
            check=True,
        )
        logger.info("Launched tmux session '%s' with %d experiments", session_name, len(queues[i]))

    logger.info("=" * 60)
    logger.info("All %d sessions launched. Monitor with:", n_sessions)
    logger.info("  tmux ls")
    logger.info("  tmux attach -t baseline_0")
    logger.info("  tail -f /tmp/baseline_logs/session_0.log")
    logger.info("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Launch baseline experiments")
    parser.add_argument("--sessions", type=int, default=8, help="Number of tmux sessions")
    parser.add_argument("--dry-run", action="store_true", help="Show plan without launching")
    args = parser.parse_args()
    launch(args.sessions, args.dry_run)


if __name__ == "__main__":
    main()

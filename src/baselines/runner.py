"""Experiment runner: processes a queue of experiment configs sequentially."""

import argparse
import json
import logging
import os
import sys
import time
import traceback
from pathlib import Path

from src.baselines.config import ExperimentConfig

logger = logging.getLogger(__name__)


def _setup_logging(session_name: str) -> None:
    """Configure logging to both console and file."""
    log_dir = Path("/tmp/baseline_logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"{session_name}.log"

    root = logging.getLogger()
    root.setLevel(logging.INFO)

    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    file_handler = logging.FileHandler(str(log_file), mode="a")
    file_handler.setFormatter(formatter)
    root.addHandler(file_handler)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    root.addHandler(console_handler)

    logger.info("Logging to %s", log_file)


def _is_completed(config: ExperimentConfig) -> bool:
    """Check if experiment already has summary.json."""
    output_dir = os.path.join(
        config.output_dir or f"/tmp/baseline_results/{config.experiment_name}",
        config.sub_experiment,
    )
    return Path(output_dir).joinpath("summary.json").exists()


def run_single_experiment(config_dict: dict, token: str) -> bool:
    """Run a single experiment from config dict.

    Args:
        config_dict: Experiment configuration dictionary.
        token: Yandex.Disk OAuth token.

    Returns:
        True on success, False on failure.
    """
    config = ExperimentConfig.from_dict(config_dict)

    if _is_completed(config):
        logger.info("SKIP (already completed): %s / %s", config.experiment_name, config.sub_experiment)
        return True

    logger.info(
        "START: %s / %s (task=%s, period=%s, mode=%s)",
        config.experiment_name, config.sub_experiment,
        config.task, config.period_name, config.mode,
    )
    t0 = time.time()

    try:
        if config.task == "link_prediction":
            from src.baselines.link_prediction import run_link_prediction
            run_link_prediction(config, token)
        elif config.task == "graph_forecasting":
            from src.baselines.graph_forecasting import run_graph_forecasting
            run_graph_forecasting(config, token)
        elif config.task == "heuristic":
            from src.baselines.heuristic_baselines import run_heuristic_baselines
            run_heuristic_baselines(config, token)
        else:
            logger.error("Unknown task: %s", config.task)
            return False

        elapsed = time.time() - t0
        logger.info(
            "DONE: %s / %s (%.1f min)",
            config.experiment_name, config.sub_experiment, elapsed / 60,
        )
        return True

    except Exception as e:
        elapsed = time.time() - t0
        logger.error(
            "FAILED: %s / %s after %.1f min: %s\n%s",
            config.experiment_name, config.sub_experiment,
            elapsed / 60, e, traceback.format_exc(),
        )
        error_dir = os.path.join(
            config.output_dir or f"/tmp/baseline_results/{config.experiment_name}",
            config.sub_experiment,
        )
        Path(error_dir).mkdir(parents=True, exist_ok=True)
        with open(os.path.join(error_dir, "error.txt"), "w") as f:
            f.write(f"Error: {e}\n\n{traceback.format_exc()}")
        return False

    finally:
        from src.baselines.data_loader import cleanup_period_data
        local_dir = config.local_data_dir
        if os.path.exists(local_dir):
            cleanup_period_data(local_dir)


def run_from_queue(queue_path: str, session_name: str = "runner") -> None:
    """Process a queue of experiment configs sequentially.

    Args:
        queue_path: Path to JSON file with list of config dicts.
        session_name: Name for logging purposes.
    """
    _setup_logging(session_name)

    token = os.environ.get("YADISK_TOKEN")
    if not token:
        logger.error("YADISK_TOKEN environment variable not set")
        sys.exit(1)

    with open(queue_path) as f:
        configs = json.load(f)

    logger.info("=" * 60)
    logger.info("Session %s: %d experiments in queue", session_name, len(configs))
    logger.info("=" * 60)

    results = {"success": 0, "skipped": 0, "failed": 0}
    session_start = time.time()

    for i, config_dict in enumerate(configs):
        config = ExperimentConfig.from_dict(config_dict)
        if _is_completed(config):
            logger.info("[%d/%d] SKIP: %s", i + 1, len(configs), config.sub_experiment)
            results["skipped"] += 1
            continue

        logger.info(
            "[%d/%d] Running: %s / %s",
            i + 1, len(configs), config.experiment_name, config.sub_experiment,
        )

        config_dict["local_data_dir"] = f"/tmp/baseline_data_{session_name}"

        success = run_single_experiment(config_dict, token)
        if success:
            results["success"] += 1
        else:
            results["failed"] += 1

    total_time = (time.time() - session_start) / 60
    logger.info("=" * 60)
    logger.info(
        "Session %s complete: %d success, %d skipped, %d failed (%.1f min total)",
        session_name, results["success"], results["skipped"], results["failed"], total_time,
    )
    logger.info("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Run baseline experiments from queue")
    parser.add_argument("--queue", required=True, help="Path to queue JSON file")
    parser.add_argument("--session", default="runner", help="Session name for logging")
    args = parser.parse_args()
    run_from_queue(args.queue, args.session)


if __name__ == "__main__":
    main()

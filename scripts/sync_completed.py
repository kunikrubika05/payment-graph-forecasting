"""Download summary.json files from Yandex.Disk for completed experiments.

This ensures runner.py skips already-completed experiments on a fresh machine.
Run this before launcher.py on a new dev machine.
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.yadisk_utils import download_file, list_remote_files


EXPERIMENTS_BASE = "orbitaal_processed/experiments"
LOCAL_BASE = "/tmp/baseline_results"


def sync_completed():
    """Download summary.json for all completed experiments from Yandex.Disk."""
    token = os.environ.get("YADISK_TOKEN")
    if not token:
        print("ERROR: YADISK_TOKEN not set")
        sys.exit(1)

    exp_dirs = [
        "exp_001_link_pred_baselines",
        "exp_002_graph_level_baselines",
        "exp_003_heuristic_baselines",
    ]
    synced = 0
    skipped = 0

    for exp_dir in exp_dirs:
        remote_exp = f"{EXPERIMENTS_BASE}/{exp_dir}"
        sub_names = list_remote_files(remote_exp, token)
        if sub_names is None:
            print(f"  SKIP {exp_dir}: not found or empty on Yandex.Disk")
            continue

        for sub_name in sub_names:
            remote_summary = f"{remote_exp}/{sub_name}/summary.json"
            local_dir = Path(LOCAL_BASE) / exp_dir / sub_name
            local_summary = local_dir / "summary.json"

            if local_summary.exists():
                print(f"  [EXISTS] {exp_dir}/{sub_name}")
                synced += 1
                continue

            local_dir.mkdir(parents=True, exist_ok=True)
            success = download_file(remote_summary, str(local_summary), token)
            if success and local_summary.exists():
                print(f"  [SYNCED] {exp_dir}/{sub_name}")
                synced += 1
            else:
                if local_summary.exists():
                    local_summary.unlink()
                skipped += 1

    print(f"\nSynced: {synced}, Skipped (no summary.json): {skipped}")


if __name__ == "__main__":
    sync_completed()

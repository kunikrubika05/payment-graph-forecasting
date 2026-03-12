"""Data loading utilities: download from Yandex.Disk, load parquet/CSV files."""

import os
import logging
import shutil
from pathlib import Path
from typing import List, Optional
from datetime import datetime, timedelta

import pandas as pd
from tqdm import tqdm

from src.yadisk_utils import download_file

logger = logging.getLogger(__name__)

GRAPH_FEATURES_LOCAL = "data/processed/graph_features.csv"
YADISK_NODE_FEATURES = "orbitaal_processed/node_features"
YADISK_DAILY_SNAPSHOTS = "orbitaal_processed/daily_snapshots"


def get_available_dates(start_date: str, end_date: str) -> List[str]:
    """Get list of dates in range that have graph data.

    Args:
        start_date: Start date (YYYY-MM-DD), inclusive.
        end_date: End date (YYYY-MM-DD), inclusive.

    Returns:
        Sorted list of date strings that exist in graph_features.csv.
    """
    gf = load_graph_features()
    all_dates = set(gf["date"].dt.strftime("%Y-%m-%d"))

    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")

    dates = []
    current = start
    while current <= end:
        ds = current.strftime("%Y-%m-%d")
        if ds in all_dates and ds != "2021-01-25":
            dates.append(ds)
        current += timedelta(days=1)
    return sorted(dates)


def download_period_data(
    dates: List[str],
    local_dir: str,
    token: str,
    need_node_features: bool = True,
    need_snapshots: bool = True,
) -> None:
    """Download node_features and/or daily_snapshots for given dates.

    Args:
        dates: List of date strings (YYYY-MM-DD).
        local_dir: Local directory to store downloaded files.
        token: Yandex.Disk OAuth token.
        need_node_features: Whether to download node features.
        need_snapshots: Whether to download daily snapshots.
    """
    if need_node_features:
        nf_dir = Path(local_dir) / "node_features"
        nf_dir.mkdir(parents=True, exist_ok=True)
        to_download = [
            d for d in dates
            if not (nf_dir / f"{d}.parquet").exists()
        ]
        if to_download:
            logger.info("Downloading %d node feature files...", len(to_download))
            for date in tqdm(to_download, desc="node_features"):
                remote = f"{YADISK_NODE_FEATURES}/{date}.parquet"
                local = str(nf_dir / f"{date}.parquet")
                success = download_file(remote, local, token)
                if not success:
                    logger.warning("Node features not available for %s (may be 0-edge day)", date)

    if need_snapshots:
        snap_dir = Path(local_dir) / "daily_snapshots"
        snap_dir.mkdir(parents=True, exist_ok=True)
        to_download = [
            d for d in dates
            if not (snap_dir / f"{d}.parquet").exists()
        ]
        if to_download:
            logger.info("Downloading %d daily snapshot files...", len(to_download))
            for date in tqdm(to_download, desc="daily_snapshots"):
                remote = f"{YADISK_DAILY_SNAPSHOTS}/{date}.parquet"
                local = str(snap_dir / f"{date}.parquet")
                if not download_file(remote, local, token):
                    logger.error("Failed to download snapshot for %s", date)


def load_node_features(date: str, local_dir: str) -> Optional[pd.DataFrame]:
    """Load node features parquet for a given date.

    Args:
        date: Date string (YYYY-MM-DD).
        local_dir: Base directory containing node_features/ subdirectory.

    Returns:
        DataFrame with node_idx as index and 25 feature columns,
        or None if file doesn't exist (0-edge day).
    """
    path = Path(local_dir) / "node_features" / f"{date}.parquet"
    if not path.exists():
        return None
    df = pd.read_parquet(path)
    if "node_idx" in df.columns:
        df = df.set_index("node_idx")
    return df


def load_daily_snapshot(date: str, local_dir: str) -> Optional[pd.DataFrame]:
    """Load daily snapshot parquet for a given date.

    Args:
        date: Date string (YYYY-MM-DD).
        local_dir: Base directory containing daily_snapshots/ subdirectory.

    Returns:
        DataFrame with columns: src_idx, dst_idx, btc, usd.
        Or None if file doesn't exist.
    """
    path = Path(local_dir) / "daily_snapshots" / f"{date}.parquet"
    if not path.exists():
        return None
    return pd.read_parquet(path)


def load_graph_features(path: str = GRAPH_FEATURES_LOCAL) -> pd.DataFrame:
    """Load graph-level features CSV.

    Args:
        path: Path to graph_features.csv.

    Returns:
        DataFrame with date column parsed as datetime.
    """
    return pd.read_csv(path, parse_dates=["date"])


def cleanup_period_data(local_dir: str) -> None:
    """Remove all downloaded data in local_dir.

    Args:
        local_dir: Directory to remove.
    """
    if os.path.exists(local_dir):
        shutil.rmtree(local_dir)
        logger.info("Cleaned up %s", local_dir)

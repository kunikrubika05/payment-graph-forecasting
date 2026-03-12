"""Experiment logging: save configs, metrics, models, predictions to disk."""

import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class NumpyEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy types."""

    def default(self, obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        return super().default(obj)


class ExperimentLogger:
    """Logger for saving all experiment artifacts to disk."""

    def __init__(self, output_dir: str):
        """Initialize logger and create output directory structure.

        Args:
            output_dir: Root directory for this experiment's artifacts.
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "model").mkdir(exist_ok=True)
        (self.output_dir / "predictions").mkdir(exist_ok=True)

        self._metrics_path = self.output_dir / "metrics.jsonl"
        self._metrics_file = open(self._metrics_path, "a")
        self._start_time = datetime.now()

    def close(self):
        """Close open file handles."""
        if self._metrics_file and not self._metrics_file.closed:
            self._metrics_file.close()

    def log_config(self, config: dict) -> None:
        """Write config.json.

        Args:
            config: Configuration dictionary.
        """
        config_with_meta = {
            **config,
            "started_at": self._start_time.isoformat(),
        }
        path = self.output_dir / "config.json"
        with open(path, "w") as f:
            json.dump(config_with_meta, f, indent=2, ensure_ascii=False, cls=NumpyEncoder)
        logger.info("Config saved to %s", path)

    def log_metrics(self, metrics: dict) -> None:
        """Append one metrics entry to metrics.jsonl. Flushes immediately.

        Args:
            metrics: Dictionary of metric values for one evaluation step.
        """
        metrics_with_ts = {
            "timestamp": datetime.now().isoformat(),
            **metrics,
        }
        line = json.dumps(metrics_with_ts, ensure_ascii=False, cls=NumpyEncoder)
        self._metrics_file.write(line + "\n")
        self._metrics_file.flush()

    def log_hp_search(self, results: List[dict]) -> None:
        """Write hyperparameter search results.

        Args:
            results: List of dicts, each with params and val metrics.
        """
        path = self.output_dir / "hp_search_results.json"
        with open(path, "w") as f:
            json.dump(results, f, indent=2, ensure_ascii=False, cls=NumpyEncoder)
        logger.info("HP search results saved to %s", path)

    def log_feature_importance(self, importance: dict) -> None:
        """Write feature importance.

        Args:
            importance: Dict mapping feature names to importance values.
        """
        path = self.output_dir / "feature_importance.json"
        with open(path, "w") as f:
            json.dump(importance, f, indent=2, ensure_ascii=False, cls=NumpyEncoder)

    def log_feature_correlations(self, corr_df: pd.DataFrame) -> None:
        """Write feature correlation matrix.

        Args:
            corr_df: Correlation matrix as DataFrame.
        """
        path = self.output_dir / "feature_correlations.csv"
        corr_df.to_csv(path)

    def log_high_correlations(self, corr_df: pd.DataFrame, threshold: float = 0.95) -> None:
        """Log pairs of features with correlation above threshold.

        Args:
            corr_df: Correlation matrix as DataFrame.
            threshold: Absolute correlation threshold.
        """
        pairs = []
        cols = corr_df.columns
        for i in range(len(cols)):
            for j in range(i + 1, len(cols)):
                val = abs(corr_df.iloc[i, j])
                if val >= threshold:
                    pairs.append({
                        "feature_a": cols[i],
                        "feature_b": cols[j],
                        "correlation": float(corr_df.iloc[i, j]),
                    })
        path = self.output_dir / "high_correlations.json"
        with open(path, "w") as f:
            json.dump(pairs, f, indent=2)
        if pairs:
            logger.info("Found %d highly correlated feature pairs (|r| >= %.2f)", len(pairs), threshold)

    def save_model(self, model: Any, name: str) -> None:
        """Save model to model/ directory.

        Args:
            model: Trained model object.
            name: Model filename (e.g., 'best_logreg.pkl', 'best_catboost.cbm').
        """
        path = self.output_dir / "model" / name
        if name.endswith(".cbm"):
            model.save_model(str(path))
        else:
            import joblib
            joblib.dump(model, str(path))
        logger.info("Model saved to %s", path)

    def save_predictions(self, predictions_df: pd.DataFrame, date: str) -> None:
        """Save predictions to predictions/<date>.parquet.

        Args:
            predictions_df: DataFrame with columns: src, dst, true_label, pred_proba.
            date: Date string for the filename.
        """
        path = self.output_dir / "predictions" / f"{date}.parquet"
        predictions_df.to_parquet(path, index=False)

    def write_summary(self, summary: dict) -> None:
        """Write summary.json with aggregated metrics.

        Args:
            summary: Dictionary with summary statistics.
        """
        summary_with_meta = {
            **summary,
            "started_at": self._start_time.isoformat(),
            "completed_at": datetime.now().isoformat(),
            "duration_seconds": (datetime.now() - self._start_time).total_seconds(),
        }
        path = self.output_dir / "summary.json"
        with open(path, "w") as f:
            json.dump(summary_with_meta, f, indent=2, ensure_ascii=False, cls=NumpyEncoder)
        logger.info("Summary saved to %s", path)

    def is_completed(self) -> bool:
        """Check if experiment already has summary.json (completed)."""
        return (self.output_dir / "summary.json").exists()

    def upload_to_yadisk(self, remote_dir: str, token: str) -> None:
        """Upload entire output_dir to Yandex.Disk.

        Args:
            remote_dir: Remote directory path on Yandex.Disk.
            token: Yandex.Disk OAuth token.
        """
        from src.yadisk_utils import upload_directory
        count = upload_directory(str(self.output_dir), remote_dir, token)
        logger.info("Uploaded %d files to %s", count, remote_dir)

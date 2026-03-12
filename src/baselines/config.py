"""Experiment configuration definitions."""

from dataclasses import dataclass, field, asdict
from typing import List, Optional
import json


PERIODS = {
    "early_2012q1": {"start": "2012-01-01", "end": "2012-03-31"},
    "early_2013q1": {"start": "2013-01-01", "end": "2013-03-31"},
    "mid_2014q3": {"start": "2014-07-01", "end": "2014-09-30"},
    "mid_2015q3": {"start": "2015-07-01", "end": "2015-09-30"},
    "growth_2016q3": {"start": "2016-07-01", "end": "2016-09-30"},
    "growth_2017q1": {"start": "2017-01-01", "end": "2017-03-31"},
    "peak_2018q2": {"start": "2018-06-01", "end": "2018-08-31"},
    "post_peak_2019q1": {"start": "2019-01-01", "end": "2019-03-31"},
    "mature_2020q2": {"start": "2020-06-01", "end": "2020-08-31"},
    "late_2020q4": {"start": "2020-10-01", "end": "2020-12-31"},
}

WINDOW_SIZES = [3, 7, 14, 30]

NODE_FEATURE_COLUMNS = [
    "in_degree", "out_degree", "total_degree",
    "weighted_in_btc", "weighted_out_btc",
    "weighted_in_usd", "weighted_out_usd",
    "balance_btc", "balance_usd",
    "avg_in_btc", "avg_out_btc",
    "median_in_btc", "median_out_btc",
    "max_in_btc", "max_out_btc",
    "min_in_btc", "min_out_btc",
    "std_in_btc", "std_out_btc",
    "unique_in_counterparties", "unique_out_counterparties",
    "pagerank", "clustering_coeff", "k_core", "triangle_count",
]

GRAPH_FORECAST_TARGETS = ["num_nodes", "num_edges", "total_btc", "total_usd"]

LOGREG_HP_GRID = {
    "C": [0.01, 0.1, 1.0, 10.0, 100.0],
    "penalty": ["l1", "l2"],
}

CATBOOST_HP_GRID = {
    "iterations": [100, 300, 500],
    "depth": [4, 6, 8],
    "learning_rate": [0.03, 0.05, 0.1],
}

RF_HP_GRID = {
    "n_estimators": [100, 200, 300],
    "max_depth": [None, 10, 20],
    "min_samples_leaf": [1, 5, 10],
}

K_VALUES = [100, 500, 1000, 5000, 10000]


@dataclass
class ExperimentConfig:
    """Configuration for a single baseline experiment."""

    experiment_name: str = "exp_001_link_pred_baselines"
    sub_experiment: str = ""
    task: str = "link_prediction"
    period_name: str = "mid_2015q3"
    period_start: str = ""
    period_end: str = ""
    window_size: int = 7
    aggregation: str = "mean"
    decay_lambda: float = 0.3
    negative_ratio: int = 5
    negative_strategy: str = "random"
    mode: str = "A"
    models: List[str] = field(default_factory=lambda: ["logreg", "catboost", "rf"])
    feature_mode: str = "extended"
    max_train_samples: int = 5_000_000
    train_ratio: float = 0.6
    val_ratio: float = 0.2
    random_seed: int = 42
    target_variables: List[str] = field(
        default_factory=lambda: ["num_nodes", "num_edges", "total_btc", "total_usd"]
    )
    local_data_dir: str = "/tmp/baseline_data"
    output_dir: str = ""
    yadisk_experiments_base: str = "orbitaal_processed/experiments"

    def __post_init__(self):
        if not self.period_start and self.period_name in PERIODS:
            self.period_start = PERIODS[self.period_name]["start"]
            self.period_end = PERIODS[self.period_name]["end"]
        if not self.sub_experiment:
            parts = [
                f"period_{self.period_name}",
                f"w{self.window_size}",
                self.aggregation,
                f"{self.negative_strategy}neg",
                f"mode{self.mode}",
            ]
            self.sub_experiment = "_".join(parts)

    def to_dict(self) -> dict:
        """Serialize config to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "ExperimentConfig":
        """Deserialize config from dictionary."""
        valid_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in d.items() if k in valid_fields}
        return cls(**filtered)

    def save(self, path: str) -> None:
        """Save config to JSON file."""
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)

    @classmethod
    def load(cls, path: str) -> "ExperimentConfig":
        """Load config from JSON file."""
        with open(path) as f:
            return cls.from_dict(json.load(f))

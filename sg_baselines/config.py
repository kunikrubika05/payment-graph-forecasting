"""Configuration for stream graph baseline experiments.

Split protocol (NO data leakage):
- Period = first {fraction} of stream graph edges (chronological)
- Train = first 70% of period (features + adjacency computed here)
- Val = next 15% of period (HP search by ranking MRR)
- Test = last 15% of period (final evaluation, NEVER used for HP search)

Evaluation protocol (TGB-style):
- Per-source ranking: for each positive (src, dst_true), build candidate set
  {dst_true} ∪ {negatives}, score all, rank dst_true → MRR, Hits@K
- Negatives: 50% historical (train neighbors of src, absent in eval set) + 50% random
- n_negatives = 100 per positive edge (eval), negative_ratio = 5 (train)
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any


PERIODS = {
    "period_10": {"fraction": 0.10, "label": "10"},
    "period_25": {"fraction": 0.25, "label": "25"},
}

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15

LOGREG_HP_GRID = {
    "C": [0.01, 0.1, 1.0, 10.0],
    "penalty": ["l1", "l2"],
}

CATBOOST_HP_GRID = {
    "iterations": [200, 500],
    "depth": [4, 6, 8],
    "learning_rate": [0.05, 0.1],
}

RF_HP_GRID = {
    "n_estimators": [100, 300],
    "max_depth": [10, 20, 30],
    "min_samples_leaf": [1, 5],
}

HP_GRIDS: Dict[str, Dict[str, list]] = {
    "logreg": LOGREG_HP_GRID,
    "catboost": CATBOOST_HP_GRID,
    "rf": RF_HP_GRID,
}

HEURISTICS = ["cn", "jaccard", "aa", "pa"]

ML_MODELS = ["logreg", "catboost", "rf"]

YADISK_STREAM_GRAPH = "orbitaal_processed/stream_graph/2020-06-01__2020-08-31.parquet"
YADISK_STREAM_DIR = "orbitaal_processed/stream_graph"
YADISK_EXPERIMENTS_BASE = "orbitaal_processed/experiments"


@dataclass
class ExperimentConfig:
    """Configuration for a single baseline experiment."""

    period_name: str
    fraction: float
    label: str
    train_ratio: float = TRAIN_RATIO
    val_ratio: float = VAL_RATIO
    n_negatives: int = 100
    negative_ratio: int = 5
    max_train_samples: int = 2_000_000
    hp_search_max_samples: int = 500_000
    random_seed: int = 42
    models: List[str] = field(default_factory=lambda: list(ML_MODELS))
    heuristics: List[str] = field(default_factory=lambda: list(HEURISTICS))
    local_data_dir: str = "/tmp/sg_baselines_data"
    output_dir: str = "/tmp/sg_baselines_results"
    upload: bool = False

    @property
    def test_ratio(self) -> float:
        return 1.0 - self.train_ratio - self.val_ratio

    def to_dict(self) -> Dict[str, Any]:
        return {
            "period_name": self.period_name,
            "fraction": self.fraction,
            "label": self.label,
            "train_ratio": self.train_ratio,
            "val_ratio": self.val_ratio,
            "test_ratio": self.test_ratio,
            "n_negatives": self.n_negatives,
            "negative_ratio": self.negative_ratio,
            "max_train_samples": self.max_train_samples,
            "hp_search_max_samples": self.hp_search_max_samples,
            "random_seed": self.random_seed,
            "models": self.models,
            "heuristics": self.heuristics,
        }


def get_experiment_configs() -> List[ExperimentConfig]:
    """Generate experiment configs for all periods."""
    configs = []
    for name, params in PERIODS.items():
        configs.append(ExperimentConfig(
            period_name=name,
            fraction=params["fraction"],
            label=params["label"],
        ))
    return configs

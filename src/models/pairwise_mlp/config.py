"""Configuration for PairwiseMLP experiments.

Split protocol is IDENTICAL to sg_baselines:
  period = first {fraction} edges of stream graph (chronological)
  train  = first 70% of period edges
  val    = next  15% of period edges
  test   = last  15% of period edges

Eval seeds are IDENTICAL to sg_baselines/run.py:
  val  seed = random_seed + 10  (= 52 with default seed=42)
  test seed = random_seed + 20  (= 62 with default seed=42)

CLI feature selection examples (all resolved via resolve_feature_indices()):
  --features cn_uu                     → [0]
  --features cn_uu log1p_cn_uu aa_uu   → [0, 1, 2]
  --feature-indices 0 1 2              → [0, 1, 2]
  (omit both)                          → all 7 features
"""

import os
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional


# ---------------------------------------------------------------------------
# Yandex.Disk paths (same as sg_baselines)
# ---------------------------------------------------------------------------
YADISK_STREAM_GRAPH     = "orbitaal_processed/stream_graph/2020-06-01__2020-08-31.parquet"
YADISK_STREAM_DIR       = "orbitaal_processed/stream_graph"
YADISK_EXPERIMENTS_BASE = "orbitaal_processed/experiments"

TRAIN_RATIO = 0.70
VAL_RATIO   = 0.15

PERIODS = {
    "period_10": {"fraction": 0.10, "label": "10"},
    "period_25": {"fraction": 0.25, "label": "25"},
}

# ---------------------------------------------------------------------------
# 7 pairwise features — index here = column index in precomputed .npy files
# ---------------------------------------------------------------------------
FEATURE_NAMES: List[str] = [
    "cn_uu",          # 0: Common Neighbors, undirected (integer count ≥ 0)
    "log1p_cn_uu",    # 1: log1p(cn_uu)
    "aa_uu",          # 2: Adamic-Adar, undirected (float ≥ 0)
    "cn_dir",         # 3: CN directed: common out-neighbors of src and dst
    "aa_dir",         # 4: AA directed
    "jaccard_uu",     # 5: Jaccard undirected ∈ [0, 1]
    "log1p_pa_uu",    # 6: log1p(deg_src_uu × deg_dst_uu)
]
N_FEATURES = len(FEATURE_NAMES)  # 7

# Predefined ablation presets (convenient shortcuts for --features argument)
FEATURE_PRESETS: Dict[str, List[str]] = {
    "E1": ["cn_uu"],
    "E2": ["cn_uu", "log1p_cn_uu", "aa_uu"],
    "E3": FEATURE_NAMES,      # all 7
    "all": FEATURE_NAMES,
}

VALID_LOSSES = ("bpr", "bce")


def resolve_feature_indices(
    feature_names: Optional[List[str]] = None,
    feature_indices: Optional[List[int]] = None,
) -> List[int]:
    """Resolve which feature columns to use.

    Priority:
      1. feature_indices (explicit 0-based column numbers) if provided.
      2. feature_names (list of strings from FEATURE_NAMES) if provided.
      3. If neither: all N_FEATURES columns.

    Also accepts preset keys from FEATURE_PRESETS (e.g. "E1", "E2", "E3").

    Args:
        feature_names:   List of feature name strings or a single preset key.
        feature_indices: List of int column indices.

    Returns:
        Sorted list of unique column indices in [0, N_FEATURES).

    Raises:
        ValueError: on unknown feature names or out-of-range indices.
    """
    if feature_indices:
        for i in feature_indices:
            if not (0 <= i < N_FEATURES):
                raise ValueError(
                    f"Feature index {i} out of range [0, {N_FEATURES}). "
                    f"Available: {list(range(N_FEATURES))}"
                )
        return sorted(set(feature_indices))

    if feature_names:
        # Check if it's a single preset key
        if len(feature_names) == 1 and feature_names[0] in FEATURE_PRESETS:
            feature_names = FEATURE_PRESETS[feature_names[0]]
        indices = []
        for name in feature_names:
            if name not in FEATURE_NAMES:
                raise ValueError(
                    f"Unknown feature '{name}'. "
                    f"Valid names: {FEATURE_NAMES}\n"
                    f"Valid presets: {list(FEATURE_PRESETS)}"
                )
            indices.append(FEATURE_NAMES.index(name))
        return sorted(set(indices))

    return list(range(N_FEATURES))  # default: all 7


@dataclass
class PairMLPConfig:
    """Full configuration for one PairMLP experiment.

    All parameters are expressible via CLI args in run.py so that
    new experiments require no code changes — only different arguments.
    """

    # --- Data ---
    period_name: str   = "period_10"
    fraction:    float = 0.10
    label:       str   = "10"
    train_ratio: float = TRAIN_RATIO
    val_ratio:   float = VAL_RATIO
    random_seed: int   = 42

    # --- Precompute (CPU) ---
    k_neg_train:      int = 20        # negatives per positive for BPR training
    k_hist_max:       int = 5         # max historical negatives (rest = random)
    n_jobs:           int = -1        # joblib workers (-1 = all cores)
    precompute_batch: int = 50_000    # pairs per scipy batch

    # --- Feature selection (resolved by resolve_feature_indices()) ---
    # Empty list = use all N_FEATURES. Populated by resolve_feature_indices().
    active_feature_indices: List[int] = field(default_factory=list)

    # --- Loss function ---
    loss: str = "bpr"  # "bpr" or "bce"

    # --- Training (GPU) ---
    hidden_dims:   List[int] = field(default_factory=lambda: [64, 32])
    dropout:       float     = 0.0   # dropout rate after each hidden ReLU (0 = off)
    lr:            float     = 1e-3
    weight_decay:  float     = 1e-4
    batch_size:    int       = 4096
    n_epochs:      int       = 50
    patience:      int       = 10
    grad_clip:     float     = 1.0
    eval_every:    int       = 2     # epochs between val evaluations
    k_neg_sample:  int       = 0     # subsample K' negatives per step (0 = use all K)

    # --- LR Scheduler ---
    scheduler:          str   = ""     # "", "cosine", "plateau"
    scheduler_min_lr:   float = 1e-6   # cosine eta_min / plateau min_lr
    scheduler_patience: int   = 3      # plateau: epochs without improvement before decay
    scheduler_factor:   float = 0.5    # plateau: lr multiplier on decay

    # --- Node features ---
    use_node_features: bool = False    # concat 15-dim src + 15-dim dst to pair features

    # --- Eval (identical to sg_baselines) ---
    n_negatives:    int = 100
    max_eval_queries: int = 50_000

    # --- Experiment tag (for output directory naming) ---
    exp_tag: str = ""   # e.g. "E1", "E2_bce" — appended to output dir name

    # --- Paths ---
    local_data_dir:       str  = "/tmp/pairmlp_data"
    local_precompute_dir: str  = "/tmp/pairmlp_precompute"
    local_output_dir:     str  = "/tmp/pairmlp_results"
    upload:               bool = False

    # ---------------------------------------------------------------------------
    # Derived properties
    # ---------------------------------------------------------------------------

    @property
    def test_ratio(self) -> float:
        return 1.0 - self.train_ratio - self.val_ratio

    @property
    def val_seed(self) -> int:
        return self.random_seed + 10

    @property
    def test_seed(self) -> int:
        return self.random_seed + 20

    @property
    def n_input_features(self) -> int:
        """Number of features actually fed to the MLP.

        = n_pair_features + (30 if use_node_features else 0)
        where n_pair_features is the number of selected pairwise structural features.
        """
        idx = self.active_feature_indices
        n_pair = len(idx) if idx else N_FEATURES
        return n_pair + (30 if self.use_node_features else 0)

    @property
    def selected_feature_names(self) -> List[str]:
        """Names of the features actually fed to the MLP."""
        idx = self.active_feature_indices
        if not idx:
            return FEATURE_NAMES
        return [FEATURE_NAMES[i] for i in idx]

    @property
    def exp_name(self) -> str:
        """Unique experiment directory name."""
        base = f"exp_pairmlp_{self.label}"
        if self.exp_tag:
            base = f"{base}_{self.exp_tag}"
        return base

    @property
    def precompute_artifact_dir(self) -> str:
        """Local dir containing precomputed .npy files (from precompute.py)."""
        return os.path.join(
            self.local_precompute_dir, f"pairmlp_precompute_{self.label}"
        )

    @property
    def yadisk_precompute_dir(self) -> str:
        return f"{YADISK_EXPERIMENTS_BASE}/pairmlp_precompute_{self.label}"

    @property
    def yadisk_results_dir(self) -> str:
        return f"{YADISK_EXPERIMENTS_BASE}/{self.exp_name}"

    # ---------------------------------------------------------------------------
    # Serialisation
    # ---------------------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        return {
            "period_name":            self.period_name,
            "fraction":               self.fraction,
            "label":                  self.label,
            "train_ratio":            self.train_ratio,
            "val_ratio":              self.val_ratio,
            "test_ratio":             self.test_ratio,
            "random_seed":            self.random_seed,
            "k_neg_train":            self.k_neg_train,
            "k_hist_max":             self.k_hist_max,
            "loss":                   self.loss,
            "active_feature_indices": self.active_feature_indices,
            "selected_feature_names": self.selected_feature_names,
            "n_input_features":       self.n_input_features,
            "hidden_dims":            self.hidden_dims,
            "dropout":                self.dropout,
            "lr":                     self.lr,
            "weight_decay":           self.weight_decay,
            "batch_size":             self.batch_size,
            "n_epochs":               self.n_epochs,
            "patience":               self.patience,
            "n_negatives":            self.n_negatives,
            "max_eval_queries":       self.max_eval_queries,
            "scheduler":              self.scheduler,
            "scheduler_min_lr":       self.scheduler_min_lr,
            "scheduler_patience":     self.scheduler_patience,
            "scheduler_factor":       self.scheduler_factor,
            "use_node_features":      self.use_node_features,
            "exp_tag":                self.exp_tag,
            "exp_name":               self.exp_name,
        }

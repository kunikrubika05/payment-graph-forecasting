"""Unified evaluation sampling strategies."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


VALID_SAMPLING_STRATEGIES = {"tgb_mixed"}


@dataclass(slots=True)
class NegativeSamplingStrategy:
    """Canonical representation of val/test negative sampling."""

    name: str = "tgb_mixed"
    n_random_neg: int = 50
    n_hist_neg: int = 50

    def __post_init__(self) -> None:
        if self.name not in VALID_SAMPLING_STRATEGIES:
            raise ValueError(
                f"Unknown sampling strategy '{self.name}'. "
                f"Known strategies: {sorted(VALID_SAMPLING_STRATEGIES)}"
            )
        if self.n_random_neg < 0 or self.n_hist_neg < 0:
            raise ValueError("Negative sampling counts must be non-negative")

    @property
    def total_negatives(self) -> int:
        return self.n_random_neg + self.n_hist_neg

    def as_mixed_kwargs(self) -> dict[str, int]:
        """Return explicit historical/random kwargs for evaluators."""

        return {
            "n_hist_neg": self.n_hist_neg,
            "n_random_neg": self.n_random_neg,
        }

    def as_total_kwargs(self) -> dict[str, int]:
        """Return total-negative kwargs for evaluators exposing only one count."""

        return {"n_negatives": self.total_negatives}


def sampling_strategy_from_config(config: Any) -> NegativeSamplingStrategy:
    """Build a canonical strategy from a sampling config-like object."""

    return NegativeSamplingStrategy(
        name=getattr(config, "strategy", "tgb_mixed"),
        n_random_neg=int(getattr(config, "n_random_neg", 50)),
        n_hist_neg=int(getattr(config, "n_hist_neg", 50)),
    )


DEFAULT_TGB_MIXED = NegativeSamplingStrategy()

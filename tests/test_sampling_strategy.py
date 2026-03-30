import pytest

from payment_graph_forecasting.sampling.strategy import (
    NegativeSamplingStrategy,
    sampling_strategy_from_config,
)


class _Cfg:
    strategy = "tgb_mixed"
    n_random_neg = 12
    n_hist_neg = 8


def test_sampling_strategy_total_and_kwargs():
    strategy = NegativeSamplingStrategy(n_random_neg=12, n_hist_neg=8)

    assert strategy.total_negatives == 20
    assert strategy.as_mixed_kwargs() == {"n_hist_neg": 8, "n_random_neg": 12}
    assert strategy.as_total_kwargs() == {"n_negatives": 20}


def test_sampling_strategy_from_config():
    strategy = sampling_strategy_from_config(_Cfg())

    assert strategy.name == "tgb_mixed"
    assert strategy.n_random_neg == 12
    assert strategy.n_hist_neg == 8


def test_sampling_strategy_rejects_invalid_input():
    with pytest.raises(ValueError):
        NegativeSamplingStrategy(name="unknown")

    with pytest.raises(ValueError):
        NegativeSamplingStrategy(n_random_neg=-1)

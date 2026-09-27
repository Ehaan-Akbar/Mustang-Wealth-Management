import numpy as np
import pandas as pd
import pytest

from black_litterman import BlackLittermanModel, MeanVarianceOptimizer, View


@pytest.fixture
def small_cov():
    tickers = ["A", "B", "C"]
    # Simple positive-definite covariance matrix.
    corr = np.array([[1.0, 0.3, 0.1], [0.3, 1.0, 0.2], [0.1, 0.2, 1.0]])
    vols = np.array([0.20, 0.25, 0.15])
    cov = np.outer(vols, vols) * corr
    return pd.DataFrame(cov, index=tickers, columns=tickers)


@pytest.fixture
def market_weights():
    return pd.Series({"A": 0.5, "B": 0.3, "C": 0.2})


def test_prior_matches_reverse_optimization(small_cov, market_weights):
    delta = 3.0
    model = BlackLittermanModel(small_cov, market_weights, risk_aversion=delta, tau=0.05)
    expected_pi = delta * small_cov.values @ market_weights.values
    np.testing.assert_allclose(model.pi.values, expected_pi)


def test_no_views_posterior_equals_prior(small_cov, market_weights):
    model = BlackLittermanModel(small_cov, market_weights)
    posterior_returns, posterior_cov = model.posterior()
    np.testing.assert_allclose(posterior_returns.values, model.pi.values)
    np.testing.assert_allclose(posterior_cov.values, small_cov.values)


def test_absolute_view_shifts_return_towards_view(small_cov, market_weights):
    model = BlackLittermanModel(small_cov, market_weights, risk_aversion=3.0, tau=0.05)
    prior_a = model.pi["A"]
    # A high-confidence view well above the prior should pull the posterior up.
    model.add_absolute_view("A", expected_return=prior_a + 0.20, confidence=0.9)
    posterior_returns, _ = model.posterior()
    assert posterior_returns["A"] > prior_a


def test_relative_view_direction(small_cov, market_weights):
    model = BlackLittermanModel(small_cov, market_weights)
    model.add_relative_view("B", "C", expected_return=0.05, confidence=0.8)
    posterior_returns, _ = model.posterior()
    # B should end up expected to outperform C given the view.
    assert posterior_returns["B"] - posterior_returns["C"] > (model.pi["B"] - model.pi["C"])


def test_invalid_view_asset_raises(small_cov, market_weights):
    model = BlackLittermanModel(small_cov, market_weights)
    with pytest.raises(ValueError):
        model.add_absolute_view("Z", expected_return=0.05)


def test_view_confidence_bounds():
    with pytest.raises(ValueError):
        View(assets={"A": 1.0}, expected_return=0.05, confidence=0)
    with pytest.raises(ValueError):
        View(assets={"A": 1.0}, expected_return=0.05, confidence=1.5)


def test_optimizer_weights_sum_to_one(small_cov, market_weights):
    model = BlackLittermanModel(small_cov, market_weights)
    posterior_returns, posterior_cov = model.posterior()
    opt = MeanVarianceOptimizer(posterior_returns, posterior_cov)

    for weights in (opt.max_sharpe(), opt.min_volatility(), opt.target_return(0.05)):
        assert pytest.approx(weights.sum(), abs=1e-6) == 1.0
        assert (weights >= -1e-6).all()  # long-only by default


def test_min_volatility_has_lowest_variance(small_cov, market_weights):
    model = BlackLittermanModel(small_cov, market_weights)
    posterior_returns, posterior_cov = model.posterior()
    opt = MeanVarianceOptimizer(posterior_returns, posterior_cov)

    min_vol_w = opt.min_volatility()
    max_sharpe_w = opt.max_sharpe()

    min_vol_variance = min_vol_w.values @ posterior_cov.values @ min_vol_w.values
    max_sharpe_variance = max_sharpe_w.values @ posterior_cov.values @ max_sharpe_w.values

    assert min_vol_variance <= max_sharpe_variance + 1e-8

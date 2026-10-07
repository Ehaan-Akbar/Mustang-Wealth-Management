import numpy as np
import pandas as pd
import pytest

from black_litterman.portfolio_lab import fit_model, prepare_returns, simulate_projection


def history():
    return pd.DataFrame(np.random.default_rng(7).normal(0.0003, 0.01, (800, 2)),
                        columns=["SPY", "BND"], index=pd.bdate_range("2020-01-01", periods=800))


def test_price_gaps_are_not_filled_and_universe_order_is_preserved():
    prices = 100 * (1 + history()).cumprod()
    prices.iloc[10, 0] = np.nan
    result = prepare_returns(prices, ("BND", "SPY"), prices=True)
    assert list(result.columns) == ["BND", "SPY"]
    assert prices.index[10] not in result.index
    assert prices.index[11] not in result.index
    assert len(result) == len(prices) - 3


def test_data_requires_all_requested_assets_and_valid_dates():
    with pytest.raises(ValueError, match="No data for: AAPL"):
        prepare_returns(history(), ("SPY", "AAPL"), prices=False)
    invalid = history()
    invalid.index = ["bad"] * len(invalid)
    with pytest.raises(ValueError, match="valid dates"):
        prepare_returns(invalid, ("SPY", "BND"), prices=False)


def test_absolute_views_are_total_returns_and_position_caps_apply():
    result = fit_model(history(), pd.Series({"SPY": 50, "BND": 50}),
                       [{"Asset": "SPY", "Return (%)": 12, "Confidence (%)": 100}], [],
                       2.5, 0.05, 0.04, 0.6, "Maximum Sharpe", 0.05)
    assert result["returns"]["SPY"] == pytest.approx(0.12, abs=1e-4)
    assert result["weights"].max() <= 0.6 + 1e-8
    assert result["weights"].sum() == pytest.approx(1)


@pytest.mark.parametrize("method", ["Monte Carlo", "Block bootstrap"])
def test_cash_flow_timing_and_failure_flags(method):
    # Opposite asset returns cancel under daily rebalancing. Blocks must retain
    # both moves together; independently resampling assets would add false risk.
    values = history()
    values["BND"] = -values["SPY"]
    result = simulate_projection(values, pd.Series({"SPY": 0.5, "BND": 0.5}),
                                 method=method, n_sims=7, years=4, initial_value=100,
                                 annual_contribution=10, annual_withdrawal=60, withdrawal_start=2)
    np.testing.assert_allclose(result.values, np.tile([100, 110, 60, 10, 0], (7, 1)))
    assert not result.funded.any()
    assert result.summary(50)["Target reached"] == 0


def test_monte_carlo_model_assumptions_and_reproducibility():
    args = dict(returns=history(), weights=pd.Series({"SPY": 0.5, "BND": 0.5}),
                method="Monte Carlo", n_sims=100, years=2, annual_mean=0.1, annual_volatility=0.0)
    result = simulate_projection(**args)
    np.testing.assert_allclose(result.values.iloc[:, -1], 100_000 * np.exp(0.2))
    args["annual_volatility"] = 0.2
    pd.testing.assert_frame_equal(simulate_projection(**args).values, simulate_projection(**args).values)


def test_block_bootstrap_seed_and_block_validation():
    args = dict(returns=history(), weights=pd.Series({"SPY": 0.5, "BND": 0.5}),
                method="Block bootstrap", n_sims=25, years=2)
    pd.testing.assert_frame_equal(simulate_projection(**args).values, simulate_projection(**args).values)
    assert not simulate_projection(**args, seed=12).values.equals(simulate_projection(**args).values)
    with pytest.raises(ValueError, match="Block size"):
        simulate_projection(**args, block_size=900)


@pytest.mark.parametrize("method", ["Monte Carlo", "Block bootstrap"])
def test_daily_paths_are_the_actual_paths_used_for_summary_and_progress(method):
    updates = []
    result = simulate_projection(history(), pd.Series({"SPY": 0.5, "BND": 0.5}),
                                 method=method, n_sims=450, years=2, initial_value=100,
                                 annual_contribution=10, annual_withdrawal=5,
                                 retained_paths=120, progress=lambda done, total: updates.append((done, total)))
    assert updates == [(200, 450), (400, 450), (450, 450)]
    assert result.daily_paths.shape == (120, 505)
    np.testing.assert_allclose(result.daily_paths.iloc[:, ::252], result.values.iloc[:120])
    assert result.daily_paths.iloc[:, -1].nunique() == 120
    assert np.isfinite(result.daily_paths).all().all()
    assert result.summary(100)["Target reached"] == (result.values.iloc[:, -1] >= 100).mean()


def test_chart_only_uses_real_retained_path_values():
    pytest.importorskip("streamlit")
    from dashboard_simulations import path_chart_frame

    result = simulate_projection(history(), pd.Series({"SPY": 0.5, "BND": 0.5}),
                                 method="Monte Carlo", n_sims=100, years=3)
    plotted = path_chart_frame(result, count=3, highlighted=50)
    assert set(plotted["Path"]) == {1, 2, 3, 50}
    for path_id, group in plotted.groupby("Path"):
        np.testing.assert_array_equal(group["Balance"], result.daily_paths.loc[path_id, group["Year"]])

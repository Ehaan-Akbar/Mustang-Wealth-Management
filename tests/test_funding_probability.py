"""Funding inputs must remain isolated between dashboard sessions."""

import numpy as np
import pytest

from black_litterman import funding_probability as funding


def test_custom_liabilities_drive_payments_and_yearly_results():
    schedule = {2033: 100.0, 2042: 250.0}
    assumptions = dict(
        initial_portfolio=300.0, second_year_contribution=0.0,
        equity_return=0.0, bond_return=0.0, cash_return=0.0,
        equity_volatility=0.0, bond_volatility=0.0, cash_volatility=0.0,
        liabilities=schedule,
    )
    baseline = funding.LIABILITIES.copy()
    success, path = funding.simulate_portfolio_path(**assumptions, rng=np.random.default_rng(42))
    assert not success
    assert path["Payment Made"].sum() == 300.0
    assert path["Shortfall"].sum() == 50.0
    yearly = funding.calculate_yearly_funding_probability(n_simulations=2, **assumptions)
    assert yearly["Year"].tolist() == [2033, 2042]
    assert yearly["Liability"].tolist() == [100.0, 250.0]
    assert yearly["Funding Probability"].tolist() == [1.0, 0.0]
    assert funding.LIABILITIES == baseline
    assert schedule == {2033: 100.0, 2042: 250.0}


@pytest.mark.parametrize("method", [funding.run_sensitivity_analysis, funding.run_two_variable_sensitivity])
def test_sensitivity_never_changes_shared_liabilities_even_on_failure(monkeypatch, method):
    baseline = funding.LIABILITIES.copy()

    def fail(**kwargs):
        assert funding.LIABILITIES == baseline
        assert kwargs["liabilities"][2033] == 40_000.0
        raise RuntimeError("Interrupted calculation")

    monkeypatch.setattr(funding, "calculate_funding_probability", fail)
    with pytest.raises(RuntimeError, match="Interrupted calculation"):
        method(n_simulations=1)
    assert funding.LIABILITIES == baseline


@pytest.mark.parametrize("schedule", [{2033: -1}, {2033: float("nan")}, {2050: 100}])
def test_invalid_liabilities_are_rejected(schedule):
    with pytest.raises(ValueError):
        funding.simulate_portfolio_path(liabilities=schedule)

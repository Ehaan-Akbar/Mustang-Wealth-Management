"""
Historical stress testing for the Mustang Wealth Management project.

Replays historical crisis periods using actual asset returns and a given
portfolio allocation.

The primary scenario is the 2008 Global Financial Crisis.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
import pandas as pd


TRADING_DAYS_PER_YEAR = 252


SCENARIOS = {
    "2008 Financial Crisis": (
        "2008-01-01",
        "2008-12-31",
    ),
    "COVID Crash": (
        "2020-02-19",
        "2020-03-23",
    ),
    "2022 Bear Market": (
        "2022-01-03",
        "2022-10-12",
    ),
}


@dataclass
class StressTestResult:
    """Results from a historical stress test."""

    scenario: str
    start_date: str
    end_date: str
    portfolio_return: float
    annualized_volatility: float
    maximum_drawdown: float
    worst_day: float
    ending_value: float
    asset_contributions: pd.Series
    daily_returns: pd.Series
    cumulative_value: pd.Series


def validate_inputs(
    returns: pd.DataFrame,
    weights: pd.Series,
) -> tuple[pd.DataFrame, pd.Series]:

    if not isinstance(returns, pd.DataFrame):
        raise TypeError("returns must be a pandas DataFrame.")

    if returns.empty:
        raise ValueError("returns cannot be empty.")

    if not isinstance(weights, pd.Series):
        weights = pd.Series(weights, index=returns.columns)

    weights = weights.reindex(returns.columns)

    if weights.isna().any():
        missing = weights[weights.isna()].index.tolist()
        raise ValueError(
            f"weights are missing assets: {missing}"
        )

    if not np.isclose(weights.sum(), 1.0, atol=1e-6):
        raise ValueError(
            f"weights must sum to 1. Current sum = {weights.sum():.6f}"
        )

    clean_returns = (
        returns
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
        .sort_index()
    )

    return clean_returns, weights.astype(float)


def calculate_max_drawdown(values: pd.Series) -> float:
    """Calculate the maximum peak-to-trough decline."""

    running_peak = values.cummax()
    drawdowns = values / running_peak - 1.0

    return float(drawdowns.min())


def run_stress_test(
    returns: pd.DataFrame,
    weights: pd.Series,
    scenario: str = "2008 Financial Crisis",
    initial_value: float = 100_000.0,
) -> StressTestResult:
    """
    Replay a historical stress scenario.

    Parameters
    ----------
    returns:
        Historical daily SIMPLE returns.

    weights:
        Portfolio weights from the Black-Litterman optimizer.

    scenario:
        Name of a scenario in SCENARIOS.

    initial_value:
        Starting portfolio value used to calculate the ending value.
    """

    if scenario not in SCENARIOS:
        raise ValueError(
            f"Unknown scenario '{scenario}'. "
            f"Available scenarios: {list(SCENARIOS)}"
        )

    returns, weights = validate_inputs(returns, weights)

    start_date, end_date = SCENARIOS[scenario]

    scenario_returns = returns.loc[
        start_date:end_date
    ].copy()

    if scenario_returns.empty:
        raise ValueError(
            f"No return data found for {scenario} "
            f"({start_date} to {end_date})."
        )

    # Portfolio daily return = weighted sum of asset returns.
    portfolio_returns = scenario_returns.dot(weights)

    # Build portfolio value path.
    cumulative_value = (
        initial_value
        * (1.0 + portfolio_returns).cumprod()
    )

    total_return = (
        cumulative_value.iloc[-1] / initial_value
        - 1.0
    )

    annualized_volatility = (
        portfolio_returns.std()
        * np.sqrt(TRADING_DAYS_PER_YEAR)
    )

    maximum_drawdown = calculate_max_drawdown(
        cumulative_value
    )

    worst_day = float(
        portfolio_returns.min()
    )

    # Contribution of each asset to the total scenario return.
    asset_contributions = (
        scenario_returns.sum()
        * weights
    )

    return StressTestResult(
        scenario=scenario,
        start_date=start_date,
        end_date=end_date,
        portfolio_return=float(total_return),
        annualized_volatility=float(annualized_volatility),
        maximum_drawdown=float(maximum_drawdown),
        worst_day=worst_day,
        ending_value=float(cumulative_value.iloc[-1]),
        asset_contributions=asset_contributions.sort_values(),
        daily_returns=portfolio_returns,
        cumulative_value=cumulative_value,
    )


def compare_stress_tests(
    returns: pd.DataFrame,
    portfolios: Dict[str, pd.Series],
    scenarios: list[str] | None = None,
    initial_value: float = 100_000.0,
) -> pd.DataFrame:
    """
    Compare multiple portfolios across historical stress scenarios.
    """

    if scenarios is None:
        scenarios = list(SCENARIOS)

    rows = []

    for portfolio_name, weights in portfolios.items():

        for scenario in scenarios:

            result = run_stress_test(
                returns=returns,
                weights=weights,
                scenario=scenario,
                initial_value=initial_value,
            )

            rows.append(
                {
                    "Portfolio": portfolio_name,
                    "Scenario": scenario,
                    "Total Return": result.portfolio_return,
                    "Annualized Volatility": result.annualized_volatility,
                    "Maximum Drawdown": result.maximum_drawdown,
                    "Worst Day": result.worst_day,
                    "Ending Value": result.ending_value,
                }
            )

    return pd.DataFrame(rows)

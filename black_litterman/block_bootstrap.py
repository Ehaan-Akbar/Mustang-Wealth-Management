
"""
Block-bootstrap simulation

Inputs:
    - historical asset returns from black_litterman.data_utils.compute_returns()
    - a portfolio weight Series from MeanVarianceOptimizer.max_sharpe()
      or MeanVarianceOptimizer.min_volatility()

Recommended:
    Use SIMPLE daily returns for wealth simulation:
        returns = compute_returns(prices, method="simple")

The simulation:
    1. Resamples historical daily return blocks jointly across all assets.
    2. Preserves cross-asset relationships within each sampled block.
    3. Simulates 2027-2042.
    4. Applies Laura's cash flows at the beginning of each year:
           2027: +$300,000
           2028: +$150,000
           2033-2042: -$50,000/year
    5. Reports the probability of funding all required payments.

This is the first-stage fixed-weight simulation. The dynamic glide path
can be added after this baseline is validated.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


START_YEAR = 2027
END_YEAR = 2042
TRADING_DAYS_PER_YEAR = 252


def laura_cash_flows() -> Dict[int, float]:
    """Beginning-of-year cash flows from the case."""
    flows = {
        2027: 300_000.0,
        2028: 150_000.0,
    }

    for year in range(2033, 2043):
        flows[year] = -50_000.0

    return flows


def _validate_returns_and_weights(
    returns: pd.DataFrame,
    weights: pd.Series,
) -> Tuple[pd.DataFrame, pd.Series]:
    """Validate and align historical returns and portfolio weights."""

    if not isinstance(returns, pd.DataFrame) or returns.empty:
        raise ValueError("returns must be a non-empty pandas DataFrame.")

    if not isinstance(weights, pd.Series):
        weights = pd.Series(weights, index=returns.columns)

    returns = returns.copy()
    weights = weights.copy()

    # Keep only numeric return columns.
    returns = returns.select_dtypes(include=[np.number])

    if returns.empty:
        raise ValueError("returns must contain numeric asset-return columns.")

    missing = [ticker for ticker in returns.columns if ticker not in weights.index]
    if missing:
        raise ValueError(f"weights are missing assets: {missing}")

    weights = weights.reindex(returns.columns).astype(float)

    if weights.isna().any():
        raise ValueError("weights contain missing values.")

    if not np.all(np.isfinite(weights.values)):
        raise ValueError("weights contain NaN or infinite values.")

    if not np.isclose(weights.sum(), 1.0, atol=1e-6):
        raise ValueError(
            f"weights must sum to 1. Current sum = {weights.sum():.6f}"
        )

    returns = returns.replace([np.inf, -np.inf], np.nan).dropna()

    if len(returns) < 2 * TRADING_DAYS_PER_YEAR:
        raise ValueError(
            "At least 2 years of historical observations are recommended."
        )

    if not returns.index.is_monotonic_increasing:
        returns = returns.sort_index()

    return returns, weights


def block_bootstrap(
    returns: pd.DataFrame,
    n_sims: int = 10_000,
    n_years: int = 16,
    block_size: int = 21,
    trading_days_per_year: int = TRADING_DAYS_PER_YEAR,
    seed: int = 42,
) -> np.ndarray:
    """
    Generate bootstrapped daily asset returns.

    Uses a moving block bootstrap. Each block contains consecutive historical
    observations, and all assets are sampled together so their historical
    cross-asset relationships are retained.

    block_size=21 is approximately one trading month.
    """
    if n_sims <= 0:
        raise ValueError("n_sims must be positive.")
    if n_years <= 0:
        raise ValueError("n_years must be positive.")
    if block_size <= 0:
        raise ValueError("block_size must be positive.")

    X = returns.to_numpy(dtype=float)
    n_obs, n_assets = X.shape

    if block_size > n_obs:
        raise ValueError("block_size cannot exceed the number of observations.")

    n_days = n_years * trading_days_per_year
    n_blocks = int(np.ceil(n_days / block_size))

    # Circular moving blocks let blocks wrap around the historical sample.
    rng = np.random.default_rng(seed)
    start_indices = rng.integers(
        0, n_obs, size=(n_sims, n_blocks), endpoint=False
    )

    simulated = np.empty(
        (n_sims, n_blocks * block_size, n_assets),
        dtype=np.float32,
    )

    for b in range(n_blocks):
        starts = start_indices[:, b]

        # Build each block with wrap-around.
        block_indices = (
            starts[:, None] + np.arange(block_size)[None, :]
        ) % n_obs

        simulated[
            :,
            b * block_size:(b + 1) * block_size,
            :
        ] = X[block_indices]

    return simulated[:, :n_days, :]


@dataclass
class BootstrapResult:
    funding_probability: float
    failure_probability: float
    simulated_portfolio_returns: np.ndarray
    year_start_values: pd.DataFrame
    success_flags: np.ndarray
    pre_payment_2033: np.ndarray
    post_payment_2033: np.ndarray
    ending_2042: np.ndarray


def run_simulation(
    returns: pd.DataFrame,
    weights: pd.Series,
    n_sims: int = 10_000,
    block_size: int = 21,
    seed: int = 42,
    cash_flows: Optional[Dict[int, float]] = None,
) -> BootstrapResult:
    """
    Run the 2027-2042 block-bootstrap wealth simulation.

    IMPORTANT:
        `returns` should be SIMPLE returns, not log returns.
    """
    returns, weights = _validate_returns_and_weights(returns, weights)

    if cash_flows is None:
        cash_flows = laura_cash_flows()

    years = list(range(START_YEAR, END_YEAR + 1))
    n_years = len(years)

    sampled_asset_returns = block_bootstrap(
        returns=returns,
        n_sims=n_sims,
        n_years=n_years,
        block_size=block_size,
        trading_days_per_year=TRADING_DAYS_PER_YEAR,
        seed=seed,
    )

    # Convert each simulated day of asset returns to portfolio return.
    simulated_portfolio_returns = np.einsum(
        "sda,a->sd",
        sampled_asset_returns,
        weights.values,
    )

    values = np.zeros(n_sims, dtype=float)
    success = np.ones(n_sims, dtype=bool)
    year_start_values = np.empty((n_sims, n_years), dtype=float)

    pre_payment_2033 = None
    post_payment_2033 = None

    for year_index, year in enumerate(years):
        cash_flow = cash_flows.get(year, 0.0)

        # Record the value before the 2033 operating payment.
        if year == 2033:
            pre_payment_2033 = values.copy()

        if cash_flow > 0:
            values += cash_flow

        elif cash_flow < 0:
            required_payment = -cash_flow

            can_fully_pay = values >= required_payment
            success &= can_fully_pay

            # Failed paths cannot continue funding the operating commitment.
            values = np.where(
                can_fully_pay,
                values - required_payment,
                0.0,
            )

        # Record the value after the beginning-of-year cash flow.
        year_start_values[:, year_index] = values

        # Apply that year's market returns.
        day_start = year_index * TRADING_DAYS_PER_YEAR
        day_end = day_start + TRADING_DAYS_PER_YEAR

        annual_daily_returns = simulated_portfolio_returns[
            :, day_start:day_end
        ]

        values *= np.prod(1.0 + annual_daily_returns, axis=1)

        if year == 2033:
            post_payment_2033 = year_start_values[:, year_index].copy()

    year_start_df = pd.DataFrame(
        year_start_values,
        columns=years,
    )

    return BootstrapResult(
        funding_probability=float(success.mean()),
        failure_probability=float(1.0 - success.mean()),
        simulated_portfolio_returns=simulated_portfolio_returns,
        year_start_values=year_start_df,
        success_flags=success,
        pre_payment_2033=pre_payment_2033,
        post_payment_2033=post_payment_2033,
        ending_2042=values.copy(),
    )


def summary(result: BootstrapResult) -> pd.Series:
    """Return key competition metrics."""

    return pd.Series(
        {
            "Funding Probability": result.funding_probability,
            "Failure Probability": result.failure_probability,
            "2033 Pre-Payment 5th Percentile":
                np.percentile(result.pre_payment_2033, 5),
            "2033 Pre-Payment Median":
                np.percentile(result.pre_payment_2033, 50),
            "2033 Pre-Payment 95th Percentile":
                np.percentile(result.pre_payment_2033, 95),
            "2042 Ending 5th Percentile":
                np.percentile(result.ending_2042, 5),
            "2042 Ending Median":
                np.percentile(result.ending_2042, 50),
            "2042 Ending 95th Percentile":
                np.percentile(result.ending_2042, 95),
        }
    )


def compare_portfolios(
    returns: pd.DataFrame,
    portfolios: Dict[str, pd.Series],
    n_sims: int = 10_000,
    block_size: int = 21,
    seed: int = 42,
) -> Tuple[pd.DataFrame, Dict[str, BootstrapResult]]:
    """
    Compare existing Black-Litterman portfolios.

    Example:
        portfolios = {
            "BL Max Sharpe": max_sharpe_weights,
            "BL Min Volatility": min_vol_weights,
        }
    """
    results = {}
    rows = {}

    for i, (name, weights) in enumerate(portfolios.items()):
        result = run_simulation(
            returns=returns,
            weights=weights,
            n_sims=n_sims,
            block_size=block_size,
            seed=seed + i,
        )
        results[name] = result
        rows[name] = summary(result)

    return pd.DataFrame(rows).T, results


def plot_2033_distribution(
    result: BootstrapResult,
    title: str = "Simulated Portfolio Value at Beginning of 2033",
) -> None:
    """Plot the simulated portfolio-value distribution immediately before the first payment."""
    values = result.pre_payment_2033

    plt.figure(figsize=(9, 5))
    plt.hist(values, bins=60)
    plt.axvline(
        np.percentile(values, 5),
        linestyle="--",
        label="5th percentile",
    )
    plt.axvline(
        np.median(values),
        linestyle="--",
        label="Median",
    )
    plt.title(title)
    plt.xlabel("Portfolio Value ($)")
    plt.ylabel("Number of Simulations")
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    print("Block bootstrap module is ready.")
    print(
        "Use simple daily returns from compute_returns(..., method='simple') "
        "and pass in your BL optimizer weights."
    )

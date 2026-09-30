"""
Block-bootstrap simulation for the Mustang Wealth Management project.

This simulation:
    - Uses historical daily SIMPLE returns
    - Resamples consecutive blocks of historical returns
    - Preserves cross-asset relationships within each block
    - Simulates Laura's portfolio from 2027 through 2042
    - Applies the required cash flows:
        2027: +$300,000
        2028: +$150,000
        2033-2042: -$50,000 each year
    - Calculates the probability of successfully funding all required payments

This version processes simulations in batches to reduce memory usage.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd


START_YEAR = 2027
END_YEAR = 2042
TRADING_DAYS_PER_YEAR = 252


def laura_cash_flows() -> Dict[int, float]:
    """
    Beginning-of-year cash flows from Laura's case.
    """

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
    """
    Validate and align historical returns and portfolio weights.
    """

    if not isinstance(returns, pd.DataFrame) or returns.empty:
        raise ValueError(
            "returns must be a non-empty pandas DataFrame."
        )

    returns = returns.select_dtypes(
        include=[np.number]
    ).copy()

    if returns.empty:
        raise ValueError(
            "returns must contain numeric asset-return columns."
        )

    if not isinstance(weights, pd.Series):
        weights = pd.Series(
            weights,
            index=returns.columns
        )
    else:
        weights = weights.copy()

    missing = [
        asset
        for asset in returns.columns
        if asset not in weights.index
    ]

    if missing:
        raise ValueError(
            f"weights are missing assets: {missing}"
        )

    weights = weights.reindex(
        returns.columns
    ).astype(float)

    if weights.isna().any():
        raise ValueError(
            "weights contain missing values."
        )

    if not np.all(
        np.isfinite(weights.values)
    ):
        raise ValueError(
            "weights contain NaN or infinite values."
        )

    if not np.isclose(
        weights.sum(),
        1.0,
        atol=1e-6
    ):
        raise ValueError(
            f"weights must sum to 1. "
            f"Current sum = {weights.sum():.6f}"
        )

    returns = returns.replace(
        [np.inf, -np.inf],
        np.nan
    ).dropna()

    if len(returns) < (
        2 * TRADING_DAYS_PER_YEAR
    ):
        raise ValueError(
            "At least two years of historical observations "
            "are recommended."
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
    batch_size: int = 250,
):
    """
    Generate block-bootstrap daily portfolio return paths in batches.

    Parameters
    ----------
    returns:
        Historical SIMPLE daily asset returns.

    n_sims:
        Number of simulations.

    n_years:
        Number of years to simulate.

    block_size:
        Number of consecutive trading days in each block.
        21 is approximately one trading month.

    trading_days_per_year:
        Number of trading days per year.

    seed:
        Random seed for reproducibility.

    batch_size:
        Number of simulation paths generated at once.
    """

    if n_sims <= 0:
        raise ValueError(
            "n_sims must be positive."
        )

    if n_years <= 0:
        raise ValueError(
            "n_years must be positive."
        )

    if block_size <= 0:
        raise ValueError(
            "block_size must be positive."
        )

    if batch_size <= 0:
        raise ValueError(
            "batch_size must be positive."
        )

    X = returns.to_numpy(
        dtype=np.float32
    )

    n_obs = X.shape[0]

    if block_size > n_obs:
        raise ValueError(
            "block_size cannot exceed the number "
            "of historical observations."
        )

    n_days = (
        n_years *
        trading_days_per_year
    )

    n_blocks = int(
        np.ceil(
            n_days / block_size
        )
    )

    rng = np.random.default_rng(
        seed
    )

    for start in range(
        0,
        n_sims,
        batch_size
    ):

        current_batch = min(
            batch_size,
            n_sims - start
        )

        start_indices = rng.integers(
            0,
            n_obs,
            size=(
                current_batch,
                n_blocks
            ),
        )

        simulated = np.empty(
            (
                current_batch,
                n_blocks * block_size,
                X.shape[1],
            ),
            dtype=np.float32,
        )

        for block_index in range(
            n_blocks
        ):

            indices = (
                start_indices[
                    :,
                    block_index,
                    None
                ]
                +
                np.arange(
                    block_size
                )[None, :]
            ) % n_obs

            simulated[
                :,
                block_index * block_size:
                (block_index + 1) * block_size,
                :
            ] = X[indices]

        yield simulated[
            :,
            :n_days,
            :
        ]


@dataclass
class BootstrapResult:
    """
    Stores the results of the simulation.
    """

    funding_probability: float

    failure_probability: float

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
    batch_size: int = 250,
    cash_flows: Optional[
        Dict[int, float]
    ] = None,
) -> BootstrapResult:
    """
    Run the 2027-2042 block-bootstrap simulation.

    `returns` should contain SIMPLE daily returns.
    """

    returns, weights = (
        _validate_returns_and_weights(
            returns,
            weights
        )
    )

    if cash_flows is None:
        cash_flows = (
            laura_cash_flows()
        )

    years = list(
        range(
            START_YEAR,
            END_YEAR + 1
        )
    )

    n_years = len(years)

    year_start_values = np.empty(
        (
            n_sims,
            n_years
        ),
        dtype=np.float64,
    )

    success_flags = np.ones(
        n_sims,
        dtype=bool
    )

    pre_payment_2033 = np.empty(
        n_sims,
        dtype=np.float64
    )

    post_payment_2033 = np.empty(
        n_sims,
        dtype=np.float64
    )

    ending_2042 = np.empty(
        n_sims,
        dtype=np.float64
    )

    simulation_index = 0

    for simulated_asset_returns in block_bootstrap(
        returns=returns,
        n_sims=n_sims,
        n_years=n_years,
        block_size=block_size,
        trading_days_per_year=TRADING_DAYS_PER_YEAR,
        seed=seed,
        batch_size=batch_size,
    ):

        current_batch = (
            simulated_asset_returns.shape[0]
        )

        # Convert asset returns into portfolio returns.
        portfolio_returns = np.einsum(
            "sda,a->sd",
            simulated_asset_returns,
            weights.values,
        )

        values = np.zeros(
            current_batch,
            dtype=np.float64
        )

        batch_success = np.ones(
            current_batch,
            dtype=bool
        )

        for year_index, year in enumerate(
            years
        ):

            cash_flow = cash_flows.get(
                year,
                0.0
            )

            # Value immediately before the
            # first operating payment in 2033.
            if year == 2033:

                pre_payment_2033[
                    simulation_index:
                    simulation_index +
                    current_batch
                ] = values

            # Beginning-of-year contribution
            if cash_flow > 0:

                values += cash_flow

            # Beginning-of-year withdrawal
            elif cash_flow < 0:

                required_payment = (
                    -cash_flow
                )

                can_fully_pay = (
                    values >= required_payment
                )

                batch_success &= (
                    can_fully_pay
                )

                # If the portfolio cannot fully
                # make the required payment,
                # mark that simulation as failed.
                values = np.where(
                    can_fully_pay,
                    values - required_payment,
                    0.0,
                )

            # Value immediately after
            # the beginning-of-year payment.
            if year == 2033:

                post_payment_2033[
                    simulation_index:
                    simulation_index +
                    current_batch
                ] = values

            year_start_values[
                simulation_index:
                simulation_index +
                current_batch,
                year_index,
            ] = values

            # Apply the simulated market returns
            # during this year.
            day_start = (
                year_index *
                TRADING_DAYS_PER_YEAR
            )

            day_end = (
                day_start +
                TRADING_DAYS_PER_YEAR
            )

            annual_returns = (
                portfolio_returns[
                    :,
                    day_start:day_end
                ]
            )

            values *= np.prod(
                1.0 + annual_returns,
                axis=1
            )

        success_flags[
            simulation_index:
            simulation_index +
            current_batch
        ] = batch_success

        ending_2042[
            simulation_index:
            simulation_index +
            current_batch
        ] = values

        simulation_index += (
            current_batch
        )

    year_start_df = pd.DataFrame(
        year_start_values,
        columns=years
    )

    funding_probability = float(
        success_flags.mean()
    )

    return BootstrapResult(
        funding_probability=(
            funding_probability
        ),

        failure_probability=(
            1.0 -
            funding_probability
        ),

        year_start_values=(
            year_start_df
        ),

        success_flags=(
            success_flags
        ),

        pre_payment_2033=(
            pre_payment_2033
        ),

        post_payment_2033=(
            post_payment_2033
        ),

        ending_2042=(
            ending_2042
        ),
    )


def summary(
    result: BootstrapResult
) -> pd.Series:
    """
    Return key simulation statistics.
    """

    values_2033 = (
        result.pre_payment_2033
    )

    values_2042 = (
        result.ending_2042
    )

    return pd.Series(
        {
            "Funding Probability":
                result.funding_probability,

            "Failure Probability":
                result.failure_probability,

            "2033 Pre-Payment 5th Percentile":
                np.percentile(
                    values_2033,
                    5
                ),

            "2033 Pre-Payment Median":
                np.percentile(
                    values_2033,
                    50
                ),

            "2033 Pre-Payment 95th Percentile":
                np.percentile(
                    values_2033,
                    95
                ),

            "2042 Ending 5th Percentile":
                np.percentile(
                    values_2042,
                    5
                ),

            "2042 Ending Median":
                np.percentile(
                    values_2042,
                    50
                ),

            "2042 Ending 95th Percentile":
                np.percentile(
                    values_2042,
                    95
                ),
        }
    )


def compare_portfolios(
    returns: pd.DataFrame,
    portfolios: Dict[
        str,
        pd.Series
    ],
    n_sims: int = 10_000,
    block_size: int = 21,
    seed: int = 42,
    batch_size: int = 250,
) -> Tuple[
    pd.DataFrame,
    Dict[str, BootstrapResult]
]:
    """
    Compare multiple portfolio allocations.

    Example:

        portfolios = {
            "BL Max Sharpe":
                max_sharpe_weights,

            "BL Min Volatility":
                min_vol_weights,
        }
    """

    results = {}

    rows = {}

    for i, (
        name,
        weights
    ) in enumerate(
        portfolios.items()
    ):

        result = run_simulation(
            returns=returns,
            weights=weights,
            n_sims=n_sims,
            block_size=block_size,
            seed=seed + i,
            batch_size=batch_size,
        )

        results[name] = result

        rows[name] = summary(
            result
        )

    return (
        pd.DataFrame(rows).T,
        results
    )


def plot_2033_distribution(
    result: BootstrapResult,
    title: str =
        "Simulated Portfolio Value at Beginning of 2033",
) -> None:
    """
    Plot the simulated distribution
    of portfolio values before the first
    2033 operating payment.
    """

    import matplotlib.pyplot as plt

    values = (
        result.pre_payment_2033
    )

    plt.figure(
        figsize=(9, 5)
    )

    plt.hist(
        values,
        bins=60
    )

    plt.axvline(
        np.percentile(
            values,
            5
        ),
        linestyle="--",
        label="5th percentile",
    )

    plt.axvline(
        np.median(values),
        linestyle="--",
        label="Median",
    )

    plt.title(title)

    plt.xlabel(
        "Portfolio Value ($)"
    )

    plt.ylabel(
        "Number of Simulations"
    )

    plt.legend()

    plt.tight_layout()

    plt.show()


if __name__ == "__main__":

    print(
        "Block bootstrap module is ready."
    )

    print(
        "Use simple daily returns from "
        "compute_returns(..., method='simple') "
        "and pass in your BL optimizer weights."
    )

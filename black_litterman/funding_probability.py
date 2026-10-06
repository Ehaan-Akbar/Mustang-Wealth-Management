"""
Funding Probability and Sensitivity Analysis.

This module evaluates how likely the portfolio is to fully fund the
$50,000 annual liabilities beginning in 2033 and continuing through 2042.

It also performs sensitivity analysis by changing key assumptions such as:
    - Annual liability
    - Expected equity return
    - Expected bond return
    - Portfolio volatility

The analysis uses Monte Carlo simulation and the existing dynamic glide
path from glide_path.py.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from black_litterman.glide_path import (
    START_YEAR,
    LIABILITY_START_YEAR,
    END_YEAR,
    ANNUAL_LIABILITY,
    calculate_glide_path,
)


# ---------------------------------------------------------------------
# Portfolio assumptions
# ---------------------------------------------------------------------

INITIAL_PORTFOLIO = 300_000.0
SECOND_YEAR_CONTRIBUTION = 150_000.0

DEFAULT_EQUITY_RETURN = 0.08
DEFAULT_BOND_RETURN = 0.04
DEFAULT_CASH_RETURN = 0.02

DEFAULT_EQUITY_VOLATILITY = 0.18
DEFAULT_BOND_VOLATILITY = 0.06
DEFAULT_CASH_VOLATILITY = 0.01

DEFAULT_CORRELATION = 0.20

NUM_SIMULATIONS = 10_000
RANDOM_SEED = 42


# ---------------------------------------------------------------------
# Monte Carlo simulation
# ---------------------------------------------------------------------

def simulate_portfolio_path(
    annual_liability: float = ANNUAL_LIABILITY,
    equity_return: float = DEFAULT_EQUITY_RETURN,
    bond_return: float = DEFAULT_BOND_RETURN,
    cash_return: float = DEFAULT_CASH_RETURN,
    equity_volatility: float = DEFAULT_EQUITY_VOLATILITY,
    bond_volatility: float = DEFAULT_BOND_VOLATILITY,
    cash_volatility: float = DEFAULT_CASH_VOLATILITY,
    correlation: float = DEFAULT_CORRELATION,
    rng: np.random.Generator | None = None,
) -> tuple[bool, pd.DataFrame]:
    """
    Simulate one portfolio path from 2027 through 2042.

    Returns
    -------
    tuple
        success:
            True if every liability can be fully funded.

        path:
            DataFrame containing the portfolio value and liability
            information for every year.
    """

    if annual_liability < 0:
        raise ValueError("annual_liability cannot be negative.")

    if rng is None:
        rng = np.random.default_rng()

    years = range(START_YEAR, END_YEAR + 1)

    # Correlation matrix for asset classes.
    correlation_matrix = np.array(
        [
            [1.0, correlation, correlation],
            [correlation, 1.0, correlation],
            [correlation, correlation, 1.0],
        ]
    )

    volatilities = np.array(
        [
            equity_volatility,
            bond_volatility,
            cash_volatility,
        ]
    )

    covariance_matrix = (
        np.outer(volatilities, volatilities)
        * correlation_matrix
    )

    # Generate correlated annual shocks.
    shocks = rng.multivariate_normal(
        mean=[0.0, 0.0, 0.0],
        cov=covariance_matrix,
        size=len(list(years)),
    )

    portfolio_value = INITIAL_PORTFOLIO

    rows = []
    success = True

    for index, year in enumerate(years):

        # Add the second contribution beginning in 2028.
        if year == 2028:
            portfolio_value += SECOND_YEAR_CONTRIBUTION

        allocation = calculate_glide_path(year)

        equity_actual_return = (
            equity_return + shocks[index, 0]
        )

        bond_actual_return = (
            bond_return + shocks[index, 1]
        )

        cash_actual_return = (
            cash_return + shocks[index, 2]
        )

        portfolio_return = (
            allocation.equity * equity_actual_return
            + allocation.bonds * bond_actual_return
            + allocation.cash * cash_actual_return
        )

        portfolio_value *= 1 + portfolio_return

        # Prevent the model from producing negative portfolio values.
        portfolio_value = max(portfolio_value, 0.0)

        liability = (
            annual_liability
            if year >= LIABILITY_START_YEAR
            else 0.0
        )

        beginning_value = portfolio_value

        # Pay the liability at the end of each liability year.
        if liability > 0:
            portfolio_value -= liability

            if portfolio_value < 0:
                success = False
                portfolio_value = 0.0

        coverage_ratio = (
            beginning_value / liability
            if liability > 0
            else np.nan
        )

        rows.append(
            {
                "Year": year,
                "Equity Allocation": allocation.equity,
                "Bond Allocation": allocation.bonds,
                "Cash Allocation": allocation.cash,
                "Portfolio Return": portfolio_return,
                "Beginning Value": beginning_value,
                "Liability": liability,
                "Ending Value": portfolio_value,
                "Coverage Ratio": coverage_ratio,
            }
        )

    return success, pd.DataFrame(rows)


# ---------------------------------------------------------------------
# Funding probability
# ---------------------------------------------------------------------

def calculate_funding_probability(
    num_simulations: int = NUM_SIMULATIONS,
    annual_liability: float = ANNUAL_LIABILITY,
    equity_return: float = DEFAULT_EQUITY_RETURN,
    bond_return: float = DEFAULT_BOND_RETURN,
    cash_return: float = DEFAULT_CASH_RETURN,
    equity_volatility: float = DEFAULT_EQUITY_VOLATILITY,
    bond_volatility: float = DEFAULT_BOND_VOLATILITY,
    cash_volatility: float = DEFAULT_CASH_VOLATILITY,
    correlation: float = DEFAULT_CORRELATION,
    seed: int = RANDOM_SEED,
) -> float:
    """
    Calculate the probability that the portfolio can fund every
    liability from 2033 through 2042.
    """

    if num_simulations <= 0:
        raise ValueError(
            "num_simulations must be greater than zero."
        )

    rng = np.random.default_rng(seed)

    successful_simulations = 0

    for _ in range(num_simulations):

        success, _ = simulate_portfolio_path(
            annual_liability=annual_liability,
            equity_return=equity_return,
            bond_return=bond_return,
            cash_return=cash_return,
            equity_volatility=equity_volatility,
            bond_volatility=bond_volatility,
            cash_volatility=cash_volatility,
            correlation=correlation,
            rng=rng,
        )

        if success:
            successful_simulations += 1

    return successful_simulations / num_simulations


# ---------------------------------------------------------------------
# Sensitivity analysis
# ---------------------------------------------------------------------

def run_sensitivity_analysis(
    num_simulations: int = 5_000,
    seed: int = RANDOM_SEED,
) -> pd.DataFrame:
    """
    Test how changes in important assumptions affect funding
    probability.

    Each sensitivity test changes one assumption while keeping
    the other assumptions at their base values.
    """

    scenarios = []

    # -------------------------------------------------------------
    # Annual liability sensitivity
    # -------------------------------------------------------------

    liability_values = [
        40_000,
        45_000,
        50_000,
        55_000,
        60_000,
    ]

    for liability in liability_values:

        probability = calculate_funding_probability(
            num_simulations=num_simulations,
            annual_liability=liability,
            seed=seed,
        )

        scenarios.append(
            {
                "Sensitivity": "Annual Liability",
                "Assumption": f"${liability:,.0f}",
                "Value": liability,
                "Funding Probability": probability,
            }
        )

    # -------------------------------------------------------------
    # Equity return sensitivity
    # -------------------------------------------------------------

    equity_returns = [
        0.04,
        0.06,
        0.08,
        0.10,
        0.12,
    ]

    for expected_return in equity_returns:

        probability = calculate_funding_probability(
            num_simulations=num_simulations,
            equity_return=expected_return,
            seed=seed,
        )

        scenarios.append(
            {
                "Sensitivity": "Equity Return",
                "Assumption": f"{expected_return:.0%}",
                "Value": expected_return,
                "Funding Probability": probability,
            }
        )

    # -------------------------------------------------------------
    # Bond return sensitivity
    # -------------------------------------------------------------

    bond_returns = [
        0.02,
        0.03,
        0.04,
        0.05,
        0.06,
    ]

    for expected_return in bond_returns:

        probability = calculate_funding_probability(
            num_simulations=num_simulations,
            bond_return=expected_return,
            seed=seed,
        )

        scenarios.append(
            {
                "Sensitivity": "Bond Return",
                "Assumption": f"{expected_return:.0%}",
                "Value": expected_return,
                "Funding Probability": probability,
            }
        )

    # -------------------------------------------------------------
    # Equity volatility sensitivity
    # -------------------------------------------------------------

    equity_volatilities = [
        0.12,
        0.15,
        0.18,
        0.21,
        0.24,
    ]

    for volatility in equity_volatilities:

        probability = calculate_funding_probability(
            num_simulations=num_simulations,
            equity_volatility=volatility,
            seed=seed,
        )

        scenarios.append(
            {
                "Sensitivity": "Equity Volatility",
                "Assumption": f"{volatility:.0%}",
                "Value": volatility,
                "Funding Probability": probability,
            }
        )

    return pd.DataFrame(scenarios)


# ---------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------

def generate_funding_summary(
    num_simulations: int = NUM_SIMULATIONS,
) -> dict:
    """
    Generate a summary of the base-case funding analysis.
    """

    funding_probability = calculate_funding_probability(
        num_simulations=num_simulations,
    )

    total_liabilities = (
        ANNUAL_LIABILITY
        * (
            END_YEAR
            - LIABILITY_START_YEAR
            + 1
        )
    )

    return {
        "Funding Probability": funding_probability,
        "Initial Portfolio": INITIAL_PORTFOLIO,
        "2028 Contribution": SECOND_YEAR_CONTRIBUTION,
        "Annual Liability": ANNUAL_LIABILITY,
        "Liability Start Year": LIABILITY_START_YEAR,
        "Liability End Year": END_YEAR,
        "Total Scheduled Liabilities": total_liabilities,
        "Number of Simulations": num_simulations,
    }


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

if __name__ == "__main__":

    print("=" * 70)
    print("FUNDING PROBABILITY AND SENSITIVITY ANALYSIS")
    print("=" * 70)

    # Base-case funding probability.
    probability = calculate_funding_probability()

    print("\nBase Case")
    print("-" * 70)
    print(
        f"Probability of Fully Funding "
        f"2033-2042 Liabilities: {probability:.2%}"
    )

    # Summary information.
    summary = generate_funding_summary()

    print("\nPortfolio Assumptions")
    print("-" * 70)

    for key, value in summary.items():

        if key == "Funding Probability":
            print(f"{key}: {value:.2%}")

        elif isinstance(value, float):
            print(f"{key}: ${value:,.2f}")

        else:
            print(f"{key}: {value}")

    # Sensitivity analysis.
    print("\nSensitivity Analysis")
    print("-" * 70)

    sensitivity = run_sensitivity_analysis()

    display_table = sensitivity.copy()

    display_table["Funding Probability"] = (
        display_table["Funding Probability"]
        .map(lambda x: f"{x:.2%}")
    )

    print(display_table.to_string(index=False))

"""
Funding Probability and Sensitivity Analysis.

This module measures the probability that the portfolio can fully fund
the required $50,000 annual liabilities from 2033 through 2042.

It also performs sensitivity analysis to determine how changes in
portfolio assumptions affect funding success.

The model uses the Dynamic Glide Path from glide_path.py to gradually
reduce portfolio risk as the liability deadline approaches.
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import pandas as pd

from .glide_path import (
    END_YEAR,
    LIABILITIES,
    START_YEAR,
    calculate_glide_path,
)


# ============================================================
# PORTFOLIO ASSUMPTIONS
# ============================================================

INITIAL_PORTFOLIO = 300_000.0
SECOND_YEAR_CONTRIBUTION = 150_000.0

EQUITY_RETURN = 0.08
BOND_RETURN = 0.04
CASH_RETURN = 0.02

EQUITY_VOLATILITY = 0.18
BOND_VOLATILITY = 0.06
CASH_VOLATILITY = 0.01

# Simplified correlation assumption between asset classes.
CORRELATION = 0.20

N_SIMULATIONS = 10_000
RANDOM_SEED = 42


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def build_covariance_matrix(
    equity_volatility: float,
    bond_volatility: float,
    cash_volatility: float,
    correlation: float,
) -> np.ndarray:
    """
    Build the covariance matrix for equities, bonds, and cash.
    """

    if equity_volatility < 0:
        raise ValueError("equity_volatility cannot be negative.")

    if bond_volatility < 0:
        raise ValueError("bond_volatility cannot be negative.")

    if cash_volatility < 0:
        raise ValueError("cash_volatility cannot be negative.")

    if not -1 <= correlation <= 1:
        raise ValueError("correlation must be between -1 and 1.")

    volatilities = np.array(
        [
            equity_volatility,
            bond_volatility,
            cash_volatility,
        ]
    )

    correlation_matrix = np.full(
        (3, 3),
        correlation,
        dtype=float,
    )

    np.fill_diagonal(correlation_matrix, 1.0)

    covariance_matrix = (
        np.outer(volatilities, volatilities)
        * correlation_matrix
    )

    return covariance_matrix


def generate_asset_returns(
    rng: np.random.Generator,
    equity_return: float,
    bond_return: float,
    cash_return: float,
    equity_volatility: float,
    bond_volatility: float,
    cash_volatility: float,
    correlation: float,
) -> np.ndarray:
    """
    Generate one year's correlated returns for equities,
    bonds, and cash.
    """

    expected_returns = np.array(
        [
            equity_return,
            bond_return,
            cash_return,
        ]
    )

    covariance_matrix = build_covariance_matrix(
        equity_volatility=equity_volatility,
        bond_volatility=bond_volatility,
        cash_volatility=cash_volatility,
        correlation=correlation,
    )

    return rng.multivariate_normal(
        mean=expected_returns,
        cov=covariance_matrix,
    )


# ============================================================
# PORTFOLIO SIMULATION
# ============================================================

def simulate_portfolio_path(
    initial_portfolio: float = INITIAL_PORTFOLIO,
    second_year_contribution: float = SECOND_YEAR_CONTRIBUTION,
    equity_return: float = EQUITY_RETURN,
    bond_return: float = BOND_RETURN,
    cash_return: float = CASH_RETURN,
    equity_volatility: float = EQUITY_VOLATILITY,
    bond_volatility: float = BOND_VOLATILITY,
    cash_volatility: float = CASH_VOLATILITY,
    correlation: float = CORRELATION,
    rng: np.random.Generator | None = None,
) -> Tuple[bool, pd.DataFrame]:
    """
    Simulate one portfolio path from 2027 through 2042.

    Returns
    -------
    tuple
        success : bool
            True if every liability can be fully funded.
        DataFrame
            Year-by-year portfolio results.
    """

    if initial_portfolio < 0:
        raise ValueError("initial_portfolio cannot be negative.")

    if second_year_contribution < 0:
        raise ValueError(
            "second_year_contribution cannot be negative."
        )

    if rng is None:
        rng = np.random.default_rng()

    portfolio_value = initial_portfolio

    rows = []

    for year in range(START_YEAR, END_YEAR + 1):

        # ----------------------------------------------------
        # Contributions
        # ----------------------------------------------------

        contribution = 0.0

        if year == START_YEAR:
            contribution = initial_portfolio

        elif year == START_YEAR + 1:
            contribution = second_year_contribution

        beginning_value = portfolio_value

        portfolio_value += contribution

        value_before_return = portfolio_value

        # ----------------------------------------------------
        # Dynamic Glide Path
        # ----------------------------------------------------

        allocation = calculate_glide_path(year)

        # ----------------------------------------------------
        # Simulate asset returns
        # ----------------------------------------------------

        asset_returns = generate_asset_returns(
            rng=rng,
            equity_return=equity_return,
            bond_return=bond_return,
            cash_return=cash_return,
            equity_volatility=equity_volatility,
            bond_volatility=bond_volatility,
            cash_volatility=cash_return * 0 + cash_volatility,
            correlation=correlation,
        )

        equity_random_return = asset_returns[0]
        bond_random_return = asset_returns[1]
        cash_random_return = asset_returns[2]

        portfolio_return = (
            allocation.equity * equity_random_return
            + allocation.bonds * bond_random_return
            + allocation.cash * cash_random_return
        )

        investment_gain = (
            portfolio_value * portfolio_return
        )

        portfolio_value += investment_gain

        value_before_liability = portfolio_value

        # ----------------------------------------------------
        # Liability Payment
        # ----------------------------------------------------

        liability = LIABILITIES.get(year, 0.0)

        payment_made = min(
            portfolio_value,
            liability,
        )

        shortfall = max(
            liability - portfolio_value,
            0.0,
        )

        portfolio_value = max(
            portfolio_value - liability,
            0.0,
        )

        funded = shortfall == 0.0

        rows.append(
            {
                "Year": year,
                "Beginning Value": beginning_value,
                "Contribution": contribution,
                "Value Before Return": value_before_return,
                "Equity Return": equity_random_return,
                "Bond Return": bond_random_return,
                "Cash Return": cash_random_return,
                "Portfolio Return": portfolio_return,
                "Investment Gain": investment_gain,
                "Value Before Liability": value_before_liability,
                "Liability": liability,
                "Payment Made": payment_made,
                "Shortfall": shortfall,
                "Funded": funded,
                "Ending Value": portfolio_value,
                "Equity Allocation": allocation.equity,
                "Bond Allocation": allocation.bonds,
                "Cash Allocation": allocation.cash,
            }
        )

    results = pd.DataFrame(rows)

    success = bool(results["Funded"].all())

    return success, results


# ============================================================
# OVERALL FUNDING PROBABILITY
# ============================================================

def calculate_funding_probability(
    n_simulations: int = N_SIMULATIONS,
    random_seed: int = RANDOM_SEED,
    **simulation_kwargs,
) -> float:
    """
    Calculate the probability that the portfolio funds
    every liability from 2033 through 2042.
    """

    if n_simulations <= 0:
        raise ValueError(
            "n_simulations must be greater than zero."
        )

    rng = np.random.default_rng(random_seed)

    successful_simulations = 0

    for _ in range(n_simulations):

        success, _ = simulate_portfolio_path(
            rng=rng,
            **simulation_kwargs,
        )

        if success:
            successful_simulations += 1

    return successful_simulations / n_simulations


# ============================================================
# YEAR-BY-YEAR FUNDING PROBABILITY
# ============================================================

def calculate_yearly_funding_probability(
    n_simulations: int = N_SIMULATIONS,
    random_seed: int = RANDOM_SEED,
    **simulation_kwargs,
) -> pd.DataFrame:
    """
    Calculate the probability of successfully funding each
    individual liability year.

    Returns
    -------
    pandas.DataFrame
        Probability of successfully funding the liability
        in every year from 2033 through 2042.
    """

    if n_simulations <= 0:
        raise ValueError(
            "n_simulations must be greater than zero."
        )

    rng = np.random.default_rng(random_seed)

    funding_counts: Dict[int, int] = {
        year: 0
        for year in LIABILITIES
    }

    for _ in range(n_simulations):

        _, results = simulate_portfolio_path(
            rng=rng,
            **simulation_kwargs,
        )

        liability_results = results[
            results["Liability"] > 0
        ]

        for _, row in liability_results.iterrows():

            year = int(row["Year"])

            if row["Funded"]:
                funding_counts[year] += 1

    rows = []

    for year, count in funding_counts.items():

        probability = count / n_simulations

        rows.append(
            {
                "Year": year,
                "Liability": LIABILITIES[year],
                "Funding Probability": probability,
                "Funding Probability (%)": probability * 100,
            }
        )

    return pd.DataFrame(rows)


# ============================================================
# SHORTFALL ANALYSIS
# ============================================================

def calculate_shortfall_statistics(
    n_simulations: int = N_SIMULATIONS,
    random_seed: int = RANDOM_SEED,
    **simulation_kwargs,
) -> Dict[str, float]:
    """
    Calculate shortfall statistics across Monte Carlo simulations.

    Measures the maximum shortfall experienced in each simulation.
    """

    if n_simulations <= 0:
        raise ValueError(
            "n_simulations must be greater than zero."
        )

    rng = np.random.default_rng(random_seed)

    maximum_shortfalls = []

    for _ in range(n_simulations):

        _, results = simulate_portfolio_path(
            rng=rng,
            **simulation_kwargs,
        )

        liability_results = results[
            results["Liability"] > 0
        ]

        max_shortfall = float(
            liability_results["Shortfall"].max()
        )

        maximum_shortfalls.append(max_shortfall)

    shortfalls = np.array(maximum_shortfalls)

    simulations_with_shortfall = shortfalls[shortfalls > 0]

    if len(simulations_with_shortfall) > 0:

        average_failure_shortfall = float(
            np.mean(simulations_with_shortfall)
        )

        median_failure_shortfall = float(
            np.median(simulations_with_shortfall)
        )

        worst_shortfall = float(
            np.max(simulations_with_shortfall)
        )

    else:

        average_failure_shortfall = 0.0
        median_failure_shortfall = 0.0
        worst_shortfall = 0.0

    return {
        "Average Maximum Shortfall": average_failure_shortfall,
        "Median Maximum Shortfall": median_failure_shortfall,
        "Worst Shortfall": worst_shortfall,
        "Probability of Any Shortfall": float(
            np.mean(shortfalls > 0)
        ),
    }


# ============================================================
# SENSITIVITY ANALYSIS
# ============================================================

def run_sensitivity_analysis(
    n_simulations: int = N_SIMULATIONS,
    random_seed: int = RANDOM_SEED,
) -> pd.DataFrame:
    """
    Test how changes in key assumptions affect funding probability.

    Each sensitivity scenario changes one assumption while
    holding the others constant.
    """

    scenarios = []

    # --------------------------------------------------------
    # Liability Sensitivity
    # --------------------------------------------------------

    liability_values = [
        40_000,
        45_000,
        50_000,
        55_000,
        60_000,
    ]

    for liability in liability_values:

        original_liabilities = LIABILITIES.copy()

        LIABILITIES.update(
            {
                year: float(liability)
                for year in LIABILITIES
            }
        )

        probability = calculate_funding_probability(
            n_simulations=n_simulations,
            random_seed=random_seed,
        )

        LIABILITIES.clear()
        LIABILITIES.update(original_liabilities)

        scenarios.append(
            {
                "Variable": "Annual Liability",
                "Value": liability,
                "Funding Probability": probability,
                "Funding Probability (%)": probability * 100,
            }
        )

    # --------------------------------------------------------
    # Equity Return Sensitivity
    # --------------------------------------------------------

    equity_returns = [
        0.04,
        0.06,
        0.08,
        0.10,
        0.12,
    ]

    for value in equity_returns:

        probability = calculate_funding_probability(
            n_simulations=n_simulations,
            random_seed=random_seed,
            equity_return=value,
        )

        scenarios.append(
            {
                "Variable": "Equity Return",
                "Value": value,
                "Funding Probability": probability,
                "Funding Probability (%)": probability * 100,
            }
        )

    # --------------------------------------------------------
    # Bond Return Sensitivity
    # --------------------------------------------------------

    bond_returns = [
        0.02,
        0.03,
        0.04,
        0.05,
        0.06,
    ]

    for value in bond_returns:

        probability = calculate_funding_probability(
            n_simulations=n_simulations,
            random_seed=random_seed,
            bond_return=value,
        )

        scenarios.append(
            {
                "Variable": "Bond Return",
                "Value": value,
                "Funding Probability": probability,
                "Funding Probability (%)": probability * 100,
            }
        )

    # --------------------------------------------------------
    # Equity Volatility Sensitivity
    # --------------------------------------------------------

    equity_volatilities = [
        0.12,
        0.15,
        0.18,
        0.21,
        0.24,
    ]

    for value in equity_volatilities:

        probability = calculate_funding_probability(
            n_simulations=n_simulations,
            random_seed=random_seed,
            equity_volatility=value,
        )

        scenarios.append(
            {
                "Variable": "Equity Volatility",
                "Value": value,
                "Funding Probability": probability,
                "Funding Probability (%)": probability * 100,
            }
        )

    return pd.DataFrame(scenarios)


# ============================================================
# TWO-VARIABLE SENSITIVITY ANALYSIS
# ============================================================

def run_two_variable_sensitivity(
    n_simulations: int = N_SIMULATIONS,
    random_seed: int = RANDOM_SEED,
) -> pd.DataFrame:
    """
    Test two assumptions simultaneously:

        Equity Return × Annual Liability

    This shows how combinations of assumptions affect
    funding probability.
    """

    equity_returns = [
        0.06,
        0.08,
        0.10,
    ]

    annual_liabilities = [
        40_000,
        50_000,
        60_000,
    ]

    rows = []

    for equity_return in equity_returns:

        for annual_liability in annual_liabilities:

            original_liabilities = LIABILITIES.copy()

            LIABILITIES.update(
                {
                    year: float(annual_liability)
                    for year in LIABILITIES
                }
            )

            probability = calculate_funding_probability(
                n_simulations=n_simulations,
                random_seed=random_seed,
                equity_return=equity_return,
            )

            LIABILITIES.clear()
            LIABILITIES.update(original_liabilities)

            rows.append(
                {
                    "Equity Return": equity_return,
                    "Annual Liability": annual_liability,
                    "Funding Probability": probability,
                    "Funding Probability (%)": probability * 100,
                }
            )

    return pd.DataFrame(rows)


# ============================================================
# SCENARIO ANALYSIS
# ============================================================

def run_scenario_analysis(
    n_simulations: int = N_SIMULATIONS,
    random_seed: int = RANDOM_SEED,
) -> pd.DataFrame:
    """
    Run optimistic, base, and pessimistic scenarios.
    """

    scenarios = {
        "Optimistic": {
            "equity_return": 0.10,
            "bond_return": 0.05,
            "equity_volatility": 0.16,
        },
        "Base": {
            "equity_return": 0.08,
            "bond_return": 0.04,
            "equity_volatility": 0.18,
        },
        "Pessimistic": {
            "equity_return": 0.06,
            "bond_return": 0.03,
            "equity_volatility": 0.22,
        },
    }

    rows = []

    for scenario_name, assumptions in scenarios.items():

        probability = calculate_funding_probability(
            n_simulations=n_simulations,
            random_seed=random_seed,
            **assumptions,
        )

        shortfall_stats = calculate_shortfall_statistics(
            n_simulations=n_simulations,
            random_seed=random_seed,
            **assumptions,
        )

        rows.append(
            {
                "Scenario": scenario_name,
                "Equity Return": assumptions[
                    "equity_return"
                ],
                "Bond Return": assumptions[
                    "bond_return"
                ],
                "Equity Volatility": assumptions[
                    "equity_volatility"
                ],
                "Funding Probability": probability,
                "Funding Probability (%)": probability * 100,
                "Average Maximum Shortfall": shortfall_stats[
                    "Average Maximum Shortfall"
                ],
                "Worst Shortfall": shortfall_stats[
                    "Worst Shortfall"
                ],
            }
        )

    return pd.DataFrame(rows)


# ============================================================
# FUNDING SUMMARY
# ============================================================

def generate_funding_summary(
    n_simulations: int = N_SIMULATIONS,
    random_seed: int = RANDOM_SEED,
) -> Dict[str, object]:
    """
    Generate a complete summary of funding certainty.
    """

    probability = calculate_funding_probability(
        n_simulations=n_simulations,
        random_seed=random_seed,
    )

    yearly_probability = calculate_yearly_funding_probability(
        n_simulations=n_simulations,
        random_seed=random_seed,
    )

    shortfall_statistics = calculate_shortfall_statistics(
        n_simulations=n_simulations,
        random_seed=random_seed,
    )

    sensitivity = run_sensitivity_analysis(
        n_simulations=n_simulations,
        random_seed=random_seed,
    )

    two_variable_sensitivity = run_two_variable_sensitivity(
        n_simulations=n_simulations,
        random_seed=random_seed,
    )

    scenarios = run_scenario_analysis(
        n_simulations=n_simulations,
        random_seed=random_seed,
    )

    return {
        "Funding Probability": probability,
        "Yearly Funding Probability": yearly_probability,
        "Shortfall Statistics": shortfall_statistics,
        "Sensitivity Analysis": sensitivity,
        "Two Variable Sensitivity": two_variable_sensitivity,
        "Scenario Analysis": scenarios,
    }


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print("=" * 70)
    print("FUNDING PROBABILITY AND SENSITIVITY ANALYSIS")
    print("=" * 70)

    print("\nPortfolio Assumptions")
    print("-" * 70)

    print(f"Initial Portfolio: ${INITIAL_PORTFOLIO:,.0f}")
    print(
        f"2028 Contribution: "
        f"${SECOND_YEAR_CONTRIBUTION:,.0f}"
    )

    print(f"Equity Return: {EQUITY_RETURN:.1%}")
    print(f"Bond Return: {BOND_RETURN:.1%}")
    print(f"Cash Return: {CASH_RETURN:.1%}")

    print(
        f"Equity Volatility: "
        f"{EQUITY_VOLATILITY:.1%}"
    )

    print(
        f"Bond Volatility: "
        f"{BOND_VOLATILITY:.1%}"
    )

    print(
        f"Cash Volatility: "
        f"{CASH_VOLATILITY:.1%}"
    )

    print(f"Simulations: {N_SIMULATIONS:,}")

    # --------------------------------------------------------
    # Overall Funding Probability
    # --------------------------------------------------------

    funding_probability = calculate_funding_probability()

    print("\nOverall Funding Probability")
    print("-" * 70)

    print(
        f"Probability of funding all liabilities "
        f"through 2042: "
        f"{funding_probability:.2%}"
    )

    # --------------------------------------------------------
    # Year-by-Year Funding Probability
    # --------------------------------------------------------

    yearly_probability = (
        calculate_yearly_funding_probability()
    )

    print("\nYear-by-Year Funding Probability")
    print("-" * 70)

    print(
        yearly_probability[
            [
                "Year",
                "Liability",
                "Funding Probability (%)",
            ]
        ].to_string(index=False)
    )

    # --------------------------------------------------------
    # Shortfall Analysis
    # --------------------------------------------------------

    shortfall_statistics = (
        calculate_shortfall_statistics()
    )

    print("\nShortfall Analysis")
    print("-" * 70)

    print(
        "Average Maximum Shortfall: "
        f"${shortfall_statistics['Average Maximum Shortfall']:,.2f}"
    )

    print(
        "Median Maximum Shortfall: "
        f"${shortfall_statistics['Median Maximum Shortfall']:,.2f}"
    )

    print(
        "Worst Shortfall: "
        f"${shortfall_statistics['Worst Shortfall']:,.2f}"
    )

    print(
        "Probability of Any Shortfall: "
        f"{shortfall_statistics['Probability of Any Shortfall']:.2%}"
    )

    # --------------------------------------------------------
    # Sensitivity Analysis
    # --------------------------------------------------------

    sensitivity = run_sensitivity_analysis()

    print("\nSensitivity Analysis")
    print("-" * 70)

    print(
        sensitivity[
            [
                "Variable",
                "Value",
                "Funding Probability (%)",
            ]
        ].to_string(index=False)
    )

    # --------------------------------------------------------
    # Two-Variable Sensitivity
    # --------------------------------------------------------

    two_variable = run_two_variable_sensitivity()

    print("\nTwo-Variable Sensitivity")
    print("-" * 70)

    print(
        two_variable[
            [
                "Equity Return",
                "Annual Liability",
                "Funding Probability (%)",
            ]
        ].to_string(index=False)
    )

    # --------------------------------------------------------
    # Scenario Analysis
    # --------------------------------------------------------

    scenarios = run_scenario_analysis()

    print("\nScenario Analysis")
    print("-" * 70)

    print(
        scenarios[
            [
                "Scenario",
                "Equity Return",
                "Bond Return",
                "Equity Volatility",
                "Funding Probability (%)",
                "Average Maximum Shortfall",
                "Worst Shortfall",
            ]
        ].to_string(index=False)
    )

    print("\n" + "=" * 70)
    print("Analysis complete.")
    print("=" * 70)

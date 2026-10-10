"""Funding Probability, Operating Reserve, and Facility Contribution Analysis.

This module is a planning model for Laura Gao's case study. It is designed to
run directly as part of the ``black_litterman`` package and does NOT launch the
Streamlit UI.

CASE-STUDY CASH FLOWS
---------------------
* $300,000 is invested at the beginning of 2027.
* An additional $150,000 is invested at the beginning of 2028.
* There are no other contributions or withdrawals before 2033.
* Ten fixed $50,000 operating payments occur at the beginning of each year
  from 2033 through 2042.
* The operating reserve is established before the 2033 payment and before any
  facility contribution.

MODEL ASSUMPTIONS, NOT CASE-STUDY REQUIREMENTS
---------------------------------------------
* Annual arithmetic return assumptions, volatilities, and correlations for
  equity, bonds, and cash are set below.
* Asset returns are sampled from a multivariate normal distribution. Returns
  below -99% are clipped to -99%. This is a simplifying model assumption,
  not a claim that financial returns are normally distributed.
* The report defaults to a conservative 5% equity / 15% bond / 80% cash
  accumulation candidate because it produced the highest modeled funding
  probability among the explicitly listed initial candidates in this model.
  This is NOT an exhaustive allocation optimizer and must be validated against
  implementation constraints and historical data. The existing dynamic
  glide path remains a comparison candidate.
* The operating reserve has a separate, conservative allocation schedule.
* The default reserve target aims to cover all ten payments in 95% of modeled
  future reserve-return paths. This is a team-selected target, not a percentage
  specified by the case study.

IMPORTANT INTERPRETATION
------------------------
The reserve target being affordable, the reserve-only strategy succeeding, and
all payments remaining fundable after a facility contribution are different
quantities. This module reports them separately.

For the no-facility-contribution benchmark, all portfolio assets at the start
of 2033 remain available to support the operating payments. For a proposed
facility contribution, the contribution is made ONLY if the portfolio can
first cover the full reserve target plus that contribution. Otherwise the
facility contribution is zero. All assets remaining after the decision are
modeled as available for the operating payments and invested according to the
reserve allocation schedule. This is deliberately protective of the operating
commitment; any different treatment of surplus assets should be stated and
modeled as a separate policy.

This is an asset-class model, not a direct simulation of the selected sample
stocks or the Black-Litterman optimized weights. Use block-bootstrap analysis
as a complementary check when historical returns and matching weights are
available.
"""

from __future__ import annotations

import argparse
from typing import Dict, Mapping, Sequence

import numpy as np
import pandas as pd

try:  # Supports both ``python -m black_litterman.funding_probability`` and direct execution.
    from .glide_path import (
        END_YEAR,
        LIABILITIES,
        LIABILITY_START_YEAR,
        START_YEAR,
        calculate_glide_path,
    )
except ImportError:  # pragma: no cover - convenience for direct file execution
    from glide_path import (  # type: ignore[no-redef]
        END_YEAR,
        LIABILITIES,
        LIABILITY_START_YEAR,
        START_YEAR,
        calculate_glide_path,
    )


# ---------------------------------------------------------------------------
# 1. CASE-STUDY INPUTS AND EXPLICIT MODEL ASSUMPTIONS
# ---------------------------------------------------------------------------

INITIAL_PORTFOLIO = 300_000.0
SECOND_YEAR_CONTRIBUTION = 150_000.0

# Illustrative annual arithmetic return assumptions. The case study does not
# prescribe these values; justify and stress-test them in the final report.
EQUITY_RETURN = 0.08
BOND_RETURN = 0.04
CASH_RETURN = 0.02
EQUITY_VOLATILITY = 0.18
BOND_VOLATILITY = 0.06
CASH_VOLATILITY = 0.01
CORRELATION = 0.20

N_SIMULATIONS = 10_000
RANDOM_SEED = 42
DEFAULT_RESERVE_CONFIDENCE_LEVEL = 0.95
FACILITY_RANGE_LOWER_QUANTILE = 0.05
FACILITY_RANGE_UPPER_QUANTILE = 0.95

# Proposed reserve-only mix. These weights are team assumptions, not case-study
# requirements. Payments are made at the beginning of each year; returns apply
# to the remaining assets after that year's payment.
RESERVE_ALLOCATION_SCHEDULE: dict[int, tuple[float, float, float]] = {
    2033: (0.10, 0.60, 0.30),
    2034: (0.08, 0.60, 0.32),
    2035: (0.06, 0.60, 0.34),
    2036: (0.04, 0.60, 0.36),
    2037: (0.02, 0.60, 0.38),
    2038: (0.00, 0.60, 0.40),
    2039: (0.00, 0.60, 0.40),
    2040: (0.00, 0.60, 0.40),
    2041: (0.00, 0.60, 0.40),
    2042: (0.00, 0.60, 0.40),
}

# Illustrative alternative accumulation policies for sensitivity/comparison.
# Static allocations are comparisons, not automatic recommendations. The
# ``Current dynamic glide path`` uses the actual glide_path.py allocation.
ACCUMULATION_POLICY_ALTERNATIVES: dict[str, tuple[float, float, float] | None] = {
    "Current dynamic glide path": None,
    "Conservative 5/15/80 (equity/bonds/cash)": (0.05, 0.15, 0.80),
    "Bond-heavy 10/70/20 (equity/bonds/cash)": (0.10, 0.70, 0.20),
    "Balanced 30/50/20 (equity/bonds/cash)": (0.30, 0.50, 0.20),
    "Equity-heavy 70/25/5 (equity/bonds/cash)": (0.70, 0.25, 0.05),
}
# This is a funding-first candidate, selected from the limited comparison set
# above. It is not the result of an exhaustive optimization over all possible
# allocations and must be reconsidered if the available assets/implementation
# constraints do not allow a bond/cash allocation of this kind.
DEFAULT_ACCUMULATION_POLICY = "Conservative 5/15/80 (equity/bonds/cash)"

# Alternative reserve policies for comparison. The reserve target and the
# payment-funding probability are recalculated for each policy.
RESERVE_POLICY_ALTERNATIVES: dict[str, tuple[float, float, float] | None] = {
    "Proposed declining-equity reserve glide path": None,
    "Bond/cash benchmark 80/20": (0.00, 0.80, 0.20),
    "All-cash benchmark": (0.00, 0.00, 1.00),
}
# For the stated goal of maximizing operating-payment reliability, default to
# the all-cash reserve among the reserve alternatives tested. This is still a
# simplified assumption: future cash yields are uncertain, so compare against
# a dated high-quality bond ladder before making a final recommendation.
DEFAULT_RESERVE_POLICY = "Proposed declining-equity reserve glide path"


# ---------------------------------------------------------------------------
# 2. INPUT VALIDATION AND ALLOCATION HELPERS
# ---------------------------------------------------------------------------


def _validate_assumptions(
    equity_return: float,
    bond_return: float,
    cash_return: float,
    equity_volatility: float,
    bond_volatility: float,
    cash_volatility: float,
    correlation: float,
) -> None:
    returns = (equity_return, bond_return, cash_return)
    volatilities = (equity_volatility, bond_volatility, cash_volatility)
    if not all(np.isfinite(value) and value > -1.0 for value in returns):
        raise ValueError("Asset returns must be finite and greater than -100%.")
    if not all(np.isfinite(value) and value >= 0.0 for value in volatilities):
        raise ValueError("Asset volatilities must be finite and non-negative.")
    # A 3-asset equicorrelation matrix is positive semidefinite for this range.
    if not np.isfinite(correlation) or not -0.5 <= correlation <= 1.0:
        raise ValueError(
            "With three equally correlated asset classes, correlation must be between -0.5 and 1.0."
        )


def _validate_simulation_inputs(
    n_simulations: int,
    initial_portfolio: float,
    second_year_contribution: float,
    reserve_confidence_level: float,
    portfolio_value_at_2031: float | None,
) -> None:
    if not isinstance(n_simulations, (int, np.integer)) or n_simulations <= 0:
        raise ValueError("n_simulations must be a positive integer.")
    if not np.isfinite(initial_portfolio) or initial_portfolio < 0:
        raise ValueError("initial_portfolio must be finite and non-negative.")
    if not np.isfinite(second_year_contribution) or second_year_contribution < 0:
        raise ValueError("second_year_contribution must be finite and non-negative.")
    if not 0.50 <= reserve_confidence_level < 1.0:
        raise ValueError("reserve_confidence_level must be at least 0.50 and less than 1.0.")
    if portfolio_value_at_2031 is not None and (
        not np.isfinite(portfolio_value_at_2031) or portfolio_value_at_2031 < 0
    ):
        raise ValueError("portfolio_value_at_2031 must be finite and non-negative.")


def _validate_liabilities(liabilities: Mapping[int, float] | None) -> Dict[int, float]:
    supplied = LIABILITIES if liabilities is None else liabilities
    result = {year: 0.0 for year in range(LIABILITY_START_YEAR, END_YEAR + 1)}
    for raw_year, raw_amount in supplied.items():
        year = int(raw_year)
        amount = float(raw_amount)
        if year < LIABILITY_START_YEAR or year > END_YEAR:
            raise ValueError(
                f"Liability years must be between {LIABILITY_START_YEAR} and {END_YEAR}."
            )
        if not np.isfinite(amount) or amount < 0:
            raise ValueError("Liabilities must be finite, non-negative amounts.")
        result[year] = amount
    if not any(amount > 0 for amount in result.values()):
        raise ValueError("At least one positive operating liability is required.")
    return result


def _validate_allocation(
    allocation: Sequence[float],
    name: str = "allocation",
) -> tuple[float, float, float]:
    values = tuple(float(value) for value in allocation)
    if len(values) != 3:
        raise ValueError(f"{name} must contain equity, bonds, and cash weights.")
    if not all(np.isfinite(value) and value >= 0.0 for value in values):
        raise ValueError(f"{name} weights must be finite and non-negative.")
    if not np.isclose(sum(values), 1.0, atol=1e-9):
        raise ValueError(f"{name} weights must sum to 100%; got {sum(values):.6f}.")
    return values  # type: ignore[return-value]


def _normalize_accumulation_schedule(
    allocation_schedule: Mapping[int, Sequence[float]] | None,
) -> dict[int, tuple[float, float, float]] | None:
    if allocation_schedule is None:
        return None
    expected_years = set(range(START_YEAR, LIABILITY_START_YEAR))
    supplied_years = {int(year) for year in allocation_schedule}
    if not expected_years.issubset(supplied_years):
        missing = sorted(expected_years - supplied_years)
        raise ValueError(f"Accumulation allocation schedule is missing years: {missing}.")
    return {
        year: _validate_allocation(allocation_schedule[year], f"allocation {year}")
        for year in expected_years
    }


def _normalize_reserve_schedule(
    allocation_schedule: Mapping[int, Sequence[float]] | None,
) -> dict[int, tuple[float, float, float]]:
    if allocation_schedule is None:
        return {
            year: _validate_allocation(allocation, f"reserve allocation {year}")
            for year, allocation in RESERVE_ALLOCATION_SCHEDULE.items()
        }
    expected_years = set(range(LIABILITY_START_YEAR, END_YEAR + 1))
    supplied_years = {int(year) for year in allocation_schedule}
    if not expected_years.issubset(supplied_years):
        missing = sorted(expected_years - supplied_years)
        raise ValueError(f"Reserve allocation schedule is missing years: {missing}.")
    return {
        year: _validate_allocation(allocation_schedule[year], f"reserve allocation {year}")
        for year in expected_years
    }


def _resolve_reserve_schedule(
    reserve_policy: str,
    allocation_schedule: Mapping[int, Sequence[float]] | None,
) -> dict[int, tuple[float, float, float]]:
    """Resolve a named reserve policy or validate an explicitly supplied schedule."""
    if allocation_schedule is not None:
        return _normalize_reserve_schedule(allocation_schedule)
    if reserve_policy not in RESERVE_POLICY_ALTERNATIVES:
        choices = ", ".join(RESERVE_POLICY_ALTERNATIVES)
        raise ValueError(f"Unknown reserve_policy. Choose one of: {choices}.")
    allocation = RESERVE_POLICY_ALTERNATIVES[reserve_policy]
    if allocation is None:
        return _normalize_reserve_schedule(None)  # proposed schedule in module constants
    return _build_static_schedule(
        allocation, range(LIABILITY_START_YEAR, END_YEAR + 1)
    )


def _build_static_schedule(
    allocation: Sequence[float],
    years: range,
) -> dict[int, tuple[float, float, float]]:
    checked = _validate_allocation(allocation)
    return {year: checked for year in years}


def _resolve_accumulation_schedule(
    accumulation_policy: str,
    allocation_schedule: Mapping[int, Sequence[float]] | None,
) -> dict[int, tuple[float, float, float]] | None:
    """Resolve a named accumulation policy or validate an explicitly supplied schedule."""
    if allocation_schedule is not None:
        return _normalize_accumulation_schedule(allocation_schedule)
    if accumulation_policy not in ACCUMULATION_POLICY_ALTERNATIVES:
        choices = ", ".join(ACCUMULATION_POLICY_ALTERNATIVES)
        raise ValueError(f"Unknown accumulation_policy. Choose one of: {choices}.")
    allocation = ACCUMULATION_POLICY_ALTERNATIVES[accumulation_policy]
    if allocation is None:
        return None  # use the existing dynamic schedule in glide_path.py
    return _build_static_schedule(allocation, range(START_YEAR, LIABILITY_START_YEAR))


def build_covariance_matrix(
    equity_volatility: float = EQUITY_VOLATILITY,
    bond_volatility: float = BOND_VOLATILITY,
    cash_volatility: float = CASH_VOLATILITY,
    correlation: float = CORRELATION,
) -> np.ndarray:
    """Build the annual covariance matrix for equity, bonds, and cash."""
    _validate_assumptions(
        EQUITY_RETURN, BOND_RETURN, CASH_RETURN,
        equity_volatility, bond_volatility, cash_volatility, correlation,
    )
    volatilities = np.array(
        [equity_volatility, bond_volatility, cash_volatility], dtype=float
    )
    correlation_matrix = np.full((3, 3), correlation, dtype=float)
    np.fill_diagonal(correlation_matrix, 1.0)
    return np.outer(volatilities, volatilities) * correlation_matrix


def generate_asset_returns(
    rng: np.random.Generator,
    equity_return: float = EQUITY_RETURN,
    bond_return: float = BOND_RETURN,
    cash_return: float = CASH_RETURN,
    equity_volatility: float = EQUITY_VOLATILITY,
    bond_volatility: float = BOND_VOLATILITY,
    cash_volatility: float = CASH_VOLATILITY,
    correlation: float = CORRELATION,
    size: int | None = None,
) -> np.ndarray:
    """Generate correlated equity/bond/cash simple returns using vectorized NumPy.

    ``size=10_000`` generates 10,000 correlated annual return triplets in one
    vectorized call. Values below -99% are clipped as a simplifying safeguard.
    """
    _validate_assumptions(
        equity_return, bond_return, cash_return,
        equity_volatility, bond_volatility, cash_volatility, correlation,
    )
    means = np.array([equity_return, bond_return, cash_return], dtype=float)
    covariance = build_covariance_matrix(
        equity_volatility, bond_volatility, cash_volatility, correlation,
    )
    sample = rng.multivariate_normal(
        mean=means, cov=covariance, size=size, check_valid="raise"
    )
    return np.maximum(sample, -0.99)


def get_reserve_allocation(year: int) -> tuple[float, float, float]:
    """Return the proposed equity/bond/cash mix for one reserve year."""
    if year < LIABILITY_START_YEAR or year > END_YEAR:
        raise ValueError(f"Reserve year must be between {LIABILITY_START_YEAR} and {END_YEAR}.")
    return _normalize_reserve_schedule(None)[year]


def generate_reserve_glide_path(
    liabilities: Mapping[int, float] | None = None,
    reserve_allocation_schedule: Mapping[int, Sequence[float]] | None = None,
) -> pd.DataFrame:
    """Return the reserve allocation and liability schedule for 2033–2042."""
    schedule = _validate_liabilities(liabilities)
    allocations = _normalize_reserve_schedule(reserve_allocation_schedule)
    return pd.DataFrame([
        {
            "Year": year,
            "Equity": allocations[year][0],
            "Bonds": allocations[year][1],
            "Cash": allocations[year][2],
            "Liability": float(schedule[year]),
        }
        for year in range(LIABILITY_START_YEAR, END_YEAR + 1)
    ])


# ---------------------------------------------------------------------------
# 3. RETURN-PATH SIMULATION AND RESERVE MATHEMATICS
# ---------------------------------------------------------------------------


def _portfolio_returns_from_asset_draws(
    asset_draws: np.ndarray,
    allocation: Sequence[float],
) -> np.ndarray:
    weights = np.asarray(_validate_allocation(allocation), dtype=float)
    return asset_draws @ weights


def _simulate_portfolio_values_at_2033(
    n_simulations: int,
    rng: np.random.Generator,
    initial_portfolio: float,
    second_year_contribution: float,
    assumptions: Mapping[str, float],
    portfolio_value_at_2031: float | None = None,
    accumulation_allocation_schedule: Mapping[int, Sequence[float]] | None = None,
) -> np.ndarray:
    """Simulate portfolio value at the beginning of 2033, before the first payment.

    Without a 2031 value, the path starts at the beginning of 2027 and applies
    the two case-study contributions. With ``portfolio_value_at_2031``, that
    value is interpreted as the beginning-of-2031 value; only 2031 and 2032
    returns are simulated, with no additional contributions.
    """
    custom_schedule = _normalize_accumulation_schedule(accumulation_allocation_schedule)
    if portfolio_value_at_2031 is None:
        values = np.zeros(n_simulations, dtype=float)
        years = range(START_YEAR, LIABILITY_START_YEAR)
        contributions = {
            START_YEAR: initial_portfolio,
            START_YEAR + 1: second_year_contribution,
        }
    else:
        values = np.full(n_simulations, float(portfolio_value_at_2031), dtype=float)
        years = range(2031, LIABILITY_START_YEAR)
        contributions = {}

    for year in years:
        values += contributions.get(year, 0.0)  # beginning-of-year contributions
        if custom_schedule is None:
            allocation_object = calculate_glide_path(year)
            allocation = (
                allocation_object.equity,
                allocation_object.bonds,
                allocation_object.cash,
            )
        else:
            allocation = custom_schedule[year]
        asset_draws = generate_asset_returns(rng=rng, size=n_simulations, **assumptions)
        portfolio_returns = _portfolio_returns_from_asset_draws(asset_draws, allocation)
        values *= 1.0 + portfolio_returns
    return values


def _simulate_reserve_portfolio_returns(
    n_simulations: int,
    rng: np.random.Generator,
    assumptions: Mapping[str, float],
    reserve_allocation_schedule: Mapping[int, Sequence[float]] | None = None,
) -> np.ndarray:
    """Simulate reserve returns for 2033–2041 (the 2042 payment is final)."""
    allocations = _normalize_reserve_schedule(reserve_allocation_schedule)
    years = range(LIABILITY_START_YEAR, END_YEAR)
    values = np.empty((n_simulations, len(list(years))), dtype=float)
    for column, year in enumerate(years):
        draws = generate_asset_returns(rng=rng, size=n_simulations, **assumptions)
        values[:, column] = _portfolio_returns_from_asset_draws(draws, allocations[year])
    return values


def calculate_required_reserve_for_return_path(
    annual_portfolio_returns: Mapping[int, float],
    liabilities: Mapping[int, float] | None = None,
) -> float:
    """Calculate beginning-of-2033 reserve required for one future return path.

    With zero returns and ten $50,000 beginning-of-year payments, this returns
    exactly $500,000. Returns after the beginning-of-2042 payment are irrelevant.
    """
    schedule = _validate_liabilities(liabilities)
    required = schedule[END_YEAR]
    for year in range(END_YEAR - 1, LIABILITY_START_YEAR - 1, -1):
        annual_return = float(annual_portfolio_returns.get(year, 0.0))
        if not np.isfinite(annual_return) or annual_return <= -1.0:
            raise ValueError("Annual portfolio returns must be finite and greater than -100%.")
        required = schedule[year] + required / (1.0 + annual_return)
    return float(required)


def _required_reserve_distribution(
    reserve_returns: np.ndarray,
    liabilities: Mapping[int, float],
) -> np.ndarray:
    """Vectorized backward liability match for every simulated return path."""
    required = np.full(reserve_returns.shape[0], liabilities[END_YEAR], dtype=float)
    # Columns are returns during years 2033, 2034, ..., 2041.
    for year in range(END_YEAR - 1, LIABILITY_START_YEAR - 1, -1):
        column = year - LIABILITY_START_YEAR
        required = liabilities[year] + required / (1.0 + reserve_returns[:, column])
    return required


def _simulate_reserve_payments(
    starting_assets: np.ndarray,
    reserve_returns: np.ndarray,
    liabilities: Mapping[int, float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Simulate payments at the start of each year and returns afterward.

    The pool contains all assets designated to support the operating commitment
    under the policy being tested. Returns for 2033–2041 are applied after each
    payment; no return is needed after the final 2042 payment.
    """
    balances = np.asarray(starting_assets, dtype=float).copy()
    n_simulations = len(balances)
    year_count = END_YEAR - LIABILITY_START_YEAR + 1
    funded = np.ones((n_simulations, year_count), dtype=bool)
    shortfalls = np.zeros((n_simulations, year_count), dtype=float)
    year_values = np.zeros((n_simulations, year_count), dtype=float)

    for year in range(LIABILITY_START_YEAR, END_YEAR + 1):
        column = year - LIABILITY_START_YEAR
        liability = liabilities[year]
        year_values[:, column] = balances
        funded[:, column] = balances + 1e-9 >= liability
        shortfalls[:, column] = np.maximum(liability - balances, 0.0)
        balances = np.maximum(balances - liability, 0.0)
        if year < END_YEAR:
            balances *= 1.0 + reserve_returns[:, column]
    return funded, shortfalls, year_values


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
    liabilities: Mapping[int, float] | None = None,
    reserve_allocation_schedule: Mapping[int, Sequence[float]] | None = None,
) -> tuple[bool, pd.DataFrame]:
    """Simulate one whole portfolio path; retained for diagnostics and old tests.

    This low-level compatibility helper applies the dynamic glide-path weights
    (or a supplied reserve schedule after 2033) and reports the cash flow at
    each year. The authoritative policy comparisons and reserve statistics are
    calculated by ``_run_analysis_arrays`` and ``run_facility_contribution_sensitivity``.
    """
    _validate_assumptions(
        equity_return, bond_return, cash_return,
        equity_volatility, bond_volatility, cash_volatility, correlation,
    )
    if not np.isfinite(initial_portfolio) or initial_portfolio < 0:
        raise ValueError("initial_portfolio must be finite and non-negative.")
    if not np.isfinite(second_year_contribution) or second_year_contribution < 0:
        raise ValueError("second_year_contribution must be finite and non-negative.")
    schedule = _validate_liabilities(liabilities)
    custom_reserve_schedule = (
        _normalize_reserve_schedule(reserve_allocation_schedule)
        if reserve_allocation_schedule is not None else None
    )
    if rng is None:
        rng = np.random.default_rng()
    assumptions = {
        "equity_return": equity_return,
        "bond_return": bond_return,
        "cash_return": cash_return,
        "equity_volatility": equity_volatility,
        "bond_volatility": bond_volatility,
        "cash_volatility": cash_volatility,
        "correlation": correlation,
    }
    portfolio_value = 0.0
    rows: list[dict[str, object]] = []
    for year in range(START_YEAR, END_YEAR + 1):
        contribution = (
            initial_portfolio if year == START_YEAR
            else second_year_contribution if year == START_YEAR + 1
            else 0.0
        )
        beginning_value = portfolio_value
        portfolio_value += contribution
        value_before_liability = portfolio_value
        liability = schedule.get(year, 0.0)
        payment_made = min(portfolio_value, liability)
        shortfall = max(liability - portfolio_value, 0.0)
        portfolio_value = max(portfolio_value - liability, 0.0)
        funded = shortfall <= 1e-9
        value_before_return = portfolio_value

        if custom_reserve_schedule is not None and year >= LIABILITY_START_YEAR:
            allocation = custom_reserve_schedule[year]
        else:
            target = calculate_glide_path(year)
            allocation = (target.equity, target.bonds, target.cash)
        asset_returns = generate_asset_returns(rng=rng, **assumptions)
        portfolio_return = float(_portfolio_returns_from_asset_draws(
            np.asarray(asset_returns).reshape(1, 3), allocation
        )[0])
        investment_gain = portfolio_value * portfolio_return
        portfolio_value += investment_gain
        rows.append({
            "Year": year,
            "Beginning Value": beginning_value,
            "Contribution": contribution,
            "Value Before Return": value_before_return,
            "Portfolio Return": portfolio_return,
            "Investment Gain": investment_gain,
            "Value Before Liability": value_before_liability,
            "Liability": liability,
            "Payment Made": payment_made,
            "Shortfall": shortfall,
            "Funded": funded,
            "Ending Value": portfolio_value,
        })
    results = pd.DataFrame(rows)
    return bool(results["Funded"].all()), results


def _shortfall_statistics(shortfalls: np.ndarray) -> Dict[str, float]:
    """Summarize single-payment and cumulative shortfalls across simulations."""
    if shortfalls.size == 0:
        return {
            "Average Maximum Shortfall": 0.0,
            "Median Maximum Shortfall": 0.0,
            "Worst Shortfall": 0.0,
            "Probability of Any Shortfall": 0.0,
            "Average Cumulative Shortfall (All Simulations)": 0.0,
            "Average Cumulative Shortfall (Failed Simulations)": 0.0,
            "Median Cumulative Shortfall (Failed Simulations)": 0.0,
            "Worst Cumulative Shortfall": 0.0,
            "Average Number of Payments Missed": 0.0,
            "Average Number of Payments Missed (Failed Simulations)": 0.0,
        }

    maximum_shortfalls = shortfalls.max(axis=1)
    cumulative_shortfalls = shortfalls.sum(axis=1)
    failed_mask = cumulative_shortfalls > 0.0
    failed_maximum = maximum_shortfalls[failed_mask]
    failed_cumulative = cumulative_shortfalls[failed_mask]
    missed_count = (shortfalls > 0.0).sum(axis=1)
    failed_missed_count = missed_count[failed_mask]
    return {
        # These two statistics condition on a simulation having a failure.
        "Average Maximum Shortfall": float(failed_maximum.mean()) if failed_maximum.size else 0.0,
        "Median Maximum Shortfall": float(np.median(failed_maximum)) if failed_maximum.size else 0.0,
        "Worst Shortfall": float(maximum_shortfalls.max()) if maximum_shortfalls.size else 0.0,
        "Probability of Any Shortfall": float(failed_mask.mean()),
        # Cumulative amount includes every missed annual liability; first value includes zeros.
        "Average Cumulative Shortfall (All Simulations)": float(cumulative_shortfalls.mean()),
        "Average Cumulative Shortfall (Failed Simulations)": float(failed_cumulative.mean()) if failed_cumulative.size else 0.0,
        "Median Cumulative Shortfall (Failed Simulations)": float(np.median(failed_cumulative)) if failed_cumulative.size else 0.0,
        "Worst Cumulative Shortfall": float(cumulative_shortfalls.max()),
        "Average Number of Payments Missed": float(missed_count.mean()),
        "Average Number of Payments Missed (Failed Simulations)": float(failed_missed_count.mean()) if failed_missed_count.size else 0.0,
    }


def _run_analysis_arrays(
    n_simulations: int = N_SIMULATIONS,
    random_seed: int = RANDOM_SEED,
    initial_portfolio: float = INITIAL_PORTFOLIO,
    second_year_contribution: float = SECOND_YEAR_CONTRIBUTION,
    equity_return: float = EQUITY_RETURN,
    bond_return: float = BOND_RETURN,
    cash_return: float = CASH_RETURN,
    equity_volatility: float = EQUITY_VOLATILITY,
    bond_volatility: float = BOND_VOLATILITY,
    cash_volatility: float = CASH_VOLATILITY,
    correlation: float = CORRELATION,
    reserve_confidence_level: float = DEFAULT_RESERVE_CONFIDENCE_LEVEL,
    liabilities: Mapping[int, float] | None = None,
    portfolio_value_at_2031: float | None = None,
    accumulation_policy: str = DEFAULT_ACCUMULATION_POLICY,
    reserve_policy: str = DEFAULT_RESERVE_POLICY,
    accumulation_allocation_schedule: Mapping[int, Sequence[float]] | None = None,
    reserve_allocation_schedule: Mapping[int, Sequence[float]] | None = None,
) -> Dict[str, object]:
    """Run the core simulation once and return detailed arrays and metrics."""
    _validate_assumptions(
        equity_return, bond_return, cash_return,
        equity_volatility, bond_volatility, cash_volatility, correlation,
    )
    _validate_simulation_inputs(
        n_simulations, initial_portfolio, second_year_contribution,
        reserve_confidence_level, portfolio_value_at_2031,
    )
    schedule = _validate_liabilities(liabilities)
    accumulation_schedule = _resolve_accumulation_schedule(
        accumulation_policy, accumulation_allocation_schedule
    )
    reserve_schedule = _resolve_reserve_schedule(reserve_policy, reserve_allocation_schedule)
    assumptions = {
        "equity_return": equity_return,
        "bond_return": bond_return,
        "cash_return": cash_return,
        "equity_volatility": equity_volatility,
        "bond_volatility": bond_volatility,
        "cash_volatility": cash_volatility,
        "correlation": correlation,
    }

    # Separate reproducible random streams keep the pre-2033 accumulation,
    # reserve sizing, and future payment-return simulations independent.
    growth_seed, requirement_seed, payment_seed = np.random.SeedSequence(int(random_seed)).spawn(3)
    portfolio_values = _simulate_portfolio_values_at_2033(
        n_simulations, np.random.default_rng(growth_seed), initial_portfolio,
        second_year_contribution, assumptions,
        portfolio_value_at_2031=portfolio_value_at_2031,
        accumulation_allocation_schedule=accumulation_schedule,
    )
    requirement_returns = _simulate_reserve_portfolio_returns(
        n_simulations, np.random.default_rng(requirement_seed), assumptions, reserve_schedule,
    )
    required_reserve_amounts = _required_reserve_distribution(requirement_returns, schedule)
    reserve_target = float(np.quantile(
        required_reserve_amounts, reserve_confidence_level, method="higher"
    ))
    payment_returns = _simulate_reserve_portfolio_returns(
        n_simulations, np.random.default_rng(payment_seed), assumptions, reserve_schedule,
    )

    # Main no-facility policy: every portfolio dollar remains available to fund
    # operating payments. This is the appropriate no-facility benchmark, not a
    # guarantee that the target reserve is affordable in every path.
    no_facility_flags, no_facility_shortfalls, no_facility_year_values = _simulate_reserve_payments(
        portfolio_values, payment_returns, schedule,
    )
    no_facility_success = no_facility_flags.all(axis=1)

    # Diagnostic: ring-fenced target reserve only. Surplus assets above the
    # target are NOT available as a backup in this deliberately stricter variant.
    ring_fenced_start = np.minimum(portfolio_values, reserve_target)
    reserve_only_flags, reserve_only_shortfalls, reserve_only_year_values = _simulate_reserve_payments(
        ring_fenced_start, payment_returns, schedule,
    )
    reserve_only_success = reserve_only_flags.all(axis=1)

    reserve_affordable = portfolio_values >= reserve_target
    best_case_probability = float(no_facility_success.mean())
    capacity = np.maximum(portfolio_values - reserve_target, 0.0)
    low, median, high = np.quantile(
        capacity,
        [FACILITY_RANGE_LOWER_QUANTILE, 0.50, FACILITY_RANGE_UPPER_QUANTILE],
        method="linear",
    )
    within_range = (capacity >= low) & (capacity <= high)
    portfolio_quantiles = np.quantile(portfolio_values, [0.05, 0.50, 0.95])

    yearly_rows = []
    for year in range(LIABILITY_START_YEAR, END_YEAR + 1):
        liability = schedule[year]
        if liability <= 0:
            continue
        column = year - LIABILITY_START_YEAR
        probability = float(no_facility_flags[:, column].mean())
        yearly_rows.append({
            "Year": year,
            "Liability": float(liability),
            "Funding Probability": probability,
            "Funding Probability (%)": probability * 100.0,
        })

    return {
        "portfolio_values_2033": portfolio_values,
        "required_reserve_amounts": required_reserve_amounts,
        "reserve_target": reserve_target,
        "reserve_confidence_level": reserve_confidence_level,
        "reserve_success_probability_at_target": float(np.mean(required_reserve_amounts <= reserve_target + 1e-9)),
        "probability_reserve_target_affordable": float(reserve_affordable.mean()),
        "probability_all_payments_funded_no_facility": best_case_probability,
        "probability_all_payments_funded_ring_fenced_reserve_only": float(reserve_only_success.mean()),
        "yearly_funding_probability": pd.DataFrame(yearly_rows),
        "shortfall_statistics_no_facility": _shortfall_statistics(no_facility_shortfalls),
        "shortfall_statistics_ring_fenced_reserve_only": _shortfall_statistics(reserve_only_shortfalls),
        "facility_contribution_range": {
            "Lower Bound": float(low),
            "Median Capacity": float(median),
            "Upper Bound": float(high),
            "Predictive Range Coverage": float(within_range.mean()),
            "Predictive Range Confidence Level": FACILITY_RANGE_UPPER_QUANTILE - FACILITY_RANGE_LOWER_QUANTILE,
            "Lower Quantile": FACILITY_RANGE_LOWER_QUANTILE,
            "Upper Quantile": FACILITY_RANGE_UPPER_QUANTILE,
            "Probability of Zero Facility Capacity": float(np.mean(capacity <= 1e-9)),
        },
        "portfolio_2033_quantiles": {
            "5th Percentile": float(portfolio_quantiles[0]),
            "Median": float(portfolio_quantiles[1]),
            "95th Percentile": float(portfolio_quantiles[2]),
        },
        "facility_capacity_samples": capacity,
        "reserve_requirement_samples": required_reserve_amounts,
        "payment_flags_no_facility": no_facility_flags,
        "shortfalls_no_facility": no_facility_shortfalls,
        "reserve_values_no_facility": no_facility_year_values,
        "payment_flags_ring_fenced_reserve_only": reserve_only_flags,
        "shortfalls_ring_fenced_reserve_only": reserve_only_shortfalls,
        "reserve_values_ring_fenced_reserve_only": reserve_only_year_values,
        "portfolio_values": portfolio_values,
        "payment_returns": payment_returns,
        "liabilities": schedule,
        "accumulation_policy": accumulation_policy,
        "reserve_policy": reserve_policy,
        "accumulation_allocation_schedule": accumulation_schedule,
        "reserve_allocation_schedule": reserve_schedule,
    }


# ---------------------------------------------------------------------------
# 4. PUBLIC ANALYSIS FUNCTIONS
# ---------------------------------------------------------------------------


def run_financial_goal_analysis(**kwargs) -> Dict[str, object]:
    """Return the principal metrics for reserve sizing and payment funding.

    To forecast from a beginning-of-2031 portfolio value, pass
    ``portfolio_value_at_2031=...``. Otherwise, the model projects from the
    2027/2028 contributions and the result is a long-range planning estimate.
    """
    results = _run_analysis_arrays(**kwargs)
    return {
        "Recommended Operating Reserve": results["reserve_target"],
        "Reserve Confidence Target": results["reserve_confidence_level"],
        "Reserve Funding Probability at Target": results["reserve_success_probability_at_target"],
        "Probability Reserve Target Is Affordable": results["probability_reserve_target_affordable"],
        "Overall Funding Probability": results["probability_all_payments_funded_no_facility"],
        "Ring-Fenced Reserve-Only Funding Probability": results["probability_all_payments_funded_ring_fenced_reserve_only"],
        "Best-Case Payment Funding Probability (No Facility Contribution)": results["probability_all_payments_funded_no_facility"],
        "Yearly Funding Probability": results["yearly_funding_probability"],
        "Shortfall Statistics": results["shortfall_statistics_no_facility"],
        "Ring-Fenced Reserve Shortfall Statistics": results["shortfall_statistics_ring_fenced_reserve_only"],
        "Portfolio Value at Beginning of 2033": results["portfolio_2033_quantiles"],
        "Facility Contribution Range": results["facility_contribution_range"],
        "Reserve Allocation Schedule": generate_reserve_glide_path(
            results["liabilities"], results["reserve_allocation_schedule"]
        ),
    }


def calculate_funding_probability(
    n_simulations: int = N_SIMULATIONS,
    random_seed: int = RANDOM_SEED,
    **simulation_kwargs,
) -> float:
    """Estimate the chance all operating payments are funded with no facility contribution."""
    return float(_run_analysis_arrays(
        n_simulations=n_simulations, random_seed=random_seed, **simulation_kwargs
    )["probability_all_payments_funded_no_facility"])


def calculate_yearly_funding_probability(
    n_simulations: int = N_SIMULATIONS,
    random_seed: int = RANDOM_SEED,
    **simulation_kwargs,
) -> pd.DataFrame:
    """Estimate the chance each annual payment can be funded, without a facility contribution."""
    return _run_analysis_arrays(
        n_simulations=n_simulations, random_seed=random_seed, **simulation_kwargs
    )["yearly_funding_probability"]


def calculate_shortfall_statistics(
    n_simulations: int = N_SIMULATIONS,
    random_seed: int = RANDOM_SEED,
    **simulation_kwargs,
) -> Dict[str, float]:
    """Summarize annual and cumulative operating-payment shortfalls, no facility contribution."""
    return _run_analysis_arrays(
        n_simulations=n_simulations, random_seed=random_seed, **simulation_kwargs
    )["shortfall_statistics_no_facility"]


def run_facility_contribution_sensitivity(
    
    facility_contributions: Sequence[float] = (
        0.0,
        2_500.0,
        4_000.0,
        5_000.0,
        7_500.0,
        10_000.0,
        15_000.0,
        20_000.0,
        25_000.0,
        35_000.0,
        50_000.0,
        75_000.0,
        100_000.0,
    ),
    n_simulations: int = N_SIMULATIONS,
    random_seed: int = RANDOM_SEED,
    **simulation_kwargs,
) -> pd.DataFrame:
    """Measure both contribution affordability and payment success after a contribution policy.

    Policy tested: contribute the requested amount only when the beginning-of-
    2033 portfolio can first cover the entire reserve target plus that amount.
    Otherwise, contribute $0. All remaining assets stay available for operating
    payments and use the tested reserve allocation schedule.

    This is conditional, reserve-first funding. It is not a promise that the
    requested contribution will be paid in every market outcome.
    """
    results = _run_analysis_arrays(
        n_simulations=n_simulations, random_seed=random_seed, **simulation_kwargs
    )
    portfolio_values = results["portfolio_values"]
    reserve_target = float(results["reserve_target"])
    payment_returns = results["payment_returns"]
    liabilities = results["liabilities"]
    rows = []

    for raw_contribution in facility_contributions:
        contribution = float(raw_contribution)
        if not np.isfinite(contribution) or contribution < 0:
            raise ValueError("Facility contributions must be finite and non-negative.")
        affordable_mask = portfolio_values >= reserve_target + contribution - 1e-9
        actual_contributions = np.where(affordable_mask, contribution, 0.0)
        operating_pool_after_contribution = np.maximum(portfolio_values - actual_contributions, 0.0)
        flags, shortfalls, _ = _simulate_reserve_payments(
            operating_pool_after_contribution, payment_returns, liabilities,
        )
        stats = _shortfall_statistics(shortfalls)
        rows.append({
            "Facility Contribution Requested (2033)": contribution,
            "Probability Reserve + Requested Contribution Are Affordable": float(affordable_mask.mean()),
            "Probability Full Requested Contribution Is Made": float(1.0 if contribution == 0.0 else affordable_mask.mean()),
            "Probability No Facility Contribution Is Made": float((~affordable_mask).mean()) if contribution > 0 else 0.0,
            "Mean Actual Facility Contribution": float(actual_contributions.mean()),
            "Median Actual Facility Contribution": float(np.median(actual_contributions)),
            "Payment Funding Probability After Policy": float(flags.all(axis=1).mean()),
            "Probability of Any Payment Shortfall": stats["Probability of Any Shortfall"],
            "Average Cumulative Shortfall (All Simulations)": stats["Average Cumulative Shortfall (All Simulations)"],
            "Worst Cumulative Shortfall": stats["Worst Cumulative Shortfall"],
        })
    return pd.DataFrame(rows)


def run_sensitivity_analysis(
    n_simulations: int = N_SIMULATIONS,
    random_seed: int = RANDOM_SEED,
    portfolio_value_at_2031: float | None = None,
    accumulation_policy: str = DEFAULT_ACCUMULATION_POLICY,
    reserve_policy: str = DEFAULT_RESERVE_POLICY,
) -> pd.DataFrame:
    """Change one assumption at a time and recalculate the reserve target."""
    rows: list[dict[str, object]] = []
    base_schedules = {
        "Annual Liability": [40_000, 45_000, 50_000, 55_000, 60_000],
        "Equity Return": [0.04, 0.06, 0.08, 0.10, 0.12],
        "Bond Return": [0.02, 0.03, 0.04, 0.05, 0.06],
        "Equity Volatility": [0.12, 0.15, 0.18, 0.21, 0.24],
    }
    for value in base_schedules["Annual Liability"]:
        liabilities = {year: float(value) for year in LIABILITIES}
        funding_probability = calculate_funding_probability(
            n_simulations=n_simulations, random_seed=random_seed, liabilities=liabilities,
            portfolio_value_at_2031=portfolio_value_at_2031,
            accumulation_policy=accumulation_policy, reserve_policy=reserve_policy,
        )
        result = _run_analysis_arrays(
            n_simulations=n_simulations, random_seed=random_seed, liabilities=liabilities,
            portfolio_value_at_2031=portfolio_value_at_2031,
            accumulation_policy=accumulation_policy, reserve_policy=reserve_policy,
        )
        rows.append({
            "Variable": "Annual Liability", "Value": value,
            "Funding Probability": funding_probability,
            "Funding Probability (%)": funding_probability * 100.0,
            "Recommended Reserve": result["reserve_target"],
            "Reserve Affordable Probability": result["probability_reserve_target_affordable"],
        })
    for name, parameter, values in (
        ("Equity Return", "equity_return", base_schedules["Equity Return"]),
        ("Bond Return", "bond_return", base_schedules["Bond Return"]),
        ("Equity Volatility", "equity_volatility", base_schedules["Equity Volatility"]),
    ):
        for value in values:
            funding_probability = calculate_funding_probability(
                n_simulations=n_simulations, random_seed=random_seed,
                portfolio_value_at_2031=portfolio_value_at_2031,
                accumulation_policy=accumulation_policy, reserve_policy=reserve_policy,
                **{parameter: value},
            )
            result = _run_analysis_arrays(
                n_simulations=n_simulations, random_seed=random_seed,
                portfolio_value_at_2031=portfolio_value_at_2031,
                accumulation_policy=accumulation_policy, reserve_policy=reserve_policy,
                **{parameter: value},
            )
            rows.append({
                "Variable": name, "Value": value,
                "Funding Probability": funding_probability,
                "Funding Probability (%)": funding_probability * 100.0,
                "Recommended Reserve": result["reserve_target"],
                "Reserve Affordable Probability": result["probability_reserve_target_affordable"],
            })
    return pd.DataFrame(rows)


def run_two_variable_sensitivity(
    n_simulations: int = N_SIMULATIONS,
    random_seed: int = RANDOM_SEED,
    portfolio_value_at_2031: float | None = None,
    accumulation_policy: str = DEFAULT_ACCUMULATION_POLICY,
    reserve_policy: str = DEFAULT_RESERVE_POLICY,
) -> pd.DataFrame:
    """Test combined changes in equity return and fixed annual operating liability."""
    rows: list[dict[str, object]] = []
    for equity_return in (0.06, 0.08, 0.10):
        for annual_liability in (40_000, 50_000, 60_000):
            liabilities = {year: float(annual_liability) for year in LIABILITIES}
            probability = calculate_funding_probability(
                n_simulations=n_simulations, random_seed=random_seed,
                equity_return=equity_return, liabilities=liabilities,
                portfolio_value_at_2031=portfolio_value_at_2031,
                accumulation_policy=accumulation_policy, reserve_policy=reserve_policy,
            )
            rows.append({
                "Equity Return": equity_return,
                "Annual Liability": annual_liability,
                "Funding Probability": probability,
                "Funding Probability (%)": probability * 100.0,
            })
    return pd.DataFrame(rows)


def run_scenario_analysis(
    n_simulations: int = N_SIMULATIONS,
    random_seed: int = RANDOM_SEED,
    portfolio_value_at_2031: float | None = None,
    accumulation_policy: str = DEFAULT_ACCUMULATION_POLICY,
    reserve_policy: str = DEFAULT_RESERVE_POLICY,
) -> pd.DataFrame:
    """Compare illustrative optimistic, base, and pessimistic return assumptions."""
    scenarios = {
        "Optimistic": {"equity_return": 0.10, "bond_return": 0.05, "equity_volatility": 0.16},
        "Base": {"equity_return": 0.08, "bond_return": 0.04, "equity_volatility": 0.18},
        "Pessimistic": {"equity_return": 0.06, "bond_return": 0.03, "equity_volatility": 0.22},
    }
    rows: list[dict[str, object]] = []
    for name, assumptions in scenarios.items():
        result = _run_analysis_arrays(
            n_simulations=n_simulations, random_seed=random_seed,
            portfolio_value_at_2031=portfolio_value_at_2031,
            accumulation_policy=accumulation_policy, reserve_policy=reserve_policy,
            **assumptions,
        )
        stats = result["shortfall_statistics_no_facility"]
        rows.append({
            "Scenario": name,
            "Equity Return": assumptions["equity_return"],
            "Bond Return": assumptions["bond_return"],
            "Equity Volatility": assumptions["equity_volatility"],
            "Funding Probability": result["probability_all_payments_funded_no_facility"],
            "Funding Probability (%)": result["probability_all_payments_funded_no_facility"] * 100.0,
            "Average Cumulative Shortfall": stats["Average Cumulative Shortfall (All Simulations)"],
            "Worst Cumulative Shortfall": stats["Worst Cumulative Shortfall"],
            "Recommended Operating Reserve": result["reserve_target"],
            "Reserve Affordable Probability": result["probability_reserve_target_affordable"],
        })
    return pd.DataFrame(rows)


def run_accumulation_policy_comparison(
    n_simulations: int = N_SIMULATIONS,
    random_seed: int = RANDOM_SEED,
    portfolio_value_at_2031: float | None = None,
    reserve_policy: str = DEFAULT_RESERVE_POLICY,
) -> pd.DataFrame:
    """Compare alternative 2027–2032 allocation policies under one return model.

    This is a model-based comparison, not a mathematically exhaustive optimizer.
    Static policy rows are illustrative choices; they must also be judged against
    the team's implementation constraints and a historical-return cross-check.
    """
    rows: list[dict[str, object]] = []
    for name, allocation in ACCUMULATION_POLICY_ALTERNATIVES.items():
        result = _run_analysis_arrays(
            n_simulations=n_simulations, random_seed=random_seed,
            portfolio_value_at_2031=portfolio_value_at_2031,
            accumulation_policy=name, reserve_policy=reserve_policy,
        )
        rows.append({
            "Accumulation Policy": name,
            "No-Facility Payment Funding Probability": result["probability_all_payments_funded_no_facility"],
            "Reserve Affordable Probability": result["probability_reserve_target_affordable"],
            "Recommended Operating Reserve": result["reserve_target"],
            "2033 Portfolio Median": result["portfolio_2033_quantiles"]["Median"],
            "2033 Portfolio 5th Percentile": result["portfolio_2033_quantiles"]["5th Percentile"],
        })
    return pd.DataFrame(rows).sort_values(
        "No-Facility Payment Funding Probability", ascending=False
    ).reset_index(drop=True)


def run_reserve_policy_comparison(
    n_simulations: int = N_SIMULATIONS,
    random_seed: int = RANDOM_SEED,
    portfolio_value_at_2031: float | None = None,
    accumulation_policy: str = DEFAULT_ACCUMULATION_POLICY,
    reserve_policy: str = DEFAULT_RESERVE_POLICY,
) -> pd.DataFrame:
    """Compare proposed, bond/cash, and all-cash reserve mixes."""
    rows: list[dict[str, object]] = []
    for name, static_allocation in RESERVE_POLICY_ALTERNATIVES.items():
        result = _run_analysis_arrays(
            n_simulations=n_simulations, random_seed=random_seed,
            portfolio_value_at_2031=portfolio_value_at_2031,
            accumulation_policy=accumulation_policy,
            reserve_policy=name,
        )
        rows.append({
            "Reserve Policy": name,
            "No-Facility Payment Funding Probability": result["probability_all_payments_funded_no_facility"],
            "Reserve Target": result["reserve_target"],
            "Reserve Target Affordable Probability": result["probability_reserve_target_affordable"],
            "Median 2033 Portfolio": result["portfolio_2033_quantiles"]["Median"],
        })
    return pd.DataFrame(rows).sort_values(
        "No-Facility Payment Funding Probability", ascending=False
    ).reset_index(drop=True)


def generate_funding_summary(
    n_simulations: int = N_SIMULATIONS,
    random_seed: int = RANDOM_SEED,
    include_comparisons: bool = True,
    accumulation_policy: str = DEFAULT_ACCUMULATION_POLICY,
    reserve_policy: str = DEFAULT_RESERVE_POLICY,
    **simulation_kwargs,
) -> Dict[str, object]:
    """Generate the main funding results and all requested comparison tables."""
    results = _run_analysis_arrays(
        n_simulations=n_simulations, random_seed=random_seed,
        accumulation_policy=accumulation_policy, reserve_policy=reserve_policy,
        **simulation_kwargs
    )
    portfolio_value_at_2031 = simulation_kwargs.get("portfolio_value_at_2031")
    summary: Dict[str, object] = {
        "Funding Probability": results["probability_all_payments_funded_no_facility"],
        "Best-Case Payment Funding Probability (No Facility Contribution)": results["probability_all_payments_funded_no_facility"],
        "Ring-Fenced Reserve-Only Funding Probability": results["probability_all_payments_funded_ring_fenced_reserve_only"],
        "Yearly Funding Probability": results["yearly_funding_probability"],
        "Shortfall Statistics": results["shortfall_statistics_no_facility"],
        "Ring-Fenced Reserve Shortfall Statistics": results["shortfall_statistics_ring_fenced_reserve_only"],
        "Operating Reserve Analysis": {
            "Recommended Operating Reserve": results["reserve_target"],
            "Reserve Confidence Target": results["reserve_confidence_level"],
            "Reserve Funding Probability at Target": results["reserve_success_probability_at_target"],
            "Probability Reserve Target Is Affordable": results["probability_reserve_target_affordable"],
            "Portfolio Value at Beginning of 2033": results["portfolio_2033_quantiles"],
        },
        "Facility Contribution Range": results["facility_contribution_range"],
        "Reserve Allocation Schedule": generate_reserve_glide_path(
            results["liabilities"], results["reserve_allocation_schedule"]
        ),
        "Facility Contribution Affordability": run_facility_contribution_sensitivity(
            n_simulations=n_simulations, random_seed=random_seed,
            accumulation_policy=accumulation_policy, reserve_policy=reserve_policy,
            **simulation_kwargs
        ),
        "Accumulation Policy": accumulation_policy,
        "Reserve Policy": reserve_policy,
        "Forecast Type": (
            "2031 as-of forecast" if portfolio_value_at_2031 is not None
            else "long-range planning projection from the 2027/2028 contributions"
        ),
    }
    if include_comparisons:
        summary.update({
            "Sensitivity Analysis": run_sensitivity_analysis(
                n_simulations, random_seed, portfolio_value_at_2031=portfolio_value_at_2031,
                accumulation_policy=accumulation_policy, reserve_policy=reserve_policy,
            ),
            "Two Variable Sensitivity": run_two_variable_sensitivity(
                n_simulations, random_seed, portfolio_value_at_2031=portfolio_value_at_2031,
                accumulation_policy=accumulation_policy, reserve_policy=reserve_policy,
            ),
            "Scenario Analysis": run_scenario_analysis(
                n_simulations, random_seed, portfolio_value_at_2031=portfolio_value_at_2031,
                accumulation_policy=accumulation_policy, reserve_policy=reserve_policy,
            ),
            "Accumulation Policy Comparison": run_accumulation_policy_comparison(
                n_simulations, random_seed, portfolio_value_at_2031=portfolio_value_at_2031,
                reserve_policy=reserve_policy,
            ),
            "Reserve Policy Comparison": run_reserve_policy_comparison(
                n_simulations, random_seed, portfolio_value_at_2031=portfolio_value_at_2031,
                accumulation_policy=accumulation_policy,
            ),
        })
    return summary


# ---------------------------------------------------------------------------
# 5. COMMAND-LINE REPORT
# ---------------------------------------------------------------------------


def _money(value: float) -> str:
    return f"${value:,.0f}"


def _percentage(value: float) -> str:
    return f"{value:.1%}"


def _print_table(
    table: pd.DataFrame,
    percentage_columns: Sequence[str] = (),
    money_columns: Sequence[str] = (),
    float_columns: Sequence[str] = (),
) -> None:
    """Print a presentation-ready copy of a DataFrame without changing its data."""
    display = table.copy()
    for column in percentage_columns:
        if column in display.columns:
            display[column] = display[column].map(
                lambda value: _percentage(float(value)) if pd.notna(value) else ""
            )
    for column in money_columns:
        if column in display.columns:
            display[column] = display[column].map(
                lambda value: _money(float(value)) if pd.notna(value) else ""
            )
    for column in float_columns:
        if column in display.columns:
            display[column] = display[column].map(
                lambda value: f"{float(value):.2f}" if pd.notna(value) else ""
            )
    print(display.to_string(index=False))


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Monte Carlo operating-reserve and facility-contribution analysis for Laura Gao."
    )
    parser.add_argument("--simulations", type=int, default=N_SIMULATIONS,
                        help=f"simulation paths per scenario (default: {N_SIMULATIONS})")
    parser.add_argument("--seed", type=int, default=RANDOM_SEED,
                        help=f"random seed for reproducible results (default: {RANDOM_SEED})")
    parser.add_argument("--portfolio-value-2031", type=float, default=None,
                        help="portfolio value at the BEGINNING of 2031; when supplied, only 2031 and 2032 are projected")
    parser.add_argument("--accumulation-policy", choices=list(ACCUMULATION_POLICY_ALTERNATIVES),
                        default=DEFAULT_ACCUMULATION_POLICY,
                        help="pre-2033 asset allocation policy to use for the main report")
    parser.add_argument("--reserve-policy", choices=list(RESERVE_POLICY_ALTERNATIVES),
                        default=DEFAULT_RESERVE_POLICY,
                        help="operating-reserve allocation policy to use for the main report")
    parser.add_argument("--skip-comparisons", action="store_true",
                        help="skip the sensitivity and allocation-comparison tables for a quicker main report")
    args = parser.parse_args(argv)

    if args.simulations <= 0:
        parser.error("--simulations must be a positive integer")

    report = generate_funding_summary(
        n_simulations=args.simulations,
        random_seed=args.seed,
        include_comparisons=not args.skip_comparisons,
        accumulation_policy=args.accumulation_policy,
        reserve_policy=args.reserve_policy,
        portfolio_value_at_2031=args.portfolio_value_2031,
    )
    reserve = report["Operating Reserve Analysis"]
    facility = report["Facility Contribution Range"]
    portfolio_distribution = reserve["Portfolio Value at Beginning of 2033"]

    print("=" * 84)
    print("MUSTANG WEALTH MANAGEMENT")
    print("LAURA GAO: OPERATING RESERVE AND FACILITY CONTRIBUTION ANALYSIS")
    print("=" * 84)
    print(f"Forecast type: {report['Forecast Type']}")
    print(f"Accumulation policy: {report['Accumulation Policy']}")
    print(f"Reserve policy: {report['Reserve Policy']}")
    print(f"Simulation paths per scenario: {args.simulations:,}")
    print(f"Random seed: {args.seed}")
    print("\nCASE-STUDY CASH FLOWS AND MODEL ASSUMPTIONS")
    print("-" * 84)
    print(f"Initial investment, beginning of 2027: {_money(INITIAL_PORTFOLIO)}")
    print(f"Additional investment, beginning of 2028: {_money(SECOND_YEAR_CONTRIBUTION)}")
    print("Operating payments: $50,000 at the beginning of each year, 2033–2042")
    print(f"Equity return / volatility assumption: {EQUITY_RETURN:.1%} / {EQUITY_VOLATILITY:.1%}")
    print(f"Bond return / volatility assumption: {BOND_RETURN:.1%} / {BOND_VOLATILITY:.1%}")
    print(f"Cash return / volatility assumption: {CASH_RETURN:.1%} / {CASH_VOLATILITY:.1%}")
    print(f"Common pairwise asset-class correlation assumption: {CORRELATION:.2f}")
    print("Returns are simulated asset-class assumptions, not direct returns for your sample stocks.")

    print("\nOPERATING RESERVE ANALYSIS")
    print("-" * 84)
    print(f"Recommended reserve at beginning of 2033: {_money(reserve['Recommended Operating Reserve'])}")
    print(f"Reserve confidence target: {_percentage(reserve['Reserve Confidence Target'])}")
    print(f"Empirical reserve sufficiency at target: {_percentage(reserve['Reserve Funding Probability at Target'])}")
    print(f"Probability full reserve target is affordable: {_percentage(reserve['Probability Reserve Target Is Affordable'])}")
    print("Reserve sufficiency is conditional on having the full reserve amount available.")

    print("\nPROJECTED PORTFOLIO VALUE AT BEGINNING OF 2033, BEFORE FIRST PAYMENT")
    print("-" * 84)
    for label, value in portfolio_distribution.items():
        print(f"{label}: {_money(value)}")

    print("\nPAYMENT-FUNDING RESULTS")
    print("-" * 84)
    print(f"All payments funded, no facility contribution (all remaining assets available): {_percentage(report['Funding Probability'])}")
    print(f"Ring-fenced target-reserve-only diagnostic (surplus assets cannot backstop reserve): {_percentage(report['Ring-Fenced Reserve-Only Funding Probability'])}")
    print("The first is the primary no-facility benchmark; the second is a stricter diagnostic policy.")
    print("\nYear-by-year payment funding probability, no facility contribution:")
    yearly_display = report["Yearly Funding Probability"][["Year", "Liability", "Funding Probability"]].copy()
    yearly_display["Liability"] = yearly_display["Liability"].map(_money)
    yearly_display["Funding Probability"] = yearly_display["Funding Probability"].map(_percentage)
    print(yearly_display.to_string(index=False))

    print("\nSHORTFALL ANALYSIS, NO FACILITY CONTRIBUTION")
    print("-" * 84)
    for label, value in report["Shortfall Statistics"].items():
        if "Probability" in label:
            print(f"{label}: {_percentage(value)}")
        elif "Number of Payments" in label:
            print(f"{label}: {value:.2f}")
        else:
            print(f"{label}: {_money(value)}")

    print("\nFACILITY CONTRIBUTION CAPACITY BEFORE A FACILITY DECISION")
    print("-" * 84)
    print(
        f"Nominal {facility['Predictive Range Confidence Level']:.0%} quantile interval "
        f"({facility['Lower Quantile']:.0%}–{facility['Upper Quantile']:.0%} quantiles): "
        f"{_money(facility['Lower Bound'])} to {_money(facility['Upper Bound'])}"
    )
    print(f"Median capacity after target reserve: {_money(facility['Median Capacity'])}")
    print(f"Probability contribution capacity is $0: {_percentage(facility['Probability of Zero Facility Capacity'])}")
    print(f"Empirical inclusive interval coverage: {_percentage(facility['Predictive Range Coverage'])}")
    print("A zero lower bound does not support promising co-sponsors a positive minimum contribution at this range level.")

    print("\nRESERVE ALLOCATION SCHEDULE")
    print("-" * 84)
    reserve_table = report["Reserve Allocation Schedule"].copy()
    for column in ("Equity", "Bonds", "Cash"):
        reserve_table[column] *= 100.0
        reserve_table[column] = reserve_table[column].map(lambda value: f"{value:.0f}%")
    reserve_table["Liability"] = reserve_table["Liability"].map(_money)
    print(reserve_table.to_string(index=False))

    print("\nFACILITY CONTRIBUTION POLICY COMPARISON")
    print("-" * 84)
    contribution_table = report["Facility Contribution Affordability"].copy()
    _print_table(
        contribution_table,
        percentage_columns=(
            "Probability Reserve + Requested Contribution Are Affordable",
            "Probability Full Requested Contribution Is Made",
            "Probability No Facility Contribution Is Made",
            "Payment Funding Probability After Policy",
            "Probability of Any Payment Shortfall",
        ),
        money_columns=(
            "Facility Contribution Requested (2033)",
            "Mean Actual Facility Contribution",
            "Median Actual Facility Contribution",
            "Average Cumulative Shortfall (All Simulations)",
            "Worst Cumulative Shortfall",
        ),
    )
    print("Conditional policy: contribute only after the reserve target plus requested contribution is affordable.")
    print("All assets remaining after any contribution remain available to support operating payments in this model.")

    if not args.skip_comparisons:
        print("\nACCUMULATION POLICY COMPARISON (ILLUSTRATIVE, NOT AN EXHAUSTIVE OPTIMIZER)")
        print("-" * 84)
        _print_table(
            report["Accumulation Policy Comparison"],
            percentage_columns=("No-Facility Payment Funding Probability", "Reserve Affordable Probability"),
            money_columns=("Recommended Operating Reserve", "2033 Portfolio Median", "2033 Portfolio 5th Percentile"),
        )
        print("The top row has the highest modeled no-facility funding probability among the listed candidate policies only.")

        print("\nRESERVE POLICY COMPARISON")
        print("-" * 84)
        _print_table(
            report["Reserve Policy Comparison"],
            percentage_columns=("No-Facility Payment Funding Probability", "Reserve Target Affordable Probability"),
            money_columns=("Reserve Target", "Median 2033 Portfolio"),
        )

        print("\nONE-VARIABLE SENSITIVITY ANALYSIS")
        print("-" * 84)
        sensitivity = report["Sensitivity Analysis"].copy()
        sensitivity["Value"] = [
            _money(value) if variable == "Annual Liability" else f"{value:.1%}"
            for variable, value in zip(sensitivity["Variable"], sensitivity["Value"])
        ]
        _print_table(
            sensitivity[["Variable", "Value", "Funding Probability", "Recommended Reserve", "Reserve Affordable Probability"]],
            percentage_columns=("Funding Probability", "Reserve Affordable Probability"),
            money_columns=("Recommended Reserve",),
        )

        print("\nTWO-VARIABLE SENSITIVITY: EQUITY RETURN × ANNUAL LIABILITY")
        print("-" * 84)
        two_variable_display = report["Two Variable Sensitivity"][[
            "Equity Return", "Annual Liability", "Funding Probability"
        ]].copy()
        _print_table(
            two_variable_display,
            percentage_columns=("Equity Return", "Funding Probability"),
            money_columns=("Annual Liability",),
        )

        print("\nOPTIMISTIC / BASE / PESSIMISTIC RETURN SCENARIOS")
        print("-" * 84)
        scenario_display = report["Scenario Analysis"][ [
            "Scenario", "Equity Return", "Bond Return", "Equity Volatility",
            "Funding Probability", "Average Cumulative Shortfall", "Worst Cumulative Shortfall",
            "Recommended Operating Reserve", "Reserve Affordable Probability"
        ]].copy()
        _print_table(
            scenario_display,
            percentage_columns=("Equity Return", "Bond Return", "Equity Volatility", "Funding Probability", "Reserve Affordable Probability"),
            money_columns=("Average Cumulative Shortfall", "Worst Cumulative Shortfall", "Recommended Operating Reserve"),
        )

    print("\nINTERPRETATION AND LIMITATIONS")
    print("-" * 84)
    print("1. The model uses assumed equity/bond/cash distributions; it is not a direct sample-stock simulation.")
    print("2. The 95% reserve target is conditional and does not ensure that the portfolio can afford that reserve.")
    print("3. The facility policy tested is conditional: no contribution is made unless the target reserve plus that amount is affordable.")
    print("4. The contribution interval is based on projected 2033 assets and is not a final 2031 co-sponsor forecast unless a 2031 value is supplied.")
    print("5. Compare these normal-return results with the existing historical block-bootstrap and stress-testing models.")
    print("6. Higher funding probability under a candidate policy is not, by itself, proof that the policy is suitable or implementable.")
    print("\n" + "=" * 84)
    print("ANALYSIS COMPLETE")
    print("=" * 84)


if __name__ == "__main__":
    main()

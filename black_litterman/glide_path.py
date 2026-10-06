"""
Liability-Driven Investing (LDI) and Dynamic Glide Path.

This module gradually reduces portfolio risk as the portfolio approaches
the 2033 liability deadline and then focuses on protecting assets needed
for annual $50,000 liabilities through 2042.

The glide path is designed to complement the Black-Litterman model:
Black-Litterman determines the portfolio's growth-oriented allocation,
while the glide path determines how that allocation should become more
conservative as liabilities approach.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


START_YEAR = 2027
LIABILITY_START_YEAR = 2033
END_YEAR = 2042

ANNUAL_LIABILITY = 50_000.0


# Ten annual $50,000 payments from 2033 through 2042.
LIABILITIES = {
    year: ANNUAL_LIABILITY
    for year in range(LIABILITY_START_YEAR, END_YEAR + 1)
}


@dataclass
class GlidePathAllocation:
    """Target portfolio allocation for a given year."""

    year: int
    equity: float
    bonds: float
    cash: float

    @property
    def total(self) -> float:
        """Return the total portfolio allocation."""
        return self.equity + self.bonds + self.cash


def validate_year(year: int) -> None:
    """Validate that a year falls within the modeled period."""

    if year < START_YEAR or year > END_YEAR:
        raise ValueError(
            f"Year must be between {START_YEAR} and {END_YEAR}. "
            f"Received {year}."
        )


def get_liability(year: int) -> float:
    """
    Return the liability due in a given year.

    Years before 2033 have no scheduled liability.
    """

    validate_year(year)

    return LIABILITIES.get(year, 0.0)


def get_remaining_liabilities(year: int) -> pd.Series:
    """
    Return all liabilities from the given year through 2042.
    """

    validate_year(year)

    remaining = {
        liability_year: amount
        for liability_year, amount in LIABILITIES.items()
        if liability_year >= year
    }

    return pd.Series(
        remaining,
        dtype=float,
        name="Liability",
    )


def calculate_years_to_first_liability(year: int) -> int:
    """
    Calculate the number of years until the first liability.

    Before 2033, this is the number of years until 2033.
    From 2033 onward, the value is zero.
    """

    validate_year(year)

    return max(LIABILITY_START_YEAR - year, 0)


def calculate_glide_path(year: int) -> GlidePathAllocation:
    """
    Calculate the target allocation for a given year.

    The portfolio becomes progressively more conservative as 2033
    approaches.

    Allocation schedule:

        2027: 70% equity / 25% bonds / 5% cash
        2028: 68% equity / 27% bonds / 5% cash
        2029: 65% equity / 30% bonds / 5% cash
        2030: 60% equity / 35% bonds / 5% cash
        2031: 50% equity / 43% bonds / 7% cash
        2032: 35% equity / 55% bonds / 10% cash
        2033: 20% equity / 65% bonds / 15% cash

    After 2033, equity exposure gradually declines while the portfolio
    becomes increasingly focused on bonds and cash to support upcoming
    liabilities.
    """

    validate_year(year)

    if year <= 2030:
        allocations = {
            2027: (0.70, 0.25, 0.05),
            2028: (0.68, 0.27, 0.05),
            2029: (0.65, 0.30, 0.05),
            2030: (0.60, 0.35, 0.05),
        }

        equity, bonds, cash = allocations[year]

    elif year == 2031:
        equity = 0.50
        bonds = 0.43
        cash = 0.07

    elif year == 2032:
        equity = 0.35
        bonds = 0.55
        cash = 0.10

    else:
        # After the first liability begins, continue reducing equity
        # exposure by 2 percentage points per year.
        years_after_2033 = year - LIABILITY_START_YEAR

        equity = max(
            0.20 - 0.02 * years_after_2033,
            0.02,
        )

        # Keep a larger cash reserve as liabilities become closer.
        cash = min(
            0.15 + 0.01 * years_after_2033,
            0.25,
        )

        bonds = 1.0 - equity - cash

    allocation = GlidePathAllocation(
        year=year,
        equity=equity,
        bonds=bonds,
        cash=cash,
    )

    if abs(allocation.total - 1.0) > 1e-9:
        raise ValueError(
            f"Allocation for {year} does not sum to 100%."
        )

    return allocation


def generate_glide_path(
    start_year: int = START_YEAR,
    end_year: int = END_YEAR,
) -> pd.DataFrame:
    """
    Generate the complete dynamic glide path.

    Returns
    -------
    pandas.DataFrame
        One row per year containing target equity, bond, and cash
        allocations along with liability information.
    """

    if start_year < START_YEAR:
        raise ValueError(
            f"start_year cannot be earlier than {START_YEAR}."
        )

    if end_year > END_YEAR:
        raise ValueError(
            f"end_year cannot be later than {END_YEAR}."
        )

    if start_year > end_year:
        raise ValueError(
            "start_year must be less than or equal to end_year."
        )

    rows = []

    for year in range(start_year, end_year + 1):

        allocation = calculate_glide_path(year)

        liability = get_liability(year)

        years_to_liability = calculate_years_to_first_liability(
            year
        )

        rows.append(
            {
                "Year": year,
                "Equity": allocation.equity,
                "Bonds": allocation.bonds,
                "Cash": allocation.cash,
                "Liability": liability,
                "Years to First Liability": years_to_liability,
            }
        )

    return pd.DataFrame(rows)


def calculate_liability_coverage(
    portfolio_value: float,
    protected_assets: float,
    year: int,
) -> float:
    """
    Calculate the percentage of the year's liability covered
    by protected assets.

    Example
    -------
    A $65,000 protected portfolio against a $50,000 liability
    produces a 130% coverage ratio.
    """

    validate_year(year)

    if portfolio_value < 0:
        raise ValueError(
            "portfolio_value cannot be negative."
        )

    if protected_assets < 0:
        raise ValueError(
            "protected_assets cannot be negative."
        )

    liability = get_liability(year)

    if liability == 0:
        return float("inf")

    return protected_assets / liability


def calculate_required_protected_assets(
    year: int,
    coverage_target: float = 1.0,
) -> float:
    """
    Calculate the amount of protected assets required to cover
    a year's liability.

    A coverage target of 1.0 means 100% coverage.

    A coverage target of 1.2 means 120% coverage.
    """

    validate_year(year)

    if coverage_target <= 0:
        raise ValueError(
            "coverage_target must be greater than zero."
        )

    liability = get_liability(year)

    return liability * coverage_target


def calculate_cumulative_liabilities(
    start_year: int = LIABILITY_START_YEAR,
    end_year: int = END_YEAR,
) -> float:
    """
    Calculate the total liabilities over a specified period.
    """

    if start_year < LIABILITY_START_YEAR:
        raise ValueError(
            f"start_year cannot be earlier than "
            f"{LIABILITY_START_YEAR}."
        )

    if end_year > END_YEAR:
        raise ValueError(
            f"end_year cannot be later than {END_YEAR}."
        )

    if start_year > end_year:
        raise ValueError(
            "start_year must be less than or equal to end_year."
        )

    return sum(
        LIABILITIES[year]
        for year in range(start_year, end_year + 1)
    )


def get_liability_schedule() -> pd.DataFrame:
    """
    Return the complete liability schedule.

    Includes the annual liability and cumulative remaining liabilities.
    """

    rows = []

    cumulative = 0.0

    for year in range(LIABILITY_START_YEAR, END_YEAR + 1):

        liability = LIABILITIES[year]

        cumulative += liability

        rows.append(
            {
                "Year": year,
                "Annual Liability": liability,
                "Cumulative Liabilities": cumulative,
                "Remaining Liabilities": (
                    calculate_cumulative_liabilities(
                        year,
                        END_YEAR,
                    )
                ),
            }
        )

    return pd.DataFrame(rows)

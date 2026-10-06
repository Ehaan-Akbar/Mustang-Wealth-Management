import numpy as np

from black_litterman.glide_path import (
    END_YEAR,
    LIABILITIES,
    LIABILITY_START_YEAR,
    START_YEAR,
    calculate_cumulative_liabilities,
    calculate_glide_path,
    calculate_liability_coverage,
    calculate_required_protected_assets,
    generate_glide_path,
    get_liability,
    get_liability_schedule,
)


def test_liability_schedule():
    """There should be ten $50,000 liabilities from 2033-2042."""

    assert len(LIABILITIES) == 10

    assert all(
        amount == 50_000
        for amount in LIABILITIES.values()
    )

    assert min(LIABILITIES) == LIABILITY_START_YEAR
    assert max(LIABILITIES) == END_YEAR


def test_liability_before_2033():
    """There should be no liability before 2033."""

    assert get_liability(2027) == 0
    assert get_liability(2032) == 0


def test_liability_from_2033():
    """Each year from 2033-2042 has a $50,000 liability."""

    for year in range(2033, 2043):
        assert get_liability(year) == 50_000


def test_glide_path_sums_to_one():
    """Every yearly allocation must sum to 100%."""

    for year in range(START_YEAR, END_YEAR + 1):

        allocation = calculate_glide_path(year)

        assert np.isclose(
            allocation.total,
            1.0,
        )


def test_risk_decreases_over_time():
    """Equity exposure should decline as liabilities approach."""

    early = calculate_glide_path(2027)
    late = calculate_glide_path(2033)

    assert late.equity < early.equity
    assert late.bonds > early.bonds
    assert late.cash > early.cash


def test_glide_path_dataframe():
    """The complete glide path should contain one row per year."""

    glide_path = generate_glide_path()

    assert len(glide_path) == 16

    assert glide_path["Year"].iloc[0] == 2027
    assert glide_path["Year"].iloc[-1] == 2042


def test_total_liabilities():
    """Total liabilities from 2033-2042 should equal $500,000."""

    total = calculate_cumulative_liabilities()

    assert total == 500_000


def test_required_protected_assets():
    """A 120% coverage target requires $60,000."""

    required = calculate_required_protected_assets(
        2033,
        coverage_target=1.20,
    )

    assert required == 60_000


def test_liability_coverage():
    """$65,000 protects a $50,000 liability at 130% coverage."""

    coverage = calculate_liability_coverage(
        portfolio_value=100_000,
        protected_assets=65_000,
        year=2033,
    )

    assert np.isclose(
        coverage,
        1.30,
    )


def test_liability_schedule_dataframe():

    schedule = get_liability_schedule()

    assert len(schedule) == 10

    assert (
        schedule["Annual Liability"].sum()
        == 500_000
    )

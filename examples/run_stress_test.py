"""
Historical stress-test example.

Replays the 2008 Financial Crisis against the portfolios
produced by the Black-Litterman optimizer.
"""

from black_litterman import (
    BlackLittermanModel,
    MeanVarianceOptimizer,
    compute_covariance,
    compute_returns,
    download_prices,
    market_cap_weights,
)

from black_litterman.stress_testing import (
    run_stress_test,
)


TICKERS = [
    "AAPL",
    "MSFT",
    "GOOGL",
    "AMZN",
    "JPM",
    "XOM",
]

MARKET_CAPS = {
    "AAPL": 3400,
    "MSFT": 3100,
    "GOOGL": 2100,
    "AMZN": 2000,
    "JPM": 600,
    "XOM": 500,
}


def main():

    # Use data extending through the 2008 crisis.
    prices = download_prices(
        TICKERS,
        start="2007-01-01",
        end="2009-01-01",
    )

    returns = compute_returns(
        prices,
        method="simple",
    )

    covariance = compute_covariance(returns)

    model = BlackLittermanModel(
        covariance,
        market_cap_weights(MARKET_CAPS),
        risk_aversion=2.5,
        tau=0.05,
    )

    model.add_absolute_view(
        "AAPL",
        expected_return=0.10,
        confidence=0.60,
    )

    model.add_relative_view(
        "JPM",
        "XOM",
        expected_return=0.04,
        confidence=0.40,
    )

    posterior_returns, posterior_covariance = model.posterior()

    optimizer = MeanVarianceOptimizer(
        posterior_returns,
        posterior_covariance,
        risk_free_rate=0.02,
        allow_short=False,
    )

    max_sharpe = optimizer.max_sharpe()
    min_volatility = optimizer.min_volatility()

    portfolios = {
        "Maximum Sharpe": max_sharpe,
        "Minimum Volatility": min_volatility,
    }

    for name, weights in portfolios.items():

        result = run_stress_test(
            returns,
            weights,
            scenario="2008 Financial Crisis",
            initial_value=100_000,
        )

        print(f"\n=== {name} ===")
        print(f"2008 return: {result.portfolio_return:.2%}")
        print(
            f"Annualized volatility: "
            f"{result.annualized_volatility:.2%}"
        )
        print(
            f"Maximum drawdown: "
            f"{result.maximum_drawdown:.2%}"
        )
        print(f"Worst day: {result.worst_day:.2%}")
        print(
            f"$100,000 ending value: "
            f"${result.ending_value:,.2f}"
        )

        print("\nAsset return contributions:")
        print(result.asset_contributions)


if __name__ == "__main__":
    main()

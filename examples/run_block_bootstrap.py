
"""
Example: connect the block bootstrap to the existing
Mustang Wealth Management Black-Litterman workflow.
"""

from black_litterman import (
    BlackLittermanModel,
    MeanVarianceOptimizer,
    compute_covariance,
    compute_returns,
    download_prices,
    market_cap_weights,
)

from black_litterman.block_bootstrap import (
    compare_portfolios,
    plot_2033_distribution,
)


TICKERS = ["AAPL", "MSFT", "GOOGL", "AMZN", "JPM", "XOM"]

MARKET_CAPS = {
    "AAPL": 3400,
    "MSFT": 3100,
    "GOOGL": 2100,
    "AMZN": 2000,
    "JPM": 600,
    "XOM": 500,
}


def main():
    # Use the SAME historical period/data philosophy as your BL model.
    prices = download_prices(
        TICKERS,
        start="2018-01-01",
    )

    # IMPORTANT:
    # Use simple returns for the wealth simulation.
    returns = compute_returns(
        prices,
        method="simple",
    )

    # BL still uses your existing covariance calculation.
    cov_matrix = compute_covariance(returns)

    market_weights = market_cap_weights(MARKET_CAPS)

    bl = BlackLittermanModel(
        cov_matrix=cov_matrix,
        market_weights=market_weights,
        risk_aversion=2.5,
        tau=0.05,
    )

    # Keep your team's existing views here.
    # Example only:
    # bl.add_absolute_view("AAPL", expected_return=0.10, confidence=0.60)
    # bl.add_relative_view("JPM", "XOM", expected_return=0.04, confidence=0.40)

    posterior_returns, posterior_cov = bl.posterior()

    optimizer = MeanVarianceOptimizer(
        expected_returns=posterior_returns,
        cov_matrix=posterior_cov,
        risk_free_rate=0.02,
        allow_short=False,
    )

    max_sharpe_weights = optimizer.max_sharpe()
    min_vol_weights = optimizer.min_volatility()

    portfolios = {
        "BL Max Sharpe": max_sharpe_weights,
        "BL Min Volatility": min_vol_weights,
    }

    results_table, results = compare_portfolios(
        returns=returns,
        portfolios=portfolios,
        n_sims=10_000,
        block_size=21,   # about one trading month
        seed=42,
    )

    print("\n=== Block Bootstrap Results ===")
    print(results_table.round(4))

    print("\n=== Portfolio Weights ===")
    print(max_sharpe_weights.round(4))
    print(min_vol_weights.round(4))

    plot_2033_distribution(
        results["BL Max Sharpe"]
    )


if __name__ == "__main__":
    main()

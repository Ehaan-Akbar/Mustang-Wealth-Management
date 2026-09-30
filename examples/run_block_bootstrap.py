"""
Run the block-bootstrap simulation using
the existing Mustang Wealth Management
Black-Litterman workflow.
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

    # ---------------------------------------------------------
    # 1. Download historical prices
    # ---------------------------------------------------------

    prices = download_prices(
        TICKERS,
        start="2018-01-01",
    )


    # ---------------------------------------------------------
    # 2. Convert prices to SIMPLE daily returns
    # ---------------------------------------------------------

    returns = compute_returns(
        prices,
        method="simple",
    )


    # ---------------------------------------------------------
    # 3. Calculate covariance for Black-Litterman
    # ---------------------------------------------------------

    cov_matrix = compute_covariance(
        returns
    )


    # ---------------------------------------------------------
    # 4. Market-cap weights
    # ---------------------------------------------------------

    market_weights = market_cap_weights(
        MARKET_CAPS
    )


    # ---------------------------------------------------------
    # 5. Build your existing Black-Litterman model
    # ---------------------------------------------------------

    bl = BlackLittermanModel(
        cov_matrix=cov_matrix,
        market_weights=market_weights,
        risk_aversion=2.5,
        tau=0.05,
    )


    # ---------------------------------------------------------
    # IMPORTANT:
    #
    # Put the SAME investment views that your team
    # already uses in your existing Black-Litterman model.
    #
    # Example:
    #
    # bl.add_absolute_view(
    #     "AAPL",
    #     expected_return=0.10,
    #     confidence=0.60
    # )
    #
    # bl.add_relative_view(
    #     "JPM",
    #     "XOM",
    #     expected_return=0.04,
    #     confidence=0.40
    # )
    #
    # DO NOT copy these examples unless they are
    # actually your team's views.
    # ---------------------------------------------------------


    # ---------------------------------------------------------
    # 6. Get Black-Litterman posterior returns/covariance
    # ---------------------------------------------------------

    posterior_returns, posterior_cov = (
        bl.posterior()
    )


    # ---------------------------------------------------------
    # 7. Run your existing portfolio optimizer
    # ---------------------------------------------------------

    optimizer = MeanVarianceOptimizer(
        expected_returns=posterior_returns,
        cov_matrix=posterior_cov,
        risk_free_rate=0.02,
        allow_short=False,
    )


    # ---------------------------------------------------------
    # 8. Get Max Sharpe and Minimum Volatility portfolios
    # ---------------------------------------------------------

    max_sharpe_weights = (
        optimizer.max_sharpe()
    )

    min_vol_weights = (
        optimizer.min_volatility()
    )


    # ---------------------------------------------------------
    # 9. Put portfolios into a dictionary
    # ---------------------------------------------------------

    portfolios = {

        "BL Max Sharpe":
            max_sharpe_weights,

        "BL Min Volatility":
            min_vol_weights,
    }


    # ---------------------------------------------------------
    # 10. Run Block Bootstrap
    # ---------------------------------------------------------

    results_table, results = (
        compare_portfolios(

            returns=returns,

            portfolios=portfolios,

            n_sims=10_000,

            block_size=21,

            batch_size=250,

            seed=42,
        )
    )


    # ---------------------------------------------------------
    # 11. Display results
    # ---------------------------------------------------------

    print(
        "\n=== BLOCK BOOTSTRAP RESULTS ==="
    )

    print(
        results_table.round(4)
    )


    # ---------------------------------------------------------
    # 12. Display portfolio weights
    # ---------------------------------------------------------

    print(
        "\n=== BL MAX SHARPE WEIGHTS ==="
    )

    print(
        max_sharpe_weights.round(4)
    )


    print(
        "\n=== BL MINIMUM VOLATILITY WEIGHTS ==="
    )

    print(
        min_vol_weights.round(4)
    )


    # ---------------------------------------------------------
    # 13. Plot the 2033 distribution
    # ---------------------------------------------------------

    plot_2033_distribution(
        results[
            "BL Max Sharpe"
        ]
    )


if __name__ == "__main__":

    main()

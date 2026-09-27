"""
End-to-end example of the Black-Litterman workflow.

By default this uses synthetic price data so it runs anywhere, offline,
with no API keys. Set USE_YFINANCE = True to instead download real
historical prices for the tickers below.
"""

import numpy as np
import pandas as pd

from black_litterman import (
    BlackLittermanModel,
    MeanVarianceOptimizer,
    compute_covariance,
    compute_returns,
    download_prices,
    market_cap_weights,
)

USE_YFINANCE = False

TICKERS = ["AAPL", "MSFT", "GOOGL", "AMZN", "JPM", "XOM"]

# Rough illustrative market caps (USD billions) — replace with real data.
MARKET_CAPS = {
    "AAPL": 3400,
    "MSFT": 3100,
    "GOOGL": 2100,
    "AMZN": 2000,
    "JPM": 600,
    "XOM": 500,
}


def load_synthetic_returns(tickers, n_days=1500, seed=42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n = len(tickers)
    # A simple factor model so assets are correlated in a plausible way.
    market_factor = rng.normal(0.0004, 0.011, size=n_days)
    idio = rng.normal(0, 0.01, size=(n_days, n))
    betas = rng.uniform(0.7, 1.4, size=n)
    daily_returns = idio + np.outer(market_factor, betas)
    dates = pd.bdate_range(end=pd.Timestamp.today(), periods=n_days + 5)[-n_days:]
    return pd.DataFrame(daily_returns, index=dates, columns=tickers)


def main():
    if USE_YFINANCE:
        prices = download_prices(TICKERS, start="2018-01-01")
        returns = compute_returns(prices)
    else:
        returns = load_synthetic_returns(TICKERS)

    cov_matrix = compute_covariance(returns)
    weights_mkt = market_cap_weights(MARKET_CAPS)

    # 1. Build the model from market-implied equilibrium returns.
    bl = BlackLittermanModel(
        cov_matrix=cov_matrix,
        market_weights=weights_mkt,
        risk_aversion=2.5,
        tau=0.05,
    )

    print("=== Implied equilibrium (prior) returns ===")
    print(bl.pi.round(4))

    # 2. Add investor views.
    #    Absolute view: AAPL will return 10% annually, moderate confidence.
    bl.add_absolute_view("AAPL", expected_return=0.10, confidence=0.6)
    #    Relative view: JPM will outperform XOM by 4% annually, lower confidence.
    bl.add_relative_view("JPM", "XOM", expected_return=0.04, confidence=0.4)

    # 3. Compute the posterior (blended) returns and covariance.
    posterior_returns, posterior_cov = bl.posterior()

    print("\n=== Prior vs. posterior returns ===")
    print(bl.summary().round(4))

    # 4. Feed the posterior into a mean-variance optimizer.
    optimizer = MeanVarianceOptimizer(
        expected_returns=posterior_returns,
        cov_matrix=posterior_cov,
        risk_free_rate=0.02,
        allow_short=False,
    )

    max_sharpe_weights = optimizer.max_sharpe()
    min_vol_weights = optimizer.min_volatility()

    print("\n=== Max Sharpe portfolio weights ===")
    print(max_sharpe_weights.round(4))
    ret, vol, sharpe = optimizer.performance(max_sharpe_weights)
    print(f"Expected return: {ret:.2%}  Volatility: {vol:.2%}  Sharpe: {sharpe:.2f}")

    print("\n=== Minimum volatility portfolio weights ===")
    print(min_vol_weights.round(4))
    ret, vol, sharpe = optimizer.performance(min_vol_weights)
    print(f"Expected return: {ret:.2%}  Volatility: {vol:.2%}  Sharpe: {sharpe:.2f}")


if __name__ == "__main__":
    main()

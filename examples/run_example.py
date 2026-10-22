"""
End-to-end Black-Litterman portfolio optimization.

Uses real historical prices by default.
The prior weights are equal-weighted because this universe mixes
individual stocks and ETFs whose holdings overlap.

IMPORTANT:
- Equal-weight prior is a custom prior, NOT a market-cap-weighted
  equilibrium portfolio.
- All covariance and return estimates are annualized.
- Every asset must have at least 1% portfolio allocation.
- Investor views below are illustrative assumptions, not forecasts.
- The Black-Litterman model works in EXCESS returns. Views are converted
  to excess returns going in, and posterior returns are converted back to
  total returns before optimizing, so the risk-free rate is subtracted
  exactly once (inside the Sharpe ratio).
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from black_litterman import (
    BlackLittermanModel,
    compute_returns,
    download_prices,
)


# ============================================================
# CONFIGURATION
# ============================================================

USE_YFINANCE = True
HISTORICAL_START = "2018-01-01"

TICKERS = [
    "MRK", "LMT", "JNJ", "KO", "ABT", "SO", "CSCO",
    "FLO", "AS", "MDLZ", "SCHL", "TSM", "NVDA", "NWS",
    "ADBE", "V", "LLY", "GOOG", "ASML", "DIS", "NTDOY", 
    "META", "AMZN", "ABNB", "DAL", "TER",
    "VXUS", "SCHD", "NTES", "CP", "VOO", "SCHG",
    "USMV", "QUAL", "VTV", "VEA", "VEU", "XLU", "XLP", 
    "ESGD", "ESGV"
]

# Black-Litterman parameters.
RISK_AVERSION = 2.5
TAU = 0.05
RISK_FREE_RATE = 0.02

# Minimum percentage invested in every asset.
MIN_WEIGHT = 0.01  # at least 1.5% in each asset
MAX_WEIGHT = 0.10   # at most 10% in each asset

# Illustrative investor views (TOTAL annual returns, as you would state them).
# Change these tickers and assumptions to reflect your actual views.
ABSOLUTE_VIEW_TICKER = "KO"
ABSOLUTE_VIEW_RETURN = 0.08        # 8% total annual return
ABSOLUTE_VIEW_CONFIDENCE = 0.50

RELATIVE_VIEW_TICKER_1 = "KO"
RELATIVE_VIEW_TICKER_2 = "MDLZ"
RELATIVE_VIEW_RETURN = 0.02        # KO beats MDLZ by 2%
RELATIVE_VIEW_CONFIDENCE = 0.40

TRADING_DAYS_PER_YEAR = 252
MIN_RETURN_OBSERVATIONS = 60


# ============================================================
# PRIOR WEIGHTS
# ============================================================

def build_prior_weights(tickers):
    """
    Create an equal-weighted custom prior.

    This avoids mixing company market capitalizations with ETF AUM.
    The weights sum to 100%, and each asset starts with the same weight.

    These are custom prior weights, not true market-cap weights.
    """

    if not tickers:
        raise ValueError("TICKERS cannot be empty.")

    weight = 1.0 / len(tickers)

    return {
        ticker: weight
        for ticker in tickers
    }


# ============================================================
# SYNTHETIC RETURNS
# ============================================================

def load_synthetic_returns(
    tickers,
    n_days=1500,
    seed=42,
):
    """
    Generate illustrative daily returns.

    This is simulated data, NOT actual historical performance.
    """

    rng = np.random.default_rng(seed)
    n_assets = len(tickers)

    market_factor = rng.normal(
        0.0004,
        0.011,
        size=n_days,
    )

    idio = rng.normal(
        0,
        0.01,
        size=(n_days, n_assets),
    )

    betas = rng.uniform(
        0.7,
        1.4,
        size=n_assets,
    )

    daily_returns = idio + np.outer(
        market_factor,
        betas,
    )

    dates = pd.bdate_range(
        end=pd.Timestamp.today().normalize(),
        periods=n_days,
    )

    return pd.DataFrame(
        daily_returns,
        index=dates,
        columns=tickers,
    )


# ============================================================
# DATA LOADING AND VALIDATION
# ============================================================

def validate_configuration():
    """Validate ticker names and portfolio settings."""

    if not TICKERS:
        raise ValueError("TICKERS cannot be empty.")

    if len(TICKERS) != len(set(TICKERS)):
        raise ValueError("TICKERS contains duplicate symbols.")

    if len(TICKERS) * MIN_WEIGHT > 1.0:
        raise ValueError(
            "The minimum allocation is infeasible: "
            "number of assets x minimum weight exceeds 100%."
        )

    required_view_tickers = [
        ABSOLUTE_VIEW_TICKER,
        RELATIVE_VIEW_TICKER_1,
        RELATIVE_VIEW_TICKER_2,
    ]

    missing_views = [
        ticker
        for ticker in required_view_tickers
        if ticker not in TICKERS
    ]

    if missing_views:
        raise ValueError(
            "View tickers must be in TICKERS: "
            + ", ".join(missing_views)
        )

    if not 0 <= ABSOLUTE_VIEW_CONFIDENCE <= 1:
        raise ValueError(
            "ABSOLUTE_VIEW_CONFIDENCE must be between 0 and 1."
        )

    if not 0 <= RELATIVE_VIEW_CONFIDENCE <= 1:
        raise ValueError(
            "RELATIVE_VIEW_CONFIDENCE must be between 0 and 1."
        )


def load_historical_returns(tickers):
    """Download prices and calculate daily returns."""

    print("Downloading historical prices...")

    prices = download_prices(
        tickers,
        start=HISTORICAL_START,
    )

    if not isinstance(prices, pd.DataFrame):
        raise TypeError(
            "download_prices() did not return a pandas DataFrame."
        )

    # Detect tickers with no downloaded price history.
    missing_tickers = [
        ticker
        for ticker in tickers
        if ticker not in prices.columns
        or prices[ticker].notna().sum() == 0
    ]

    if missing_tickers:
        raise ValueError(
            "Historical prices are unavailable for: "
            + ", ".join(missing_tickers)
            + ". Verify these symbols with your data provider."
        )

    prices = prices[tickers]

    # Show which tickers have the shortest history, since they set the
    # length of the shared sample used for the covariance matrix.
    first_dates = prices.apply(lambda col: col.first_valid_index())
    shortest = first_dates.sort_values(ascending=False).head(3)
    print("Latest-starting tickers (these limit the shared sample):")
    for ticker, date in shortest.items():
        print(f"  {ticker}: first price on {date.date()}")

    # Keep dates where all selected assets have prices, so the
    # covariance calculation uses the same observation dates.
    prices = prices.replace(
        [np.inf, -np.inf],
        np.nan,
    ).dropna(axis=0, how="any")

    if len(prices) < MIN_RETURN_OBSERVATIONS + 1:
        raise ValueError(
            "Too few complete price observations remain after aligning "
            "all tickers. Check the symbols or select a shorter-history "
            "asset universe."
        )

    returns = compute_returns(prices)

    returns = returns.reindex(columns=tickers)
    returns = returns.replace(
        [np.inf, -np.inf],
        np.nan,
    ).dropna(axis=0, how="any")

    if len(returns) < MIN_RETURN_OBSERVATIONS:
        raise ValueError(
            f"Only {len(returns)} complete return observations are available; "
            f"at least {MIN_RETURN_OBSERVATIONS} are required."
        )

    if not np.isfinite(returns.to_numpy()).all():
        raise ValueError("Return data contains non-finite values.")

    return returns


# ============================================================
# CONSTRAINED PORTFOLIO OPTIMIZATION
# ============================================================

def optimize_portfolio(
    expected_returns,
    covariance,
    tickers,
    risk_free_rate=0.02,
    min_weight=0.01,
    objective="sharpe",
):
    """
    Optimize portfolio weights with:
      - Weights summing to 100%.
      - No short selling.
      - A minimum allocation for every asset.

    objective:
      "sharpe"     -> maximize the Sharpe ratio
      "volatility" -> minimize portfolio variance

    expected_returns must be TOTAL returns (not excess), because the
    Sharpe ratio subtracts risk_free_rate here.
    Expected returns and covariance must use consistent annual units.
    """

    n_assets = len(tickers)

    if n_assets * min_weight > 1.0:
        raise ValueError(
            "Minimum weight is infeasible for this number of assets."
        )

    if objective not in ("sharpe", "volatility"):
        raise ValueError(
            "objective must be 'sharpe' or 'volatility'."
        )

    # Align data with the requested ticker order.
    if isinstance(expected_returns, pd.Series):
        mu = expected_returns.reindex(tickers).to_numpy(dtype=float)

    elif isinstance(expected_returns, pd.DataFrame):
        if set(tickers).issubset(expected_returns.index):
            mu = expected_returns.reindex(tickers).iloc[:, 0].to_numpy(
                dtype=float
            )
        elif set(tickers).issubset(expected_returns.columns):
            mu = expected_returns.iloc[0].reindex(tickers).to_numpy(
                dtype=float
            )
        else:
            mu = expected_returns.to_numpy(dtype=float).reshape(-1)

    else:
        mu = np.asarray(
            expected_returns,
            dtype=float,
        ).reshape(-1)

    if isinstance(covariance, pd.DataFrame):
        sigma = covariance.reindex(
            index=tickers,
            columns=tickers,
        ).to_numpy(dtype=float)
    else:
        sigma = np.asarray(covariance, dtype=float)

    if mu.size != n_assets:
        raise ValueError(
            "Expected-return vector does not match the ticker count."
        )

    if sigma.shape != (n_assets, n_assets):
        raise ValueError(
            "Covariance matrix dimensions do not match the ticker count."
        )

    if not np.isfinite(mu).all() or not np.isfinite(sigma).all():
        raise ValueError(
            "Expected returns or covariance contain missing/non-finite values."
        )

    # Remove tiny numerical asymmetries in the covariance matrix.
    sigma = (sigma + sigma.T) / 2.0

    # Objective functions with analytic gradients. Gradients make SLSQP
    # far more stable than finite differences on 30+ assets.
    def sharpe_obj(w):
        vol = np.sqrt(max(w @ sigma @ w, 1e-16))
        return -(w @ mu - risk_free_rate) / vol

    def sharpe_grad(w):
        vol = np.sqrt(max(w @ sigma @ w, 1e-16))
        excess = w @ mu - risk_free_rate
        return -(mu / vol - excess * (sigma @ w) / vol ** 3)

    def var_obj(w):
        return float(w @ sigma @ w)

    def var_grad(w):
        return 2.0 * sigma @ w

    if objective == "sharpe":
        fun, jac = sharpe_obj, sharpe_grad
    else:
        fun, jac = var_obj, var_grad

    # An equal-weight initial guess satisfies the constraints because
    # MIN_WEIGHT is no greater than 1 / number of assets.
    initial_weights = np.full(
        n_assets,
        1.0 / n_assets,
    )

    result = minimize(
        fun,
        x0=initial_weights,
        jac=jac,
        method="SLSQP",
        bounds=[(min_weight, 1.0)] * n_assets,
        constraints=[
            {
                "type": "eq",
                "fun": lambda w: np.sum(w) - 1.0,
                "jac": lambda w: np.ones_like(w),
            }
        ],
        options={
            "maxiter": 1000,
            "ftol": 1e-9,
        },
    )

    weights = result.x

    # Judge the result by whether the constraints hold, not only by
    # SLSQP's success flag, which is strict about convergence tolerance.
    if not np.isfinite(weights).all():
        raise RuntimeError(
            f"Portfolio optimization failed: {result.message}"
        )

    if not np.isclose(weights.sum(), 1.0, atol=1e-6):
        raise RuntimeError(
            "Optimized weights do not sum to 100%."
        )

    if weights.min() < min_weight - 1e-6:
        raise RuntimeError(
            "One or more optimized weights violate the minimum allocation."
        )

    if not result.success:
        print(
            f"Warning ({objective}): optimizer reported '{result.message}', "
            "but the weights satisfy all constraints."
        )

    return pd.Series(
        weights,
        index=tickers,
        name="Weight",
    )


def print_portfolio_results(
    title,
    weights,
    expected_returns,
    covariance,
    risk_free_rate,
):
    """Print allocations and estimated portfolio performance."""

    if isinstance(expected_returns, pd.Series):
        mu = expected_returns.reindex(weights.index).to_numpy(dtype=float)
    else:
        mu = np.asarray(expected_returns, dtype=float).reshape(-1)

    if isinstance(covariance, pd.DataFrame):
        sigma = covariance.reindex(
            index=weights.index,
            columns=weights.index,
        ).to_numpy(dtype=float)
    else:
        sigma = np.asarray(covariance, dtype=float)

    w = weights.to_numpy(dtype=float)

    portfolio_return = float(w @ mu)
    variance = max(float(w @ sigma @ w), 0.0)
    volatility = np.sqrt(variance)

    sharpe = (
        (portfolio_return - risk_free_rate) / volatility
        if volatility > 1e-12
        else np.nan
    )

    print(f"\n=== {title} ===")
    print(
        weights.sort_values(ascending=False)
        .map(lambda weight: f"{weight:.2%}")
        .to_string()
    )

    print(f"\nTotal allocation:    {weights.sum():.2%}")
    print(f"Minimum allocation:  {weights.min():.2%}")
    print(f"Expected return:     {portfolio_return:.2%}")
    print(f"Volatility:          {volatility:.2%}")
    print(f"Sharpe ratio:        {sharpe:.2f}")


# ============================================================
# MAIN WORKFLOW
# ============================================================

def main():
    validate_configuration()

    print("=" * 65)
    print("BLACK-LITTERMAN PORTFOLIO OPTIMIZATION")
    print("=" * 65)

    print(f"\nNumber of assets: {len(TICKERS)}")
    print(f"Minimum allocation per asset: {MIN_WEIGHT:.1%}")

    # --------------------------------------------------------
    # 1. Load daily returns
    # --------------------------------------------------------

    print("\nLoading return data...")

    if USE_YFINANCE:
        returns = load_historical_returns(TICKERS)
        print("Using downloaded historical prices.")
    else:
        returns = load_synthetic_returns(TICKERS)
        print("Using synthetic returns.")
        print("Synthetic data is not actual historical performance.")

    print(f"Assets: {len(returns.columns)}")
    print(f"Return observations: {len(returns)}")
    print(f"Return dates: {returns.index.min()} to {returns.index.max()}")

    # --------------------------------------------------------
    # 2. Calculate annualized covariance
    # --------------------------------------------------------

    # Returns are daily, so annualize their covariance.
    cov_matrix = returns.cov() * TRADING_DAYS_PER_YEAR
    cov_matrix = cov_matrix.reindex(
        index=TICKERS,
        columns=TICKERS,
    )

    # --------------------------------------------------------
    # 3. Build a consistent custom prior
    # --------------------------------------------------------

    prior_weights = build_prior_weights(TICKERS)

    print("\nPrior weight methodology: equal-weighted custom prior")
    print(
        "Note: these are not traditional market-cap equilibrium weights."
    )
    print(
        f"Each prior weight: {1.0 / len(TICKERS):.2%}"
    )

    # The BlackLittermanModel API calls this argument market_weights.
    # We pass equal-weight custom prior weights here to avoid mixing
    # individual company market caps and ETF AUM.
    bl = BlackLittermanModel(
        cov_matrix=cov_matrix,
        market_weights=prior_weights,
        risk_aversion=RISK_AVERSION,
        tau=TAU,
    )

    print("\n=== Prior Implied EXCESS Returns ===")
    print(bl.pi.round(4))

    # --------------------------------------------------------
    # 4. Add investor views
    # --------------------------------------------------------

    # The model works in excess returns, so convert the absolute view
    # (a total return) by subtracting the risk-free rate.
    bl.add_absolute_view(
        ABSOLUTE_VIEW_TICKER,
        expected_return=ABSOLUTE_VIEW_RETURN - RISK_FREE_RATE,
        confidence=ABSOLUTE_VIEW_CONFIDENCE,
    )

    print(
        f"\nAbsolute view: {ABSOLUTE_VIEW_TICKER} "
        f"has an assumed annual return of {ABSOLUTE_VIEW_RETURN:.1%} "
        f"({ABSOLUTE_VIEW_RETURN - RISK_FREE_RATE:.1%} excess)."
    )

    # A relative view is a difference between two returns, so the
    # risk-free rate cancels out and no adjustment is needed.
    bl.add_relative_view(
        RELATIVE_VIEW_TICKER_1,
        RELATIVE_VIEW_TICKER_2,
        expected_return=RELATIVE_VIEW_RETURN,
        confidence=RELATIVE_VIEW_CONFIDENCE,
    )

    print(
        f"Relative view: {RELATIVE_VIEW_TICKER_1} is assumed to outperform "
        f"{RELATIVE_VIEW_TICKER_2} by {RELATIVE_VIEW_RETURN:.1%} annually."
    )

    # --------------------------------------------------------
    # 5. Calculate posterior returns and covariance
    # --------------------------------------------------------

    posterior_excess, posterior_cov = bl.posterior()

    print("\n=== Prior vs. Posterior EXCESS Returns ===")
    print(bl.summary().round(4))

    # Convert back to total returns. The Sharpe ratio below subtracts
    # the risk-free rate, so this keeps it from being subtracted twice.
    posterior_returns = posterior_excess + RISK_FREE_RATE

    # --------------------------------------------------------
    # 6. Optimize the portfolio
    # --------------------------------------------------------

    max_sharpe_weights = optimize_portfolio(
        expected_returns=posterior_returns,
        covariance=posterior_cov,
        tickers=TICKERS,
        risk_free_rate=RISK_FREE_RATE,
        min_weight=MIN_WEIGHT,
        objective="sharpe",
    )

    print_portfolio_results(
        title="Maximum Sharpe Ratio Portfolio",
        weights=max_sharpe_weights,
        expected_returns=posterior_returns,
        covariance=posterior_cov,
        risk_free_rate=RISK_FREE_RATE,
    )

    min_vol_weights = optimize_portfolio(
        expected_returns=posterior_returns,
        covariance=posterior_cov,
        tickers=TICKERS,
        risk_free_rate=RISK_FREE_RATE,
        min_weight=MIN_WEIGHT,
        objective="volatility",
    )

    print_portfolio_results(
        title="Minimum Volatility Portfolio",
        weights=min_vol_weights,
        expected_returns=posterior_returns,
        covariance=posterior_cov,
        risk_free_rate=RISK_FREE_RATE,
    )

    # --------------------------------------------------------
    # 7. Save portfolio allocations
    # --------------------------------------------------------

    results = pd.DataFrame({
        "Maximum_Sharpe": max_sharpe_weights,
        "Minimum_Volatility": min_vol_weights,
    })

    results.to_csv("black_litterman_allocations.csv")

    print("\nAllocations saved to black_litterman_allocations.csv")
    print("Portfolio optimization completed.")


if __name__ == "__main__":
    main()

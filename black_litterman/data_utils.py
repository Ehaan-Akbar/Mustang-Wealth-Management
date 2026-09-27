"""
Helpers for pulling price data and turning it into the inputs the
Black-Litterman model needs: a covariance matrix and market-cap weights.

Downloading requires the optional `yfinance` dependency. Everything else
in this module works on plain pandas DataFrames, so you can supply your
own price data if you'd rather not depend on yfinance.
"""

from typing import Dict, List, Optional

import numpy as np
import pandas as pd


def download_prices(
    tickers: List[str], start: str, end: Optional[str] = None, interval: str = "1d"
) -> pd.DataFrame:
    """
    Download adjusted close prices for `tickers` using yfinance.

    Returns a DataFrame indexed by date with one column per ticker.
    """
    try:
        import yfinance as yf
    except ImportError as exc:
        raise ImportError(
            "yfinance is required for download_prices(). Install it with "
            "`pip install yfinance`, or load your own price DataFrame instead."
        ) from exc

    data = yf.download(
        tickers, start=start, end=end, interval=interval, auto_adjust=True, progress=False
    )
    if isinstance(data.columns, pd.MultiIndex):
        prices = data["Close"]
    else:
        prices = data[["Close"]]
        prices.columns = tickers
    return prices.dropna(how="all")


def compute_returns(prices: pd.DataFrame, method: str = "log") -> pd.DataFrame:
    """Convert a price DataFrame into periodic returns."""
    if method == "log":
        returns = np.log(prices / prices.shift(1))
    elif method == "simple":
        returns = prices.pct_change()
    else:
        raise ValueError("method must be 'log' or 'simple'")
    return returns.dropna(how="all")


def compute_covariance(returns: pd.DataFrame, periods_per_year: int = 252) -> pd.DataFrame:
    """Annualized sample covariance matrix from periodic returns."""
    return returns.cov() * periods_per_year


def market_cap_weights(market_caps: Dict[str, float]) -> pd.Series:
    """Normalize a dict of {ticker: market_cap} into weights summing to 1."""
    s = pd.Series(market_caps, dtype=float)
    if (s < 0).any():
        raise ValueError("market caps must be non-negative")
    if s.sum() == 0:
        raise ValueError("market caps sum to zero")
    return s / s.sum()


def implied_risk_aversion(
    market_return: float, market_variance: float, risk_free_rate: float = 0.0
) -> float:
    """
    Standard estimate of the market risk-aversion coefficient (delta):
        delta = (E[R_market] - risk_free_rate) / Var(R_market)
    """
    if market_variance <= 0:
        raise ValueError("market_variance must be positive")
    return (market_return - risk_free_rate) / market_variance

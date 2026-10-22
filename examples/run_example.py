"""End-to-end Black-Litterman portfolio optimization with Wall Street views.

Historical prices/returns are downloaded at run time. Wall Street views are a
STATIC SNAPSHOT CHECKED ON 2026-10-21 and are NOT fetched dynamically. See
WALL_STREET_DATA for consensus ratings, analyst counts, average 12-month price
targets, reported upside, source links, and notes.

MANUAL_VIEW_RETURNS is the editable input section. Values are decimals:
0.10 means a +10% 12-month price-return view; -0.05 means -5%. Values are
pre-filled from the reported analyst-consensus target upside where a usable
target exists. No target is invented for NWS or NTDOY because the chosen source
did not supply a consensus target for those symbols; their qualitative ratings
are shown, but their numeric view is None until you enter one.

ETFs are included in the portfolio universe, but do not have single-company
analyst consensus price targets. They receive no Wall Street view unless you
explicitly add a numeric value to MANUAL_VIEW_RETURNS.

The model uses the standard Black-Litterman diagonal Omega assumption
Omega = diag(tau * P * Sigma * P.T), rather than a hand-set confidence for each
stock. Analyst price-target upside is a price-return estimate and does not add
dividends. The model works in excess returns; the risk-free rate is subtracted
from views before the posterior step and added back before Sharpe optimization.

This is informational research, not a guarantee of future performance.
"""

import datetime as dt

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from black_litterman import (
    compute_returns,
    download_prices,
)


# ============================================================
# CONFIGURATION
# ============================================================

USE_YFINANCE = True
HISTORICAL_START = "2018-01-01"

TICKERS = [
    "MRK", "LMT", "JNJ", "KO", "SO", 
    "SCHL", "TSM", "NVDA", "NWS",
    "ADBE", "V", "LLY", "GOOG", "ASML", "NTDOY",
    "META", "AMZN",
    "VXUS", "SCHD", "NTES", "VOO", "SCHG",
    "USMV", "QUAL", "VTV", "VEA", "VEU", "XLU", "XLP",
    "ESGD", "ESGV",
]

# ETFs have no analyst price targets, so they receive no views.
# Also used to report the stock / ETF split.
ETF_TICKERS = [
    "VXUS", "SCHD", "VOO", "SCHG", "USMV", "QUAL", "VTV",
    "VEA", "VEU", "XLU", "XLP", "ESGD", "ESGV",
]

# Black-Litterman parameters.
RISK_AVERSION = 2.5
TAU = 0.05
RISK_FREE_RATE = 0.02

# Allocation limits for every asset.
MIN_WEIGHT = 0.0    # no economic minimum allocation is imposed
MAX_WEIGHT = 0.10   # at most 10% in each asset

# Soft diversification regularization; this is NOT a minimum weight.
# It penalizes portfolios that depart too far from the equal-weight prior
# using KL divergence. Treat 0.10 as a starting value and inspect the
# sensitivity of the results to smaller/larger values before relying on them.
DIVERSIFICATION_PENALTY = 1.0

# Wall Street snapshot metadata. Source links are included for verification.
# Most estimates below are the StockAnalysis/S&P Global consensus page values.
# SCHL, NWS, and NTDOY use MarketBeat because a comparable StockAnalysis page
# did not provide suitable current target data for those tickers.
VIEW_DATA_CHECK_DATE = "2026-10-21"
USE_WALL_STREET_VIEWS = True

WALL_STREET_DATA = {
    "MRK": {"company": "Merck & Co.", "rating": "Buy", "analysts": 27, "current_price": 148.50, "target_price": 158.44, "target_return": 0.0669, "target_low": 105.00, "target_high": 186.00, "source": "StockAnalysis / S&P Global", "source_url": "https://stockanalysis.com/stocks/mrk/forecast/", "data_as_of": "2026-10-21", "notes": "12-month analyst consensus target."},
    "LMT": {"company": "Lockheed Martin", "rating": "Buy", "analysts": 23, "current_price": 548.65, "target_price": 618.86, "target_return": 0.1280, "target_low": 486.00, "target_high": 756.00, "source": "StockAnalysis / S&P Global", "source_url": "https://stockanalysis.com/stocks/lmt/forecast/", "data_as_of": "2026-10-21", "notes": "12-month analyst consensus target."},
    "JNJ": {"company": "Johnson & Johnson", "rating": "Buy", "analysts": 23, "current_price": 256.03, "target_price": 279.41, "target_return": 0.0913, "target_low": 190.00, "target_high": 320.00, "source": "StockAnalysis / S&P Global", "source_url": "https://stockanalysis.com/stocks/jnj/forecast/", "data_as_of": "2026-10-21", "notes": "12-month analyst consensus target."},
    "KO": {"company": "The Coca-Cola Company", "rating": "Buy", "analysts": 24, "current_price": 92.68, "target_price": 94.74, "target_return": 0.0222, "target_low": 75.00, "target_high": 104.00, "source": "StockAnalysis / S&P Global", "source_url": "https://stockanalysis.com/stocks/ko/forecast/", "data_as_of": "2026-10-21", "notes": "12-month analyst consensus target."},
    "ABT": {"company": "Abbott Laboratories", "rating": "Strong Buy", "analysts": 28, "current_price": 100.51, "target_price": 120.72, "target_return": 0.2011, "target_low": 103.00, "target_high": 135.00, "source": "StockAnalysis / S&P Global", "source_url": "https://stockanalysis.com/stocks/abt/forecast/", "data_as_of": "2026-10-21", "notes": "12-month analyst consensus target."},
    "SO": {"company": "The Southern Company", "rating": "Hold", "analysts": 23, "current_price": 90.39, "target_price": 96.29, "target_return": 0.0653, "target_low": 74.00, "target_high": 112.00, "source": "StockAnalysis / S&P Global", "source_url": "https://stockanalysis.com/stocks/so/forecast/", "data_as_of": "2026-10-21", "notes": "12-month analyst consensus target."},
    "CSCO": {"company": "Cisco Systems", "rating": "Buy", "analysts": 29, "current_price": 107.26, "target_price": 136.16, "target_return": 0.2694, "target_low": 110.00, "target_high": 170.00, "source": "StockAnalysis / S&P Global", "source_url": "https://stockanalysis.com/stocks/csco/forecast/", "data_as_of": "2026-10-21", "notes": "12-month analyst consensus target."},
    "FLO": {"company": "Flowers Foods", "rating": "Hold", "analysts": 5, "current_price": 5.81, "target_price": 7.40, "target_return": 0.2737, "target_low": 5.00, "target_high": 12.00, "source": "StockAnalysis / S&P Global", "source_url": "https://stockanalysis.com/stocks/flo/forecast/", "data_as_of": "2026-10-21", "notes": "Only 5 analysts; target range is wide."},
    "AS": {"company": "Amer Sports", "rating": "Strong Buy", "analysts": 27, "current_price": 29.58, "target_price": 47.24, "target_return": 0.5970, "target_low": 31.00, "target_high": 67.00, "source": "StockAnalysis / S&P Global", "source_url": "https://stockanalysis.com/stocks/as/forecast/", "data_as_of": "2026-10-21", "notes": "Very high target-implied upside; broad target range."},
    "MDLZ": {"company": "Mondelez International", "rating": "Buy", "analysts": 23, "current_price": 62.74, "target_price": 69.05, "target_return": 0.1006, "target_low": 55.00, "target_high": 77.00, "source": "StockAnalysis / S&P Global", "source_url": "https://stockanalysis.com/stocks/mdlz/forecast/", "data_as_of": "2026-10-21", "notes": "12-month analyst consensus target."},
    "SCHL": {"company": "Scholastic", "rating": "Hold", "analysts": 3, "current_price": 38.94, "target_price": 35.00, "target_return": -0.1012, "target_low": 35.00, "target_high": 35.00, "source": "MarketBeat", "source_url": "https://www.marketbeat.com/stocks/NASDAQ/SCHL/forecast/", "data_as_of": "2026-10-14", "notes": "Only 3 ratings; quoted price data on the source page was dated 2026-10-14."},
    "TSM": {"company": "Taiwan Semiconductor Manufacturing Company (ADR)", "rating": "Strong Buy", "analysts": 21, "current_price": 431.89, "target_price": 584.39, "target_return": 0.3531, "target_low": 440.00, "target_high": 1000.00, "source": "StockAnalysis / S&P Global", "source_url": "https://stockanalysis.com/stocks/tsm/forecast/", "data_as_of": "2026-10-21", "notes": "Wide high/low target range."},
    "NVDA": {"company": "NVIDIA", "rating": "Strong Buy", "analysts": 62, "current_price": 221.97, "target_price": 328.72, "target_return": 0.4809, "target_low": 180.00, "target_high": 515.00, "source": "StockAnalysis / S&P Global", "source_url": "https://stockanalysis.com/stocks/nvda/forecast/", "data_as_of": "2026-10-21", "notes": "12-month analyst consensus target."},
    "NWS": {"company": "News Corporation Class B", "rating": "Moderate Buy", "analysts": 2, "current_price": 32.01, "target_price": None, "target_return": None, "target_low": None, "target_high": None, "source": "MarketBeat", "source_url": "https://www.marketbeat.com/stocks/NASDAQ/NWS/forecast/", "data_as_of": "2026-10-09", "notes": "MarketBeat showed no consensus price target; not replaced with the NWSA Class A target."},
    "ADBE": {"company": "Adobe", "rating": "Hold", "analysts": 39, "current_price": 255.06, "target_price": 272.31, "target_return": 0.0676, "target_low": 195.00, "target_high": 373.00, "source": "StockAnalysis / S&P Global", "source_url": "https://stockanalysis.com/stocks/adbe/forecast/", "data_as_of": "2026-10-21", "notes": "12-month analyst consensus target."},
    "V": {"company": "Visa", "rating": "Strong Buy", "analysts": 40, "current_price": 407.85, "target_price": 422.77, "target_return": 0.0366, "target_low": 330.00, "target_high": 475.00, "source": "StockAnalysis / S&P Global", "source_url": "https://stockanalysis.com/stocks/v/forecast/", "data_as_of": "2026-10-21", "notes": "Reported target-implied upside is 3.66%."},
    "LLY": {"company": "Eli Lilly", "rating": "Buy", "analysts": 31, "current_price": 1270.75, "target_price": 1328.00, "target_return": 0.0451, "target_low": 930.00, "target_high": 1600.00, "source": "StockAnalysis / S&P Global", "source_url": "https://stockanalysis.com/stocks/lly/forecast/", "data_as_of": "2026-10-21", "notes": "12-month analyst consensus target."},
    "GOOG": {"company": "Alphabet Class C", "rating": "Strong Buy", "analysts": 62, "current_price": 330.95, "target_price": 420.79, "target_return": 0.2715, "target_low": 340.00, "target_high": 485.00, "source": "StockAnalysis / S&P Global", "source_url": "https://stockanalysis.com/stocks/goog/forecast/", "data_as_of": "2026-10-21", "notes": "Target forecasts are for Alphabet; ticker used here is GOOG Class C."},
    "ASML": {"company": "ASML Holding", "rating": "Strong Buy", "analysts": 41, "current_price": 1676.69, "target_price": 2152.00, "target_return": 0.2837, "target_low": 868.56, "target_high": 2886.00, "source": "StockAnalysis / S&P Global", "source_url": "https://stockanalysis.com/stocks/asml/forecast/", "data_as_of": "2026-10-21", "notes": "Wide target range; prices in USD for the traded listing."},
    "DIS": {"company": "Walt Disney Company", "rating": "Strong Buy", "analysts": 33, "current_price": 111.83, "target_price": 126.65, "target_return": 0.1325, "target_low": 88.00, "target_high": 144.00, "source": "StockAnalysis / S&P Global", "source_url": "https://stockanalysis.com/stocks/dis/forecast/", "data_as_of": "2026-10-21", "notes": "12-month analyst consensus target."},
    "NTDOY": {"company": "Nintendo ADR", "rating": "Hold", "analysts": 9, "current_price": 12.32, "target_price": None, "target_return": None, "target_low": None, "target_high": None, "source": "MarketBeat", "source_url": "https://www.marketbeat.com/stocks/OTCMKTS/NTDOY/forecast/", "data_as_of": "2026-10-21", "notes": "MarketBeat reported no consensus 12-month price target; numerical view left blank rather than using a sparse/stale estimate."},
    "META": {"company": "Meta Platforms", "rating": "Strong Buy", "analysts": 63, "current_price": 697.52, "target_price": 814.51, "target_return": 0.1677, "target_low": 580.00, "target_high": 1050.00, "source": "StockAnalysis / S&P Global", "source_url": "https://stockanalysis.com/stocks/meta/forecast/", "data_as_of": "2026-10-21", "notes": "12-month analyst consensus target."},
    "AMZN": {"company": "Amazon", "rating": "Strong Buy", "analysts": 60, "current_price": 262.37, "target_price": 331.46, "target_return": 0.2633, "target_low": 230.00, "target_high": 405.00, "source": "StockAnalysis / S&P Global", "source_url": "https://stockanalysis.com/stocks/amzn/forecast/", "data_as_of": "2026-10-21", "notes": "12-month analyst consensus target."},
    "ABNB": {"company": "Airbnb", "rating": "Buy", "analysts": 44, "current_price": 173.69, "target_price": 183.15, "target_return": 0.0545, "target_low": 135.00, "target_high": 220.00, "source": "StockAnalysis / S&P Global", "source_url": "https://stockanalysis.com/stocks/abnb/forecast/", "data_as_of": "2026-10-21", "notes": "12-month analyst consensus target."},
    "DAL": {"company": "Delta Air Lines", "rating": "Strong Buy", "analysts": 25, "current_price": 77.67, "target_price": 101.07, "target_return": 0.3013, "target_low": 53.00, "target_high": 125.00, "source": "StockAnalysis / S&P Global", "source_url": "https://stockanalysis.com/stocks/dal/forecast/", "data_as_of": "2026-10-21", "notes": "12-month analyst consensus target."},
    "TER": {"company": "Teradyne", "rating": "Buy", "analysts": 17, "current_price": 345.49, "target_price": 446.47, "target_return": 0.2923, "target_low": 350.00, "target_high": 550.00, "source": "StockAnalysis / S&P Global", "source_url": "https://stockanalysis.com/stocks/ter/forecast/", "data_as_of": "2026-10-21", "notes": "12-month analyst consensus target."},
    "NTES": {"company": "NetEase", "rating": "Strong Buy", "analysts": 33, "current_price": 122.98, "target_price": 162.81, "target_return": 0.3239, "target_low": 132.66, "target_high": 200.61, "source": "StockAnalysis / S&P Global", "source_url": "https://stockanalysis.com/stocks/ntes/forecast/", "data_as_of": "2026-10-21", "notes": "Source page last quote update was dated 2026-10-21."},
    "CP": {"company": "Canadian Pacific Kansas City", "rating": "Buy", "analysts": 30, "current_price": 87.36, "target_price": 98.27, "target_return": 0.1154, "target_low": 74.30, "target_high": 105.44, "source": "StockAnalysis / S&P Global", "source_url": "https://stockanalysis.com/stocks/cp/forecast/", "data_as_of": "2026-10-16", "notes": "Latest quote shown by the source page was dated 2026-10-16."},
}

# EDIT THESE VALUES TO ENTER YOUR OWN VIEWS MANUALLY.
# Decimal expected 12-month PRICE return: 0.12 = +12%, -0.08 = -8%.
# Seeded from the Wall Street consensus target upside listed above.
# None means the selected source had no consensus target, so no numeric view
# is fabricated; you can fill it yourself after reviewing the rating/source.
MANUAL_VIEW_RETURNS = {
    "MRK": 0.0669,
    "LMT": 0.1280,
    "JNJ": 0.0913,
    "KO": 0.0222,
    "SO": 0.0653,
    "SCHL": -0.1012,
    "TSM": 0.3531,
    "NVDA": 0.4809,
    "NWS": None,     # MarketBeat consensus target unavailable
    "ADBE": 0.0676,
    "V": 0.0366,
    "LLY": 0.0451,
    "GOOG": 0.2715,
    "ASML": 0.2837,
    "NTDOY": None,   # MarketBeat consensus target unavailable
    "META": 0.1677,
    "AMZN": 0.2633,
    "NTES": 0.3239,
    # Add a ticker here if you also want to specify an explicit ETF view.
}

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

    unknown_etfs = [t for t in ETF_TICKERS if t not in TICKERS]
    if unknown_etfs:
        raise ValueError(
            "ETF_TICKERS has tickers not in TICKERS: "
            + ", ".join(unknown_etfs)
        )

    if not 0 <= MIN_WEIGHT <= MAX_WEIGHT <= 1:
        raise ValueError("Need 0 <= MIN_WEIGHT <= MAX_WEIGHT <= 1.")

    if len(TICKERS) * MIN_WEIGHT > 1.0:
        raise ValueError(
            "The minimum allocation is infeasible: "
            "number of assets x minimum weight exceeds 100%."
        )

    if len(TICKERS) * MAX_WEIGHT < 1.0:
        raise ValueError(
            "The maximum allocation is infeasible: "
            "number of assets x maximum weight is below 100%. "
            "Raise MAX_WEIGHT or add more assets."
        )

    unknown_manual = [t for t in MANUAL_VIEW_RETURNS if t not in TICKERS]
    if unknown_manual:
        raise ValueError(
            "MANUAL_VIEW_RETURNS has tickers not in TICKERS: "
            + ", ".join(unknown_manual)
        )

    for ticker, view in MANUAL_VIEW_RETURNS.items():
        if view is not None and not np.isfinite(float(view)):
            raise ValueError(f"Manual view for {ticker} must be a finite decimal or None.")


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
# WALL STREET VIEW TABLE + BLACK-LITTERMAN POSTERIOR
# ============================================================

def build_wall_street_table():
    """Return the researched snapshot, including coverage gaps and ETF rows."""
    rows = []
    for ticker in TICKERS:
        row = WALL_STREET_DATA.get(ticker)
        if row is None:
            # ETFs do not have a comparable single-company analyst consensus target.
            rows.append({
                "ticker": ticker,
                "asset_type": "ETF",
                "company": ticker,
                "consensus_rating": "N/A (ETF)",
                "analyst_count": np.nan,
                "current_price": np.nan,
                "consensus_target": np.nan,
                "target_low": np.nan,
                "target_high": np.nan,
                "target_upside_pct": np.nan,
                "manual_model_view_pct": np.nan,
                "source": "N/A",
                "source_url": "",
                "data_as_of": VIEW_DATA_CHECK_DATE,
                "notes": "Diversified fund; no single-company analyst target is used.",
            })
            continue

        manual_view = MANUAL_VIEW_RETURNS.get(ticker)
        rows.append({
            "ticker": ticker,
            "asset_type": "Stock/ADR",
            "company": row["company"],
            "consensus_rating": row["rating"],
            "analyst_count": row["analysts"],
            "current_price": row["current_price"],
            "consensus_target": row["target_price"],
            "target_low": row["target_low"],
            "target_high": row["target_high"],
            "target_upside_pct": (
                100.0 * row["target_return"]
                if row["target_return"] is not None else np.nan
            ),
            "manual_model_view_pct": (
                100.0 * float(manual_view)
                if manual_view is not None else np.nan
            ),
            "source": row["source"],
            "source_url": row["source_url"],
            "data_as_of": row["data_as_of"],
            "notes": row["notes"],
        })
    return pd.DataFrame(rows)


def black_litterman_posterior(
    covariance,
    prior_weights,
    tickers,
    manual_view_returns,
    risk_aversion=RISK_AVERSION,
    tau=TAU,
    risk_free_rate=RISK_FREE_RATE,
):
    """Compute Black-Litterman posterior using the standard diagonal Omega rule.

    The editable view inputs are total price-return views. They are converted
    into excess-return views by subtracting the risk-free rate. Omega is set to
    diag(P @ (tau * Sigma) @ P.T), a common Black-Litterman default, so there
    is no arbitrary per-stock confidence number in this script.
    """
    sigma = covariance.reindex(index=tickers, columns=tickers).to_numpy(dtype=float)
    w_prior = pd.Series(prior_weights).reindex(tickers).to_numpy(dtype=float)

    if not np.isfinite(sigma).all() or not np.isfinite(w_prior).all():
        raise ValueError("Covariance/prior contains missing or non-finite values.")

    pi = risk_aversion * sigma @ w_prior  # equilibrium implied excess returns
    tau_sigma = tau * sigma

    active_tickers = [
        t for t in tickers
        if t in manual_view_returns and manual_view_returns[t] is not None
    ]

    if not active_tickers:
        posterior_excess = pi.copy()
        posterior_cov = sigma + tau_sigma
    else:
        ticker_positions = {ticker: i for i, ticker in enumerate(tickers)}
        p = np.zeros((len(active_tickers), len(tickers)), dtype=float)
        for row_index, ticker in enumerate(active_tickers):
            p[row_index, ticker_positions[ticker]] = 1.0

        q_total = np.array(
            [float(manual_view_returns[t]) for t in active_tickers],
            dtype=float,
        )
        q_excess = q_total - risk_free_rate

        view_covariance = p @ tau_sigma @ p.T
        omega = np.diag(np.maximum(np.diag(view_covariance), 1e-12))
        middle = view_covariance + omega
        middle_inverse = np.linalg.pinv(middle)

        posterior_excess = (
            pi
            + tau_sigma @ p.T @ middle_inverse @ (q_excess - p @ pi)
        )
        posterior_cov = (
            sigma
            + tau_sigma
            - tau_sigma @ p.T @ middle_inverse @ p @ tau_sigma
        )

    posterior_cov = (posterior_cov + posterior_cov.T) / 2.0
    return (
        pd.Series(pi, index=tickers, name="Prior_Implied_Excess_Return"),
        pd.Series(posterior_excess, index=tickers, name="Posterior_Excess_Return"),
        pd.DataFrame(posterior_cov, index=tickers, columns=tickers),
        active_tickers,
    )


# ============================================================
# CONSTRAINED PORTFOLIO OPTIMIZATION
# ============================================================

def optimize_portfolio(
    expected_returns,
    covariance,
    tickers,
    prior_weights=None,
    risk_free_rate=0.02,
    min_weight=0.0,
    max_weight=0.10,
    objective="sharpe",
    diversification_penalty=DIVERSIFICATION_PENALTY,
):
    """Optimize a portfolio with a soft KL-divergence diversification penalty.

    No meaningful minimum allocation is imposed. Instead, KL(w || prior)
    penalizes concentration relative to the supplied prior. With positive prior
    weights and a positive penalty, this discourages dropping assets entirely.

    objective:
      "sharpe"     -> maximize Sharpe ratio, penalized for divergence from prior
      "volatility" -> minimize log portfolio volatility, with the same penalty

    Expected returns and covariance must use consistent annual units.
    """
    n_assets = len(tickers)

    if not 0 <= min_weight <= max_weight <= 1:
        raise ValueError("Need 0 <= min_weight <= max_weight <= 1.")
    if n_assets * min_weight > 1.0 or n_assets * max_weight < 1.0:
        raise ValueError("Allocation limits are infeasible for this number of assets.")
    if objective not in ("sharpe", "volatility"):
        raise ValueError("objective must be 'sharpe' or 'volatility'.")
    if diversification_penalty < 0:
        raise ValueError("diversification_penalty must be non-negative.")

    # Align expected returns with ticker order.
    if isinstance(expected_returns, pd.Series):
        mu = expected_returns.reindex(tickers).to_numpy(dtype=float)
    elif isinstance(expected_returns, pd.DataFrame):
        if set(tickers).issubset(expected_returns.index):
            mu = expected_returns.reindex(tickers).iloc[:, 0].to_numpy(dtype=float)
        elif set(tickers).issubset(expected_returns.columns):
            mu = expected_returns.iloc[0].reindex(tickers).to_numpy(dtype=float)
        else:
            mu = expected_returns.to_numpy(dtype=float).reshape(-1)
    else:
        mu = np.asarray(expected_returns, dtype=float).reshape(-1)

    if isinstance(covariance, pd.DataFrame):
        sigma = covariance.reindex(index=tickers, columns=tickers).to_numpy(dtype=float)
    else:
        sigma = np.asarray(covariance, dtype=float)

    if mu.size != n_assets:
        raise ValueError("Expected-return vector does not match the ticker count.")
    if sigma.shape != (n_assets, n_assets):
        raise ValueError("Covariance matrix dimensions do not match the ticker count.")
    if not np.isfinite(mu).all() or not np.isfinite(sigma).all():
        raise ValueError("Expected returns or covariance contain missing/non-finite values.")

    sigma = (sigma + sigma.T) / 2.0

    # Use the actual portfolio prior as the regularization target.
    if prior_weights is None:
        prior = np.full(n_assets, 1.0 / n_assets)
    elif isinstance(prior_weights, pd.Series):
        prior = prior_weights.reindex(tickers).to_numpy(dtype=float)
    elif isinstance(prior_weights, dict):
        prior = np.array([prior_weights[t] for t in tickers], dtype=float)
    else:
        prior = np.asarray(prior_weights, dtype=float).reshape(-1)

    if prior.size != n_assets or not np.isfinite(prior).all() or np.any(prior <= 0):
        raise ValueError("Every prior weight must be finite and strictly positive.")
    prior = prior / prior.sum()

    # Tiny numerical guard only for log evaluation; this is not an allocation floor.
    tiny = 1e-300

    def kl_divergence(w):
        w_safe = np.maximum(w, tiny)
        return float(np.sum(w_safe * np.log(w_safe / prior)))

    def kl_gradient(w):
        w_safe = np.maximum(w, tiny)
        return np.log(w_safe / prior) + 1.0

    def sharpe_base(w):
        variance = max(float(w @ sigma @ w), 1e-16)
        volatility = np.sqrt(variance)
        return -(float(w @ mu) - risk_free_rate) / volatility

    def sharpe_base_grad(w):
        variance = max(float(w @ sigma @ w), 1e-16)
        volatility = np.sqrt(variance)
        excess = float(w @ mu) - risk_free_rate
        return -(mu / volatility - excess * (sigma @ w) / volatility**3)

    def log_volatility(w):
        variance = max(float(w @ sigma @ w), 1e-16)
        return 0.5 * np.log(variance)

    def log_volatility_grad(w):
        variance = max(float(w @ sigma @ w), 1e-16)
        return (sigma @ w) / variance

    if objective == "sharpe":
        fun = lambda w: sharpe_base(w) + diversification_penalty * kl_divergence(w)
        jac = lambda w: sharpe_base_grad(w) + diversification_penalty * kl_gradient(w)
    else:
        fun = lambda w: log_volatility(w) + diversification_penalty * kl_divergence(w)
        jac = lambda w: log_volatility_grad(w) + diversification_penalty * kl_gradient(w)

    # Start from the prior, which is feasible for this asset universe.
    initial_weights = prior.copy()
    initial_weights = np.clip(initial_weights, min_weight, max_weight)
    initial_weights = initial_weights / initial_weights.sum()

    result = minimize(
        fun,
        x0=initial_weights,
        jac=jac,
        method="SLSQP",
        bounds=[(min_weight, max_weight)] * n_assets,
        constraints=[{
            "type": "eq",
            "fun": lambda w: np.sum(w) - 1.0,
            "jac": lambda w: np.ones_like(w),
        }],
        options={"maxiter": 2000, "ftol": 1e-10},
    )

    weights = result.x
    if not np.isfinite(weights).all():
        raise RuntimeError(f"Portfolio optimization failed: {result.message}")
    if not np.isclose(weights.sum(), 1.0, atol=1e-6):
        raise RuntimeError("Optimized weights do not sum to 100%.")
    if weights.min() < min_weight - 1e-6 or weights.max() > max_weight + 1e-6:
        raise RuntimeError("Optimized weights violate the allocation bounds.")
    if not result.success:
        print(f"Warning ({objective}): optimizer reported '{result.message}'.")

    return pd.Series(weights, index=tickers, name="Weight")


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
        .map(lambda weight: f"{weight:.4%}")
        .to_string()
    )

    etfs = [t for t in ETF_TICKERS if t in weights.index]
    etf_share = float(weights[etfs].sum())

    print(f"\nTotal allocation:    {weights.sum():.2%}")
    print(f"Smallest allocation: {weights.min():.4%}")
    print(f"Largest allocation:  {weights.max():.2%}")
    print(f"Individual stocks:   {1.0 - etf_share:.2%}")
    print(f"ETFs:                {etf_share:.2%}")
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
    print(
        f"Allocation limits per asset: "
        f"{MIN_WEIGHT:.1%} to {MAX_WEIGHT:.1%}"
    )

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
    # 3. Build the custom prior and display the hard-coded Wall Street snapshot
    # --------------------------------------------------------

    prior_weights = build_prior_weights(TICKERS)

    print("\nPrior weight methodology: equal-weighted custom prior")
    print("Note: these are not traditional market-cap equilibrium weights.")
    print(f"Each prior weight: {1.0 / len(TICKERS):.2%}")
    print(f"KL diversification penalty: {DIVERSIFICATION_PENALTY:.3f}")
    print("This soft penalty favors diversification without imposing a meaningful minimum allocation.")

    wall_street_table = build_wall_street_table()

    print(f"\n=== Wall Street analyst snapshot (checked {VIEW_DATA_CHECK_DATE}) ===")
    display_columns = [
        "ticker", "consensus_rating", "analyst_count", "current_price",
        "consensus_target", "target_upside_pct", "manual_model_view_pct",
    ]
    print(wall_street_table[display_columns].to_string(index=False, na_rep="N/A"))
    print("Full source URLs and notes are hard-coded in WALL_STREET_DATA.")

    # --------------------------------------------------------
    # 4. Add manually editable Wall Street views
    # --------------------------------------------------------

    active_numeric_views = {
        ticker: float(view)
        for ticker, view in MANUAL_VIEW_RETURNS.items()
        if view is not None and ticker in TICKERS
    } if USE_WALL_STREET_VIEWS else {}

    missing_numeric_views = [
        ticker for ticker in TICKERS
        if ticker not in ETF_TICKERS
        and MANUAL_VIEW_RETURNS.get(ticker) is None
    ]
    if missing_numeric_views:
        print(
            "\nNo numeric consensus target found for these stocks: "
            + ", ".join(missing_numeric_views)
            + ". Their qualitative Wall Street rating is still listed above; "
              "enter a decimal in MANUAL_VIEW_RETURNS if you want to add a view."
        )

    if not USE_WALL_STREET_VIEWS:
        print("\nUSE_WALL_STREET_VIEWS=False: no analyst views will be applied.")
    else:
        print(f"\nApplying {len(active_numeric_views)} manually editable numeric views.")
        print("No hard-coded per-stock confidence is used.")
        print(
            "Omega uses diag(tau * P * Sigma * P.T), a standard "
            "Black-Litterman uncertainty assumption."
        )

    # --------------------------------------------------------
    # 5. Calculate posterior returns and covariance
    # --------------------------------------------------------

    prior_excess, posterior_excess, posterior_cov, used_view_tickers = (
        black_litterman_posterior(
            covariance=cov_matrix,
            prior_weights=prior_weights,
            tickers=TICKERS,
            manual_view_returns=active_numeric_views,
            risk_aversion=RISK_AVERSION,
            tau=TAU,
            risk_free_rate=RISK_FREE_RATE,
        )
    )

    print("\n=== Prior vs. Posterior EXCESS Returns ===")
    comparison = pd.DataFrame({
        "Prior_Implied_Excess_Return": prior_excess,
        "Posterior_Excess_Return": posterior_excess,
    })
    comparison["Change"] = (
        comparison["Posterior_Excess_Return"]
        - comparison["Prior_Implied_Excess_Return"]
    )
    print(comparison.round(4).to_string())
    print(f"\nBlack-Litterman numeric views used: {len(used_view_tickers)}")

    # Convert back to total returns. The Sharpe ratio below subtracts
    # risk_free_rate, so it is deducted only once.
    posterior_returns = posterior_excess + RISK_FREE_RATE

    # --------------------------------------------------------
    # 6. Optimize the portfolio
    # --------------------------------------------------------

    max_sharpe_weights = optimize_portfolio(
        expected_returns=posterior_returns,
        covariance=posterior_cov,
        tickers=TICKERS,
        prior_weights=prior_weights,
        risk_free_rate=RISK_FREE_RATE,
        min_weight=MIN_WEIGHT,
        max_weight=MAX_WEIGHT,
        objective="sharpe",
        diversification_penalty=DIVERSIFICATION_PENALTY,
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
        prior_weights=prior_weights,
        risk_free_rate=RISK_FREE_RATE,
        min_weight=MIN_WEIGHT,
        max_weight=MAX_WEIGHT,
        objective="volatility",
        diversification_penalty=DIVERSIFICATION_PENALTY,
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

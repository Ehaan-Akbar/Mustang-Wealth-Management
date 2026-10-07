"""Data preparation and configurable simulations for the portfolio lab."""

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd

from .block_bootstrap import block_bootstrap
from .black_litterman import BlackLittermanModel
from .optimizer import MeanVarianceOptimizer


def fit_model(returns, reference, absolute_views, relative_views, risk_aversion,
              tau, risk_free_rate, max_weight, objective, shrinkage):
    """Fit BL using excess-return views, then optimize total expected returns."""
    reference = reference.reindex(returns.columns).astype(float)
    if not np.isfinite(reference).all() or (reference < 0).any() or reference.sum() <= 0:
        raise ValueError("Reference weights must be non-negative with a positive total.")
    if max_weight * len(reference) < 1 - 1e-9:
        raise ValueError(f"Maximum position must be at least {100 / len(reference):.1f}% for {len(reference)} assets.")
    reference /= reference.sum()
    covariance = returns.cov() * 252
    covariance = (1 - shrinkage) * covariance + shrinkage * np.diag(np.diag(covariance))
    model = BlackLittermanModel(covariance, reference, risk_aversion, tau)
    for view in absolute_views:
        model.add_absolute_view(view["Asset"], view["Return (%)"] / 100 - risk_free_rate, view["Confidence (%)"] / 100)
    for view in relative_views:
        if view["Asset"] == view["Compared with"]:
            raise ValueError("A relative view needs two different assets.")
        model.add_relative_view(view["Asset"], view["Compared with"], view["Outperformance (%)"] / 100, view["Confidence (%)"] / 100)
    posterior, covariance = model.posterior()
    total_returns = posterior + risk_free_rate
    optimizer = MeanVarianceOptimizer(total_returns, covariance, risk_free_rate=risk_free_rate, max_weight=max_weight)
    weights = optimizer.max_sharpe() if objective == "Maximum Sharpe" else optimizer.min_volatility()
    weights = weights.clip(lower=0)
    weights /= weights.sum()
    summary = pd.DataFrame({
        "Reference weight": reference, "Optimized weight": weights,
        "Prior return": model.pi + risk_free_rate, "Posterior return": total_returns,
    })
    return dict(weights=weights, reference=reference, summary=summary, returns=total_returns,
                covariance=covariance, performance=optimizer.performance(weights),
                reference_performance=optimizer.performance(reference))


def prepare_returns(frame: pd.DataFrame, tickers: tuple[str, ...], prices: bool) -> pd.DataFrame:
    """Align the requested universe without filling missing prices or dropping assets."""
    if not tickers:
        raise ValueError("Choose at least one stock or ETF.")
    frame = frame.copy()
    frame.columns = frame.columns.astype(str).str.strip().str.upper()
    if frame.columns.duplicated().any():
        raise ValueError("The data contains duplicate ticker columns.")
    missing = [ticker for ticker in tickers if ticker not in frame or frame[ticker].dropna().empty]
    if missing:
        raise ValueError(f"No data for: {', '.join(missing)}. Check the symbols or choose a different date range.")
    frame.index = pd.to_datetime(frame.index, errors="coerce", utc=True).tz_convert(None)
    if frame.index.isna().any() or frame.index.duplicated().any():
        raise ValueError("Use unique, valid dates in the first CSV column.")
    frame = frame.loc[:, list(tickers)].apply(pd.to_numeric, errors="coerce").sort_index()
    frame = frame.replace([np.inf, -np.inf], np.nan)
    if prices:
        if (frame <= 0).any().any():
            raise ValueError("Prices must be positive.")
        frame = frame.pct_change(fill_method=None)
    frame = frame.dropna()
    if len(frame) < 60:
        raise ValueError("At least 60 overlapping daily returns are required. Try a longer date range.")
    if (frame <= -1).any().any():
        raise ValueError("Daily simple returns must be greater than -100%. Enter returns as decimals, e.g. 0.01 for 1%.")
    if (frame.std() <= 1e-12).any():
        raise ValueError("Each asset must have some return variation. Check the uploaded data.")
    return frame


@dataclass
class Projection:
    """Year-end balances and pathwise withdrawal success for one simulation method."""

    values: pd.DataFrame
    funded: np.ndarray
    daily_paths: pd.DataFrame

    def summary(self, target: float) -> dict:
        ending = self.values.iloc[:, -1]
        return {
            "Median ending value": float(ending.median()),
            "5th percentile": float(ending.quantile(0.05)),
            "95th percentile": float(ending.quantile(0.95)),
            "Target reached": float((ending >= target).mean()),
            "All withdrawals funded": float(self.funded.mean()),
        }


def simulate_projection(
    returns: pd.DataFrame,
    weights: pd.Series,
    *,
    method: str,
    n_sims: int = 1000,
    years: int = 10,
    initial_value: float = 100_000,
    annual_contribution: float = 0,
    annual_withdrawal: float = 0,
    withdrawal_start: int = 1,
    block_size: int = 21,
    seed: int = 42,
    annual_mean: float | None = None,
    annual_volatility: float | None = None,
    retained_paths: int = 200,
    progress: Callable[[int, int], None] | None = None,
) -> Projection:
    """Compare lognormal Monte Carlo with circular blocks of historical returns.

    Both methods hold constant portfolio weights (daily rebalancing). Contributions
    and withdrawals occur at each year's start, before returns. Historical Monte
    Carlo fits log returns of the weighted portfolio. With model assumptions it
    uses annual arithmetic drift minus half the variance for the log drift.
    """
    if method not in {"Monte Carlo", "Block bootstrap"}:
        raise ValueError("Choose Monte Carlo or Block bootstrap.")
    if not 1 <= n_sims <= 100_000 or not 1 <= years <= 40:
        raise ValueError("Use 1–100,000 paths and a horizon of 1–40 years.")
    if not 1 <= withdrawal_start <= years:
        raise ValueError("The first withdrawal year must be within the horizon.")
    if not 1 <= retained_paths <= 200:
        raise ValueError("Retain between 1 and 200 daily paths.")
    cash_values = [initial_value, annual_contribution, annual_withdrawal]
    if not np.isfinite(cash_values).all() or min(cash_values) < 0:
        raise ValueError("Balances and cash flows must be finite and non-negative.")
    weights = weights.reindex(returns.columns)
    if not np.isfinite(weights).all() or (weights < 0).any() or not np.isclose(weights.sum(), 1):
        raise ValueError("Portfolio weights must be non-negative and sum to 100%.")
    if len(returns) < 2 or not np.isfinite(returns.to_numpy()).all() or (returns <= -1).any().any():
        raise ValueError("Provide finite daily returns greater than -100%.")
    portfolio = returns.dot(weights)
    rng = np.random.default_rng(seed)
    batch_size = 200
    if method == "Block bootstrap":
        if not 1 <= block_size <= len(returns):
            raise ValueError("Block size must be between 1 and the number of observations.")
        # Fixed-weight daily returns commute with block selection. Sampling the
        # weighted series preserves joint asset moves without storing asset cubes.
        batches = (
            1 + batch[:, :, 0].astype(float)
            for batch in block_bootstrap(
                portfolio.to_frame("Portfolio"), n_sims=n_sims, n_years=years,
                block_size=block_size, seed=seed, batch_size=batch_size,
            )
        )
    else:
        if (annual_mean is None) != (annual_volatility is None):
            raise ValueError("Supply both model return and volatility, or neither.")
        if annual_mean is None:
            log_returns = np.log1p(portfolio)
            drift = float(log_returns.mean() * 252)
            volatility = float(log_returns.std() * np.sqrt(252))
        else:
            if not np.isfinite([annual_mean, annual_volatility]).all() or annual_volatility < 0:
                raise ValueError("Model return and volatility must be finite; volatility cannot be negative.")
            drift = annual_mean - 0.5 * annual_volatility ** 2
            volatility = annual_volatility
        # Generate actual daily draws; retain the same draws used for summaries.
        batches = (
            np.exp(rng.normal(drift / 252, volatility / np.sqrt(252),
                              size=(min(batch_size, n_sims - start), years * 252)))
            for start in range(0, n_sims, batch_size)
        )
    values = np.empty((n_sims, years + 1))
    funded = np.ones(n_sims, dtype=bool)
    retained_count = min(retained_paths, n_sims)
    daily_paths = np.empty((retained_count, years * 252 + 1))
    offset = 0
    for growth in batches:
        count = len(growth)
        balance = np.full(count, initial_value, dtype=float)
        success = np.ones(count, dtype=bool)
        values[offset:offset + count, 0] = balance
        keep = max(0, min(count, retained_count - offset))
        if keep:
            daily_paths[offset:offset + keep, 0] = balance[:keep]
        for year in range(1, years + 1):
            balance += annual_contribution
            due = annual_withdrawal if year >= withdrawal_start else 0
            success &= balance >= due
            balance = np.maximum(balance - due, 0)
            daily_growth = np.cumprod(growth[:, (year - 1) * 252:year * 252], axis=1)
            if keep:
                daily_paths[offset:offset + keep, (year - 1) * 252 + 1:year * 252 + 1] = (
                    balance[:keep, None] * daily_growth[:keep]
                )
            balance *= daily_growth[:, -1]
            values[offset:offset + count, year] = balance
        funded[offset:offset + count] = success
        offset += count
        if progress is not None:
            progress(offset, n_sims)
    return Projection(
        pd.DataFrame(values, columns=pd.Index(range(years + 1), name="Year")), funded,
        pd.DataFrame(daily_paths, index=pd.Index(range(1, retained_count + 1), name="Path"),
                     columns=pd.Index(np.arange(years * 252 + 1) / 252, name="Year")),
    )

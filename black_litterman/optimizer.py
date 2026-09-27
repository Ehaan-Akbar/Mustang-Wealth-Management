"""
Mean-variance optimizer that turns (expected returns, covariance) into
portfolio weights. Designed to consume the posterior output of
BlackLittermanModel.posterior(), but works with any (mu, Sigma) pair.
"""

from typing import Optional, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import minimize


class MeanVarianceOptimizer:
    def __init__(
        self,
        expected_returns: pd.Series,
        cov_matrix: pd.DataFrame,
        risk_free_rate: float = 0.0,
        allow_short: bool = False,
        max_weight: Optional[float] = None,
    ):
        self.tickers = list(expected_returns.index)
        self.mu = expected_returns.loc[self.tickers].values
        self.sigma = cov_matrix.loc[self.tickers, self.tickers].values
        self.rf = risk_free_rate
        self.n = len(self.tickers)

        lower = -1.0 if allow_short else 0.0
        upper = max_weight if max_weight is not None else 1.0
        self.bounds = tuple((lower, upper) for _ in range(self.n))

    # ------------------------------------------------------------------
    # Core metrics
    # ------------------------------------------------------------------
    def portfolio_return(self, weights: np.ndarray) -> float:
        return float(weights @ self.mu)

    def portfolio_volatility(self, weights: np.ndarray) -> float:
        return float(np.sqrt(weights @ self.sigma @ weights))

    def sharpe_ratio(self, weights: np.ndarray) -> float:
        vol = self.portfolio_volatility(weights)
        if vol == 0:
            return 0.0
        return (self.portfolio_return(weights) - self.rf) / vol

    def _equal_weights(self) -> np.ndarray:
        return np.full(self.n, 1.0 / self.n)

    def _base_constraints(self):
        return [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]

    # ------------------------------------------------------------------
    # Optimization targets
    # ------------------------------------------------------------------
    def max_sharpe(self) -> pd.Series:
        def neg_sharpe(w):
            return -self.sharpe_ratio(w)

        result = minimize(
            neg_sharpe,
            self._equal_weights(),
            method="SLSQP",
            bounds=self.bounds,
            constraints=self._base_constraints(),
        )
        if not result.success:
            raise RuntimeError(f"max_sharpe optimization failed: {result.message}")
        return pd.Series(result.x, index=self.tickers, name="weight")

    def min_volatility(self) -> pd.Series:
        def variance(w):
            return w @ self.sigma @ w

        result = minimize(
            variance,
            self._equal_weights(),
            method="SLSQP",
            bounds=self.bounds,
            constraints=self._base_constraints(),
        )
        if not result.success:
            raise RuntimeError(f"min_volatility optimization failed: {result.message}")
        return pd.Series(result.x, index=self.tickers, name="weight")

    def target_return(self, target: float) -> pd.Series:
        """Minimum-variance portfolio achieving at least `target` return."""

        def variance(w):
            return w @ self.sigma @ w

        constraints = self._base_constraints() + [
            {"type": "eq", "fun": lambda w: self.portfolio_return(w) - target}
        ]
        result = minimize(
            variance,
            self._equal_weights(),
            method="SLSQP",
            bounds=self.bounds,
            constraints=constraints,
        )
        if not result.success:
            raise RuntimeError(f"target_return optimization failed: {result.message}")
        return pd.Series(result.x, index=self.tickers, name="weight")

    def efficient_frontier(self, n_points: int = 25) -> pd.DataFrame:
        """Sample the efficient frontier between min-vol and max-return assets."""
        min_vol_weights = self.min_volatility()
        min_ret = self.portfolio_return(min_vol_weights.values)
        max_ret = float(np.max(self.mu))

        targets = np.linspace(min_ret, max_ret * 0.999, n_points)
        rows = []
        for t in targets:
            try:
                w = self.target_return(t)
            except RuntimeError:
                continue
            rows.append(
                {
                    "target_return": t,
                    "return": self.portfolio_return(w.values),
                    "volatility": self.portfolio_volatility(w.values),
                    "sharpe": self.sharpe_ratio(w.values),
                }
            )
        return pd.DataFrame(rows)

    def performance(self, weights: pd.Series) -> Tuple[float, float, float]:
        w = weights.loc[self.tickers].values
        return (
            self.portfolio_return(w),
            self.portfolio_volatility(w),
            self.sharpe_ratio(w),
        )

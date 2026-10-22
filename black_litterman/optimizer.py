"""Mean-variance optimizer with a diversification penalty and a stock/ETF band."""

from typing import Mapping, Optional, Tuple

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
        asset_types: Optional[Mapping[str, str]] = None,
        stock_share_range: Tuple[float, float] = (0.35, 0.40),
        diversify: bool = True,
    ):
        self.tickers = list(expected_returns.index)
        self.mu = expected_returns.loc[self.tickers].values
        self.sigma = cov_matrix.loc[self.tickers, self.tickers].values
        self.rf = risk_free_rate
        self.n = len(self.tickers)
        self.bounds = tuple(((-1.0 if allow_short else 0.0), max_weight or 1.0) for _ in range(self.n))

        types = {t: str(k).strip().lower() for t, k in (asset_types or {}).items()}
        self.is_stock = np.array([types.get(t, "stock") != "etf" for t in self.tickers])
        self.has_band = bool(self.is_stock.any() and (~self.is_stock).any())
        self.lo, self.hi = stock_share_range

        # Equal share within each sleeve; the penalty only judges the spread
        # inside a sleeve, so the stock/ETF split itself is left to risk/return.
        s, e = self.is_stock, ~self.is_stock
        self.share = np.where(s, 1.0 / max(s.sum(), 1), 1.0 / max(e.sum(), 1))
        mid = (self.lo + self.hi) / 2 if self.has_band else 1.0
        self.target = self.share * np.where(s, mid, 1 - mid)

        avg_var = float(np.trace(self.sigma) / self.n)
        self.lam = avg_var / float(self.target @ self.target) if diversify else 0.0
        self.kappa = avg_var / self.n if self.has_band else 0.0

    # ------------------------------------------------------------------
    # Penalized risk
    # ------------------------------------------------------------------
    def _gaps(self, w):
        s = float(w[self.is_stock].sum())
        return max(s - self.lo, 1e-12), max(self.hi - s, 1e-12)

    def _spread(self, w):
        s = float(w[self.is_stock].sum())
        return w - self.share * np.where(self.is_stock, s, float(w.sum()) - s)

    def effective_variance(self, w: np.ndarray) -> float:
        v = w @ self.sigma @ w + self.lam * np.sum(self._spread(w) ** 2)
        if self.has_band:
            a, b = self._gaps(w)
            v -= self.kappa * (np.log(a) + np.log(b))
        return float(v)

    def _effective_variance_grad(self, w: np.ndarray) -> np.ndarray:
        g = 2 * self.sigma @ w + 2 * self.lam * self._spread(w)
        if self.has_band:
            a, b = self._gaps(w)
            g = g + self.kappa * (1 / b - 1 / a) * self.is_stock
        return g

    # ------------------------------------------------------------------
    # Core metrics
    # ------------------------------------------------------------------
    def portfolio_return(self, weights: np.ndarray) -> float:
        return float(weights @ self.mu)

    def portfolio_volatility(self, weights: np.ndarray) -> float:
        return float(np.sqrt(weights @ self.sigma @ weights))

    def sharpe_ratio(self, weights: np.ndarray) -> float:
        vol = self.portfolio_volatility(weights)
        return 0.0 if vol == 0 else (self.portfolio_return(weights) - self.rf) / vol

    def _base_constraints(self):
        cons = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0, "jac": lambda w: np.ones(self.n)}]
        if self.has_band:
            m = self.is_stock.astype(float)
            cons += [
                {"type": "ineq", "fun": lambda w: w @ m - self.lo, "jac": lambda w: m},
                {"type": "ineq", "fun": lambda w: self.hi - w @ m, "jac": lambda w: -m},
            ]
        return cons

    def _solve(self, fun, jac, constraints, name: str) -> pd.Series:
        res = minimize(fun, self.target, jac=jac, method="SLSQP", bounds=self.bounds,
                       constraints=constraints, options={"ftol": 1e-12, "maxiter": 1000})
        if not res.success:
            raise RuntimeError(f"{name} optimization failed: {res.message}")
        return pd.Series(res.x, index=self.tickers, name="weight")

    # ------------------------------------------------------------------
    # Optimization targets
    # ------------------------------------------------------------------
    def max_sharpe(self) -> pd.Series:
        def f(w):
            return -(self.portfolio_return(w) - self.rf) / np.sqrt(self.effective_variance(w))

        def g(w):
            v, ex = self.effective_variance(w), self.portfolio_return(w) - self.rf
            return -(self.mu / np.sqrt(v) - ex * self._effective_variance_grad(w) / (2 * v**1.5))

        return self._solve(f, g, self._base_constraints(), "max_sharpe")

    def min_volatility(self) -> pd.Series:
        return self._solve(self.effective_variance, self._effective_variance_grad,
                           self._base_constraints(), "min_volatility")

    def target_return(self, target: float) -> pd.Series:
        cons = self._base_constraints() + [
            {"type": "eq", "fun": lambda w: self.portfolio_return(w) - target, "jac": lambda w: self.mu}
        ]
        return self._solve(self.effective_variance, self._effective_variance_grad, cons, "target_return")

    def efficient_frontier(self, n_points: int = 25) -> pd.DataFrame:
        min_ret = self.portfolio_return(self.min_volatility().values)
        rows = []
        for t in np.linspace(min_ret, float(np.max(self.mu)) * 0.999, n_points):
            try:
                w = self.target_return(t).values
            except RuntimeError:
                continue
            rows.append({"target_return": t, "return": self.portfolio_return(w),
                         "volatility": self.portfolio_volatility(w), "sharpe": self.sharpe_ratio(w)})
        return pd.DataFrame(rows)

    def performance(self, weights: pd.Series) -> Tuple[float, float, float]:
        w = weights.loc[self.tickers].values
        return self.portfolio_return(w), self.portfolio_volatility(w), self.sharpe_ratio(w)

    def sleeve_weights(self, weights: pd.Series) -> pd.Series:
        w = weights.loc[self.tickers].values
        return pd.Series({"Stocks": w[self.is_stock].sum(), "ETFs": w[~self.is_stock].sum()})

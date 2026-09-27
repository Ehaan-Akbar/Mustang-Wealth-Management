"""
Core Black-Litterman model.

Implements the standard Black-Litterman (1992) framework:

    Prior (equilibrium) returns:
        pi = delta * Sigma @ w_mkt

    Posterior (combined) returns:
        E[R] = M @ (tau_Sigma_inv @ pi + P.T @ Omega_inv @ Q)
        M    = inv(tau_Sigma_inv + P.T @ Omega_inv @ P)

    Posterior covariance:
        Sigma_posterior = Sigma + M

where:
    Sigma  = asset covariance matrix (n x n)
    w_mkt  = market-cap weights (n,)
    delta  = risk aversion coefficient
    tau    = scalar reflecting uncertainty in the prior (typically 0.01-0.05)
    P      = view "pick" matrix (k x n)
    Q      = view return vector (k,)
    Omega  = view uncertainty covariance matrix (k x k)
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd


@dataclass
class View:
    """
    A single investor view.

    Absolute view example: "AAPL will return 8% annually"
        View(assets={"AAPL": 1.0}, expected_return=0.08, confidence=0.6)

    Relative view example: "AAPL will outperform MSFT by 3%"
        View(assets={"AAPL": 1.0, "MSFT": -1.0}, expected_return=0.03, confidence=0.4)

    confidence in (0, 1]: higher = more certain. Used to scale the view's
    entry in the Omega uncertainty matrix (via the He-Litterman approach).
    """

    assets: Dict[str, float]
    expected_return: float
    confidence: float = 0.5

    def __post_init__(self):
        if not (0 < self.confidence <= 1):
            raise ValueError("confidence must be in (0, 1]")
        if not self.assets:
            raise ValueError("a view must reference at least one asset")


class BlackLittermanModel:
    def __init__(
        self,
        cov_matrix: pd.DataFrame,
        market_weights: Optional[Union[pd.Series, Dict[str, float]]] = None,
        risk_aversion: float = 2.5,
        tau: float = 0.05,
        risk_free_rate: float = 0.0,
    ):
        """
        Parameters
        ----------
        cov_matrix : DataFrame (n x n)
            Annualized covariance matrix of asset returns. Index/columns
            must be the asset tickers.
        market_weights : Series or dict, optional
            Market-capitalization weights used to derive the equilibrium
            (prior) returns via reverse optimization. If omitted, an
            equal-weighted prior is used.
        risk_aversion : float
            Market risk-aversion coefficient (delta). A common estimate is
            (market excess return) / (market variance). Default 2.5 is a
            reasonable generic value.
        tau : float
            Scalar controlling how much weight is placed on the prior
            relative to the views. Typical values: 0.01-0.05.
        risk_free_rate : float
            Used only when back-deriving implied returns from a market
            excess-return context; kept for convenience/reporting.
        """
        if not isinstance(cov_matrix, pd.DataFrame):
            raise TypeError("cov_matrix must be a pandas DataFrame")
        if cov_matrix.shape[0] != cov_matrix.shape[1]:
            raise ValueError("cov_matrix must be square")

        self.tickers: List[str] = list(cov_matrix.columns)
        self.sigma: pd.DataFrame = cov_matrix.loc[self.tickers, self.tickers]
        self.n = len(self.tickers)
        self.delta = risk_aversion
        self.tau = tau
        self.risk_free_rate = risk_free_rate

        if market_weights is None:
            w = pd.Series(1.0 / self.n, index=self.tickers)
        else:
            w = pd.Series(market_weights).reindex(self.tickers)
            if w.isnull().any():
                missing = w[w.isnull()].index.tolist()
                raise ValueError(f"market_weights missing entries for: {missing}")
            w = w / w.sum()
        self.market_weights = w

        # Implied equilibrium (prior) excess returns: pi = delta * Sigma @ w
        self.pi = self.delta * self.sigma.values @ self.market_weights.values
        self.pi = pd.Series(self.pi, index=self.tickers, name="implied_return")

        self.views: List[View] = []

    # ------------------------------------------------------------------
    # View management
    # ------------------------------------------------------------------
    def add_view(self, view: View) -> "BlackLittermanModel":
        for asset in view.assets:
            if asset not in self.tickers:
                raise ValueError(f"view references unknown asset '{asset}'")
        self.views.append(view)
        return self

    def add_absolute_view(
        self, asset: str, expected_return: float, confidence: float = 0.5
    ) -> "BlackLittermanModel":
        return self.add_view(
            View(assets={asset: 1.0}, expected_return=expected_return, confidence=confidence)
        )

    def add_relative_view(
        self,
        outperformer: str,
        underperformer: str,
        expected_return: float,
        confidence: float = 0.5,
    ) -> "BlackLittermanModel":
        return self.add_view(
            View(
                assets={outperformer: 1.0, underperformer: -1.0},
                expected_return=expected_return,
                confidence=confidence,
            )
        )

    def clear_views(self) -> None:
        self.views = []

    # ------------------------------------------------------------------
    # Matrix construction
    # ------------------------------------------------------------------
    def _build_view_matrices(self):
        k = len(self.views)
        P = np.zeros((k, self.n))
        Q = np.zeros(k)
        confidences = np.zeros(k)

        col_index = {t: i for i, t in enumerate(self.tickers)}
        for row, view in enumerate(self.views):
            for asset, weight in view.assets.items():
                P[row, col_index[asset]] = weight
            Q[row] = view.expected_return
            confidences[row] = view.confidence
        return P, Q, confidences

    def _build_omega(self, P: np.ndarray, confidences: np.ndarray) -> np.ndarray:
        """
        He-Litterman-style Omega: the diagonal variance of each view is
        proportional to tau * P Sigma P.T, then scaled inversely by the
        stated confidence (confidence=1 -> near-zero uncertainty,
        confidence->0 -> huge uncertainty i.e. view is nearly ignored).
        """
        raw_view_var = np.diag(P @ (self.tau * self.sigma.values) @ P.T)
        # Avoid division by zero; clip confidence away from the extremes.
        conf = np.clip(confidences, 1e-4, 1.0)
        scaled_var = raw_view_var * (1.0 - conf) / conf
        # Guard against exactly-zero variance views (e.g. degenerate P rows).
        scaled_var = np.where(scaled_var <= 0, 1e-8, scaled_var)
        return np.diag(scaled_var)

    # ------------------------------------------------------------------
    # Posterior computation
    # ------------------------------------------------------------------
    def posterior(self):
        """
        Returns
        -------
        posterior_returns : pd.Series
            Combined (posterior) expected excess returns per asset.
        posterior_cov : pd.DataFrame
            Combined (posterior) covariance matrix, i.e. Sigma + M, suitable
            for feeding into a mean-variance optimizer.
        """
        sigma = self.sigma.values
        tau_sigma_inv = np.linalg.inv(self.tau * sigma)

        if not self.views:
            # No views: posterior collapses to the prior.
            posterior_returns = self.pi.copy()
            posterior_cov = self.sigma.copy()
            return posterior_returns, posterior_cov

        P, Q, confidences = self._build_view_matrices()
        omega = self._build_omega(P, confidences)
        omega_inv = np.linalg.inv(omega)

        M = np.linalg.inv(tau_sigma_inv + P.T @ omega_inv @ P)
        combined_mean = M @ (tau_sigma_inv @ self.pi.values + P.T @ omega_inv @ Q)

        posterior_returns = pd.Series(combined_mean, index=self.tickers, name="posterior_return")
        posterior_cov = pd.DataFrame(sigma + M, index=self.tickers, columns=self.tickers)
        return posterior_returns, posterior_cov

    def summary(self) -> pd.DataFrame:
        """Convenience table comparing prior vs. posterior expected returns."""
        posterior_returns, _ = self.posterior()
        df = pd.DataFrame(
            {
                "market_weight": self.market_weights,
                "prior_return": self.pi,
                "posterior_return": posterior_returns,
                "delta": posterior_returns - self.pi,
            }
        )
        return df.sort_values("posterior_return", ascending=False)

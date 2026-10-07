# Black-Litterman Portfolio Optimization

A small, dependency-light Python toolkit implementing the Black-Litterman
asset allocation model, plus a mean-variance optimizer to turn the model's
output into portfolio weights.

## What's inside

```
black_litterman/
├── black_litterman.py   # Core BL model: priors, views, posterior returns/cov
├── optimizer.py          # Mean-variance optimizer (max Sharpe, min vol, frontier)
├── data_utils.py          # Price download, returns, covariance, market-cap weights
examples/
└── run_example.py        # End-to-end demo (works offline with synthetic data)
tests/
└── test_black_litterman.py
```

## Install

```bash
git clone <this-repo-url>
cd black-litterman-portfolio
pip install -r requirements.txt
pip install -e .          # optional, installs the package for `import black_litterman`
```

Or, once cloned, add it straight to your own project's dependencies:

```bash
pip install git+https://github.com/<you>/black-litterman-portfolio.git
```

## Quickstart

```python
from black_litterman import BlackLittermanModel, MeanVarianceOptimizer

# cov_matrix: an (n x n) pandas DataFrame of annualized asset covariances
# market_weights: a pandas Series/dict of market-cap weights, index = tickers
model = BlackLittermanModel(
    cov_matrix=cov_matrix,
    market_weights=market_weights,
    risk_aversion=2.5,   # delta
    tau=0.05,
)

# Add views
model.add_absolute_view("AAPL", expected_return=0.10, confidence=0.6)
model.add_relative_view("JPM", "XOM", expected_return=0.04, confidence=0.4)

posterior_returns, posterior_cov = model.posterior()

optimizer = MeanVarianceOptimizer(posterior_returns, posterior_cov, risk_free_rate=0.02)
weights = optimizer.max_sharpe()
print(weights)
```

Run the full worked example (no API keys or network needed):

```bash
python examples/run_example.py
```

## Interactive dashboard

Install the dependencies and launch the Streamlit interface:

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

The dashboard works offline with synthetic correlated returns, or accepts a
CSV of daily prices/returns. It includes editable market assumptions and
investor views, maximum-Sharpe/minimum-volatility optimization, and a
block-bootstrap Monte Carlo projection. The **Iterations** control supports
100 to 100,000 simulation paths.

Use the sidebar to switch between **Portfolio analysis**, **Glide path &
liabilities**, and **Funding & sensitivity**. The glide-path view shows the
annual equity/bond/cash targets and liability schedule. Funding analysis accepts
contributions, annual liabilities, asset-class returns, volatility, correlation,
simulation count, and seed; results include overall and yearly funding
probabilities, shortfalls, and CSV downloads.

Run one-variable sensitivity, two-variable sensitivity, and optimistic/base/
pessimistic comparisons separately. These comparisons use the module's baseline
assumptions, displayed in the dashboard, independently of custom funding inputs.
Results remain available while navigating between planning screens. Larger
comparison runs can take several minutes.

The funding model applies annual returns before paying liabilities using the
dynamic glide path. The block-bootstrap model uses fixed optimized ticker weights
and beginning-of-year cash flows. In Portfolio analysis, the **Historical stress
tests** tab replays the uploaded data for the 2008 crisis, COVID crash, and 2022
bear market; synthetic demo data is excluded, and missing or partial periods
are identified.

To use real historical prices instead of synthetic data, set
`USE_YFINANCE = True` at the top of `examples/run_example.py` (requires
`pip install yfinance`).

### Funding Probability & Sensitivity Analysis

Measures the probability that the portfolio can fully fund the required $50,000 annual liabilities from 2033 through 2042. Monte Carlo simulations are used to estimate both overall and year-by-year funding probabilities.

The analysis also tests how changes in key assumptions, including annual liabilities, equity returns, bond returns, and equity volatility, affect funding certainty. Additional scenario and shortfall analyses evaluate optimistic, base, and pessimistic conditions while quantifying potential funding gaps.

#### Methods
- Monte Carlo funding probability
- Year-by-year funding probability
- Shortfall analysis
- One-variable sensitivity analysis
- Two-variable sensitivity analysis
- Optimistic, base, and pessimistic scenario analysis

## How the model works

1. **Prior (equilibrium) returns** are derived by reverse-optimizing the
   market portfolio: `pi = delta * Sigma @ w_market`. This assumes the
   market-cap-weighted portfolio is itself mean-variance efficient given
   the covariance matrix `Sigma` and risk-aversion coefficient `delta`.

2. **Views** express your own opinions, either:
   - **Absolute**: "Asset X will return r%" — `add_absolute_view`
   - **Relative**: "Asset X will outperform Asset Y by r%" — `add_relative_view`

   Each view carries a `confidence` in `(0, 1]`. Higher confidence pulls
   the posterior further toward your view and shrinks that view's entry
   in the uncertainty matrix `Omega` (built via the standard
   He-Litterman proportional-variance approach).

3. **Posterior returns/covariance** blend the prior and the views using
   the closed-form Black-Litterman update:

   ```
   M = [ (tau*Sigma)^-1 + P' Omega^-1 P ]^-1
   E[R] = M [ (tau*Sigma)^-1 pi + P' Omega^-1 Q ]
   Sigma_posterior = Sigma + M
   ```

4. **Optimization**: feed `(posterior_returns, posterior_cov)` into
   `MeanVarianceOptimizer` to get max-Sharpe weights, minimum-volatility
   weights, a weight vector for a specific target return, or a sampled
   efficient frontier.

## Testing

```bash
pytest tests/
```

## Notes & caveats

- This is an educational/research implementation, not investment advice.
  Validate all inputs (covariance estimation method, market-cap data,
  view calibration) before using it for real capital allocation.
- The covariance matrix should be well-conditioned (e.g., enough
  historical observations relative to the number of assets, or a
  shrinkage estimator) — Black-Litterman inherits any instability in
  `Sigma`.
- `tau` and `risk_aversion` are the two levers most worth tuning; small
  changes in `tau` noticeably change how strongly views move the
  posterior away from the market-implied prior.

## License

MIT — see [LICENSE](LICENSE).

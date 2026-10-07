# Mustang Wealth Management

## Overview

Mustang Wealth Lab is a Python research toolkit and interactive dashboard for building stock and ETF portfolios, testing investment views, and exploring future funding outcomes. It combines Black–Litterman allocation, mean–variance optimization, daily Monte Carlo simulation, historical block bootstrap, and liability planning.

The dashboard connects asset selection to portfolio construction and simulation in one workflow. Separate planning tools evaluate a 2027–2042 contribution and liability schedule using an equity, bond, and cash glide path. The underlying models can also be used directly from Python.

## Research Workflow

1. [Load and inspect data](#data-and-return-estimation): choose stocks and ETFs, download historical prices, upload a CSV, or explore synthetic data.
2. [Express investment views](#blacklitterman-model): combine a reference allocation with absolute return expectations and relative outperformance views.
3. [Construct an allocation](#allocation-objectives): optimize expected return relative to risk, subject to position limits.
4. [Generate portfolio paths](#daily-monte-carlo): simulate daily returns using historical assumptions or the Black–Litterman posterior.
5. [Compare historical resampling](#historical-block-bootstrap): project the same portfolio using consecutive blocks of observed returns.
6. [Replay market stress](#historical-stress-tests): apply the selected allocation to historical crisis periods.
7. [Evaluate future payments](#glide-path-and-funding-analysis): estimate liability funding probabilities, shortfalls, and sensitivity to assumptions.

## Getting Started

Python 3.11 or newer is recommended. From the project directory, install the dependencies and start the dashboard:

```powershell
python -m pip install -r requirements.txt
python -m streamlit run streamlit_app.py
```

Open **http://localhost:8501**. The interface requires Streamlit 1.50 or newer; rerun the installation command when upgrading from an older version. Stop the server with **Ctrl+C**.

For a first run:

1. Open **Assets & data**, choose an example preset or enter your own symbols, and click **Load selected assets**. Select **Synthetic demo** to work offline.
2. Open **Black–Litterman** to enable views, set confidence levels, and inspect the resulting weights.
3. Open **Simulations**, choose a method and horizon, and click **Run simulations**.
4. Inspect individual paths, compare outcomes, and download results.

An offline Python example is available from the repository root:

```powershell
python -m examples.run_example
```

For development or use of the package outside this directory:

```powershell
python -m pip install -e .
```

## Dashboard Organization

| Page | Purpose |
| --- | --- |
| **Assets & data** | Select the asset universe, load prices or returns, inspect historical growth and correlations, and export aligned data. |
| **Black–Litterman** | Edit reference weights, investment views, confidence, and optimization constraints; compare prior and posterior estimates. |
| **Simulations** | Run Monte Carlo, block bootstrap, or both; inspect actual paths, distributions, and target probabilities. |
| **Historical stress tests** | Replay an optimized allocation over available historical crisis data. |
| **Glide path & liabilities** | Inspect annual equity, bond, and cash targets alongside scheduled payments. |
| **Funding & sensitivity** | Estimate funding success and shortfalls, then run assumption and scenario comparisons. |

Portfolio settings and completed simulation results stay in the current Streamlit session when switching pages. Reloading the asset universe clears the previous portfolio simulation. Session results are not a permanent saved project; use CSV exports to retain them.

## Model Configuration

Portfolio controls are available in the dashboard. Python users can pass parameters directly to the functions in [portfolio_lab.py](black_litterman/portfolio_lab.py).

| Control | Default | Effect |
| --- | --- | --- |
| Reference allocation | Equal weights | Defines the equilibrium portfolio used to derive the prior; custom inputs are normalized. |
| Optimization objective | Maximum Sharpe | Chooses maximum Sharpe or minimum volatility. |
| Maximum position | 100% | Limits an individual asset's optimized weight. |
| Risk-free rate | 2% | Converts between excess and total return expectations and enters the Sharpe calculation. |
| Risk aversion, $\delta$ | 2.5 | Scales equilibrium excess returns. |
| Prior uncertainty, $\tau$ | 0.05 | Scales uncertainty in the equilibrium prior. |
| Covariance shrinkage | 5% | Blends the sample covariance toward its diagonal. |
| Simulation method | Compare both | Executes Monte Carlo and block bootstrap separately. |
| Starting balance | $100,000 | Sets the portfolio value at time zero. |
| Projection horizon | 10 years | Supports 1–40 years. |
| Iterations | 1,000 per method | Supports 100–100,000 paths per method in the UI. |
| Annual contribution / withdrawal | $0 / $0 | Adds or withdraws cash at the start of each projection year. |
| First withdrawal year | 1 | Determines when recurring withdrawals begin. |
| Bootstrap block length | 21 trading days | Controls how much consecutive historical behavior is retained per block. |
| Random seed | 42 | Makes a run reproducible for the same data and parameters. |
| Ending balance target | $200,000 | Sets the threshold used to calculate target probability from completed paths. |

Changing model inputs immediately updates the Black–Litterman allocation. Simulations execute only when **Run simulations** is clicked. If inputs change afterward, the previous results remain visible with a notice that they are out of date.

## Data and Return Estimation

Historical prices are downloaded through `yfinance` with price adjustment enabled. The dashboard caches downloads for one hour and offers CSV input as an alternative. CSV files must have dates in the first column and ticker symbols as the remaining column names. Daily simple returns use decimal values: `0.01` means a 1% return.

The portfolio lab requires at least 60 overlapping daily returns for every selected asset. Missing symbols are reported, missing observations are excluded, and missing prices are not forward-filled. Use assets quoted in the same currency; the app does not perform currency conversion.

### Daily Returns and Covariance

For adjusted price $P_{i,t}$, an asset's daily simple return is:

$$
r_{i,t} = \frac{P_{i,t}}{P_{i,t-1}} - 1
$$

The model annualizes daily covariance using 252 trading days:

$$
\Sigma = 252\,\operatorname{Cov}(r_t)
$$

The dashboard applies diagonal shrinkage before fitting Black–Litterman:

$$
\Sigma_{\lambda} = (1-\lambda)\Sigma + \lambda\operatorname{diag}(\Sigma)
$$

Here, $\operatorname{diag}(\Sigma)$ denotes a diagonal matrix containing the original asset variances. This reduces the influence of estimated cross-asset covariance, which can help stabilize highly correlated asset selections.

Historical growth charts and the correlation matrix describe the loaded sample. Synthetic demo histories are generated examples and do not represent the actual performance of their ticker labels.

## Black–Litterman Model

Implementation: [black_litterman.py](black_litterman/black_litterman.py), with dashboard integration in [portfolio_lab.py](black_litterman/portfolio_lab.py).

### Equilibrium Prior

The prior expected excess returns are obtained from a reference portfolio:

$$
\pi = \delta\Sigma_{\lambda}w_{\mathrm{ref}}
$$

Here, $w_{\mathrm{ref}}$ is the reference allocation and $\delta$ is the risk-aversion parameter. The dashboard starts with equal weights and permits a custom allocation. For a mixed stock and ETF universe, reference weights describe the intended benchmark allocation; ETF fund size is not automatically treated as company market capitalization.

### Investor Views

Two view types are supported:

- **Absolute:** an expected annual return for one asset, such as a 10% total return for a chosen stock.
- **Relative:** an expected annual return difference, such as one ETF outperforming another by 2 percentage points.

Each enabled view has a confidence level. The dashboard accepts absolute views as total returns and subtracts the risk-free rate before passing them to the core model. Relative views already describe return differences. Direct callers of `BlackLittermanModel` should supply absolute views in excess-return terms.

Let $P$ encode the assets in each view, $Q$ contain expected excess returns or return differences, and $\Omega$ describe view uncertainty. The implementation scales each view's uncertainty using its variance and confidence, with numerical safeguards for near-zero uncertainty.

### Posterior Estimates

The model combines the prior and views using:

$$
M = \left[(\tau\Sigma_{\lambda})^{-1} + P^T\Omega^{-1}P\right]^{-1}
$$

$$
\mu_{\mathrm{excess}} = M\left[(\tau\Sigma_{\lambda})^{-1}\pi + P^T\Omega^{-1}Q\right]
$$

$$
\Sigma_{\mathrm{post}} = \Sigma_{\lambda} + M
$$

With no views, the implementation returns the equilibrium prior and input covariance. For optimization and displayed return estimates, the dashboard adds the risk-free rate back to the posterior excess returns.

## Allocation Objectives

Implementation: [optimizer.py](black_litterman/optimizer.py).

The dashboard supports two objectives, solved with SciPy's SLSQP optimizer.

**Maximum Sharpe:**

$$
\max_w \frac{w^T\mu-r_f}{\sqrt{w^T\Sigma_{\mathrm{post}}w}}
$$

**Minimum volatility:**

$$
\min_w w^T\Sigma_{\mathrm{post}}w
$$

Here, $\mu$ contains total expected returns and $r_f$ is the risk-free rate. Dashboard allocations are long-only and constrained by:

$$
\sum_i w_i=1, \qquad 0\leq w_i\leq w_{\max}
$$

The position cap must permit a fully invested portfolio. For example, two assets require a cap of at least 50%. The Python optimizer also exposes target-return and efficient-frontier methods.

## Daily Monte Carlo

Implementation: [portfolio_lab.py](black_litterman/portfolio_lab.py).

The engine generates independent daily log-return draws for the weighted portfolio, using one of two calibrations.

### Historical Calibration

First calculate the daily return of the fixed-weight portfolio:

$$
r_{p,t}=w^Tr_t, \qquad \ell_t=\log(1+r_{p,t})
$$

Simulated log returns are drawn from a normal distribution fitted to the mean and sample standard deviation of $\ell_t$.

### Black–Litterman Calibration

The posterior gives portfolio drift $\mu_p=w^T\mu$ and volatility $\sigma_p=\sqrt{w^T\Sigma_{\mathrm{post}}w}$. Daily gross returns are generated as:

$$
G_t=\exp\left[\frac{\mu_p-\tfrac12\sigma_p^2}{252}+\frac{\sigma_p}{\sqrt{252}}Z_t\right],
\qquad Z_t\sim\mathcal{N}(0,1)
$$

This is a portfolio-level lognormal model with constant parameters. Each draw updates the balance along an individual path. Contributions and withdrawals are applied at each year's start, before that year's daily returns.

## Historical Block Bootstrap

Implementation: [block_bootstrap.py](black_litterman/block_bootstrap.py), used by the configurable engine in [portfolio_lab.py](black_litterman/portfolio_lab.py).

Block bootstrap constructs future paths from consecutive sequences of observed daily returns:

1. Calculate the historical daily return of the selected fixed-weight portfolio.
2. Randomly select starting positions in that return history.
3. Copy blocks of the chosen length, wrapping around the end of the sample when necessary.
4. Join enough blocks to cover the projection horizon and compound their returns.

Sampling the weighted portfolio series preserves simultaneous asset moves for fixed weights. Consecutive days retain dependence within each block; ordering between independently selected blocks is randomized.

Both simulation methods assume constant weights maintained through daily rebalancing. Black–Litterman views can change the optimized weights used by bootstrap, but they do not replace its historical return distribution.

The original `run_simulation()` function remains available for the fixed 2027–2042 cash-flow case. The dashboard's configurable stock/ETF simulations use `simulate_projection()`.

## Simulation Outputs and Reproducibility

The **Simulations** page separates output into three views:

| View | Contents |
| --- | --- |
| **Individual paths** | Actual retained daily trajectories, selectable path numbers, ending values, lowest observed balances, and withdrawal success. |
| **Outcome comparison** | Median and 5th–95th percentile bands, ending-value distributions, target probabilities, and overall withdrawal funding. |
| **Run details** | Completed-run settings, data source, asset universe, and the weights used. |

Each run records its number, completion time, elapsed runtime, seed, and computed path count. Progress is updated as batches finish. Every click of **Run simulations** executes the engine again; navigating pages or adjusting chart controls reuses the stored run.

For memory efficiency, the first **up to 200 paths** are retained at full daily resolution. The chart can display fewer paths and a subset of actual dates. **Year-end values and summary statistics include every simulated path.** These displayed paths are drawn from the same calculations used for the summaries.

CSV exports include a selected daily path, all retained daily paths, all annual paths, and outcome comparisons. Large path exports are prepared on request. With the same input data, parameters, and seed, a run is reproducible; change the seed for another random sample.

An ending-balance target can be changed without rerunning the engine. Its probability is the fraction of completed paths that end at or above that value. Withdrawal funding instead measures whether every scheduled withdrawal was fully paid along a path.

## Historical Stress Tests

Implementation: [stress_testing.py](black_litterman/stress_testing.py).

The stress page replays the optimized allocation against available daily returns from three periods:

| Scenario | Start | End |
| --- | --- | --- |
| 2008 Financial Crisis | January 1, 2008 | December 31, 2008 |
| COVID Crash | February 19, 2020 | March 23, 2020 |
| 2022 Bear Market | January 3, 2022 | October 12, 2022 |

The dashboard uses a $100,000 starting balance and reports total return, maximum drawdown, worst daily return, and ending value. It also displays the portfolio-value path and the actual dates available in the loaded data.

Load a date range covering the desired crisis. Missing periods are identified, partial coverage is flagged, and synthetic demo histories are excluded. These are fixed-allocation historical replays, not walk-forward backtests of a strategy trained only on information available at the time.

## Glide Path and Funding Analysis

Implementation: [glide_path.py](black_litterman/glide_path.py) and [funding_probability.py](black_litterman/funding_probability.py).

This planning model uses three asset classes—equities, bonds, and cash—and a calendar-based allocation schedule. Its baseline assumes a $300,000 contribution in 2027, another $150,000 in 2028, and ten annual $50,000 liabilities from 2033 through 2042.

### Allocation Schedule

| Year | Equities | Bonds | Cash |
| --- | --- | --- | --- |
| 2027 | 70% | 25% | 5% |
| 2030 | 60% | 35% | 5% |
| 2032 | 35% | 55% | 10% |
| 2033 | 20% | 65% | 15% |
| 2042 | 2% | 74% | 24% |

The full annual schedule and liability totals are available on **Glide path & liabilities**.

### Funding Probability and Shortfall

For $N$ simulated paths, the estimated probability of funding every required payment is:

$$
\widehat{P}_{\mathrm{funded}}=\frac{1}{N}\sum_{j=1}^{N}\mathbf{1}\{\text{all liabilities are fully paid on path }j\}
$$

The funding page also reports payment success by year and the probability of any shortfall. Shortfall statistics use the largest single-year funding gap within each path; the reported average and median are conditional on paths with a shortfall.

Custom funding runs accept contributions, annual liabilities, asset-class returns and volatility, correlation, simulation count, and seed. Contributions occur at the start of the year, annual returns are applied, and liabilities are then paid. This payment timing differs from the stock/ETF simulation page, where withdrawals occur before the year's returns.

### Sensitivity and Scenario Comparisons

Comparisons run separately from custom funding inputs. They use the module's baseline: returns of 8% / 4% / 2%, volatility of 18% / 6% / 1% for equities / bonds / cash, and a common correlation of 0.20.

- **One-variable sensitivity** varies annual liabilities, equity returns, bond returns, or equity volatility.
- **Two-variable sensitivity** combines equity-return assumptions with annual liability amounts.
- **Scenario comparisons** evaluate optimistic, base, and pessimistic return/volatility assumptions.

The funding model's asset-class glide path is separate from the chosen stock/ETF allocation. Its probabilities should be interpreted using its own assumptions and cash-flow timing.

## Repository Guide

```text
Mustang-Wealth-Management/
├── streamlit_app.py                 # Dashboard entry point and page navigation
├── dashboard_portfolio.py           # Asset loading, BL controls, market-data views
├── dashboard_simulations.py         # Simulation controls, actual paths, exports
├── dashboard_planning.py            # Glide path, funding, and stress-test views
├── black_litterman/
│   ├── black_litterman.py           # Equilibrium prior, views, posterior estimates
│   ├── optimizer.py                 # Mean–variance allocation objectives
│   ├── data_utils.py                # Price downloads and return helpers
│   ├── portfolio_lab.py             # Data validation, model fitting, daily simulations
│   ├── block_bootstrap.py           # Circular block sampler and fixed-case simulation
│   ├── stress_testing.py            # Historical crisis replays
│   ├── glide_path.py                # Annual allocation and liability schedules
│   └── funding_probability.py       # Funding, shortfalls, sensitivity, scenarios
├── examples/                        # Python examples for allocation and simulations
├── tests/                           # Model and Streamlit workflow tests
├── requirements.txt                 # Dashboard, research, and test dependencies
└── pyproject.toml                   # Package metadata and optional dependencies
```

## Validation

Run the test suite from the repository root:

```powershell
python -m pytest tests -q
```

Tests cover model behavior, allocation constraints, data validation, simulation cash-flow timing, seeded reproducibility, funding inputs, and dashboard navigation. Simulation tests verify that retained daily paths agree with annual results, chart values come from actual stored paths, and navigation and exports do not trigger new runs.

## Assumptions and Limitations

- This is an educational research tool. Simulated outcomes are not guaranteed and do not constitute investment advice.
- Historical estimates depend on the chosen universe and sample period. Missing history can reduce the overlapping data available for analysis.
- Portfolio projections exclude fees, taxes, inflation, currency conversion, and trading frictions. Daily rebalancing is a modeling assumption.
- Monte Carlo assumes constant distribution parameters; block bootstrap can only resample behavior represented in the loaded history.
- Funding comparisons use a separate asset-class model and baseline. Keep its payment timing and assumptions with any reported results.
- To reproduce an exported result, retain the input data, settings, seed, and code revision. A seed alone cannot reproduce a result if the source data or implementation changes.

## License and Acknowledgment

Released under the [MIT License](LICENSE).

The organization of this README was inspired by [Gavin Ho's Wharton Investment Competition README](https://github.com/gavin-ho1/wharton-investment-comp/blob/main/README.md). The descriptions and equations here document Mustang Wealth Lab's own implementation and supported workflows.

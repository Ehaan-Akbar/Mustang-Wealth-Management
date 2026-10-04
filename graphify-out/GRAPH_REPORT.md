# Graph Report - .  (2026-10-04)

## Corpus Check
- Corpus is ~5,319 words - fits in a single context window. You may not need a graph.

## Summary
- 108 nodes · 247 edges · 8 communities
- Extraction: 98% EXTRACTED · 2% INFERRED · 0% AMBIGUOUS · INFERRED: 6 edges (avg confidence: 0.5)
- Token cost: 0 input · 0 output

## Community Hubs (Navigation)
- [[_COMMUNITY_Portfolio Optimization|Portfolio Optimization]]
- [[_COMMUNITY_Market Data Utilities|Market Data Utilities]]
- [[_COMMUNITY_Monte Carlo Simulation|Monte Carlo Simulation]]
- [[_COMMUNITY_Project Architecture|Project Architecture]]
- [[_COMMUNITY_Model Test Suite|Model Test Suite]]
- [[_COMMUNITY_Streamlit Dashboard|Streamlit Dashboard]]
- [[_COMMUNITY_Posterior Computation|Posterior Computation]]
- [[_COMMUNITY_Investor Views|Investor Views]]

## God Nodes (most connected - your core abstractions)
1. `BlackLittermanModel` - 30 edges
2. `MeanVarianceOptimizer` - 21 edges
3. `run_simulation()` - 15 edges
4. `compare_portfolios()` - 11 edges
5. `market_cap_weights()` - 10 edges
6. `compute_returns()` - 9 edges
7. `compute_covariance()` - 9 edges
8. `main()` - 9 edges
9. `View` - 8 edges
10. `download_prices()` - 8 edges

## Surprising Connections (you probably didn't know these)
- `ndarray` --uses--> `BlackLittermanModel`  [INFERRED]
  streamlit_app.py → black_litterman/black_litterman.py
- `float` --uses--> `BlackLittermanModel`  [INFERRED]
  streamlit_app.py → black_litterman/black_litterman.py
- `DataFrame` --uses--> `BlackLittermanModel`  [INFERRED]
  examples/run_example.py → black_litterman/black_litterman.py
- `str` --uses--> `BlackLittermanModel`  [INFERRED]
  streamlit_app.py → black_litterman/black_litterman.py
- `int` --uses--> `BlackLittermanModel`  [INFERRED]
  streamlit_app.py → black_litterman/black_litterman.py

## Communities (8 total, 0 thin omitted)

### Community 0 - "Portfolio Optimization"
Cohesion: 0.29
Nodes (7): MeanVarianceOptimizer, DataFrame, float, int, ndarray, Series, bool

### Community 1 - "Market Data Utilities"
Cohesion: 0.27
Nodes (14): compute_covariance(), compute_returns(), download_prices(), implied_risk_aversion(), market_cap_weights(), DataFrame, float, int (+6 more)

### Community 2 - "Monte Carlo Simulation"
Cohesion: 0.25
Nodes (15): block_bootstrap(), BootstrapResult, compare_portfolios(), laura_cash_flows(), plot_2033_distribution(), DataFrame, float, int (+7 more)

### Community 3 - "Project Architecture"
Cohesion: 0.18
Nodes (14): BlackLittermanModel, Black-Litterman Portfolio Optimization, Educational and research use, Absolute and relative investor views, Historical and synthetic market price data, MeanVarianceOptimizer, NumPy >= 1.23, pandas >= 1.5 (+6 more)

### Community 4 - "Model Test Suite"
Cohesion: 0.27
Nodes (8): BlackLittermanModel, test_absolute_view_shifts_return_towards_view(), test_invalid_view_asset_raises(), test_min_volatility_has_lowest_variance(), test_no_views_posterior_equals_prior(), test_optimizer_weights_sum_to_one(), test_prior_matches_reverse_optimization(), test_relative_view_direction()

### Community 5 - "Streamlit Dashboard"
Cohesion: 0.29
Nodes (9): load_uploaded_returns(), make_histogram(), money(), synthetic_returns(), DataFrame, float, int, ndarray (+1 more)

### Community 6 - "Posterior Computation"
Cohesion: 0.28
Nodes (4): DataFrame, float, ndarray, Series

### Community 7 - "Investor Views"
Cohesion: 0.36
Nodes (3): str, View, test_view_confidence_bounds()

## Knowledge Gaps
- **12 isolated node(s):** `ndarray`, `int`, `Series`, `bool`, `int` (+7 more)
  These have ≤1 connection - possible missing edges or undocumented components.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `BlackLittermanModel` connect `Model Test Suite` to `Market Data Utilities`, `Streamlit Dashboard`, `Posterior Computation`, `Investor Views`?**
  _High betweenness centrality (0.322) - this node is a cross-community bridge._
- **Why does `MeanVarianceOptimizer` connect `Portfolio Optimization` to `Market Data Utilities`, `Model Test Suite`, `Streamlit Dashboard`?**
  _High betweenness centrality (0.256) - this node is a cross-community bridge._
- **Why does `run_simulation()` connect `Monte Carlo Simulation` to `Market Data Utilities`, `Streamlit Dashboard`?**
  _High betweenness centrality (0.125) - this node is a cross-community bridge._
- **Are the 6 inferred relationships involving `BlackLittermanModel` (e.g. with `str` and `int`) actually correct?**
  _`BlackLittermanModel` has 6 INFERRED edges - model-reasoned connections that need verification._
- **What connects `ndarray`, `int`, `Series` to the rest of the system?**
  _12 weakly-connected nodes found - possible documentation gaps or missing edges._
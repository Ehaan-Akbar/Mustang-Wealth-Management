# Mustang Wealth Management

## Investment strategy for Laura Gao

Mustang Wealth Management is our goals-based investment strategy for the 2026 Wharton Global High School Investment Competition. We treat Laura Gao's portfolio as two connected obligations: grow capital for a 2033 creative residency in Taiwan, then reliably fund ten $50,000 operating payments from 2033 through 2042. The facility contribution is important, but it is conditional on securing those payments and preserving flexibility.

Our guiding principle is **protect the promise; grow the possibility**. Before 2033, the portfolio seeks diversified growth while steadily reducing the chance that a market decline near the commitment date disrupts the plan. In 2033, assets are divided into an operating reserve, a facility contribution, and flexible capital. We do not assume grants, co-sponsors, program fees, or future business income will cover any of the ten required payments.

## Client goals and investment implications

The case specifies a $300,000 investment at the beginning of 2027 and a $150,000 addition at the beginning of 2028. No further portfolio cash flows occur before 2033. Beginning in 2033, the portfolio must pay $50,000 at the start of each year for ten years. Laura's living expenses are covered elsewhere, giving the portfolio capacity to accept measured risk before 2033, while the fixed operating commitment limits how much risk is appropriate as the first payment approaches.

The residency reflects Laura's work as an author, educator, entrepreneur, and community builder. We express that alignment through the goals, stewardship, and decision process—not by concentrating the portfolio in technology, media, or Taiwan-related investments. Diversification is especially valuable because her career and future project already have exposure to creative and technology industries.

The case does not specify the facility's total cost, so we recommend a contribution based on what the portfolio can support rather than a predetermined target. The 2031 co-sponsor conversation is an early risk date: any range communicated then should be defensible under weaker markets as well as favorable ones.

## Three-stage portfolio plan

| Period | Portfolio role | Strategy |
| --- | --- | --- |
| **2027–2030: Grow** | Build capital while there are no planned withdrawals. | Maintain a diversified, long-only portfolio across equities and high-quality fixed income. An illustrative starting mix is 60–70% equities and 30–40% bonds and cash, subject to eligible securities and scenario analysis. Avoid relying on a single sector, issuer, or region. |
| **2031–2032: Protect** | Prepare for the co-sponsor discussion and the first payment. | Gradually reduce equity exposure and move assets needed for the operating reserve toward cash and high-quality bonds. Update projected 2033 outcomes and the contribution range as actual market results arrive. |
| **2033–2042: Match and maintain flexibility** | Pay the operating commitment and support the facility responsibly. | First fund the full operating reserve. Match the near-term payments with cash and laddered high-quality bonds maturing before each payment. Contribute only from the remaining portfolio, retaining a separate flexibility buffer. Invest any longer-horizon remainder according to its own time horizon. |

The project's dynamic glide path makes the de-risking gradual: it begins near 70% equities in 2027, falls to about 50% by 2031, and reaches about 20% in 2033, with the balance in bonds and cash. These are policy guideposts that should be reviewed against funding results and market conditions, not a promise that one allocation is optimal in every scenario. Once payments begin, the operating reserve should be managed to the payment dates rather than exposed to the growth portfolio's full volatility.

At 2033, the facility contribution follows a simple priority rule:

> **Facility contribution = portfolio value − fully funded operating reserve − flexibility buffer**

The contribution cannot fall below zero. The flexibility buffer protects against project changes, unexpected costs, and the risk of committing every remaining dollar. It should be sized from portfolio outcomes and stated as a policy choice, not as an estimate of facility cost. A preliminary planning discussion can test a 10–20% buffer of assets remaining after the operating reserve; the final recommendation should be grounded in the simulation results.

## How the models work together

Each model answers a different decision question, and their outputs feed the next step:

1. **Market data and return estimates** establish the evidence base. Adjusted prices are converted to daily returns; covariance estimates describe volatility and how assets move together. Covariance shrinkage reduces dependence on noisy historical correlations.
2. **Black–Litterman** combines a diversified reference portfolio with explicit absolute or relative investment views. Its posterior expected returns provide a disciplined input to allocation decisions while allowing views to influence the result in proportion to their confidence.
3. **Mean–variance optimization** translates expected returns and covariance into portfolio weights. Maximum Sharpe is the growth-phase objective; minimum volatility is available when protecting capital takes priority. Long-only and position-limit constraints help prevent a mathematically attractive result from becoming an impractical concentration.
4. **The glide path and liability schedule** connect those portfolio decisions to the actual calendar of contributions and payments. Equity exposure declines as 2033 approaches, while cash and bonds take on a larger role in funding near-term obligations.
5. **Block-bootstrap and Monte Carlo projections** test whether the resulting strategy can support the $50,000 payments and show the range of possible 2033 portfolio values. Block bootstrap resamples consecutive historical return periods, retaining some short-run market behavior; Monte Carlo explores a wider distribution under stated return and volatility assumptions. Funding probability is the share of simulated paths that make every payment in full.
6. **Historical stress tests and sensitivity analysis** challenge the plan under severe observed market periods and changes to assumptions such as returns, volatility, and liability size. These checks show where a strategy is vulnerable and whether the facility contribution or buffer should change.

Together, these models connect security-level views to portfolio weights, portfolio weights to a time-based risk path, and that path to the funding outcomes Laura actually cares about. No single model establishes certainty: the allocation model proposes weights, while liability matching, simulations, and stress analysis test whether those weights serve the goals.

## Why these models—and what they cannot tell us

Black–Litterman is useful when an investment team has informed views but does not want subjective forecasts to overwhelm a diversified market-based starting point. Mean–variance optimization makes the return-versus-risk tradeoff explicit. The glide path reflects the changing time horizon: growth matters early, while payment reliability matters more as 2033 approaches. Block bootstrap and Monte Carlo are complementary because one reuses observed return sequences and the other explores outcomes from parameterized distributions. Stress and sensitivity analysis make the recommendation less dependent on a single average-return forecast.

We define **high funding certainty** through two tests: (1) by the beginning of 2033, the reserve is composed of cash and high-quality instruments matched to all ten payment dates; and (2) before 2033, simulations show at least a 95% probability—preferably 97.5%—of having enough assets to establish that reserve. These are decision thresholds, not guarantees. Results depend on return, volatility, correlation, interest-rate, and timing assumptions; historical data cannot predict every future regime. The final report should disclose those assumptions, test poor early-return sequences and inflation scenarios, and distinguish the nominal fixed $50,000 case payments from the residency's potentially rising real costs.

The strategy outline also calls for VaR/CVaR optimization as a downside-risk tool. That is a prospective extension: the current repository does not implement a VaR/CVaR optimizer. Existing historical stress tests, downside outcomes in simulation, and funding shortfall analysis provide the implemented risk checks. Any VaR/CVaR result should be presented only after that component is built and validated.

The 2031 co-sponsor range should be derived from the distribution of 2033 **facility capacity after** reserving the full operating liability and flexibility buffer. A defensible lower bound can use a conservative percentile and an upper bound a moderate favorable percentile; neither should rely on an optimistic tail. The range should be refreshed in 2031 and include the confidence level from the model. If markets disappoint, reduce the facility contribution before weakening the operating reserve.

## Competition workflow

The WInS trading portfolio is the competition-period implementation and demonstration of our decision process. Long-term projections begin with the case's $300,000 in 2027 and $150,000 in 2028; WInS trading gains or losses are not added to those projections. The three deliverables should tell one consistent story: Trading Notes explain how decisions tested the strategy, the Investment Policy Statement defines its rules, and the Final Report evaluates implementation and recommends the reserve, facility contribution, flexibility buffer, and co-sponsor range.

## Research dashboard

This repository includes a Streamlit research dashboard for loading asset data, expressing Black–Litterman views, comparing allocation objectives, running simulations and stress tests, and exploring glide paths and funding sensitivity. It supports the strategy analysis; the competition recommendation should still be based on disclosed assumptions and reviewed outcomes.

```powershell
python -m pip install -r requirements.txt
python -m streamlit run streamlit_app.py
```

Use **Synthetic demo** for offline exploration. Synthetic data is illustrative and is not historical performance for the named securities. For CSV data, use dates in the first column and asset symbols in the remaining columns.

## Repository map

- `black_litterman/black_litterman.py` and `optimizer.py` — posterior return estimates and portfolio allocation.
- `black_litterman/glide_path.py` and `funding_probability.py` — annual asset mix, liabilities, funding probability, and sensitivity analysis.
- `black_litterman/block_bootstrap.py` and `portfolio_lab.py` — portfolio projections and simulation workflows.
- `black_litterman/stress_testing.py` — historical market stress tests.
- `dashboard_*.py` and `streamlit_app.py` — interactive research interface.
- `Info/` — competition case, strategy outline, and client research used to shape this strategy.

The models are decision-support tools for educational competition use. Their projections are estimates, not promises of investment performance.

"""Interactive Black-Litterman portfolio and Monte Carlo dashboard."""

from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd
import streamlit as st

from black_litterman import (
    BlackLittermanModel,
    MeanVarianceOptimizer,
    compute_covariance,
    compute_returns,
    market_cap_weights,
)
from black_litterman.block_bootstrap import run_simulation, summary
from dashboard_planning import show_funding, show_glide_path, show_stress_tests


DEFAULT_MARKET_CAPS = {
    "AAPL": 3400.0,
    "MSFT": 3100.0,
    "GOOGL": 2100.0,
    "AMZN": 2000.0,
    "JPM": 600.0,
    "XOM": 500.0,
}


@st.cache_data(show_spinner=False)
def synthetic_returns(tickers: tuple[str, ...], n_days: int, seed: int) -> pd.DataFrame:
    """Create correlated daily returns so the dashboard works offline."""
    rng = np.random.default_rng(seed)
    market_factor = rng.normal(0.0004, 0.011, size=n_days)
    idiosyncratic = rng.normal(0, 0.01, size=(n_days, len(tickers)))
    betas = rng.uniform(0.7, 1.4, size=len(tickers))
    values = idiosyncratic + np.outer(market_factor, betas)
    dates = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=n_days)
    return pd.DataFrame(values, index=dates, columns=tickers)


def load_uploaded_returns(uploaded_file) -> pd.DataFrame:
    """Load a CSV containing either daily prices or daily simple returns."""
    frame = pd.read_csv(uploaded_file, index_col=0, parse_dates=True)
    numeric = frame.select_dtypes(include=[np.number]).dropna()
    if numeric.empty:
        raise ValueError("The CSV must contain numeric asset columns.")
    return numeric.sort_index()


def make_histogram(values: np.ndarray, bins: int = 35) -> pd.DataFrame:
    counts, edges = np.histogram(values, bins=bins)
    centers = (edges[:-1] + edges[1:]) / 2
    return pd.DataFrame({"Portfolio paths": counts}, index=centers)


def money(value: float) -> str:
    return f"${value:,.0f}"


st.set_page_config(
    page_title="Mustang Wealth Lab",
    page_icon="📈",
    layout="wide",
)

st.title("Mustang Wealth Lab")
st.caption(
    "Portfolio allocation, glide-path planning, funding analysis, and historical stress tests."
)

workspace = st.sidebar.radio(
    "Workspace", ["Portfolio analysis", "Glide path & liabilities", "Funding & sensitivity"]
)
if workspace != "Portfolio analysis":
    if workspace == "Glide path & liabilities":
        show_glide_path()
    else:
        show_funding()
    st.divider()
    st.caption("Educational planning tool only — results are simulated, not guaranteed, and are not investment advice.")
    st.stop()

with st.sidebar:
    st.header("Model controls")
    data_source = st.radio("Return data", ["Synthetic demo", "Upload CSV"])
    uploaded_file = None
    upload_kind = "Daily returns"
    if data_source == "Upload CSV":
        uploaded_file = st.file_uploader("CSV file", type="csv")
        upload_kind = st.radio("CSV values", ["Daily returns", "Prices"])

    risk_aversion = st.slider("Risk aversion", 0.5, 10.0, 2.5, 0.1)
    tau = st.slider("Prior uncertainty (tau)", 0.001, 0.20, 0.05, 0.001)
    risk_free_rate = st.number_input(
        "Risk-free rate", min_value=-0.05, max_value=0.20, value=0.02, step=0.005,
        format="%.3f",
    )
    objective = st.selectbox("Portfolio objective", ["Maximum Sharpe", "Minimum volatility"])
    max_weight_pct = st.slider("Maximum position", 10, 100, 100, 5)

st.subheader("1 · Assets and market assumptions")
asset_input = pd.DataFrame(
    {"Ticker": list(DEFAULT_MARKET_CAPS), "Market cap ($B)": list(DEFAULT_MARKET_CAPS.values())}
)
asset_table = st.data_editor(
    asset_input,
    num_rows="dynamic",
    hide_index=True,
    width="stretch",
    column_config={
        "Ticker": st.column_config.TextColumn(required=True),
        "Market cap ($B)": st.column_config.NumberColumn(min_value=0.01, required=True),
    },
)

clean_assets = asset_table.dropna().copy()
clean_assets["Ticker"] = clean_assets["Ticker"].astype(str).str.strip().str.upper()
clean_assets = clean_assets[clean_assets["Ticker"] != ""].drop_duplicates("Ticker", keep="last")
tickers = tuple(clean_assets["Ticker"].tolist())
market_caps: Dict[str, float] = dict(
    zip(clean_assets["Ticker"], clean_assets["Market cap ($B)"].astype(float))
)

st.subheader("2 · Investor views")
st.caption("Add absolute return views or relative outperformance views. Leave the table empty to use the market prior.")
views = st.data_editor(
    pd.DataFrame(
        [
            {"Type": "Absolute", "Asset": "AAPL", "Compared with": "", "Expected return": 0.10, "Confidence": 0.60},
            {"Type": "Relative", "Asset": "JPM", "Compared with": "XOM", "Expected return": 0.04, "Confidence": 0.40},
        ]
    ),
    num_rows="dynamic",
    hide_index=True,
    width="stretch",
    column_config={
        "Type": st.column_config.SelectboxColumn(options=["Absolute", "Relative"], required=True),
        "Asset": st.column_config.SelectboxColumn(options=list(tickers), required=True),
        "Compared with": st.column_config.SelectboxColumn(options=[""] + list(tickers)),
        "Expected return": st.column_config.NumberColumn(format="%.2f", min_value=-1.0, max_value=1.0),
        "Confidence": st.column_config.NumberColumn(format="%.2f", min_value=0.01, max_value=1.0),
    },
)

st.subheader("3 · Block-bootstrap Monte Carlo settings")
st.caption("Resamples daily returns with fixed optimized weights. Cash flows occur at the start of each year.")
sim_a, sim_b, sim_c = st.columns(3)
with sim_a:
    iterations = st.number_input(
        "Iterations", min_value=100, max_value=100_000, value=10_000, step=1_000,
        help="More paths produce a more stable estimate but take longer.",
    )
with sim_b:
    block_size = st.number_input(
        "Block size (trading days)", min_value=1, max_value=252, value=21,
        help="Consecutive historical-return blocks preserve short-term dependence.",
    )
with sim_c:
    simulation_seed = st.number_input("Random seed", min_value=0, value=42, step=1)

with st.expander("Cash-flow schedule", expanded=False):
    st.caption("Positive values are contributions; negative values are withdrawals at the start of the year.")
    default_flows = pd.DataFrame(
        {
            "Year": list(range(2027, 2043)),
            "Cash flow": [300_000, 150_000, 0, 0, 0, 0] + [-50_000] * 10,
        }
    )
    flow_table = st.data_editor(default_flows, hide_index=True, width="stretch")

run = st.button("Run portfolio analysis", type="primary", width="stretch")

if run:
    try:
        if len(tickers) < 2:
            raise ValueError("Add at least two assets.")

        if data_source == "Upload CSV":
            if uploaded_file is None:
                raise ValueError("Upload a CSV file before running the analysis.")
            raw_data = load_uploaded_returns(uploaded_file)
            missing = [ticker for ticker in tickers if ticker not in raw_data.columns]
            if missing:
                raise ValueError(f"The CSV is missing these asset columns: {', '.join(missing)}")
            raw_data = raw_data.loc[:, tickers]
            returns = compute_returns(raw_data, method="simple") if upload_kind == "Prices" else raw_data
        else:
            returns = synthetic_returns(tickers, 1_500, int(simulation_seed))

        returns = returns.replace([np.inf, -np.inf], np.nan).dropna()
        covariance = compute_covariance(returns)
        model = BlackLittermanModel(
            covariance,
            market_cap_weights(market_caps),
            risk_aversion=float(risk_aversion),
            tau=float(tau),
        )

        for row in views.dropna(subset=["Type", "Asset", "Expected return", "Confidence"]).to_dict("records"):
            if row["Type"] == "Relative":
                compared = str(row.get("Compared with", "")).strip()
                if not compared or compared == row["Asset"]:
                    raise ValueError("Each relative view needs a different comparison asset.")
                model.add_relative_view(
                    row["Asset"], compared, float(row["Expected return"]), float(row["Confidence"])
                )
            else:
                model.add_absolute_view(
                    row["Asset"], float(row["Expected return"]), float(row["Confidence"])
                )

        posterior_returns, posterior_covariance = model.posterior()
        optimizer = MeanVarianceOptimizer(
            posterior_returns,
            posterior_covariance,
            risk_free_rate=float(risk_free_rate),
            max_weight=float(max_weight_pct) / 100,
        )
        weights = optimizer.max_sharpe() if objective == "Maximum Sharpe" else optimizer.min_volatility()
        expected_return, volatility, sharpe = optimizer.performance(weights)
        cash_flows = {
            int(row["Year"]): float(row["Cash flow"])
            for row in flow_table.dropna(subset=["Year", "Cash flow"]).to_dict("records")
        }

        with st.spinner(f"Running {int(iterations):,} Monte Carlo paths…"):
            result = run_simulation(
                returns,
                weights,
                n_sims=int(iterations),
                block_size=int(block_size),
                seed=int(simulation_seed),
                batch_size=min(500, int(iterations)),
                cash_flows=cash_flows,
            )

        st.success(f"Analysis complete using {int(iterations):,} simulated paths.")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Funding probability", f"{result.funding_probability:.1%}")
        m2.metric("Expected return", f"{expected_return:.1%}")
        m3.metric("Volatility", f"{volatility:.1%}")
        m4.metric("Sharpe ratio", f"{sharpe:.2f}")

        allocation_tab, outlook_tab, stress_tab, detail_tab = st.tabs(
            ["Allocation", "Monte Carlo outlook", "Historical stress tests", "Model detail"]
        )
        with allocation_tab:
            allocation = weights.rename("Weight").to_frame()
            st.bar_chart(allocation)
            st.dataframe(allocation.style.format("{:.2%}"), width="stretch")

        with outlook_tab:
            percentiles = result.year_start_values.quantile([0.05, 0.50, 0.95]).T
            percentiles.columns = ["5th percentile", "Median", "95th percentile"]
            st.line_chart(percentiles)
            st.caption("Portfolio value at the start of each year, after that year's cash flow.")
            st.markdown("**Ending value distribution (2042)**")
            st.bar_chart(make_histogram(result.ending_2042))

        with stress_tab:
            show_stress_tests(returns, weights, data_source)

        with detail_tab:
            left, right = st.columns(2)
            with left:
                st.markdown("**Prior vs. posterior returns**")
                st.dataframe(
                    model.summary().style.format(
                        {"market_weight": "{:.2%}", "prior_return": "{:.2%}", "posterior_return": "{:.2%}", "delta": "{:+.2%}"}
                    ),
                    width="stretch",
                )
            with right:
                st.markdown("**Simulation summary**")
                st.dataframe(
                    summary(result).rename("Value").to_frame().style.format(
                        lambda value: f"{value:.1%}" if abs(value) <= 1 else money(value)
                    ),
                    width="stretch",
                )
    except Exception as exc:
        st.error(str(exc))
        st.exception(exc)

st.divider()
st.caption("Educational planning tool only — results are simulated, not guaranteed, and are not investment advice.")

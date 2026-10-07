"""Interactive stock and ETF portfolio workbench."""

from datetime import date, timedelta
from hashlib import sha256
from io import BytesIO
import pickle

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from black_litterman.data_utils import download_prices
from black_litterman.portfolio_lab import fit_model, prepare_returns
from dashboard_planning import show_stress_tests
from dashboard_simulations import show_simulations


UNIVERSE = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "JPM", "XOM", "SPY", "VOO", "VTI", "QQQ", "VXUS", "VEA", "VWO", "BND", "AGG", "TLT", "IEF", "SGOV", "GLD", "VNQ"]
PRESETS = {
    "Stock mix": ["AAPL", "MSFT", "GOOGL", "JPM", "XOM"],
    "ETF mix": ["VTI", "VXUS", "BND", "GLD"],
    "Stocks + ETFs": ["AAPL", "MSFT", "SPY", "BND", "GLD"],
}
COLORS = ["#2563eb", "#0d9488", "#f59e0b", "#8b5cf6", "#e11d48", "#64748b"]


def fingerprint(value) -> str:
    return sha256(pickle.dumps(value)).hexdigest()[:16]


@st.cache_data(ttl=3600, show_spinner=False, max_entries=24)
def historical_prices(tickers, start, end):
    return download_prices(list(tickers), start.isoformat(), (end + timedelta(days=1)).isoformat())


@st.cache_data(show_spinner=False)
def demo_returns(tickers):
    rng = np.random.default_rng(42)
    market = rng.normal(0.0003, 0.01, 1260)
    values = market[:, None] * rng.uniform(0.4, 1.2, len(tickers)) + rng.normal(0.0001, 0.008, (1260, len(tickers)))
    return pd.DataFrame(values, index=pd.bdate_range(end=date.today(), periods=1260), columns=tickers)


def use_preset(name):
    st.session_state["lab_tickers"] = PRESETS[name]


def load_data():
    st.subheader("Choose your stocks & ETFs")
    st.caption("Choose stocks, ETFs, or a mix. Type a symbol and press Enter to add it. Presets are examples to explore.")
    for column, name in zip(st.columns(3), PRESETS):
        column.button(name, on_click=use_preset, args=(name,), width="stretch")
    with st.container(border=True):
        with st.form("market_data"):
            selected = st.multiselect("Stocks & ETFs", UNIVERSE, default=PRESETS["Stocks + ETFs"], accept_new_options=True, key="lab_tickers")
            a, b, c = st.columns([1.4, 1, 1])
            source = a.selectbox("Data source", ["Historical prices", "Synthetic demo", "Upload CSV"], key="lab_data_source")
            start = b.date_input("Start date", value=date.today() - timedelta(days=365 * 5), key="lab_data_start")
            end = c.date_input("End date", value=date.today(), key="lab_data_end")
            with st.expander("CSV upload"):
                upload = st.file_uploader("Daily data CSV", type="csv")
                kind = st.radio("CSV contains", ["Adjusted prices", "Daily returns"], horizontal=True)
                st.caption("First column: date. Remaining columns: ticker symbols. Daily returns use decimals (0.01 = 1%).")
            submitted = st.form_submit_button("Load selected assets", type="primary", width="stretch")
        st.caption("Historical prices use Yahoo Finance through yfinance and are cached for one hour. Demo data is synthetic. Use one quote currency; no currency conversion is applied.")
    if submitted:
        try:
            tickers = tuple(dict.fromkeys(t.strip().upper() for t in selected if t.strip()))
            if not 1 <= len(tickers) <= 30:
                raise ValueError("Choose between 1 and 30 symbols.")
            if start >= end:
                raise ValueError("Start date must be before end date.")
            with st.spinner("Loading and aligning your selected assets…"):
                if source == "Historical prices":
                    frame = historical_prices(tickers, start, end)
                    if frame is None or frame.empty:
                        raise ValueError("No prices were returned. Check the symbols and dates, or try CSV/demo data.")
                    returns = prepare_returns(frame, tickers, prices=True)
                elif source == "Upload CSV":
                    if upload is None:
                        raise ValueError("Choose a CSV file in the upload section.")
                    frame = pd.read_csv(BytesIO(upload.getvalue()), index_col=0)
                    returns = prepare_returns(frame, tickers, prices=kind == "Adjusted prices")
                    returns = returns.loc[str(start):str(end)]
                    if len(returns) < 60:
                        raise ValueError("The selected dates need at least 60 overlapping daily returns.")
                else:
                    returns = prepare_returns(demo_returns(tickers), tickers, prices=False)
            st.session_state["lab_data"] = dict(returns=returns, source=source)
            st.session_state.pop("lab_simulations", None)
            st.session_state.pop("lab_model", None)
            st.session_state.pop("lab_model_error", None)
            st.success(f"Loaded {len(tickers)} assets and {len(returns):,} daily observations. Open Black–Litterman to set your views, or Simulations to run a projection.")
        except Exception as exc:
            st.error(f"Could not load this selection: {exc}")
            if "lab_data" in st.session_state:
                st.info("Your last successfully loaded universe is still shown below.")
    return st.session_state.get("lab_data")


def show_model(returns):
    tickers = tuple(returns.columns)
    key = fingerprint(tickers)
    st.markdown("**Express your views. See the allocation respond.**")
    st.caption("Changes below immediately refit Black–Litterman. Absolute views are total annual returns; relative views are annual outperformance in percentage points.")
    left, right = st.columns([1.5, 1])
    with left:
        absolute = st.data_editor(
            pd.DataFrame({"Use": False, "Asset": tickers, "Return (%)": 8.0, "Confidence (%)": 60.0}),
            key=f"absolute_{key}", hide_index=True, width="stretch", disabled=["Asset"],
            column_config={"Use": st.column_config.CheckboxColumn(),
                           "Return (%)": st.column_config.NumberColumn(min_value=-99.0, max_value=100.0, step=0.5, required=True),
                           "Confidence (%)": st.column_config.NumberColumn(min_value=1.0, max_value=100.0, step=5.0, required=True)},
        )
        with st.expander("Add relative views"):
            relative = st.data_editor(
                pd.DataFrame(columns=["Asset", "Compared with", "Outperformance (%)", "Confidence (%)"]),
                key=f"relative_{key}", num_rows="dynamic", hide_index=True, width="stretch",
                column_config={
                    "Asset": st.column_config.SelectboxColumn(options=list(tickers), required=True),
                    "Compared with": st.column_config.SelectboxColumn(options=list(tickers), required=True),
                    "Outperformance (%)": st.column_config.NumberColumn(min_value=-100.0, max_value=100.0, default=2.0, required=True),
                    "Confidence (%)": st.column_config.NumberColumn(min_value=1.0, max_value=100.0, default=60.0, required=True),
                },
            )
        with st.expander("Reference allocation", expanded=False):
            st.caption("These weights define the equilibrium prior and your comparison portfolio. Equal weights are the starting point. For mixed stocks and ETFs, enter your reference allocation rather than treating ETF size as stock market capitalization. Values are normalized to 100%.")
            reference_table = st.data_editor(
                pd.DataFrame({"Asset": tickers, "Reference (%)": 100 / len(tickers)}),
                key=f"reference_{key}", disabled=["Asset"], hide_index=True, width="stretch",
                column_config={"Reference (%)": st.column_config.NumberColumn(min_value=0.0, max_value=100.0, required=True)},
            )
    with right:
        objective = st.selectbox("Portfolio objective", ["Maximum Sharpe", "Minimum volatility"], key="lab_model_objective")
        max_weight = st.slider("Maximum position (%)", 1, 100, 100, key="lab_max_weight") / 100
        risk_free = st.number_input("Risk-free rate (%)", -5.0, 20.0, 2.0, 0.25, key="lab_model_rf") / 100
        with st.expander("Model tuning"):
            risk_aversion = st.slider("Risk aversion", 0.5, 10.0, 2.5, 0.1, key="lab_model_risk")
            tau = st.slider("Prior uncertainty (tau)", 0.001, 0.2, 0.05, 0.001, key="lab_model_tau")
            shrinkage = st.slider("Covariance shrinkage (%)", 1, 100, 5,
                                  help="Blends the covariance toward its diagonal to stabilize highly correlated assets.", key="lab_model_shrinkage") / 100
    selected_absolute = absolute.loc[absolute["Use"].fillna(False)].drop(columns="Use")
    if selected_absolute.isna().any().any():
        raise ValueError("Complete every enabled absolute view.")
    if relative.isna().any().any():
        raise ValueError("Complete or delete unfinished relative views.")
    reference = reference_table.set_index("Asset")["Reference (%)"]
    result = fit_model(returns, reference, selected_absolute.to_dict("records"), relative.to_dict("records"),
                       risk_aversion, tau, risk_free, max_weight, objective, shrinkage)
    st.session_state["lab_model"] = result
    st.session_state.pop("lab_model_error", None)
    a, b, c = st.columns(3)
    expected, volatility, sharpe = result["performance"]
    baseline = result["reference_performance"]
    a.metric("Expected annual return", f"{expected:.1%}", f"{(expected - baseline[0]) * 100:+.2f} pp vs reference")
    b.metric("Annual volatility", f"{volatility:.1%}", f"{(volatility - baseline[1]) * 100:+.2f} pp vs reference", delta_color="inverse")
    c.metric("Sharpe ratio", f"{sharpe:.2f}", f"{sharpe - baseline[2]:+.2f} vs reference")
    a, b = st.columns(2)
    with a:
        st.markdown("**Reference → optimized allocation**")
        plot = (result["summary"][["Reference weight", "Optimized weight"]] * 100).rename_axis("Asset").reset_index().melt("Asset", var_name="Allocation", value_name="Weight (%)")
        chart = alt.Chart(plot).mark_bar(cornerRadiusTopLeft=3, cornerRadiusTopRight=3).encode(
            x=alt.X("Asset:N", axis=alt.Axis(labelAngle=0)), y="Weight (%):Q", xOffset="Allocation:N",
            color=alt.Color("Allocation:N", scale=alt.Scale(range=COLORS[:2])),
            tooltip=["Asset", "Allocation", alt.Tooltip("Weight (%):Q", format=".1f")],
        ).properties(height=260)
        st.altair_chart(chart, width="stretch")
    with b:
        st.markdown("**How your views change expected returns**")
        st.bar_chart(result["summary"][["Prior return", "Posterior return"]] * 100, y_label="Annual return (%)", color=COLORS[:2])
    st.dataframe(result["summary"].style.format("{:.2%}"), width="stretch")
    st.download_button("Download allocation", result["summary"].to_csv(), "portfolio_allocation.csv", "text/csv")
    return result


def show_market_data(returns, source):
    st.caption(f"{source} · {returns.index.min():%b %d, %Y} to {returns.index.max():%b %d, %Y} · {len(returns):,} aligned daily observations. Missing dates are excluded; prices are never forward-filled.")
    st.markdown("**Growth of $100 over the loaded history**")
    st.line_chart(100 * (1 + returns).cumprod(), y_label="Value ($)")
    st.caption("Historical performance of individual assets, before costs.")
    stats = pd.DataFrame({"Annualized arithmetic return": returns.mean() * 252, "Annualized volatility": returns.std() * np.sqrt(252)})
    st.dataframe(stats.style.format("{:.2%}"), width="stretch")
    corr = returns.corr().rename_axis("Asset").reset_index().melt("Asset", var_name="Compared with", value_name="Correlation")
    heatmap = alt.Chart(corr).mark_rect().encode(x="Asset:N", y="Compared with:N", color=alt.Color("Correlation:Q", scale=alt.Scale(domain=[-1, 1], scheme="redblue", reverse=True)), tooltip=["Asset", "Compared with", alt.Tooltip("Correlation", format=".2f")]).properties(height=320)
    st.markdown("**Asset correlations**")
    st.altair_chart(heatmap, width="stretch")
    st.download_button("Download aligned returns", returns.to_csv(), "daily_returns.csv", "text/csv")


def navigate(page):
    st.session_state["workspace"] = page


def show_portfolio(page):
    data = load_data() if page == "Assets & data" else st.session_state.get("lab_data")
    if not data:
        st.info("Load stocks or ETFs on Assets & data first. You can also choose Synthetic demo to explore offline.")
        if page != "Assets & data":
            st.button("Go to Assets & data", on_click=navigate, args=("Assets & data",), type="primary")
        return
    returns, source = data["returns"], data["source"]
    st.caption(f"Active portfolio: {', '.join(returns.columns)} · {source} · {len(returns):,} daily returns · {returns.index.min():%Y-%m-%d} → {returns.index.max():%Y-%m-%d}")
    if source == "Synthetic demo":
        st.info("Demo data is active. The simulations run normally, using generated example histories instead of real market prices.")
    elif len(returns) < 504:
        st.info("Less than two years of overlapping history is available. A longer sample can include more market conditions.")
    if page == "Assets & data":
        a, b = st.columns(2)
        a.button("Set portfolio views →", on_click=navigate, args=("Black–Litterman",), width="stretch")
        b.button("Run a simulation →", on_click=navigate, args=("Simulations",), width="stretch")
        st.divider()
        show_market_data(returns, source)
        return
    if page == "Black–Litterman":
        try:
            show_model(returns)
            st.button("Continue to simulations →", on_click=navigate, args=("Simulations",), type="primary")
        except (ValueError, RuntimeError, np.linalg.LinAlgError) as exc:
            st.session_state["lab_model_error"] = str(exc)
            st.error(f"Update the model inputs: {exc}")
        return
    if st.session_state.get("lab_model_error"):
        st.error(f"Resolve your Black–Litterman inputs first: {st.session_state['lab_model_error']}")
        st.button("Review model inputs", on_click=navigate, args=("Black–Litterman",))
        return
    model = st.session_state.get("lab_model")
    if model is None:
        model = fit_model(returns, pd.Series(1 / len(returns.columns), index=returns.columns),
                          [], [], 2.5, 0.05, 0.02, 1.0, "Maximum Sharpe", 0.05)
        st.caption("Using the default Black–Litterman model: equal reference weights, no investor views. Customize it on the Black–Litterman page.")
    if page == "Simulations":
        show_simulations(returns, model, source)
    else:
        show_stress_tests(returns, model["weights"], "Upload CSV" if source != "Synthetic demo" else source)

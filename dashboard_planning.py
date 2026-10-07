"""Streamlit views for the glide-path and funding analysis modules."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from black_litterman import funding_probability as funding
from black_litterman.glide_path import (
    calculate_cumulative_liabilities,
    generate_glide_path,
    get_liability_schedule,
)
from black_litterman.stress_testing import SCENARIOS, run_stress_test


def show_glide_path() -> None:
    st.subheader("Glide path & liabilities")
    st.caption(
        "The model gradually shifts from equities to bonds and cash, with "
        "$50,000 due annually from 2033 through 2042."
    )
    a, b, c = st.columns(3)
    a.metric("Total scheduled liabilities", f"${calculate_cumulative_liabilities():,.0f}")
    b.metric("First payment", "2033")
    c.metric("Final payment", "2042")
    glide = generate_glide_path().set_index("Year")
    st.markdown("**Target allocation over time**")
    st.area_chart(glide[["Equity", "Bonds", "Cash"]] * 100, y_label="Allocation (%)")
    st.dataframe(
        glide.style.format({"Equity": "{:.0%}", "Bonds": "{:.0%}", "Cash": "{:.0%}", "Liability": "${:,.0f}"}),
        width="stretch",
    )
    st.markdown("**Liability schedule**")
    schedule = get_liability_schedule().set_index("Year")
    st.dataframe(schedule.style.format("${:,.0f}"), width="stretch")
    st.download_button("Download glide path", glide.to_csv(), "glide_path.csv", "text/csv")
    st.info(
        "Funding analysis uses these equity, bond, and cash targets. Portfolio analysis "
        "uses the optimized ticker weights and its own editable cash-flow schedule."
    )


@st.cache_data(show_spinner=False, max_entries=8)
def funding_results(iterations: int, seed: int, assumptions: dict) -> dict:
    kwargs = dict(n_simulations=iterations, random_seed=seed, **assumptions)
    return {
        "probability": funding.calculate_funding_probability(**kwargs),
        "yearly": funding.calculate_yearly_funding_probability(**kwargs),
        "shortfalls": funding.calculate_shortfall_statistics(**kwargs),
    }


@st.cache_data(show_spinner=False, max_entries=8)
def comparison_results(analysis: str, iterations: int, seed: int) -> pd.DataFrame:
    methods = {
        "One-variable sensitivity": funding.run_sensitivity_analysis,
        "Two-variable sensitivity": funding.run_two_variable_sensitivity,
        "Scenarios": funding.run_scenario_analysis,
    }
    return methods[analysis](n_simulations=iterations, random_seed=seed)


def show_funding() -> None:
    st.subheader("Funding & sensitivity")
    st.caption(
        "Estimate the chance of paying every annual liability from 2033 to 2042. "
        "This model draws correlated annual equity, bond, and cash returns and follows "
        "the glide path. Contributions arrive at the start of the year; liabilities "
        "are paid after the year's investment return."
    )
    with st.form("funding_controls"):
        a, b, c = st.columns(3)
        initial = a.number_input("2027 contribution ($)", min_value=0.0, value=funding.INITIAL_PORTFOLIO, step=10_000.0)
        second = b.number_input("2028 contribution ($)", min_value=0.0, value=funding.SECOND_YEAR_CONTRIBUTION, step=10_000.0)
        liability = c.number_input("Annual liability ($), 2033–2042", min_value=1.0, value=50_000.0, step=5_000.0)
        st.markdown("**Annual asset-class assumptions (%)**")
        assumptions = {}
        for column, label, prefix in zip(st.columns(3), ["Equity", "Bond", "Cash"], ["equity", "bond", "cash"]):
            with column:
                assumptions[f"{prefix}_return"] = st.number_input(
                    f"{label} return (%)", min_value=-50.0, max_value=50.0,
                    value=getattr(funding, f"{prefix.upper()}_RETURN") * 100, step=0.5,
                ) / 100
                assumptions[f"{prefix}_volatility"] = st.number_input(
                    f"{label} volatility (%)", min_value=0.0, max_value=100.0,
                    value=getattr(funding, f"{prefix.upper()}_VOLATILITY") * 100, step=0.5,
                ) / 100
        a, b, c = st.columns(3)
        # A common correlation for three assets is positive semidefinite only at >= -0.5.
        assumptions["correlation"] = a.number_input("Asset-class correlation", min_value=-0.5, max_value=1.0, value=funding.CORRELATION, step=0.05)
        iterations = b.number_input("Funding iterations", min_value=100, max_value=100_000, value=1_000, step=100)
        seed = c.number_input("Funding random seed", min_value=0, value=funding.RANDOM_SEED, step=1)
        submitted = st.form_submit_button("Run funding analysis", type="primary", width="stretch")

    assumptions.update(
        initial_portfolio=initial,
        second_year_contribution=second,
        liabilities={year: liability for year in funding.LIABILITIES},
    )
    if submitted:
        try:
            with st.spinner(f"Calculating funding and shortfalls for {iterations:,} paths…"):
                result = funding_results(int(iterations), int(seed), assumptions)
            st.session_state["funding_result"] = (result, int(iterations), int(seed), assumptions)
        except (ValueError, TypeError) as exc:
            st.session_state.pop("funding_result", None)
            st.error(str(exc))

    saved = st.session_state.get("funding_result")
    if saved:
        result, saved_iterations, saved_seed, saved_assumptions = saved
        st.caption(
            f"Last completed run: {saved_iterations:,} paths · seed {saved_seed} · "
            f"${saved_assumptions['initial_portfolio']:,.0f} in 2027 · "
            f"${saved_assumptions['second_year_contribution']:,.0f} in 2028 · "
            f"${saved_assumptions['liabilities'][2033]:,.0f} annual liability. "
            "Submit the form to apply changes."
        )
        a, b, c = st.columns(3)
        shortfalls = result["shortfalls"]
        a.metric("All liabilities funded", f"{result['probability']:.1%}")
        b.metric("Any shortfall", f"{shortfalls['Probability of Any Shortfall']:.1%}")
        c.metric("Worst annual shortfall", f"${shortfalls['Worst Shortfall']:,.0f}")
        yearly_tab, shortfall_tab = st.tabs(["Year-by-year funding", "Shortfalls"])
        with yearly_tab:
            yearly = result["yearly"].set_index("Year")
            st.bar_chart(yearly[["Funding Probability (%)"]], y_label="Funding probability (%)")
            st.caption("Each bar measures that individual year's payment; overall success requires every payment to be funded.")
            st.dataframe(yearly[["Liability", "Funding Probability"]].style.format(
                {"Liability": "${:,.0f}", "Funding Probability": "{:.1%}"}
            ), width="stretch")
            st.download_button("Download yearly funding", yearly.to_csv(), "yearly_funding.csv", "text/csv")
        with shortfall_tab:
            a, b = st.columns(2)
            a.metric("Average maximum shortfall", f"${shortfalls['Average Maximum Shortfall']:,.0f}")
            b.metric("Median maximum shortfall", f"${shortfalls['Median Maximum Shortfall']:,.0f}")
            st.caption("Average and median use only paths with a shortfall and measure the largest single-year funding gap in each path.")

    st.divider()
    st.markdown("**Sensitivity & scenario comparisons**")
    st.caption(
        "Comparisons use the model's baseline: $300,000 in 2027, $150,000 in 2028, "
        "$50,000 annual liabilities, returns of 8% / 4% / 2%, volatility of 18% / 6% / 1%, "
        "and correlation of 0.20. Each comparison varies only its listed assumptions. "
        "The custom funding inputs above do not change this baseline."
    )
    with st.form("comparison_controls"):
        analysis = st.selectbox("Comparison", ["One-variable sensitivity", "Two-variable sensitivity", "Scenarios"])
        a, b = st.columns(2)
        comparison_iterations = a.number_input("Paths per comparison", min_value=100, max_value=100_000, value=1_000, step=100,
                                              help="Sensitivity runs multiple simulations; larger counts may take several minutes.")
        comparison_seed = b.number_input("Comparison random seed", min_value=0, value=42, step=1)
        compare = st.form_submit_button("Run comparison")
    if compare:
        try:
            with st.spinner(f"Running {analysis.lower()} with {comparison_iterations:,} paths per case…"):
                frame = comparison_results(analysis, int(comparison_iterations), int(comparison_seed))
            st.session_state["comparison_result"] = (analysis, frame, comparison_iterations, comparison_seed)
        except (ValueError, TypeError) as exc:
            st.session_state.pop("comparison_result", None)
            st.error(str(exc))
    if "comparison_result" in st.session_state:
        name, frame, count, comparison_seed = st.session_state["comparison_result"]
        st.markdown(f"**{name} results**")
        st.caption(f"{count:,} paths per case · seed {comparison_seed} · baseline assumptions")
        if name == "One-variable sensitivity":
            for variable, group in frame.groupby("Variable", sort=False):
                plot = group.set_index("Value")[["Funding Probability (%)"]]
                if variable != "Annual Liability":
                    plot.index = plot.index * 100
                st.markdown(f"**{variable}**")
                st.line_chart(plot, x_label=f"{variable} ({'$' if variable == 'Annual Liability' else '%'})", y_label="Funding probability (%)")
        elif name == "Two-variable sensitivity":
            matrix = frame.pivot(index="Equity Return", columns="Annual Liability", values="Funding Probability")
            matrix.index = [f"{value:.0%}" for value in matrix.index]
            matrix.columns = [f"${value:,.0f}" for value in matrix.columns]
            matrix.index.name = "Equity return / annual liability"
            st.dataframe(matrix.style.format("{:.1%}"), width="stretch")
        else:
            st.bar_chart(frame.set_index("Scenario")[["Funding Probability (%)"]], y_label="Funding probability (%)")
        display = frame.drop(columns=["Funding Probability (%)"]).copy()
        if "Variable" in display:
            display["Value"] = [f"${v:,.0f}" if variable == "Annual Liability" else f"{v:.1%}" for variable, v in zip(display["Variable"], display["Value"])]
        formats = {column: "{:.1%}" for column in ["Funding Probability", "Equity Return", "Bond Return", "Equity Volatility"] if column in display}
        formats.update({column: "${:,.0f}" for column in ["Annual Liability", "Average Maximum Shortfall", "Worst Shortfall"] if column in display})
        st.dataframe(display.style.format(formats), hide_index=True, width="stretch")
        st.download_button("Download comparison", frame.to_csv(index=False), "funding_comparison.csv", "text/csv")


def show_stress_tests(returns: pd.DataFrame, weights: pd.Series, data_source: str) -> None:
    st.caption("Historical scenarios replay the optimized ticker allocation with a $100,000 starting value.")
    if data_source != "Upload CSV":
        st.info("Upload historical prices or daily returns to run stress tests. Synthetic demo data cannot represent historical crises.")
        return
    if not isinstance(returns.index, pd.DatetimeIndex):
        st.warning("Historical stress testing requires valid dates in the CSV's first column.")
        return
    for scenario, (start, end) in SCENARIOS.items():
        st.markdown(f"**{scenario}** · {start} to {end}")
        window = returns.loc[start:end]
        if len(window) < 2:
            st.info("This upload does not contain enough observations for this scenario.")
            continue
        if returns.index.min() > pd.Timestamp(start) + pd.Timedelta(days=7) or returns.index.max() < pd.Timestamp(end) - pd.Timedelta(days=7):
            st.warning("Partial period: upload data covering the full scenario to compare its results.")
        result = run_stress_test(returns, weights, scenario=scenario)
        a, b, c, d = st.columns(4)
        a.metric("Total return", f"{result.portfolio_return:.1%}")
        b.metric("Maximum drawdown", f"{result.maximum_drawdown:.1%}")
        c.metric("Worst day", f"{result.worst_day:.1%}")
        d.metric("Ending value", f"${result.ending_value:,.0f}")
        st.caption(f"{len(window):,} daily observations · {window.index.min():%Y-%m-%d} to {window.index.max():%Y-%m-%d}")
        st.line_chart(result.cumulative_value.rename("Portfolio value ($)"))

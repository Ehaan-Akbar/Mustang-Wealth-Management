"""Simulation controls and charts backed by retained engine output."""

from datetime import datetime
from hashlib import sha256
import pickle
from time import perf_counter

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from black_litterman.portfolio_lab import simulate_projection


def fingerprint(value):
    return sha256(pickle.dumps(value)).hexdigest()[:16]


def path_chart_frame(projection, count, highlighted):
    """Select genuine stored paths, limiting rendered dates without fabricating points."""
    paths = projection.daily_paths
    ids = list(paths.index[:count])
    if highlighted not in ids:
        ids.append(highlighted)
    # Keep every year-end plus a regularly spaced sample of real trading days.
    positions = np.unique(np.concatenate([
        np.linspace(0, paths.shape[1] - 1, min(600, paths.shape[1]), dtype=int),
        np.arange(0, paths.shape[1], 252),
    ]))
    return (paths.loc[ids].iloc[:, positions].rename_axis(index="Path", columns="Year")
            .reset_index().melt("Path", var_name="Year", value_name="Balance"))


def show_individual_paths(saved):
    results = saved["results"]
    if st.session_state.get("lab_path_method") not in results:
        st.session_state["lab_path_method"] = next(iter(results))
    a, b, c = st.columns([1.2, 1, 1])
    method = a.selectbox("View method", list(results), key="lab_path_method")
    result = results[method]
    retained = len(result.daily_paths)
    for key in ["lab_path_count", "lab_path_selected"]:
        if st.session_state.get(key, 1) > retained:
            st.session_state[key] = retained
    count = b.slider("Paths on chart", 1, retained, min(50, retained), key="lab_path_count")
    selected = int(c.number_input("Inspect path #", min_value=1, max_value=retained, value=1, key="lab_path_selected"))
    frame = path_chart_frame(result, count, selected)
    # Explicit inline data avoids Altair's 5,000-row dataframe limit.
    base = alt.Chart(alt.Data(values=frame.to_dict("records"))).encode(
        x=alt.X("Year:Q", title="Years from start", axis=alt.Axis(tickMinStep=1)),
        y=alt.Y("Balance:Q", title="Portfolio value ($)", axis=alt.Axis(format="~s")),
        detail="Path:N", order="Year:Q",
    )
    background = base.mark_line(color="#3b82f6", opacity=0.20, strokeWidth=1)
    highlight = base.transform_filter(alt.datum.Path == selected).mark_line(
        color="#0d9488", strokeWidth=2.5
    ).encode(tooltip=[alt.Tooltip("Path:N"), alt.Tooltip("Year:Q", format=".2f"), alt.Tooltip("Balance:Q", format="$,.0f")])
    st.altair_chart((background + highlight).properties(height=420).interactive(), width="stretch")
    st.caption(
        f"{method}: showing the first {count} of {len(result.values):,} simulated paths"
        f"{' plus the selected path' if selected > count else ''}. Path {selected} is highlighted in teal. "
        f"The first {retained} paths are retained at daily resolution; long charts display a subset "
        "of actual dates. Summary statistics use every simulated path."
    )
    path = result.daily_paths.loc[selected]
    a, b, c = st.columns(3)
    a.metric(f"Path {selected} · ending value", f"${path.iloc[-1]:,.0f}")
    b.metric("Lowest observed balance", f"${path.min():,.0f}")
    c.metric("All withdrawals funded", "Yes" if result.funded[selected - 1] else "No")
    with st.expander("Inspect this path's actual values"):
        st.dataframe(path.rename("Portfolio value ($)").to_frame(), width="stretch")
        st.download_button("Download selected daily path", path.to_csv(), f"{method.lower().replace(' ', '_')}_path_{selected}.csv", "text/csv")
    with st.expander("Export simulation paths"):
        st.caption("Daily export contains retained paths. Annual export contains every path from the run.")
        if st.button("Prepare retained daily paths CSV"):
            st.session_state["lab_daily_export"] = (saved["run_id"], method, result.daily_paths.to_csv())
        daily_export = st.session_state.get("lab_daily_export")
        if daily_export and daily_export[:2] == (saved["run_id"], method):
            st.download_button("Download retained daily paths", daily_export[2], "daily_simulation_paths.csv", "text/csv")
        # Build potentially large exports only on request.
        if st.button("Prepare all annual paths CSV"):
            annual = result.values.copy()
            annual.index = pd.RangeIndex(1, len(annual) + 1, name="Path")
            st.session_state["lab_annual_export"] = (saved["run_id"], method, annual.to_csv())
        export = st.session_state.get("lab_annual_export")
        if export and export[:2] == (saved["run_id"], method):
            st.download_button("Download all annual paths", export[2], "all_annual_simulation_paths.csv", "text/csv")


def show_outcomes(results, target):
    rows = []
    for name, result in results.items():
        bands = result.values.quantile([0.05, 0.5, 0.95]).T
        for year, row in bands.iterrows():
            rows.append({"Year": year, "Method": name, "Low": row[0.05], "Median": row[0.5], "High": row[0.95]})
    base = alt.Chart(pd.DataFrame(rows)).encode(
        x=alt.X("Year:Q", axis=alt.Axis(tickMinStep=1)),
        color=alt.Color("Method:N", scale=alt.Scale(range=["#2563eb", "#0d9488"])),
    )
    band = base.mark_area(opacity=0.12).encode(y=alt.Y("Low:Q", title="Portfolio value ($)"), y2="High:Q")
    median = base.mark_line(strokeWidth=3).encode(y="Median:Q", tooltip=["Method", "Year", alt.Tooltip("Median", format="$,.0f")])
    st.altair_chart((band + median).properties(height=320).interactive(), width="stretch")
    st.caption("Lines show the median; shaded regions show the 5th–95th percentiles across every path. These are summary ranges, not individual paths.")
    summaries = pd.DataFrame({name: result.summary(target) for name, result in results.items()}).T
    st.dataframe(summaries.style.format({
        column: "{:.1%}" if column in ["Target reached", "All withdrawals funded"] else "${:,.0f}"
        for column in summaries
    }), width="stretch")
    for column, (name, result) in zip(st.columns(len(results)), results.items()):
        with column:
            st.markdown(f"**{name} · ending balances**")
            counts, edges = np.histogram(result.values.iloc[:, -1], bins=35)
            st.bar_chart(pd.DataFrame({"Paths": counts}, index=(edges[:-1] + edges[1:]) / 2), x_label="Ending balance ($)")
    st.download_button("Download outcome comparison", summaries.to_csv(), "simulation_comparison.csv", "text/csv")


def show_simulations(returns, model, source):
    st.caption("Set up a run, then explore individual paths and compare the range of outcomes. Your results stay available when you navigate away.")
    with st.container(border=True):
        st.markdown("**1 · Choose a method and portfolio**")
        a, b, c = st.columns(3)
        mode = a.selectbox("Simulation method", ["Compare both", "Monte Carlo", "Block bootstrap"], key="lab_sim_mode")
        allocation = b.selectbox("Simulate allocation", ["Black–Litterman", "Reference allocation", "Equal weight"], key="lab_sim_allocation")
        calibration = c.selectbox("Monte Carlo assumptions", ["Historical returns", "Black–Litterman posterior"], disabled=mode == "Block bootstrap", key="lab_sim_calibration")
        methods = ["Monte Carlo", "Block bootstrap"] if mode == "Compare both" else [mode]
        st.caption("Monte Carlo generates daily random returns from fitted assumptions. Block bootstrap resamples sequences of actual daily returns. Both use your selected asset weights.")
        st.markdown("**2 · Set the size and duration of the run**")
        a, b, c = st.columns(3)
        initial = a.number_input("Starting balance ($)", min_value=0.0, value=100_000.0, step=10_000.0, key="lab_sim_initial")
        years = b.slider("Horizon (years)", 1, 40, 10, key="lab_sim_years")
        iterations = c.number_input("Iterations", min_value=100, max_value=100_000, value=1_000, step=100, key="lab_sim_iterations", help="Each iteration generates one full portfolio path for each selected method.")
        with st.expander("Contributions, withdrawals & randomness"):
            a, b, c = st.columns(3)
            contribution = a.number_input("Annual contribution ($)", min_value=0.0, value=0.0, step=1_000.0, key="lab_sim_contribution")
            withdrawal = b.number_input("Annual withdrawal ($)", min_value=0.0, value=0.0, step=1_000.0, key="lab_sim_withdrawal")
            if st.session_state.get("lab_sim_withdrawal_start", 1) > years:
                st.session_state["lab_sim_withdrawal_start"] = years
            withdrawal_start = c.number_input("First withdrawal year", min_value=1, max_value=years, value=1, key="lab_sim_withdrawal_start")
            a, b = st.columns(2)
            block_size = a.number_input("Block size (trading days)", min_value=1, max_value=min(252, len(returns)), value=21, disabled=mode == "Monte Carlo", key="lab_sim_block")
            seed = b.number_input("Random seed", min_value=0, value=42, key="lab_sim_seed")
            st.caption("The same seed and inputs reproduce the same paths. Change the seed for a different random sample. Cash flows occur at the start of each year; weights are rebalanced daily. Fees, taxes, and inflation are excluded.")
        weights = {"Black–Litterman": model["weights"], "Reference allocation": model["reference"],
                   "Equal weight": pd.Series(1 / len(returns.columns), index=returns.columns)}[allocation]
        settings = dict(n_sims=int(iterations), years=years, initial_value=initial, annual_contribution=contribution,
                        annual_withdrawal=withdrawal, withdrawal_start=int(withdrawal_start), block_size=int(block_size), seed=int(seed))
        monte_carlo = {}
        if calibration == "Black–Litterman posterior" and "Monte Carlo" in methods:
            monte_carlo = dict(annual_mean=float(weights @ model["returns"]),
                               annual_volatility=float(np.sqrt(weights @ model["covariance"] @ weights)))
        signature = fingerprint((returns, weights, settings, methods, monte_carlo, source, allocation))
        run = st.button("Run simulations", type="primary", width="stretch")
    if run:
        started = perf_counter()
        progress_bar = st.progress(0, text="Starting simulation…")
        try:
            results = {}
            for i, method in enumerate(methods):
                def report(completed, total):
                    progress_bar.progress((i + completed / total) / len(methods),
                                          text=f"{method}: {completed:,} / {total:,} paths computed")
                results[method] = simulate_projection(returns, weights, method=method, **settings,
                    **(monte_carlo if method == "Monte Carlo" else {}), progress=report)
            run_id = st.session_state.get("lab_run_counter", 0) + 1
            st.session_state["lab_run_counter"] = run_id
            st.session_state["lab_simulations"] = dict(results=results, signature=signature, allocation=allocation,
                settings=settings, calibration=calibration, source=source, tickers=list(returns.columns), weights=weights,
                run_id=run_id, completed_at=datetime.now().astimezone().isoformat(timespec="seconds"),
                duration=perf_counter() - started)
            st.session_state.pop("lab_annual_export", None)
            st.session_state.pop("lab_daily_export", None)
        except (ValueError, RuntimeError) as exc:
            st.error(f"Run failed: {exc}")
        finally:
            progress_bar.empty()
    saved = st.session_state.get("lab_simulations")
    if not saved:
        st.info("No simulation has run yet. Click Run simulations to generate your portfolio paths.")
        return
    if signature != saved["signature"]:
        st.warning("Inputs have changed. Results below are from your last run; click Run simulations to update them.")
    config = saved["settings"]
    total = sum(len(result.values) for result in saved["results"].values())
    st.divider()
    st.subheader("Simulation results")
    st.success(f"Run #{saved['run_id']} complete · {total:,} paths computed across {len(saved['results'])} method(s) · {saved['duration']:.2f} seconds")
    st.caption(f"{saved['allocation']} · {config['n_sims']:,} paths per method · {config['years']} years · seed {config['seed']} · completed {saved['completed_at']}")
    paths_tab, outcomes_tab, details_tab = st.tabs(["Individual paths", "Outcome comparison", "Run details"])
    with paths_tab:
        show_individual_paths(saved)
    with outcomes_tab:
        target = st.number_input("Ending balance target ($)", min_value=0.0, value=200_000.0, step=10_000.0, key="lab_sim_target")
        st.caption("Changing the target recalculates probabilities from the stored run; it does not generate new paths.")
        show_outcomes(saved["results"], target)
        if config["annual_withdrawal"] == 0:
            st.caption("No withdrawals were scheduled. Withdrawal funding is 100% by definition.")
    with details_tab:
        st.markdown("**Inputs used for this completed run**")
        st.caption(f"Data: {saved['source']} · Assets: {', '.join(saved['tickers'])}")
        st.caption(f"Monte Carlo calibration: {saved['calibration']}. Bootstrap always uses the loaded return history; Black–Litterman views can change its weights, not that history.")
        st.dataframe(saved["weights"].rename("Simulated weight").to_frame().style.format("{:.2%}"), width="stretch")
        st.json(config)
        st.caption("Every click of Run simulations executes the engine again. Viewing paths, changing chart options, and navigating between pages reuse the completed run.")

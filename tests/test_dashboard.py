"""Exercise dashboard navigation and real backend integration."""

from pathlib import Path

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest


APP = str(Path(__file__).resolve().parents[1] / "streamlit_app.py")


def widget(elements, label):
    return next(element for element in elements if element.label == label)


def test_portfolio_and_glide_path():
    app = AppTest.from_file(APP, default_timeout=30).run()
    widget(app.selectbox, "Data source").set_value("Synthetic demo")
    widget(app.button, "Load selected assets").click().run()
    app.sidebar.radio[0].set_value("Simulations").run()
    widget(app.number_input, "Iterations").set_value(100)
    widget(app.button, "Run simulations").click().run()
    assert not app.exception
    assert not app.error
    assert any(tab.label == "Individual paths" for tab in app.tabs)
    app.sidebar.radio[0].set_value("Historical stress tests").run()
    assert any("Synthetic demo data" in info.value for info in app.info)
    app.sidebar.radio[0].set_value("Glide path & liabilities").run()
    assert not app.exception
    assert app.metric[0].value == "$500,000"
    assert len(app.dataframe[0].value) == 16


def test_portfolio_selection_reruns_and_stale_results():
    app = AppTest.from_file(APP, default_timeout=30).run()
    widget(app.button, "ETF mix").click().run()
    widget(app.selectbox, "Data source").set_value("Synthetic demo")
    widget(app.button, "Load selected assets").click().run()
    assert list(app.session_state["lab_data"]["returns"].columns) == ["VTI", "VXUS", "BND", "GLD"]
    app.sidebar.radio[0].set_value("Simulations").run()
    widget(app.number_input, "Iterations").set_value(100)
    widget(app.button, "Run simulations").click().run()
    assert not app.exception
    assert not app.error
    assert set(app.session_state["lab_simulations"]["results"]) == {"Monte Carlo", "Block bootstrap"}
    app.run()
    assert not any("Inputs have changed" in warning.value for warning in app.warning)
    widget(app.number_input, "Ending balance target ($)").set_value(0.0).run()
    assert not any("Inputs have changed" in warning.value for warning in app.warning)
    widget(app.slider, "Horizon (years)").set_value(5).run()
    assert any("Inputs have changed" in warning.value for warning in app.warning)
    assert app.session_state["lab_simulations"]["settings"]["years"] == 10
    widget(app.button, "Run simulations").click().run()
    assert not app.exception
    assert app.session_state["lab_simulations"]["settings"]["years"] == 5


def test_historical_selection_loads_all_requested_tickers(monkeypatch):
    import dashboard_portfolio
    import numpy as np
    import pandas as pd

    calls = []

    def prices(tickers, start, end):
        calls.append(tickers)
        rng = np.random.default_rng(5)
        return pd.DataFrame(100 * np.exp(np.cumsum(rng.normal(0, 0.01, (800, len(tickers))), axis=0)),
                            columns=tickers, index=pd.bdate_range("2023-01-01", periods=800))

    monkeypatch.setattr(dashboard_portfolio, "historical_prices", prices)
    app = AppTest.from_file(APP, default_timeout=30).run()
    widget(app.multiselect, "Stocks & ETFs").set_value(["SPY", "BND"])
    widget(app.button, "Load selected assets").click().run()
    assert not app.exception
    assert not app.error
    assert calls == [("SPY", "BND")]
    assert list(app.session_state["lab_data"]["returns"].columns) == ["SPY", "BND"]
    app.sidebar.radio[0].set_value("Black–Litterman").run()
    widget(app.slider, "Maximum position (%)").set_value(40).run()
    assert any("at least 50.0%" in error.value for error in app.error)
    assert not app.exception


def test_missing_ticker_does_not_silently_change_universe(monkeypatch):
    import dashboard_portfolio
    import pandas as pd

    monkeypatch.setattr(dashboard_portfolio, "historical_prices", lambda *args: pd.DataFrame({"SPY": [100, 101]}))
    app = AppTest.from_file(APP).run()
    widget(app.multiselect, "Stocks & ETFs").set_value(["SPY", "BND"])
    widget(app.button, "Load selected assets").click().run()
    assert not app.exception
    assert any("No data for: BND" in error.value for error in app.error)


def test_simulations_execute_only_on_run_and_keep_paths_across_pages(monkeypatch):
    import dashboard_simulations
    import pandas as pd

    calls = []
    original = dashboard_simulations.simulate_projection

    def record(*args, **kwargs):
        calls.append(kwargs["method"])
        return original(*args, **kwargs)

    monkeypatch.setattr(dashboard_simulations, "simulate_projection", record)
    app = AppTest.from_file(APP, default_timeout=30).run()
    widget(app.selectbox, "Data source").set_value("Synthetic demo")
    widget(app.button, "Load selected assets").click().run()
    app.sidebar.radio[0].set_value("Black–Litterman").run()
    widget(app.selectbox, "Portfolio objective").set_value("Minimum volatility").run()
    model_weights = app.session_state["lab_model"]["weights"].copy()
    app.sidebar.radio[0].set_value("Simulations").run()
    widget(app.number_input, "Iterations").set_value(100)
    widget(app.slider, "Horizon (years)").set_value(2)
    assert calls == []
    widget(app.button, "Run simulations").click().run()
    assert not app.exception
    assert not app.error
    assert calls == ["Monte Carlo", "Block bootstrap"]
    saved = app.session_state["lab_simulations"]
    paths = saved["results"]["Monte Carlo"].daily_paths.copy()
    pd.testing.assert_series_equal(saved["weights"], model_weights)
    assert paths.shape == (100, 505)
    assert saved["run_id"] == 1
    widget(app.slider, "Paths on chart").set_value(20).run()
    widget(app.number_input, "Inspect path #").set_value(7).run()
    assert calls == ["Monte Carlo", "Block bootstrap"]
    app.sidebar.radio[0].set_value("Assets & data").run()
    app.sidebar.radio[0].set_value("Black–Litterman").run()
    assert widget(app.selectbox, "Portfolio objective").value == "Minimum volatility"
    app.sidebar.radio[0].set_value("Simulations").run()
    assert widget(app.slider, "Horizon (years)").value == 2
    assert widget(app.number_input, "Iterations").value == 100
    pd.testing.assert_frame_equal(app.session_state["lab_simulations"]["results"]["Monte Carlo"].daily_paths, paths)
    assert len(calls) == 2
    widget(app.number_input, "Random seed").set_value(43).run()
    widget(app.button, "Run simulations").click().run()
    assert not app.exception
    assert not app.error
    assert len(calls) == 4
    assert app.session_state["lab_simulations"]["run_id"] == 2
    assert not app.session_state["lab_simulations"]["results"]["Monte Carlo"].daily_paths.equals(paths)
    widget(app.selectbox, "View method").set_value("Block bootstrap").run()
    widget(app.selectbox, "Simulation method").set_value("Monte Carlo").run()
    widget(app.button, "Run simulations").click().run()
    assert not app.exception
    assert widget(app.selectbox, "View method").value == "Monte Carlo"
    assert len(calls) == 5
    widget(app.button, "Prepare all annual paths CSV").click().run()
    from io import StringIO
    exported = pd.read_csv(StringIO(app.session_state["lab_annual_export"][2]), index_col=0)
    assert exported.index.tolist() == list(range(1, 101))
    assert exported.shape == (100, 3)
    widget(app.button, "Prepare retained daily paths CSV").click().run()
    exported_daily = pd.read_csv(StringIO(app.session_state["lab_daily_export"][2]), index_col=0)
    assert exported_daily.shape == (100, 505)
    assert len(calls) == 5
    assert not app.exception


def test_funding_inputs_and_results_survive_navigation():
    app = AppTest.from_file(APP, default_timeout=30).run()
    app.sidebar.radio[0].set_value("Funding & sensitivity").run()
    widget(app.number_input, "Funding iterations").set_value(100)
    widget(app.number_input, "2027 contribution ($)").set_value(0.0)
    widget(app.number_input, "2028 contribution ($)").set_value(0.0)
    widget(app.button, "Run funding analysis").click().run()
    assert not app.exception
    assert not app.error
    assert app.metric[0].value == "0.0%"
    assert app.metric[1].value == "100.0%"
    assert app.metric[2].value == "$50,000"
    app.sidebar.radio[0].set_value("Glide path & liabilities").run()
    app.sidebar.radio[0].set_value("Funding & sensitivity").run()
    assert app.metric[0].value == "0.0%"
    assert not app.exception


@pytest.mark.parametrize("comparison,rows", [("One-variable sensitivity", 20), ("Two-variable sensitivity", 9), ("Scenarios", 3)])
def test_comparisons(comparison, rows):
    app = AppTest.from_file(APP, default_timeout=60).run()
    app.sidebar.radio[0].set_value("Funding & sensitivity").run()
    widget(app.selectbox, "Comparison").set_value(comparison)
    widget(app.number_input, "Paths per comparison").set_value(100)
    widget(app.button, "Run comparison").click().run()
    assert not app.exception
    assert not app.error
    assert len(app.dataframe[-1].value) == rows


def test_historical_stress_data_and_missing_periods():
    app = AppTest.from_string('''
import numpy as np
import pandas as pd
from dashboard_planning import show_stress_tests
returns = pd.DataFrame({"A": np.full(252, -0.001), "B": np.full(252, 0.0002)}, index=pd.bdate_range("2008-01-01", periods=252))
show_stress_tests(returns, pd.Series({"A": 0.6, "B": 0.4}), "Upload CSV")
''').run()
    assert not app.exception
    assert len(app.metric) == 4
    assert app.metric[0].value.startswith("-")
    assert len(app.info) == 2

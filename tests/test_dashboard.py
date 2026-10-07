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
    widget(app.number_input, "Iterations").set_value(100)
    widget(app.button, "Run portfolio analysis").click().run()
    assert not app.exception
    assert not app.error
    assert any(tab.label == "Historical stress tests" for tab in app.tabs)
    assert any("Synthetic demo data" in info.value for info in app.info)
    app.sidebar.radio[0].set_value("Glide path & liabilities").run()
    assert not app.exception
    assert app.metric[0].value == "$500,000"
    assert len(app.dataframe[0].value) == 16


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

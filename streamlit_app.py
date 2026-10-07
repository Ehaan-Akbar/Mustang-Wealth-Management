"""Mustang Wealth Lab: interactive portfolio research and liability planning."""

import streamlit as st

from dashboard_planning import show_funding, show_glide_path
from dashboard_portfolio import show_portfolio


st.set_page_config(page_title="Mustang Wealth Lab", page_icon="📈", layout="wide")
st.markdown("""<style>
.block-container {max-width: 1440px; padding-top: 2rem;}
div[data-testid="stMetric"] {border: 1px solid #94a3b833; border-radius: 12px; padding: 16px;}
div[data-testid="stTabs"] button {font-weight: 600;}
</style>""", unsafe_allow_html=True)

# Preserve portfolio widgets when a different page is rendered. Results and
# model snapshots are ordinary session data; neither navigation nor chart
# controls execute a simulation.
for key in list(st.session_state):
    if key.startswith(("lab_", "absolute_", "relative_", "reference_")):
        st.session_state[key] = st.session_state[key]

st.sidebar.title("Mustang Wealth Lab")
st.sidebar.caption("RESEARCH & SIMULATION")

workspace = st.sidebar.radio(
    "Workspace", ["Assets & data", "Black–Litterman", "Simulations", "Historical stress tests",
                  "Glide path & liabilities", "Funding & sensitivity"], key="workspace"
)
st.sidebar.divider()
st.sidebar.markdown("**Portfolio research**")
st.sidebar.caption("1. Load assets\n\n2. Set return views\n\n3. Run and inspect simulations")
st.sidebar.markdown("**Liability planning**")
st.sidebar.caption("Glide path and funding tools use a separate equity/bond/cash model.")
data = st.session_state.get("lab_data")
if data:
    st.sidebar.divider()
    st.sidebar.markdown("**Loaded assets**")
    st.sidebar.write(" · ".join(data["returns"].columns))
    st.sidebar.caption(data["source"])
saved = st.session_state.get("lab_simulations")
if saved:
    st.sidebar.caption(f"Completed simulation: run #{saved['run_id']}")

st.title(workspace)

if workspace in ["Assets & data", "Black–Litterman", "Simulations", "Historical stress tests"]:
    show_portfolio(workspace)
elif workspace == "Glide path & liabilities":
    show_glide_path()
else:
    show_funding()

st.divider()
st.caption("Educational planning tool only — results are simulated, not guaranteed, and are not investment advice.")

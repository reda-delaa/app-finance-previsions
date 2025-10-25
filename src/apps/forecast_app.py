import os
import json
from pathlib import Path
import sys as _sys
import streamlit as st
import pandas as pd

# sys.path bootstrap to import from src/* (analytics, core, etc.)
_SRC_ROOT = Path(__file__).resolve().parents[1]
if str(_SRC_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_SRC_ROOT))

from analytics.market_intel import build_snapshot
from analytics.forecaster import forecast_ticker


st.set_page_config(page_title="Forecasts — Finance Agent", layout="wide")
st.title("🔮 Forecasts — Finance Agent")

with st.sidebar:
    st.header("Settings")
    default_watch = os.getenv("WATCHLIST", "AAPL,MSFT,NVDA,SPY")
    tickers = st.text_input("Tickers (comma)", value=default_watch)
    horizon = st.selectbox("Horizon", ["1w","1m","1y"], index=1)
    run_btn = st.button("Run")

if run_btn:
    tickers_list = [t.strip().upper() for t in tickers.split(",") if t.strip()]
    cols = st.columns([2,1])
    with cols[0]:
        st.subheader("Results")
    with cols[1]:
        st.subheader("Details")

    rows = []
    for t in tickers_list:
        with st.spinner(f"Building snapshot for {t}..."):
            snap = build_snapshot(regions=["US","INTL"], window="last_week", ticker=t, limit=150)
        feats = snap.get("features") or {}
        f = forecast_ticker(t, horizon=horizon, features=feats).to_dict()
        rows.append({"ticker": t, **{f"f_{k}": v for k, v in f.items() if k != "drivers"}})
        st.session_state[f"snap_{t}"] = snap
        st.session_state[f"fc_{t}"] = f

    df = pd.DataFrame(rows)
    with cols[0]:
        st.dataframe(df, use_container_width=True)
    with cols[1]:
        for t in tickers_list:
            st.markdown(f"### {t}")
            f = st.session_state.get(f"fc_{t}") or {}
            st.json(f)
            if st.checkbox(f"Show features for {t}"):
                st.json(st.session_state.get(f"snap_{t}", {}).get("features", {}))

st.info("Tip: run `python scripts/agent_daily.py` to generate a daily set of forecasts in data/forecast/YYYYMMDD/.")

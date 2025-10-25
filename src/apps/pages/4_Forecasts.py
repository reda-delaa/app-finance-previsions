from pathlib import Path
import sys as _sys
import streamlit as st
from ui.shell import page_header, page_footer
import pandas as pd

SRC = Path(__file__).resolve().parents[2]
if str(SRC) not in _sys.path:
    _sys.path.insert(0, str(SRC))

from analytics.market_intel import build_snapshot
from analytics.forecaster import forecast_ticker

st.set_page_config(page_title="Forecasts — Finance Agent", layout="wide")
page_header(active="user")
st.subheader("🔮 Forecasts — Multi‑tickers")

with st.sidebar:
    st.header("Paramètres")
    tickers = st.text_input("Tickers (comma)", value="AAPL,MSFT,NVDA,SPY")
    horizon = st.selectbox("Horizon", ["1w","1m","1y"], index=1)
    run = st.button("Lancer")

if run:
    rows = []
    for t in [x.strip().upper() for x in tickers.split(",") if x.strip()]:
        snap = build_snapshot(regions=["US","INTL"], window="last_week", ticker=t, limit=150)
        feats = snap.get("features", {})
        f = forecast_ticker(t, horizon=horizon, features=feats).to_dict()
        rows.append({"ticker": t, **{f"f_{k}": v for k, v in f.items() if k != "drivers"}})
    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True)
else:
    st.info("Indiquez des tickers et cliquez Lancer pour calculer des prévisions par horizon.")
page_footer()

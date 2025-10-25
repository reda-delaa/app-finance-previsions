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
from core.market_data import get_price_history
import plotly.graph_objects as go
import pandas as pd
import datetime as dt
import ta

st.set_page_config(page_title="Deep Dive — Finance Agent", layout="wide")
page_header(active="user")
st.subheader("🔎 Deep Dive — Analyse d'un titre")

with st.sidebar:
    st.header("Sélection")
    ticker = st.text_input("Ticker", value="AAPL")
    window = st.selectbox("Fenêtre news", ["24h","48h","last_week","last_month","all"], index=2)
    run = st.button("Analyser")

if run and ticker:
    with st.spinner("Construction du snapshot..."):
        snap = build_snapshot(regions=["US","INTL"], window=window, ticker=ticker.upper(), limit=200)
    feats = snap.get("features", {})
    with st.expander("Détails des features (JSON)"):
        st.json(feats)
    st.subheader("Prévisions")
    cols = st.columns(3)
    for i, h in enumerate(["1w","1m","1y"]):
        f = forecast_ticker(ticker.upper(), horizon=h, features=feats).to_dict()
        with cols[i]:
            st.metric(label=f"{h} direction", value=f["direction"], delta=f"conf={f['confidence']}")
            with st.expander("JSON prévision"):
                st.json({k:v for k,v in f.items() if k != "drivers"})
    st.subheader("News (récents)")
    news = snap.get("news", [])
    if news:
        df = pd.DataFrame(news)
        st.dataframe(df[[c for c in ["ts","source","title","sent","tickers","link"] if c in df.columns]], use_container_width=True)
    else:
        st.info("Pas de news agrégées.")

    # ---- Price & Technicals ----
    st.subheader("Prix & Techniques")
    colp1, colp2 = st.columns([1,1])
    with colp1:
        period = st.selectbox("Période", ["1y","3y","5y"], index=1)
    with colp2:
        show_rsi = st.checkbox("Afficher RSI", value=True)

    days = {"1y": 365, "3y": 365*3, "5y": 365*5}[period]
    start = (dt.datetime.utcnow().date() - dt.timedelta(days=days)).isoformat()
    hist = get_price_history(ticker.upper(), start=start)
    if hist is None or hist.empty:
        st.info("Historique introuvable pour le graphique.")
    else:
        df = hist.copy()
        # Compute technicals
        df['SMA20'] = ta.trend.sma_indicator(df['Close'], window=20)
        df['SMA50'] = ta.trend.sma_indicator(df['Close'], window=50)
        df['SMA200'] = ta.trend.sma_indicator(df['Close'], window=200)
        df['RSI'] = ta.momentum.rsi(df['Close'], window=14)

        fig = go.Figure()
        fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name='Cours'))
        fig.add_trace(go.Scatter(x=df.index, y=df['SMA20'], name='SMA20', line=dict(width=1)))
        fig.add_trace(go.Scatter(x=df.index, y=df['SMA50'], name='SMA50', line=dict(width=1)))
        fig.add_trace(go.Scatter(x=df.index, y=df['SMA200'], name='SMA200', line=dict(width=1)))
        fig.update_layout(height=500, xaxis_rangeslider_visible=False)
        st.plotly_chart(fig, use_container_width=True)

        if show_rsi:
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(x=df.index, y=df['RSI'], name='RSI'))
            fig2.add_hrect(y0=30, y1=70, fillcolor="lightgray", opacity=0.2, line_width=0)
            fig2.update_layout(height=200)
            st.plotly_chart(fig2, use_container_width=True)

    # ---- Peers (quick compare) ----
    st.subheader("Pairs / Comparaison rapide")
    default_peers = "NGD.TO,AEM.TO,ABX.TO,K.TO,GDX"
    peers = st.text_input("Tickers (comma)", value=default_peers)
    if peers:
        tickers = [t.strip().upper() for t in peers.split(',') if t.strip()]
        data = {}
        for tkr in tickers:
            ph = get_price_history(tkr, start=start)
            if ph is not None and not ph.empty:
                data[tkr] = ph['Close']
        if data:
            panel = pd.DataFrame(data).dropna(how='all')
            if not panel.empty:
                # returns snapshot
                def last_pct(s, d):
                    if len(s) > d:
                        return float(s.iloc[-1] / s.iloc[-d] - 1.0)
                    return np.nan
                import numpy as np
                rows = []
                for tkr in panel.columns:
                    s = panel[tkr].dropna()
                    rows.append({
                        'ticker': tkr,
                        'ret_1m': last_pct(s, 21),
                        'ret_3m': last_pct(s, 63),
                        'ret_6m': last_pct(s, 126),
                    })
                st.dataframe(pd.DataFrame(rows), use_container_width=True)
else:
    st.info("Entrez un ticker (ex: AAPL) puis cliquez Analyser pour afficher la fiche complète (features, prévisions, news, techniques, pairs).")
page_footer()

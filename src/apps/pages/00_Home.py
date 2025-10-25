from __future__ import annotations

from pathlib import Path
import sys as _sys
import streamlit as st
from ui.nav import render_top_nav
from ui.footer import render_footer

SRC = Path(__file__).resolve().parents[2]
if str(SRC) not in _sys.path:
    _sys.path.insert(0, str(SRC))

st.set_page_config(page_title="Home — Finance Agent", layout="wide")
st.title("🏠 Accueil — Navigation guidée")
render_top_nav(active="user")

st.caption("Sépare les pages Prévisions (utilisateur) et Administration (opérations)")

forecast_pages = [
    ("1_Dashboard", "📊 Dashboard"),
    ("4_Forecasts", "📈 Forecasts"),
    ("15_Signals", "🔎 Signals"),
    ("16_Portfolio", "💼 Portfolio"),
    ("14_Regimes", "📉 Regimes"),
    ("21_Risk", "🛡 Risk"),
    ("25_Recession", "🌧 Recession"),
    ("20_Memos", "📝 Memos"),
    ("3_Deep_Dive", "🔬 Deep Dive"),
    ("6_Backtests", "📚 Backtests"),
    ("8_Evaluation", "✅ Evaluation"),
    ("7_Reports", "📄 Reports"),
    ("18_Watchlist", "⭐ Watchlist"),
    ("9_Advisor", "🤖 Advisor"),
    ("19_Notes", "🗒 Notes"),
]

admin_pages = [
    ("10_Events", "📅 Events"),
    ("11_Quality", "🧪 Data Quality"),
    ("12_Agents", "🤖 Agents"),
    ("22_LLM_Scoreboard", "🏁 LLM Scoreboard"),
    ("23_Settings", "⚙ Settings"),
    ("24_Changes", "🧭 Changes"),
    ("26_Earnings", "📆 Earnings"),
    ("27_Agents_Status", "🛰 Agents Status"),
    ("5_Observability", "🔭 Observability"),
]

c1, c2 = st.columns(2)
with c1:
    st.subheader("Prévisions / Analyse")
    for fname, label in forecast_pages:
        st.page_link(f"pages/{fname}.py", label=label, icon=None)
with c2:
    st.subheader("Administration / Opérations")
    for fname, label in admin_pages:
        st.page_link(f"pages/{fname}.py", label=label, icon=None)

st.divider()
st.caption("Tips: utilisez la page Home pour rester orienté; l'ordre des pages du menu peut encore évoluer.")
render_footer()

from pathlib import Path
import sys as _sys
import streamlit as st
from ui.nav import render_top_nav
from ui.footer import render_footer

# Ensure src on sys.path
_SRC_ROOT = Path(__file__).resolve().parents[1]
if str(_SRC_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_SRC_ROOT))

st.set_page_config(page_title="Finance Agent — Accueil", layout="wide", initial_sidebar_state="expanded")
st.title("🤝 Finance Agent — Accueil")
render_top_nav(active="user")

with st.sidebar:
    st.header("Accueil / Navigation")
    st.page_link("pages/00_Home.py", label="🏠 Home", icon=None)
    st.divider()
    st.caption("Prévisions / Analyse")
    for p, lbl in [
        ("pages/1_Dashboard.py", "📊 Dashboard"),
        ("pages/4_Forecasts.py", "📈 Forecasts"),
        ("pages/15_Signals.py", "🔎 Signals"),
        ("pages/16_Portfolio.py", "💼 Portfolio"),
        ("pages/14_Regimes.py", "📉 Regimes"),
        ("pages/21_Risk.py", "🛡 Risk"),
        ("pages/25_Recession.py", "🌧 Recession"),
        ("pages/20_Memos.py", "📝 Memos"),
        ("pages/3_Deep_Dive.py", "🔬 Deep Dive"),
        ("pages/6_Backtests.py", "📚 Backtests"),
        ("pages/8_Evaluation.py", "✅ Evaluation"),
        ("pages/7_Reports.py", "📄 Reports"),
        ("pages/18_Watchlist.py", "⭐ Watchlist"),
        ("pages/9_Advisor.py", "🤖 Advisor"),
        ("pages/19_Notes.py", "🗒 Notes"),
    ]:
        st.page_link(p, label=lbl, icon=None)

    st.divider()
    st.caption("Administration / Opérations")
    for p, lbl in [
        ("pages/10_Events.py", "📅 Events"),
        ("pages/11_Quality.py", "🧪 Data Quality"),
        ("pages/12_Agents.py", "🤖 Agents"),
        ("pages/22_LLM_Scoreboard.py", "🏁 LLM Scoreboard"),
        ("pages/23_Settings.py", "⚙ Settings"),
        ("pages/24_Changes.py", "🧭 Changes"),
        ("pages/26_Earnings.py", "📆 Earnings"),
        ("pages/27_Agents_Status.py", "🛰 Agents Status"),
        ("pages/5_Observability.py", "🔭 Observability"),
    ]:
        st.page_link(p, label=lbl, icon=None)

st.markdown(
    """
Bienvenue. Utilisez les sections de navigation à gauche:

- Prévisions / Analyse: pages pour l’investisseur (signaux, portefeuilles, régimes, risques, memos…)
- Administration / Opérations: santé des données, agents, modèles LLM, réglages et observabilité
    """
)

st.info("Astuce: commencez par 🏠 Home pour une vue d’ensemble; port canonique UI: 5555.")
render_footer()

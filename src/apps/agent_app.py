from pathlib import Path
import sys as _sys
import streamlit as st

# Ensure src on sys.path
_SRC_ROOT = Path(__file__).resolve().parents[1]
if str(_SRC_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_SRC_ROOT))

st.set_page_config(page_title="Finance Agent — Home", layout="wide")
st.title("🤝 Finance Agent — Accueil")

st.markdown("""
Bienvenue. Utilisez le menu Pages (gauche) pour:
- Dashboard: résumé du jour et top opportunités
- News: agrégation et synthèse IA
- Deep Dive: analyse complète d'un titre
- Forecasts: prévisions 1w/1m/1y
- Backtest: stratégies et métriques (à venir)
- Observability: santé, clés, logs
""")

st.info("Astuce: pour multi-pages, Streamlit détecte automatiquement le dossier 'pages/' à côté de ce fichier.")


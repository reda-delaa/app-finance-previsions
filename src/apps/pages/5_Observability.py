from pathlib import Path
import sys as _sys
import os
import streamlit as st
from ui.shell import page_header, page_footer

SRC = Path(__file__).resolve().parents[2]
if str(SRC) not in _sys.path:
    _sys.path.insert(0, str(SRC))

st.set_page_config(page_title="Observability — Finance Agent", layout="wide")
page_header(active="admin")
st.subheader("🛠️ Observability — Santé & Clés")

keys = [
    "FIRECRAWL_API_KEY",
    "SERPER_API_KEY",
    "TAVILY_API_KEY",
    "FINNHUB_API_KEY",
    "FRED_API_KEY",
]

st.markdown("#### Clés d'API (présence seulement)")
rows = []
for i, k in enumerate(keys, start=1):
    v = os.getenv(k)
    rows.append({"clé": f"Key #{i}", "présente": bool(v)})
st.dataframe(rows, use_container_width=True)

st.markdown("#### Processus")
st.write("UI principale et pages chargées. Consultez les logs dans le dossier logs/ si nécessaire.")
page_footer()

from pathlib import Path
import sys as _sys
import os
import streamlit as st

SRC = Path(__file__).resolve().parents[2]
if str(SRC) not in _sys.path:
    _sys.path.insert(0, str(SRC))

st.set_page_config(page_title="Observability — Finance Agent", layout="wide")
st.title("🛠️ Observability — Santé & Clés")

keys = [
    "FIRECRAWL_API_KEY",
    "SERPER_API_KEY",
    "TAVILY_API_KEY",
    "FINNHUB_API_KEY",
    "FRED_API_KEY",
]

st.subheader("Clés d'API (présence seulement)")
rows = []
for k in keys:
    v = os.getenv(k)
    rows.append({"key": k, "present": bool(v)})
st.dataframe(rows, use_container_width=True)

st.subheader("Processus")
st.write("UI principale et pages chargées. Consultez les logs dans le dossier logs/ si nécessaire.")


from pathlib import Path
import sys as _sys
import json
import streamlit as st

SRC = Path(__file__).resolve().parents[2]
if str(SRC) not in _sys.path:
    _sys.path.insert(0, str(SRC))

st.set_page_config(page_title="Risk — Finance Agent", layout="wide")
st.title("🛡️ Risk Monitor — Composite")

with st.sidebar:
    st.header("Source")
    base = Path('data/risk')
    dates = sorted([p.name for p in base.glob('dt=*')], reverse=True)
    chosen = st.selectbox("Dossier date", dates, index=0 if dates else None)
    if st.button("Rafraîchir (script)"):
        st.info("Exécute: PYTHONPATH=src python scripts/run_risk_monitor.py")

if not chosen:
    st.info("Sélectionnez un dossier date")
else:
    p = Path('data/risk')/chosen/'risk.json'
    if not p.exists():
        st.info("Aucun risk.json trouvé.")
    else:
        obj = json.loads(p.read_text(encoding='utf-8'))
        st.subheader("Niveau de risque (composite)")
        c1,c2 = st.columns(2)
        with c1: st.metric("Risk Level", obj.get('risk_level'))
        with c2: st.metric("Composite (z‑approx)", f"{obj.get('composite'):.2f}")
        st.subheader("Composants")
        st.json(obj.get('components') or {})
        with st.expander("JSON brut"):
            st.json(obj)


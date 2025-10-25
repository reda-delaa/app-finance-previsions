from pathlib import Path
import sys as _sys
import json
import streamlit as st

SRC = Path(__file__).resolve().parents[2]
if str(SRC) not in _sys.path:
    _sys.path.insert(0, str(SRC))

st.set_page_config(page_title="Recession Risk — Finance Agent", layout="wide")
st.title("🌧️ Recession Risk — Probabilité")

with st.sidebar:
    st.header("Source")
    base = Path('data/macro/recession')
    dates = sorted([p.name for p in base.glob('dt=*')], reverse=True)
    chosen = st.selectbox("Dossier date", dates, index=0 if dates else None)
    if st.button("Rafraîchir (script)"):
        st.info("Exécute: PYTHONPATH=src python scripts/run_recession.py")

if not chosen:
    st.info("Sélectionnez un dossier date")
else:
    p = Path('data/macro/recession')/chosen/'recession.json'
    if not p.exists():
        st.info("Aucun recession.json trouvé.")
    else:
        obj = json.loads(p.read_text(encoding='utf-8'))
        st.subheader("Probabilité")
        st.metric("Recession (approx.)", f"{int((obj.get('probability') or 0)*100)}%")
        st.subheader("Résumé")
        st.write(obj.get('summary_fr') or "")
        st.subheader("Composants & Scores")
        col1, col2 = st.columns(2)
        with col1: st.json(obj.get('inputs') or {})
        with col2: st.json(obj.get('scores') or {})
        with st.expander("JSON brut"):
            st.json(obj)


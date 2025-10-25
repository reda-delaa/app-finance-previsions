from pathlib import Path
import sys as _sys
import json
import streamlit as st

SRC = Path(__file__).resolve().parents[2]
if str(SRC) not in _sys.path:
    _sys.path.insert(0, str(SRC))

st.set_page_config(page_title="Macro Regimes — Finance Agent", layout="wide")
st.title("🧭 Macro Regimes — Probabilités")

with st.sidebar:
    st.header("Source")
    base = Path("data/macro/regime")
    dates = sorted([p.name for p in base.glob("dt=*")], reverse=True)
    chosen = st.selectbox("Dossier date", dates, index=0 if dates else None)
    if st.button("Rafraîchir (script)"):
        st.info("Exécute: PYTHONPATH=src python scripts/run_macro_regime.py")

if chosen:
    p = Path("data/macro/regime")/chosen/"regime.json"
    if not p.exists():
        st.info("Aucun fichier regime.json trouvé.")
    else:
        obj = json.loads(p.read_text(encoding='utf-8'))
        st.subheader("Probabilités de régime")
        probs = obj.get('probs') or {}
        c1,c2,c3,c4 = st.columns(4)
        c1.metric("Expansion", f"{int((probs.get('expansion') or 0)*100)}%")
        c2.metric("Ralentissement", f"{int((probs.get('slowdown') or 0)*100)}%")
        c3.metric("Inflation", f"{int((probs.get('inflation') or 0)*100)}%")
        c4.metric("Déflation", f"{int((probs.get('deflation') or 0)*100)}%")
        st.subheader("Indicateurs clés")
        st.json(obj.get('indicators') or {})
        st.subheader("Résumé")
        st.write(obj.get('summary_fr') or "")
        with st.expander("JSON brut"):
            st.json(obj)
else:
    st.info("Sélectionnez un dossier date.")


from pathlib import Path
import sys as _sys
import json
import streamlit as st
import pandas as pd

SRC = Path(__file__).resolve().parents[2]
if str(SRC) not in _sys.path:
    _sys.path.insert(0, str(SRC))

st.set_page_config(page_title="LLM Models — Finance Agent", layout="wide")
st.title("🧠 LLM Models — G4F Working List")

from agents.g4f_model_watcher import load_working_models

with st.sidebar:
    st.header("Actions")
    quick = st.button("Tester les modèles (rapide)")
    limit = st.slider("Limite à tester", 2, 12, 6)
    st.caption("Utilise g4f; essaie quelques providers. Peut prendre 1–2 minutes.")

if quick:
    with st.spinner("Tests en cours..."):
        try:
            from agents.g4f_model_watcher import refresh
            p = refresh(limit=limit, refresh_verified=True)
            st.success(f"Mise à jour écrite: {p}")
        except Exception as e:
            st.warning(f"Échec mise à jour: {e}")

p = Path('data/llm/models/working.json')
if p.exists():
    obj = json.loads(p.read_text(encoding='utf-8'))
    st.caption(f"Dernière mise à jour: {obj.get('asof')}")
    rows = pd.DataFrame(obj.get('models') or [])
    if not rows.empty:
        rows = rows.sort_values(by=["ok","pass_rate","latency_s"], ascending=[False, False, True])
        st.dataframe(rows, use_container_width=True)
    else:
        st.info("La liste est vide. Utilisez le bouton 'Tester les modèles (rapide)'.")
else:
    st.info("Aucun fichier working.json trouvé. Utilisez le bouton 'Tester les modèles (rapide)'.")

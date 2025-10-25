from pathlib import Path
import sys as _sys
import json
import streamlit as st
from ui.shell import page_header, page_footer

SRC = Path(__file__).resolve().parents[2]
if str(SRC) not in _sys.path:
    _sys.path.insert(0, str(SRC))

st.set_page_config(page_title="Recession Risk — Finance Agent", layout="wide")
page_header(active="user")
st.subheader("🌧️ Recession Risk — Probabilité")

base = Path('data/macro/recession')
dates = sorted([p.name for p in base.glob('dt=*')], reverse=True)
if not dates:
    st.info("Aucune partition trouvée. Revenez plus tard ou consultez Admin → Agents Status.")
else:
    chosen = st.selectbox("Date", dates, index=0)
    p = base/ chosen / 'recession.json'
    if not p.exists():
        st.info("Aucun recession.json trouvé pour la date sélectionnée.")
    else:
        obj = json.loads(p.read_text(encoding='utf-8'))
        st.markdown("#### Probabilité")
        st.metric("Récession (approx.)", f"{int((obj.get('probability') or 0)*100)}%")
        st.markdown("#### Résumé")
        st.write(obj.get('summary_fr') or "")
        st.markdown("#### Composants & Scores")
        col1, col2 = st.columns(2)
        with col1: st.json(obj.get('inputs') or {})
        with col2: st.json(obj.get('scores') or {})
        with st.expander("Détails (JSON)"):
            st.json(obj)
page_footer()

from pathlib import Path
import sys as _sys
import json
import streamlit as st
from ui.shell import page_header, page_footer

SRC = Path(__file__).resolve().parents[2]
if str(SRC) not in _sys.path:
    _sys.path.insert(0, str(SRC))

st.set_page_config(page_title="Macro Regimes — Finance Agent", layout="wide")
page_header(active="user")
st.subheader("🧭 Macro Regimes — Probabilités")

base = Path("data/macro/regime")
dates = sorted([p.name for p in base.glob("dt=*")], reverse=True)
if not dates:
    st.info("Aucune partition trouvée. Revenez plus tard ou consultez Admin → Agents Status.")
else:
    chosen = st.selectbox("Date", options=dates, index=0)
    p = base/ chosen / "regime.json"
    if not p.exists():
        st.info("Aucun fichier regime.json trouvé pour la date sélectionnée.")
    else:
        obj = json.loads(p.read_text(encoding='utf-8'))
        st.markdown("#### Probabilités de régime")
        probs = obj.get('probs') or {}
        c1,c2,c3,c4 = st.columns(4)
        c1.metric("Expansion", f"{int((probs.get('expansion') or 0)*100)}%")
        c2.metric("Ralentissement", f"{int((probs.get('slowdown') or 0)*100)}%")
        c3.metric("Inflation", f"{int((probs.get('inflation') or 0)*100)}%")
        c4.metric("Déflation", f"{int((probs.get('deflation') or 0)*100)}%")
        st.markdown("#### Indicateurs clés")
        st.json(obj.get('indicators') or {})
        st.markdown("#### Résumé")
        st.write(obj.get('summary_fr') or "")
        with st.expander("Détails (JSON)"):
            st.json(obj)
page_footer()

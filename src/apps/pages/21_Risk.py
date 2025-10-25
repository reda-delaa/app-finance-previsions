from __future__ import annotations

from pathlib import Path
import sys as _sys
import json
import streamlit as st
from ui.shell import page_header, page_footer

SRC = Path(__file__).resolve().parents[2]
if str(SRC) not in _sys.path:
    _sys.path.insert(0, str(SRC))

st.set_page_config(page_title="Risk — Finance Agent", layout="wide")
page_header(active="user")
st.subheader("🛡️ Risk Monitor — Composite")

base = Path('data/risk')
dates = sorted([p.name for p in base.glob('dt=*')], reverse=True)
beginner = st.toggle("Mode débutant", value=False, help="Explications simples des indicateurs.")

if not dates:
    st.info("Aucune partition trouvée. Revenez plus tard ou consultez Admin → Agents Status.")
else:
    chosen = st.selectbox("Date", dates, index=0)
    p = base/ chosen/'risk.json'
    if not p.exists():
        st.info("Aucun risk.json trouvé pour la date sélectionnée.")
    else:
        obj = json.loads(p.read_text(encoding='utf-8'))
        st.markdown("#### Niveau de risque (composite)")
        c1,c2 = st.columns(2)
        with c1: st.metric("Risk Level", obj.get('risk_level'))
        with c2:
            try:
                st.metric("Composite (z‑approx)", f"{obj.get('composite'):.2f}")
            except Exception:
                st.metric("Composite (z‑approx)", "—")
        if beginner:
            st.caption("Inversion DGS10–DGS2 négative, spread High Yield élevé, ou NFCI élevé → risque accru.")
        st.markdown("#### Composants")
        st.json(obj.get('components') or {})
        with st.expander("Détails (JSON)"):
            st.json(obj)
page_footer()

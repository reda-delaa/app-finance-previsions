from pathlib import Path
import sys as _sys
import json
import streamlit as st
from ui.shell import page_header, page_footer

SRC = Path(__file__).resolve().parents[2]
if str(SRC) not in _sys.path:
    _sys.path.insert(0, str(SRC))

st.set_page_config(page_title="Reports — Finance Agent", layout="wide")
page_header(active="admin")
st.subheader("📝 Reports — Investigations & Summaries")

with st.sidebar:
    st.header("Source")
    base = Path("data/reports")
    dates = sorted([p.name for p in base.glob("dt=*")], reverse=True)
    chosen = st.selectbox("Dossier date", dates, index=0 if dates else None)

if chosen:
    ddir = Path("data/reports") / chosen
    files = sorted(list(ddir.glob("*.json")))
    if not files:
        st.info("Aucun rapport pour cette date.")
    else:
        for f in files:
            st.subheader(f.name)
            try:
                obj = json.loads(f.read_text(encoding="utf-8"))
                # Show macro KPIs if present
                macro = (obj or {}).get("macro") or {}
                def _is_nan(x):
                    try:
                        import math
                        return isinstance(x, float) and math.isnan(x)
                    except Exception:
                        return False
                def _fmt_pct(v):
                    try:
                        if v is None:
                            return "n/a"
                        f = float(v)
                        if _is_nan(f):
                            return "n/a"
                        return f"{f*100:.2f}%"
                    except Exception:
                        return "n/a"
                def _fmt_bp(v):
                    try:
                        if v is None:
                            return "n/a"
                        f = float(v)
                        if _is_nan(f):
                            return "n/a"
                        return f"{f:.1f} bp"
                    except Exception:
                        return "n/a"
                if macro:
                    c1, c2 = st.columns(2)
                    with c1:
                        st.metric("Dollar américain (variation sur 1 semaine)", _fmt_pct(macro.get('DXY_wow')))
                    with c2:
                        st.metric("Taux US 10 ans (écart sur 1 semaine)", _fmt_bp(macro.get('UST10Y_bp_wow')))
                    if any(_is_nan(v) or v is None for v in [macro.get('DXY_wow'), macro.get('UST10Y_bp_wow')]):
                        st.caption("Certaines données macro sont indisponibles aujourd'hui (week‑end/jour férié). Valeurs manquantes affichées en 'n/a'.")
                with st.expander("Détails (JSON)"):
                    st.json(obj)
            except Exception as e:
                st.warning(f"Failed to read {f.name}: {e}")
else:
    st.info("Sélectionnez une date pour afficher les rapports d'investigation.")
page_footer()

from pathlib import Path
import sys as _sys
import json
import streamlit as st
import pandas as pd

SRC = Path(__file__).resolve().parents[2]
if str(SRC) not in _sys.path:
    _sys.path.insert(0, str(SRC))

from agents.data_quality import scan_all, write_report

st.set_page_config(page_title="Quality — Finance Agent", layout="wide")
st.title("✅ Data Quality — Vérification & Rapports")

with st.sidebar:
    st.header("Actions")
    if st.button("Lancer un scan complet"):
        with st.spinner("Scan en cours..."):
            rep = scan_all()
            p = write_report(rep)
            st.session_state['last_quality_report'] = str(p)
            st.success(f"Rapport écrit: {p}")

def _show_section(name: str, sec: dict):
    st.subheader(name)
    ok = sec.get('ok', False)
    st.write("Statut:", "✅ OK" if ok else "⚠️ À vérifier")
    issues = pd.DataFrame(sec.get('issues') or [])
    if not issues.empty:
        st.dataframe(issues, use_container_width=True)
    else:
        st.caption("Aucun problème détecté.")

latest = sorted(Path('data/quality').glob('dt=*/report.json'))
if latest:
    try:
        obj = json.loads(Path(latest[-1]).read_text(encoding='utf-8'))
        st.caption(f"Dernier rapport: {latest[-1]}")
        _show_section("News", obj.get('news', {}))
        _show_section("Macro", obj.get('macro', {}))
        _show_section("Prices", obj.get('prices', {}))
        _show_section("Forecasts", obj.get('forecasts', {}))
        _show_section("Features", obj.get('features', {}))
        _show_section("Events", obj.get('events', {}))
        with st.expander("Voir JSON complet"):
            st.json(obj)
    except Exception as e:
        st.warning(f"Lecture rapport impossible: {e}")
else:
    st.info("Aucun rapport précédent. Cliquez sur 'Lancer un scan complet'.")


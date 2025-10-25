from pathlib import Path
import sys as _sys
import json
import streamlit as st
import pandas as pd

SRC = Path(__file__).resolve().parents[2]
if str(SRC) not in _sys.path:
    _sys.path.insert(0, str(SRC))

st.set_page_config(page_title="Alerts — Finance Agent", layout="wide")
st.title("⚠️ Alerts — Qualité & Mouvements Inhabituels")

def _latest(globpat: str) -> Path | None:
    parts = sorted(Path().glob(globpat))
    return parts[-1] if parts else None

# Data Quality issues
st.subheader("Qualité des données (dernier rapport)")
q = _latest('data/quality/dt=*/report.json')
if not q:
    st.info("Aucun rapport de qualité. Lancez la page Quality.")
else:
    try:
        obj = json.loads(Path(q).read_text(encoding='utf-8'))
        issues = []
        for sec in ['news','macro','prices','forecasts','features','events']:
            s = obj.get(sec) or {}
            for it in (s.get('issues') or []):
                issues.append({'section': sec, **it})
        if issues:
            df = pd.DataFrame(issues)
            # order by severity
            sev_order = {'error':0,'warn':1,'info':2}
            df['sev_rank'] = df['sev'].map(lambda x: sev_order.get(str(x).lower(), 9))
            df = df.sort_values(['sev_rank','section'])
            st.dataframe(df[['section','sev','msg']], use_container_width=True)
        else:
            st.success("Aucun problème détecté.")
    except Exception as e:
        st.warning(f"Lecture impossible: {e}")

st.divider()
st.subheader("Mouvements récents (watchlist)")
b = _latest('data/forecast/dt=*/brief.json')
if not b:
    st.info("Aucun brief.json récent. Lancez agent_daily.")
else:
    try:
        br = json.loads(Path(b).read_text(encoding='utf-8'))
        changes = (br or {}).get('changes') or {}
        # macro moves (d1)
        m = changes.get('macro') or {}
        bullets = []
        if 'DXY_d1' in m:
            bullets.append(f"DXY (1j): {round(float(m['DXY_d1'])*100,2)}%")
        if 'UST10Y_bp_d1' in m:
            bullets.append(f"UST10Y (1j): {round(float(m['UST10Y_bp_d1']),1)} bp")
        if 'Gold_d1' in m:
            bullets.append(f"Or (1j): {round(float(m['Gold_d1'])*100,2)}%")
        if bullets:
            for b in bullets:
                st.write(f"- {b}")
        # watchlist biggest 1‑day moves
        w = changes.get('watchlist_moves') or []
        if w:
            dfw = pd.DataFrame(w)
            dfw['abs'] = dfw['d1'].abs()
            dfw = dfw.sort_values('abs', ascending=False).drop(columns=['abs'])
            dfw['d1_%'] = (dfw['d1']*100).round(2)
            st.dataframe(dfw[['ticker','d1_%']], use_container_width=True)
    except Exception as e:
        st.warning(f"Lecture brief impossible: {e}")


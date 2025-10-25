from __future__ import annotations

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
    st.info("Aucun rapport de qualité disponible. Consultez Admin → Data Quality pour générer un rapport.")
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
            try:
                csv_bytes = df[['section','sev','msg']].to_csv(index=False).encode('utf-8')
                st.download_button("Exporter issues (CSV)", data=csv_bytes, file_name="alerts_quality.csv", mime="text/csv")
            except Exception:
                pass
        else:
            st.success("Aucun problème détecté.")
    except Exception as e:
        st.warning(f"Lecture impossible: {e}")

st.divider()
st.subheader("Mouvements récents (watchlist)")
# Load default threshold from settings if present
thr_default = 1.0
try:
    import json as _json
    from pathlib import Path as _P
    ap = _P('data/config/alerts.json')
    if ap.exists():
        js = _json.loads(ap.read_text(encoding='utf-8'))
        thr_default = float(js.get('move_abs_pct', 1.0))
except Exception:
    pass
thr = st.slider("Seuil absolu (%, 1j)", 0.0, 5.0, float(thr_default), 0.1)
b = _latest('data/forecast/dt=*/brief.json')
if not b:
    st.info("Aucun brief récent. Consultez Admin → Agents Status pour vérifier la fraîcheur des données.")
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
            filt = dfw[dfw['d1_%'].abs() >= thr]
            st.dataframe((filt if not filt.empty else dfw)[['ticker','d1_%']], use_container_width=True)
            try:
                csv_bytes = (filt if not filt.empty else dfw)[['ticker','d1_%']].to_csv(index=False).encode('utf-8')
                st.download_button("Exporter mouvements (CSV)", data=csv_bytes, file_name="alerts_moves.csv", mime="text/csv")
            except Exception:
                pass
    except Exception as e:
        st.warning(f"Lecture brief impossible: {e}")

st.divider()
st.subheader("Earnings à venir (watchlist)")
# charge le dernier snapshot earnings
try:
    parts = sorted(Path('data/earnings').glob('dt=*/earnings.json'))
    if not parts:
        st.info("Aucun earnings.json trouvé. Consultez Admin → Agents Status.")
    else:
        p = parts[-1]
        obj = json.loads(Path(p).read_text(encoding='utf-8'))
        evs = obj.get('events') or []
        rows = []
        for e in evs:
            rows.append({'ticker': e.get('ticker'), 'date': e.get('date'), 'info': e.get('info')})
        df = pd.DataFrame(rows)
        if df.empty:
            st.info("Aucun événement à afficher.")
        else:
            df['date'] = pd.to_datetime(df['date'], errors='coerce')
            days = st.slider("Fenêtre (jours à venir)", 3, 60, 21)
            now = pd.Timestamp.utcnow().normalize()
            soon = df[(df['date']>=now)&(df['date']<=now+pd.Timedelta(days=days))].copy()
            st.dataframe((soon if not soon.empty else df).sort_values('date'), use_container_width=True)
            try:
                csv_bytes = (soon if not soon.empty else df).to_csv(index=False).encode('utf-8')
                st.download_button("Exporter earnings (CSV)", data=csv_bytes, file_name="alerts_earnings.csv", mime="text/csv")
            except Exception:
                pass
except Exception as e:
    st.warning(f"Lecture earnings impossible: {e}")

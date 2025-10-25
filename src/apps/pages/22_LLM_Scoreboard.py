from pathlib import Path
import sys as _sys
import json
import streamlit as st
from ui.shell import page_header, page_footer
import pandas as pd

SRC = Path(__file__).resolve().parents[2]
if str(SRC) not in _sys.path:
    _sys.path.insert(0, str(SRC))

st.set_page_config(page_title="LLM Scoreboard — Finance Agent", layout="wide")
page_header(active="admin")
st.subheader("🏁 LLM Scoreboard — Utilisation & Accord")

with st.sidebar:
    st.header("Fenêtre")
    limit_days = st.slider("Jours à considérer", 7, 120, 60)

rows = []
for p in sorted(Path('data/forecast').glob('dt=*/llm_agents.json')):
    try:
        dt = p.parent.name.replace('dt=','')
        obj = json.loads(p.read_text(encoding='utf-8'))
        # avg_agreement applies to the ensemble; attribute to all models in that ensemble
        for it in (obj.get('tickers') or []):
            ens = (it or {}).get('ensemble') or {}
            aa = ens.get('avg_agreement')
            models = ens.get('models') or []
            for m in models:
                rows.append({'dt': dt, 'model': m, 'avg_agreement': aa})
    except Exception:
        continue

if not rows:
    st.info("Aucune donnée LLM disponible pour l'instant. Consultez Admin → Agents Status pour l'état des agents.")
else:
    df = pd.DataFrame(rows)
    # filter by date window if dt parseable
    try:
        df['dt_parsed'] = pd.to_datetime(df['dt'], errors='coerce')
        end = df['dt_parsed'].max()
        start = end - pd.Timedelta(days=limit_days)
        df = df[(df['dt_parsed']>=start)&(df['dt_parsed']<=end)]
    except Exception:
        pass
    if df.empty:
        st.info("Aucune donnée dans la fenêtre.")
    else:
        ag = df.groupby('model').agg(
            uses=('model','count'),
            avg_agreement=('avg_agreement','mean'),
            last_seen=('dt','max')
        ).reset_index()
        # Enrich with last working latency/provider from working.json
        try:
            w = Path('data/llm/models/working.json')
            if w.exists():
                wobj = json.loads(w.read_text(encoding='utf-8'))
                rows_w = {r.get('model'): r for r in (wobj.get('models') or [])}
                ag['provider'] = ag['model'].map(lambda m: (rows_w.get(m) or {}).get('provider'))
                ag['latency_s'] = ag['model'].map(lambda m: (rows_w.get(m) or {}).get('latency_s'))
                ag['source'] = ag['model'].map(lambda m: (rows_w.get(m) or {}).get('source'))
        except Exception:
            pass
        ag = ag.sort_values(['avg_agreement','uses'], ascending=[False, False])
        st.dataframe(ag, use_container_width=True)
        try:
            csv_bytes = ag.to_csv(index=False).encode('utf-8')
            st.download_button("Exporter scoreboard (CSV)", data=csv_bytes, file_name="llm_scoreboard.csv", mime="text/csv")
        except Exception:
            pass
page_footer()

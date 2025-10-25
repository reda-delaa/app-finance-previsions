from __future__ import annotations

from pathlib import Path
import sys as _sys
import json
import pandas as pd
import streamlit as st

SRC = Path(__file__).resolve().parents[2]
if str(SRC) not in _sys.path:
    _sys.path.insert(0, str(SRC))

st.set_page_config(page_title="Earnings — Finance Agent", layout="wide")
st.title("📅 Earnings — Calendrier à venir")

from core.data_store import have_files

def _latest_earnings_file() -> Path | None:
    parts = sorted(Path('data/earnings').glob('dt=*/earnings.json'))
    return parts[-1] if parts else None

def _load_events(p: Path) -> pd.DataFrame:
    try:
        obj = json.loads(p.read_text(encoding='utf-8'))
        evs = obj.get('events') or []
        rows = []
        for e in evs:
            rows.append({
                'ticker': e.get('ticker'),
                'date': e.get('date'),
                'info': e.get('info'),
            })
        df = pd.DataFrame(rows)
        if not df.empty:
            df['date'] = pd.to_datetime(df['date'], errors='coerce')
            df = df.sort_values('date')
        return df
    except Exception:
        return pd.DataFrame(columns=['ticker','date','info'])

with st.sidebar:
    st.header("Options")
    st.caption("Mettez à jour via la cible Makefile: earnings")

latest = _latest_earnings_file()
if latest is None:
    st.info("Aucune donnée d'earnings disponible. Consultez Admin → Agents Status.")
else:
    df = _load_events(latest)
    st.caption(f"Source: {latest}")
    if df.empty:
        st.warning("Aucun événement disponible dans le dernier snapshot.")
    else:
        st.dataframe(df, use_container_width=True)
        try:
            csv_bytes = df.to_csv(index=False).encode('utf-8')
            st.download_button("Télécharger (CSV)", data=csv_bytes, file_name="earnings_upcoming.csv", mime="text/csv")
        except Exception:
            pass

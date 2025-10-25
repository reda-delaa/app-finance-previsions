from pathlib import Path
import sys as _sys
import json
import streamlit as st

SRC = Path(__file__).resolve().parents[2]
if str(SRC) not in _sys.path:
    _sys.path.insert(0, str(SRC))

st.set_page_config(page_title="Reports — Finance Agent", layout="wide")
st.title("📝 Reports — Investigations & Summaries")

with st.sidebar:
    st.header("Source")
    base = Path("data/reports")
    dates = sorted([p.name for p in base.glob("dt=*")], reverse=True)
    chosen = st.selectbox("Date folder", dates, index=0 if dates else None)
    refresh = st.button("Refresh")

if chosen:
    ddir = Path("data/reports") / chosen
    files = sorted(list(ddir.glob("*.json")))
    if not files:
        st.info("No reports for this date.")
    else:
        for f in files:
            st.subheader(f.name)
            try:
                obj = json.loads(f.read_text(encoding="utf-8"))
                # Show macro KPIs if present
                macro = (obj or {}).get("macro") or {}
                if macro:
                    c1, c2 = st.columns(2)
                    with c1:
                        st.metric("DXY WoW", f"{macro.get('DXY_wow') if macro.get('DXY_wow') is not None else 'n/a'}")
                    with c2:
                        st.metric("UST10Y bp WoW", f"{macro.get('UST10Y_bp_wow') if macro.get('UST10Y_bp_wow') is not None else 'n/a'}")
                st.json(obj)
            except Exception as e:
                st.warning(f"Failed to read {f.name}: {e}")
else:
    st.info("Select a date to view investigation reports.")


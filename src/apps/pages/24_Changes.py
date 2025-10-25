from pathlib import Path
import sys as _sys
import json
import streamlit as st
import pandas as pd

SRC = Path(__file__).resolve().parents[2]
if str(SRC) not in _sys.path:
    _sys.path.insert(0, str(SRC))

st.set_page_config(page_title="What Changed — Finance Agent", layout="wide")
st.title("🔄 What Changed — Depuis la veille")

def _latest_prev(globpat: str):
    parts = sorted(Path().glob(globpat))
    if len(parts) >= 2:
        return parts[-1], parts[-2]
    elif parts:
        return parts[-1], None
    return None, None

# Macro regime
cur, prev = _latest_prev('data/macro/regime/dt=*/regime.json')
if cur:
    st.subheader("Régime macro")
    cobj = json.loads(Path(cur).read_text(encoding='utf-8'))
    pobj = json.loads(Path(prev).read_text(encoding='utf-8')) if prev and Path(prev).exists() else {}
    def _top(o):
        pr = (o.get('probs') or {})
        return (sorted(pr.items(), key=lambda x: -x[1])[0] if pr else (None,None))
    cn, cp = _top(cobj)
    pn, pp = _top(pobj)
    st.write({"current": {"name": cn, "p": cp}, "previous": {"name": pn, "p": pp}})

# Risk composite
cur, prev = _latest_prev('data/risk/dt=*/risk.json')
if cur:
    st.subheader("Risque (composite)")
    cobj = json.loads(Path(cur).read_text(encoding='utf-8'))
    pobj = json.loads(Path(prev).read_text(encoding='utf-8')) if prev and Path(prev).exists() else {}
    st.write({"current": cobj.get('composite'), "prev": pobj.get('composite')})

# Top‑N changes (final scores)
cur, prev = _latest_prev('data/forecast/dt=*/final.parquet')
if cur:
    st.subheader("Top‑N (1m) — changements")
    cdf = pd.read_parquet(cur)
    pdf = pd.read_parquet(prev) if prev and Path(prev).exists() else pd.DataFrame(columns=cdf.columns)
    ctop = cdf[cdf['horizon']=='1m'].sort_values('final_score', ascending=False).head(10)
    ptop = pdf[pdf['horizon']=='1m'].sort_values('final_score', ascending=False).head(10)
    # positions map
    pos_prev = {t:i for i,t in enumerate(list(ptop['ticker']))}
    rows=[]
    for i, r in enumerate(ctop['ticker']):
        rows.append({'ticker': r, 'pos_now': i+1, 'pos_prev': (pos_prev.get(r)+1) if r in pos_prev else None})
    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True)

# Macro brief
cur, prev = _latest_prev('data/forecast/dt=*/brief.json')
if cur:
    st.subheader("Brief macro — delta")
    cobj = json.loads(Path(cur).read_text(encoding='utf-8'))
    pobj = json.loads(Path(prev).read_text(encoding='utf-8')) if prev and Path(prev).exists() else {}
    cm = (cobj.get('changes') or {}).get('macro') or {}
    pm = (pobj.get('changes') or {}).get('macro') or {}
    diffs = {}
    for k in set(cm.keys()) | set(pm.keys()):
        if cm.get(k) != pm.get(k):
            diffs[k] = {"now": cm.get(k), "prev": pm.get(k)}
    if diffs:
        st.json(diffs)
    else:
        st.caption("Pas de changement significatif.")


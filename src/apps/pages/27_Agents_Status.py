from __future__ import annotations

from pathlib import Path
import sys as _sys
import json
import pandas as pd
import streamlit as st

SRC = Path(__file__).resolve().parents[2]
if str(SRC) not in _sys.path:
    _sys.path.insert(0, str(SRC))

st.set_page_config(page_title="Agents Status — Finance Agent", layout="wide")
st.title("🛰️ Agents — Statut et Fraîcheur")

def _latest_dt(glob_pat: str) -> tuple[str | None, Path | None]:
    parts = sorted(Path('.').glob(glob_pat))
    if not parts:
        return (None, None)
    last = parts[-1]
    try:
        # expect dt=YYYYMMDD as parent dir
        if 'dt=' in last.as_posix():
            dt = last.parent.name.replace('dt=','')
        else:
            dt = None
    except Exception:
        dt = None
    return (dt, last)

def _read_json_asof(p: Path) -> str | None:
    try:
        obj = json.loads(p.read_text(encoding='utf-8'))
        a = obj.get('asof') or obj.get('as_of')
        return str(a) if a else None
    except Exception:
        return None

def _badge(ok: bool) -> str:
    return f"✅ OK" if ok else "⚠️ Manquant"

grid = st.columns(3)

# Forecasts
dt_fc, p_fc = _latest_dt('data/forecast/dt=*/forecasts.parquet')
ok_fc = p_fc is not None and p_fc.exists()
with grid[0]:
    st.subheader("Forecasts")
    st.write(_badge(ok_fc))
    st.caption(f"Dernier dt: {dt_fc or '—'}")
    if p_fc:
        st.code(str(p_fc), language='text')

# Macro Regime
dt_reg, p_reg = _latest_dt('data/macro/regime/dt=*/regime.json')
ok_reg = p_reg is not None and p_reg.exists()
with grid[1]:
    st.subheader("Macro Regime")
    st.write(_badge(ok_reg))
    st.caption(f"Dernier dt: {dt_reg or '—'}")
    if p_reg:
        st.caption(f"asof: { _read_json_asof(p_reg) or '—' }")
        st.code(str(p_reg), language='text')

# Risk Monitor
dt_risk, p_risk = _latest_dt('data/risk/dt=*/risk.json')
ok_risk = p_risk is not None and p_risk.exists()
with grid[2]:
    st.subheader("Risk Monitor")
    st.write(_badge(ok_risk))
    st.caption(f"Dernier dt: {dt_risk or '—'}")
    if p_risk:
        st.caption(f"asof: { _read_json_asof(p_risk) or '—' }")
        st.code(str(p_risk), language='text')

grid2 = st.columns(3)

# Earnings
dt_earn, p_earn = _latest_dt('data/earnings/dt=*/earnings.json')
ok_earn = p_earn is not None and p_earn.exists()
with grid2[0]:
    st.subheader("Earnings")
    st.write(_badge(ok_earn))
    st.caption(f"Dernier dt: {dt_earn or '—'}")
    if p_earn:
        st.caption(f"asof: { _read_json_asof(p_earn) or '—' }")
        st.code(str(p_earn), language='text')

# Memos
dt_memo, p_memo_any = _latest_dt('data/memos/dt=*/*')
ok_memo = p_memo_any is not None and p_memo_any.exists()
with grid2[1]:
    st.subheader("Memos")
    st.write(_badge(ok_memo))
    if dt_memo:
        st.caption(f"Dernier dt: {dt_memo}")
    if p_memo_any:
        base = p_memo_any.parent
        count = len(list(base.glob('*.json')))
        st.caption(f"Fichiers: {count} dans {base}")
        st.code(str(base), language='text')

# Quality Report
dt_q, p_q = _latest_dt('data/quality/dt=*/report.json')
ok_q = p_q is not None and p_q.exists()
with grid2[2]:
    st.subheader("Quality Report")
    st.write(_badge(ok_q))
    st.caption(f"Dernier dt: {dt_q or '—'}")
    if p_q:
        st.caption(f"asof: { _read_json_asof(p_q) or '—' }")
        st.code(str(p_q), language='text')

st.divider()
with st.expander("(Dev) Commandes utiles", expanded=False):
    st.code(
        """
make factory-run      # pipeline one-pass (sans orchestrator)
make risk-monitor     # met à jour data/risk
make recession        # met à jour data/macro/regime
make earnings         # collecte dates pour WATCHLIST
make memos            # génère des notes d'investissement
make backfill-prices  # remplit ≥5 ans de prix
        """.strip(), language='bash')

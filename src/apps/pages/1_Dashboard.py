from __future__ import annotations

from pathlib import Path
import sys as _sys
import os
import json
import streamlit as st
import pandas as pd
from ui.shell import page_header, page_footer

SRC = Path(__file__).resolve().parents[2]
if str(SRC) not in _sys.path:
    _sys.path.insert(0, str(SRC))

from analytics.recommender import rank
from core.data_store import have_files, query_duckdb
import numpy as np


st.set_page_config(page_title="Dashboard — Finance Agent", layout="wide")
st.title("📊 Dashboard — Résumé & Picks")
page_header(active="user")
with st.sidebar:
    beginner = st.toggle("Beginner mode", value=False, help="Affiche des explications simples et des indices d'interprétation.")
st.caption("Uses Parquet/DuckDB if available for fast scanning; falls back to JSON.")
if beginner:
    st.info("Cette page résume les meilleures idées selon les signaux disponibles (prévisions, ML, consensus LLM). Utilisez-la pour un aperçu rapide.")

# Alerts badge (latest Quality report error/warn counts)
try:
    from pathlib import Path as _P
    import json as _json
    qrep = sorted(_P('data/quality').glob('dt=*/report.json'))
    if qrep:
        rep = _json.loads(_P(qrep[-1]).read_text(encoding='utf-8'))
        def _count(rep, sev):
            cnt = 0
            for sec in ['news','macro','prices','forecasts','features','events','freshness']:
                s = rep.get(sec) or {}
                for it in (s.get('issues') or []):
                    if str(it.get('sev','')).lower() == sev:
                        cnt += 1
            return cnt
        err_n = _count(rep, 'error'); warn_n = _count(rep, 'warn')
        c1, c2 = st.columns(2)
        with c1: st.metric("Alerts — Errors", err_n)
        with c2: st.metric("Alerts — Warnings", warn_n)
except Exception:
    pass

with st.sidebar:
    st.header("Source")
    base = Path("data/forecast")
    dates = sorted([p.name for p in base.glob("*/")], reverse=True)
    chosen = st.selectbox("Date folder", dates, index=0 if dates else None)
    refresh = st.button("Refresh")
    st.divider()
    st.subheader("Gold focus")
    gold_tickers = st.text_input("Gold miners (comma)", value="NGD.TO,AEM.TO,ABX.TO,K.TO,GDX")

rows = []
# Fast path: prefer aggregated Final if present (1m), fallback to forecasts parquet
try:
    # Prefer final.parquet
    from pathlib import Path as _P
    parts_f = sorted(_P('data/forecast').glob('dt=*/final.parquet'))
    if parts_f:
        _fdf = pd.read_parquet(parts_f[-1])
        _fsel = _fdf[_fdf['horizon']=='1m'].sort_values('final_score', ascending=False).head(10)
        if not _fsel.empty:
            st.subheader("Top 10 (Final)")
            st.dataframe(_fsel[['ticker','final_score']].reset_index(drop=True), use_container_width=True)
    # Fallback to forecasts parquet detailed view
    if have_files("data/forecast/dt=*/forecasts.parquet"):
        dfp = query_duckdb("""
            select * from read_parquet('data/forecast/dt=*/forecasts.parquet')
            where horizon = '1m'
        """)
        if not dfp.empty:
            dir_map = {"up": 1.0, "flat": 0.0, "down": -1.0}
            dfp["dir_base"] = dfp["direction"].map(dir_map).fillna(0.0)
            # If ML columns exist, blend into score
            if "ml_return" in dfp.columns:
                dfp["score"] = (
                    dfp["dir_base"] * dfp["confidence"].astype(float)
                    + 0.4 * dfp["expected_return"].fillna(0.0).astype(float)
                    + 0.3 * dfp["ml_return"].fillna(0.0).astype(float) * dfp.get("ml_conf", 0.6)
                )
            else:
                dfp["score"] = dfp["dir_base"] * dfp["confidence"].astype(float) + 0.5 * dfp["expected_return"].fillna(0.0).astype(float)
            cols = ["ticker","score","direction","confidence","expected_return"] + (["ml_return","ml_conf"] if "ml_return" in dfp.columns else [])
            top = dfp.sort_values("score", ascending=False).head(10)[cols]
            st.subheader("Top 10 (Parquet)")
            st.dataframe(top, use_container_width=True)
            # Gold miners focus table
            try:
                gset = set([t.strip().upper() for t in gold_tickers.split(',') if t.strip()])
                gdf = dfp[dfp["ticker"].str.upper().isin(gset)].copy()
                if not gdf.empty:
                    gdf = gdf.sort_values("score", ascending=False)[["ticker","score","direction","confidence","expected_return"]]
                    st.subheader("Gold miners focus (1m)")
                    st.dataframe(gdf, use_container_width=True)
                    # Enrich with features_flat if present
                    if have_files("data/features/dt=*/features_flat.parquet"):
                        fdf = query_duckdb("select * from read_parquet('data/features/dt=*/features_flat.parquet')")
                        if not fdf.empty:
                            fdf = fdf[fdf["ticker"].str.upper().isin(gset)]
                            cols = [c for c in ["ticker","news_count","mean_sentiment","pos_ratio","neg_ratio","y_pe","y_beta","dividend_yield"] if c in fdf.columns]
                            if len(cols) > 1:
                                st.subheader("Gold features (latest partition)")
                                # take latest dt per ticker
                                fdf = fdf.sort_values(["ticker","dt"]).groupby("ticker", as_index=False).tail(1)
                                st.dataframe(fdf[cols], use_container_width=True)
                                # Heatmap: sentiment, momentum (21d), valuation (PE), risk (beta)
                                try:
                                    import pandas as _pd
                                    import plotly.express as px
                                    # compute momentum 21d from cached prices if available
                                    def _momentum_21d(t):
                                        import os
                                        from pathlib import Path as _P
                                        p = _P("data/prices")/f"ticker={t}"/"prices.parquet"
                                        if p.exists():
                                            dfp = _pd.read_parquet(p)
                                            col_date = 'date' if 'date' in dfp.columns else None
                                            if col_date:
                                                dfp[col_date] = _pd.to_datetime(dfp[col_date], errors='coerce')
                                                dfp = dfp.set_index(col_date)
                                            if 'Close' in dfp.columns and len(dfp) > 21:
                                                return float(dfp['Close'].iloc[-1]/dfp['Close'].iloc[-21]-1.0)
                                        return None
                                    latest = fdf.set_index('ticker')
                                    heat = _pd.DataFrame(index=[t for t in latest.index])
                                    heat['sentiment'] = latest.get('mean_sentiment')
                                    heat['pe'] = latest.get('y_pe')
                                    heat['beta'] = latest.get('y_beta')
                                    heat['mom_21d'] = [ _momentum_21d(t) for t in heat.index ]
                                    # normalize columns for display (z‑score like), keep signs
                                    def _norm(s):
                                        try:
                                            s = _pd.to_numeric(s, errors='coerce')
                                            m, sd = s.mean(skipna=True), s.std(skipna=True)
                                            return (s - m)/sd if sd and sd>0 else s*0
                                        except Exception:
                                            return s
                                    vis = _pd.DataFrame({
                                        'sentiment': _norm(heat['sentiment']),
                                        'momentum': _norm(heat['mom_21d']),
                                        'pe_inv': _norm(1.0/heat['pe'].replace(0, _pd.NA)),
                                        'beta_inv': _norm(1.0/heat['beta'].replace(0, _pd.NA)),
                                    }, index=heat.index)
                                    fig = px.imshow(vis.T, color_continuous_scale='RdBu', aspect='auto', origin='lower', labels=dict(color='z-score'))
                                    fig.update_layout(title='Gold Miners Heatmap (sentiment/momentum/valuation/risk)')
                                    st.plotly_chart(fig, use_container_width=True)
                                except Exception:
                                    pass
            except Exception:
                pass
except Exception:
    pass

# ---- Final Top‑5 (1m) and Macro Regime badge + quick metrics ----
try:
    import pandas as _pd
    from pathlib import Path as _P
    # final top‑N if available
    parts = sorted(_P('data/forecast').glob('dt=*/final.parquet'))
    if parts:
        fdf = _pd.read_parquet(parts[-1])
        if not fdf.empty:
            sel = fdf[fdf['horizon']=='1m'].sort_values('final_score', ascending=False).head(5)
            if not sel.empty:
                st.subheader("Final Top‑5 (1m)")
                st.dataframe(sel[['ticker','final_score']].reset_index(drop=True), use_container_width=True)
    # macro regime badge
    reg = sorted(_P('data/macro/regime').glob('dt=*/regime.json'))
    if reg:
        import json as _json
        robj = _json.loads(_P(reg[-1]).read_text(encoding='utf-8'))
        pr = robj.get('probs') or {}
        if pr:
            top_name, top_p = sorted(pr.items(), key=lambda x: -x[1])[0]
            c1, c2 = st.columns(2)
            with c1:
                st.metric("Régime macro dominant", top_name)
            with c2:
                st.metric("Confiance (≈)", f"{int((top_p or 0)*100)}%")
    # simple backtest card (Top‑N hit rate over last 90 days)
    if have_files("data/forecast/dt=*/forecasts.parquet"):
        dfb = query_duckdb("select * from read_parquet('data/forecast/dt=*/forecasts.parquet') where horizon='1m' order by dt")
        if not dfb.empty:
            dfb['dt'] = _pd.to_datetime(dfb['dt'], errors='coerce')
            end = dfb['dt'].max(); start = end - _pd.Timedelta(days=90)
            dfb = dfb[(dfb['dt']>=start)&(dfb['dt']<=end)].copy()
            if not dfb.empty:
                dir_map = {"up":1.0,"flat":0.0,"down":-1.0}
                dfb['dir_base'] = dfb['direction'].map(dir_map).fillna(0.0)
                score = dfb['dir_base']*dfb['confidence'].astype(float) + 0.5*dfb['expected_return'].fillna(0.0).astype(float)
                dfb['score'] = score
                H=21
                def _realized(t, d):
                    p = _P('data/prices')/f"ticker={t}"/'prices.parquet'
                    if not p.exists(): return None
                    try:
                        dd = _pd.read_parquet(p)
                        if 'date' in dd.columns:
                            dd['date'] = _pd.to_datetime(dd['date'], errors='coerce'); dd = dd.set_index('date')
                        if dd.empty or 'Close' not in dd.columns: return None
                        idx = dd.index.get_loc(d, method='nearest')
                    except Exception:
                        aft = dd[dd.index>=d];
                        if aft.empty: return None
                        idx = dd.index.get_loc(aft.index[0])
                    j = min(len(dd)-1, idx+H)
                    return float(dd['Close'].iloc[j]/dd['Close'].iloc[idx] - 1.0)
                daily=[]
                for d, sdf in dfb.groupby(dfb['dt'].dt.date):
                    sdf = sdf.sort_values('score', ascending=False).head(5)
                    rets=[]
                    for _, r in sdf.iterrows():
                        rr = _realized(str(r['ticker']), _pd.Timestamp(d))
                        if rr is not None: rets.append(rr)
                    if rets:
                        daily.append(_pd.Series({'dt':str(d),'mean_ret':float(_pd.Series(rets).mean())}))
                if daily:
                    dd = _pd.DataFrame(daily)
                    hr = float((dd['mean_ret']>0).mean()) if not dd.empty else 0.0
                    avg = float(dd['mean_ret'].mean()) if not dd.empty else 0.0
                    c1, c2 = st.columns(2)
                    with c1: st.metric("Hit‑rate 90j (Top‑5)", f"{int(hr*100)}%")
                    with c2: st.metric("Moy. panier 1m/j", f"{avg*100:.2f}%")
except Exception:
    pass

# ---- Mini Alerts card (top issues & moves) ----
try:
    from pathlib import Path as _P
    import json as _json
    # Quality top issue
    q = sorted(_P('data/quality').glob('dt=*/report.json'))
    top_issue = None
    if q:
        rep = _json.loads(_P(q[-1]).read_text(encoding='utf-8'))
        sev_order = {'error':0,'warn':1,'info':2}
        items=[]
        for sec in ['news','macro','prices','forecasts','features','events']:
            for it in (rep.get(sec,{}).get('issues') or []):
                items.append((sev_order.get(str(it.get('sev','info')).lower(),9), sec, it.get('msg')))
        if items:
            items.sort(key=lambda x: x[0])
            top_issue = f"{items[0][1]}: {items[0][2]}"
    # Biggest watchlist move from brief
    b = sorted(_P('data/forecast').glob('dt=*/brief.json'))
    top_move = None
    if b:
        br = _json.loads(_P(b[-1]).read_text(encoding='utf-8'))
        w = (br.get('changes') or {}).get('watchlist_moves') or []
        if w:
            import pandas as _pd
            dfw=_pd.DataFrame(w); dfw['abs']=dfw['d1'].abs(); dfw=dfw.sort_values('abs', ascending=False)
            top_move = f"{dfw.iloc[0]['ticker']}: {round(float(dfw.iloc[0]['d1'])*100,2)}% (1j)"
    if top_issue or top_move:
        st.subheader("Alerts (résumé)")
        if top_issue: st.write(f"- Données: {top_issue}")
        if top_move: st.write(f"- Mouvement: {top_move}")
except Exception:
    pass

# Optional: Cumulative Top‑N performance chart if parquet forecasts + cached prices exist
try:
    if have_files("data/forecast/dt=*/forecasts.parquet"):
        dfp = query_duckdb("select dt, ticker, direction, confidence, expected_return from read_parquet('data/forecast/dt=*/forecasts.parquet') where horizon='1m'")
        if not dfp.empty:
            # compute daily mean return of Top‑N and cumulate (reuse Evaluation logic, simplified)
            dfp['dt'] = pd.to_datetime(dfp['dt'], errors='coerce')
            top_n = 5
            H = 21
            def _realized(ticker: str, d: pd.Timestamp, days: int) -> float | None:
                p = Path("data/prices")/f"ticker={ticker}"/"prices.parquet"
                if not p.exists():
                    return None
                try:
                    df = pd.read_parquet(p)
                    if 'date' in df.columns:
                        df['date'] = pd.to_datetime(df['date'], errors='coerce')
                        df = df.set_index('date')
                    if df.empty or 'Close' not in df.columns:
                        return None
                    idx = df.index.get_loc(d, method='nearest')
                except Exception:
                    after = df[df.index >= d]
                    if after.empty: return None
                    idx = df.index.get_loc(after.index[0])
                j = min(len(df)-1, idx+days)
                return float(df['Close'].iloc[j]/df['Close'].iloc[idx]-1.0)
            daily = []
            # simple score: direction*confidence + expected_return
            dir_map = {"up": 1.0, "flat": 0.0, "down": -1.0}
            dfp['dir_base'] = dfp['direction'].map(dir_map).fillna(0.0)
            dfp['score'] = dfp['dir_base']*dfp['confidence'].astype(float) + 0.5*dfp['expected_return'].fillna(0.0).astype(float)
            for d, sdf in dfp.groupby(dfp['dt'].dt.date):
                sdf = sdf.sort_values('score', ascending=False).head(top_n)
                rets = []
                for _, row in sdf.iterrows():
                    rr = _realized(str(row['ticker']), pd.Timestamp(d), H)
                    if rr is not None:
                        rets.append(rr)
                if rets:
                    daily.append({'dt': str(d), 'mean_return': float(np.mean(rets))})
            if daily:
                perf = pd.DataFrame(daily)
                perf['dt'] = pd.to_datetime(perf['dt'], errors='coerce')
                perf = perf.sort_values('dt')
                perf['cum_return'] = (1.0 + perf['mean_return']).cumprod() - 1.0
                st.subheader("Cumulative Performance (Top‑N basket, 1m horizon)")
                st.area_chart(perf.set_index('dt')[['cum_return']])
except Exception:
    pass
if chosen:
    date_dir = Path("data/forecast") / chosen
    for f in date_dir.glob("*.json"):
        if f.name == "summary.json":
            continue
        try:
            obj = json.loads(f.read_text(encoding="utf-8"))
            rows.append(obj)
        except Exception:
            pass

if rows:
    ranked = rank(rows)
    st.subheader("Top 10 recommandations (JSON)")
    df = pd.DataFrame([
        {"ticker": r["ticker"], "score": r["score"], "reasons": ", ".join(r["reasons"]) }
        for r in ranked[:10]
    ])
    st.dataframe(df, use_container_width=True)
else:
    st.info("Aucune prévision disponible pour cette date. Consultez Admin → Agents Status pour l'état du pipeline ou réessayez plus tard.")

# Daily brief (if any) and macro KPIs + changes
try:
    if chosen:
        brief_path = Path("data/forecast") / chosen / "brief.json"
        if brief_path.exists():
            import json as _json
            st.subheader("Daily Brief")
            brief = _json.loads(brief_path.read_text(encoding="utf-8"))
            macro = (brief or {}).get("macro") or {}
            # KPIs
            k1, k2, k3 = st.columns(3)
            with k1:
                st.metric("DXY WoW", f"{macro.get('DXY_wow') if macro.get('DXY_wow') is not None else 'n/a'}")
            with k2:
                st.metric("UST10Y bp WoW", f"{macro.get('UST10Y_bp_wow') if macro.get('UST10Y_bp_wow') is not None else 'n/a'}")
            with k3:
                st.metric("Gold WoW", f"{macro.get('Gold_wow') if macro.get('Gold_wow') is not None else 'n/a'}")
            # Changes since yesterday (simple text summary)
            changes = (brief or {}).get('changes') or {}
            if changes:
                st.subheader("Qu’est‑ce qui a changé depuis hier ?")
                # macro d1
                m = changes.get('macro') or {}
                bullets = []
                def pct(v):
                    try:
                        return f"{float(v)*100:.2f}%"
                    except Exception:
                        return "n/a"
                def bp(v):
                    try:
                        return f"{float(v):.1f} bp"
                    except Exception:
                        return "n/a"
                if 'DXY_d1' in m:
                    bullets.append(f"Dollar américain: {pct(m.get('DXY_d1'))} sur 1 jour")
                if 'UST10Y_bp_d1' in m:
                    bullets.append(f"Taux US 10 ans: {bp(m.get('UST10Y_bp_d1'))} sur 1 jour")
                if 'Gold_d1' in m:
                    bullets.append(f"Or: {pct(m.get('Gold_d1'))} sur 1 jour")
                # watchlist moves
                w = changes.get('watchlist_moves') or []
                if w:
                    top = ", ".join([f"{x['ticker']}: {pct(x['d1'])}" for x in w])
                    bullets.append(f"Watchlist principaux mouvements (1j): {top}")
                if bullets:
                    for b in bullets:
                        st.write(f"- {b}")
            st.json(brief)
except Exception:
    pass

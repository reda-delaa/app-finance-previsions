from pathlib import Path
import sys as _sys
import os
import json
import streamlit as st
import pandas as pd

SRC = Path(__file__).resolve().parents[2]
if str(SRC) not in _sys.path:
    _sys.path.insert(0, str(SRC))

from analytics.recommender import rank
from core.data_store import have_files, query_duckdb


st.set_page_config(page_title="Dashboard — Finance Agent", layout="wide")
st.title("📊 Dashboard — Résumé & Picks")
st.caption("Uses Parquet/DuckDB if available for fast scanning; falls back to JSON.")

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
# Fast path: show top from Parquet if present (1m only)
try:
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
    st.info("Aucun fichier de prévisions trouvé. Exécutez `python scripts/agent_daily.py` pour générer la journée.")

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

from __future__ import annotations

from pathlib import Path
import sys as _sys
import pandas as pd
import streamlit as st
from ui.shell import page_header, page_footer

SRC = Path(__file__).resolve().parents[2]
if str(SRC) not in _sys.path:
    _sys.path.insert(0, str(SRC))

st.set_page_config(page_title="Signals — Finance Agent", layout="wide")
page_header(active="user")
st.subheader("🔎 Signals — Indicateurs par Titre")

from core.data_store import have_files, query_duckdb

with st.sidebar:
    st.header("Paramètres")
    beginner = st.toggle("Mode débutant", value=False, help="Affiche des explications simples pour comprendre les scores.")
    horizon = st.selectbox("Horizon", ["1w","1m","1y"], index=1)
    top_n = st.slider("Top N", 1, 20, 10)
    st.caption("Pondération des composantes du signal (normalisées).")
    w_rule = st.slider("Poids règle (direction+ER)", 0.0, 1.0, 0.65, 0.05)
    w_ml = st.slider("Poids ML", 0.0, 1.0, 0.25, 0.05)
    w_llm = st.slider("Poids LLM (consensus)", 0.0, 1.0, 0.10, 0.05)

def _mom_21d(ticker: str) -> float | None:
    p = Path('data/prices')/f"ticker={ticker}"/'prices.parquet'
    if not p.exists():
        return None
    try:
        df = pd.read_parquet(p)
        col_date = 'date' if 'date' in df.columns else None
        if col_date:
            df[col_date] = pd.to_datetime(df[col_date], errors='coerce')
            df = df.set_index(col_date)
        if 'Close' in df.columns and len(df) > 21:
            return float(df['Close'].iloc[-1]/df['Close'].iloc[-21] - 1.0)
    except Exception:
        return None
    return None

if not have_files("data/forecast/dt=*/forecasts.parquet"):
    st.info("Aucune prévision consolidée disponible pour l'instant. Consultez Admin → Agents Status pour l'état du pipeline.")
else:
    df = query_duckdb(f"select * from read_parquet('data/forecast/dt=*/forecasts.parquet') where horizon='{horizon}'")
    if df.empty:
        st.info("Aucune donnée pour cet horizon.")
    else:
        dir_map = {"up":1.0,"flat":0.0,"down":-1.0}
        df['dir_base'] = df['direction'].map(dir_map).fillna(0.0)
        score = df['dir_base']*df['confidence'].astype(float) + 0.5*df['expected_return'].fillna(0.0).astype(float)
        ml_part = (df.get('ml_return', 0.0).fillna(0.0).astype(float) * df.get('ml_conf', 0.6).fillna(0.6).astype(float)) if 'ml_return' in df.columns else 0.0
        # LLM consensus (avg_agreement) if llm_agents.json available
        llm_map = {}
        try:
            # pick latest dt folder if present
            parts = sorted(Path('data/forecast').glob('dt=*/llm_agents.json'))
            if parts:
                obj = __import__('json').loads(Path(parts[-1]).read_text(encoding='utf-8'))
                for it in (obj.get('tickers') or []):
                    t = (it or {}).get('ticker')
                    ens = (it or {}).get('ensemble') or {}
                    aa = ens.get('avg_agreement')
                    if t and isinstance(aa, (int,float)):
                        llm_map[str(t)] = max(0.0, min(1.0, float(aa)))
        except Exception:
            pass
        df['llm_consensus'] = df['ticker'].map(lambda x: llm_map.get(str(x), 0.0)).astype(float)
        # normalize weights
        total_w = max(1e-9, (w_rule + w_ml + w_llm))
        wr, wm, wl = w_rule/total_w, w_ml/total_w, w_llm/total_w
        df['signal_score'] = wr*score + wm*ml_part + wl*df['llm_consensus']
        # enrich with features_flat if present
        if have_files("data/features/dt=*/features_flat.parquet"):
            fdf = query_duckdb("select * from read_parquet('data/features/dt=*/features_flat.parquet')")
            if not fdf.empty:
                # latest per ticker
                fdf = fdf.sort_values(['ticker','dt']).groupby('ticker', as_index=False).tail(1)
                cols = [c for c in ['ticker','news_count','mean_sentiment','pos_ratio','neg_ratio','y_pe','y_beta','dividend_yield'] if c in fdf.columns]
                df = df.merge(fdf[cols], on='ticker', how='left')
        # momentum 21d
        df['mom_21d'] = [ _mom_21d(t) for t in df['ticker'] ]
        # Join final_score if available for the same horizon
        try:
            import pandas as _pd
            from pathlib import Path as _P
            parts = sorted(_P('data/forecast').glob('dt=*/final.parquet'))
            if parts:
                fdf = _pd.read_parquet(parts[-1])
                fdf = fdf[fdf['horizon']==horizon][['ticker','final_score']]
                df = df.merge(fdf, on='ticker', how='left')
        except Exception:
            pass
        # show top
        top = df.sort_values(['final_score','signal_score'] if 'final_score' in df.columns else ['signal_score'], ascending=False).head(top_n)
        show_cols_base = ['ticker','final_score','signal_score','direction','confidence','expected_return','ml_return','ml_conf','llm_consensus','mean_sentiment','mom_21d','y_pe','y_beta']
        show_cols = [c for c in show_cols_base if c in top.columns]
        if beginner:
            st.caption("Le score combine: règle (direction x confiance + ER), ML (retour x confiance), et consensus LLM (0–1). Plus haut = plus favorable.")
        st.subheader("Top signaux")
        st.dataframe(top[show_cols], use_container_width=True)
        try:
            csv_bytes = top[show_cols].to_csv(index=False).encode('utf-8')
            st.download_button("Télécharger (CSV)", data=csv_bytes, file_name=f"signals_{horizon}.csv", mime="text/csv")
        except Exception:
            pass
page_footer()

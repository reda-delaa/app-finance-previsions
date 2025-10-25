from pathlib import Path
import sys as _sys
import streamlit as st
import pandas as pd

SRC = Path(__file__).resolve().parents[2]
if str(SRC) not in _sys.path:
    _sys.path.insert(0, str(SRC))

from analytics.news_aggregator import aggregate_news, summarize_news
from core.data_store import have_files
import glob
import pandas as pd

st.set_page_config(page_title="News — Finance Agent", layout="wide")
st.title("🗞️ News — Agrégation & Synthèse IA")

with st.sidebar:
    st.header("Paramètres")
    regions = st.text_input("Régions (comma)", value="US,INTL")
    window = st.selectbox("Fenêtre", ["24h","48h","last_week","last_month","all"], index=2)
    query = st.text_input("Filtre mot-clé (optionnel)", value="")
    company = st.text_input("Entreprise (optionnel)", value="")
    ticker = st.text_input("Ticker (optionnel)", value="")
    limit = st.slider("Limite", min_value=50, max_value=500, value=200, step=50)
    st.divider()
    st.subheader("Parquet (historique)")
    use_parquet = st.checkbox("Utiliser données Parquet si présentes", value=True)
    days_back = st.slider("Jours à charger", min_value=1, max_value=365, value=30, step=1)
    kw_filter = st.text_input("Mot-clé filtre (titre/summary)", value="")
    run = st.button("Agrèger & Résumer")

if run:
    regs = [r.strip() for r in regions.split(",") if r.strip()]
    news = []
    if use_parquet and have_files("data/news/dt=*/*.parquet"):
        try:
            import datetime as dt
            end = dt.datetime.utcnow().date()
            start = end - dt.timedelta(days=days_back)
            # Load files for last N days
            files = []
            for d in pd.date_range(start, end):
                files.extend(glob.glob(f"data/news/dt={d.date()}/*.parquet"))
            rows = []
            for f in files[:500]:  # safety cap
                try:
                    dfp = pd.read_parquet(f)
                    rows.append(dfp)
                except Exception:
                    pass
            if rows:
                dfa = pd.concat(rows, ignore_index=True)
                # optional keyword filter
                if kw_filter:
                    k = kw_filter.lower()
                    def _contains(s):
                        try:
                            return k in str(s or '').lower()
                        except Exception:
                            return False
                    dfa = dfa[(dfa['title'].apply(_contains)) | (dfa['summary'].apply(_contains))]
                news = dfa.to_dict(orient='records')
        except Exception:
            pass
    # Fallback to live aggregation
    if not news:
        with st.spinner("Agrégation des news (live)..."):
            out = aggregate_news(regions=regs, window=window, query=query, company=company or None, tgt_ticker=ticker or None, limit=limit)
            news = out.get("news", [])
    st.write(f"Articles agrégés: {len(news)}")
    if news:
        df = pd.DataFrame(news)
        st.dataframe(df[[c for c in ["ts","source","title","sent","tickers","link"] if c in df.columns]], use_container_width=True)
        with st.spinner("Synthèse IA..."):
            summ = summarize_news(news)
        st.subheader("Synthèse")
        st.write(summ.get("text") or "(vide)")
        if st.checkbox("Afficher JSON de synthèse"):
            st.json(summ.get("json"))
    else:
        st.info("Aucun article trouvé.")

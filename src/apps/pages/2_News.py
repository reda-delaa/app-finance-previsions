from pathlib import Path
import sys as _sys
import streamlit as st
from ui.shell import page_header, page_footer
import pandas as pd

SRC = Path(__file__).resolve().parents[2]
if str(SRC) not in _sys.path:
    _sys.path.insert(0, str(SRC))

from analytics.news_aggregator import aggregate_news, summarize_news
from core.data_store import have_files
import glob
import pandas as pd

st.set_page_config(page_title="News — Finance Agent", layout="wide")
page_header(active="user")
st.subheader("🗞️ News — Agrégation & Synthèse IA")

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
    st.subheader("Filtres avancés")
    colf1, colf2 = st.columns(2)
    with colf1:
        evt_earn = st.checkbox("Résultats (earnings)")
        evt_mna = st.checkbox("Fusions/Acquisitions (M&A)")
    with colf2:
        evt_geo = st.checkbox("Géopolitique")
        evt_macro = st.checkbox("Macro (CPI, emploi, taux)")
    sector = st.selectbox("Secteur (indice)", ["(Tous)", "gold", "energy", "financials", "technology"], index=0)
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
                # advanced filters
                def _ensure(df, col, default=False):
                    if col not in df.columns:
                        df[col] = default
                    return df
                dfa = _ensure(dfa, 'flag_earnings')
                dfa = _ensure(dfa, 'flag_mna')
                dfa = _ensure(dfa, 'flag_geopolitics')
                dfa = _ensure(dfa, 'flag_macro')
                dfa = _ensure(dfa, 'sector_hint', None)
                if evt_earn:
                    dfa = dfa[dfa['flag_earnings'] == True]
                if evt_mna:
                    dfa = dfa[dfa['flag_mna'] == True]
                if evt_geo:
                    dfa = dfa[dfa['flag_geopolitics'] == True]
                if evt_macro:
                    dfa = dfa[dfa['flag_macro'] == True]
                if sector != "(Tous)":
                    dfa = dfa[(dfa['sector_hint'].fillna('') == sector)]
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
        # Ensure enrichment columns exist for display
        for c, default in (
            ("flag_earnings", False),
            ("flag_mna", False),
            ("flag_geopolitics", False),
            ("flag_macro", False),
            ("sector_hint", None),
        ):
            if c not in df.columns:
                df[c] = default
        # Small header metrics
        colm1, colm2, colm3, colm4, colm5 = st.columns(5)
        with colm1:
            st.metric("Résultats (earnings)", int(df["flag_earnings"].astype(bool).sum()))
        with colm2:
            st.metric("Fusions/Acquisitions", int(df["flag_mna"].astype(bool).sum()))
        with colm3:
            st.metric("Géopolitique", int(df["flag_geopolitics"].astype(bool).sum()))
        with colm4:
            st.metric("Macro (CPI, emploi, taux)", int(df["flag_macro"].astype(bool).sum()))
        with colm5:
            top_sector = (df["sector_hint"].value_counts(dropna=True).head(1).index.tolist() or ["—"])[0]
            st.metric("Secteur dominant", str(top_sector or "—"))
        # Display dataframe with event/sector columns
        shown_cols = [c for c in [
            "ts","source","title","sent","tickers","sector_hint",
            "flag_earnings","flag_mna","flag_geopolitics","flag_macro","link"
        ] if c in df.columns]
        st.dataframe(df[shown_cols], use_container_width=True)
        st.caption("Légende: • Résultats = publications financières • Fusions/Acquisitions = M&A • Géopolitique = conflits/sanctions/élections • Macro = inflation/emploi/Fed")
        with st.spinner("Synthèse IA..."):
            summ = summarize_news(news)
        st.subheader("Synthèse")
        st.write(summ.get("text") or "(vide)")
        if st.checkbox("Afficher JSON de synthèse"):
            st.json(summ.get("json"))
    else:
        st.info("Aucun article trouvé.")
else:
    st.info("Saisissez vos filtres puis cliquez « Agrèger & Résumer ». Si des parquets existent (data/news/dt=*), ils seront utilisés en priorité.")
page_footer()

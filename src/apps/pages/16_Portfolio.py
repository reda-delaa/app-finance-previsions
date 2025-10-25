from pathlib import Path
import sys as _sys
import json
import pandas as pd
import streamlit as st

SRC = Path(__file__).resolve().parents[2]
if str(SRC) not in _sys.path:
    _sys.path.insert(0, str(SRC))

st.set_page_config(page_title="Portfolio — Finance Agent", layout="wide")
st.title("💼 Portfolio — Poids & Propositions")

def _load_holdings() -> pd.DataFrame:
    p = Path('data/portfolio/holdings.json')
    if not p.exists():
        return pd.DataFrame(columns=['ticker','weight'])
    try:
        obj = json.loads(p.read_text(encoding='utf-8'))
        return pd.DataFrame(obj)
    except Exception:
        return pd.DataFrame(columns=['ticker','weight'])

def _latest_final_parquet() -> Path | None:
    parts = sorted(Path('data/forecast').glob('dt=*/final.parquet'))
    return parts[-1] if parts else None

with st.sidebar:
    st.header("Actions")
    if st.button("Créer un fichier holdings d'exemple"):
        p = Path('data/portfolio'); p.mkdir(parents=True, exist_ok=True)
        sample = [{"ticker":"AAPL","weight":0.2},{"ticker":"MSFT","weight":0.2}]
        (p/'holdings.json').write_text(json.dumps(sample, ensure_ascii=False, indent=2), encoding='utf-8')
        st.success("data/portfolio/holdings.json créé")
    top_n = st.slider("Top N (proposition)", 1, 10, 5)
    mode = st.selectbox("Méthode de pondération", ["Égal‑pondéré","Proportionnel au score"], index=0)

hold = _load_holdings()
st.subheader("Positions actuelles")
if hold.empty:
    st.info("Aucune position. Ajoutez data/portfolio/holdings.json")
else:
    st.dataframe(hold, use_container_width=True)

st.subheader("Proposition (Top‑N par score final)")
fp = _latest_final_parquet()
if not fp:
    st.info("Aucun final.parquet. Lancez scripts/fuse_forecasts.py")
else:
    df = pd.read_parquet(fp)
    if df.empty:
        st.info("final.parquet vide")
    else:
        top = df[df['horizon']=='1m'].sort_values('final_score', ascending=False).head(top_n)
        if top.empty:
            st.info("Pas de données 1m")
        else:
            if mode.startswith("Égal"):
                weights = [round(1.0/len(top), 6)]*len(top)
            else:
                sc = top['final_score'].clip(lower=0).astype(float)
                ssum = float(sc.sum()) or 1.0
                weights = [round(float(v/ssum), 6) for v in sc]
            prop = pd.DataFrame({'ticker': top['ticker'], 'proposed_weight': weights, 'final_score': top['final_score'].round(4)})
            st.dataframe(prop, use_container_width=True)
            st.caption("Poids égal‑pondéré; ajustez selon votre profil de risque.")
            # export buttons
            try:
                csv_bytes = prop.to_csv(index=False).encode('utf-8')
                st.download_button("Exporter pondérations (CSV)", data=csv_bytes, file_name="portfolio_proposed.csv", mime="text/csv")
                out_json = prop.to_dict(orient='records')
                st.download_button("Exporter pondérations (JSON)", data=json.dumps(out_json, ensure_ascii=False, indent=2), file_name="portfolio_proposed.json", mime="application/json")
            except Exception:
                pass

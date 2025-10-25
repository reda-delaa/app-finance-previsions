from pathlib import Path
import sys as _sys
import json
import streamlit as st

SRC = Path(__file__).resolve().parents[2]
if str(SRC) not in _sys.path:
    _sys.path.insert(0, str(SRC))

st.set_page_config(page_title="Memos — Finance Agent", layout="wide")
st.title("📄 Investment Memos — Par Titre")

with st.sidebar:
    st.header("Source")
    base = Path('data/memos')
    dates = sorted([p.name for p in base.glob('dt=*')], reverse=True)
    chosen = st.selectbox("Dossier date", dates, index=0 if dates else None)
    if st.button("Générer pour watchlist (script)"):
        st.info("Exécute: PYTHONPATH=src python scripts/run_memos.py")

if not chosen:
    st.info("Sélectionnez un dossier date")
else:
    ddir = Path('data/memos')/chosen
    files = sorted(ddir.glob('*.json'))
    if not files:
        st.info("Aucun memo pour cette date")
    else:
        tickers = [f.stem for f in files]
        t = st.selectbox("Ticker", tickers, index=0)
        p = ddir/f"{t}.json"
        try:
            obj = json.loads(p.read_text(encoding='utf-8'))
            st.subheader(t)
            st.markdown(obj.get('answer') or "")
            with st.expander("Parsed JSON"):
                st.json(obj.get('parsed'))
            with st.expander("Ensemble (résumé)"):
                ens = obj.get('ensemble') or {}
                if ens:
                    st.write({'models': ens.get('models'), 'avg_agreement': ens.get('avg_agreement')})
                st.json(ens)
        except Exception as e:
            st.warning(f"Lecture impossible: {e}")


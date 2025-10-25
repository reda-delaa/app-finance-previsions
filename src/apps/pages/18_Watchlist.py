from pathlib import Path
import sys as _sys
import json
import os
import streamlit as st

SRC = Path(__file__).resolve().parents[2]
if str(SRC) not in _sys.path:
    _sys.path.insert(0, str(SRC))

st.set_page_config(page_title="Watchlist — Finance Agent", layout="wide")
st.title("📜 Watchlist — Gestion")

current = os.getenv('WATCHLIST') or "NGD.TO,AEM.TO,ABX.TO,K.TO,GDX"
st.caption("La plupart des scripts utilisent la variable d'environnement WATCHLIST.")

st.subheader("Actuel")
st.code(current, language='bash')

st.subheader("Modifier (local)")
wl_text = st.text_area("Tickers séparés par des virgules", value=current, height=120)
col1, col2 = st.columns(2)
with col1:
    if st.button("Enregistrer dans data/watchlist.json"):
        try:
            lst = [x.strip().upper() for x in wl_text.split(',') if x.strip()]
            Path('data').mkdir(parents=True, exist_ok=True)
            (Path('data')/'watchlist.json').write_text(json.dumps({'watchlist': lst}, ensure_ascii=False, indent=2), encoding='utf-8')
            st.success("data/watchlist.json enregistré")
        except Exception as e:
            st.warning(f"Erreur: {e}")
with col2:
    if st.button("Générer commande export"):
        try:
            lst = [x.strip().upper() for x in wl_text.split(',') if x.strip()]
            cmd = f"export WATCHLIST={','.join(lst)}"
            st.code(cmd, language='bash')
            st.caption("Copiez/collez dans votre shell pour l'utiliser dans les scripts.")
        except Exception as e:
            st.warning(f"Erreur: {e}")


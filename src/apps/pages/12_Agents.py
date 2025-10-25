from pathlib import Path
import sys as _sys
import json
import streamlit as st

SRC = Path(__file__).resolve().parents[2]
if str(SRC) not in _sys.path:
    _sys.path.insert(0, str(SRC))

st.set_page_config(page_title="Agents — Finance Agent", layout="wide")
st.title("🤖 LLM Agents — Ensemble & Arbitrage")

with st.sidebar:
    st.header("Source")
    base = Path("data/forecast")
    dates = sorted([p.name for p in base.glob("*/")], reverse=True)
    chosen = st.selectbox("Dossier date", dates, index=0 if dates else None)
    st.caption("Générer via: python scripts/run_llm_agents.py")

if chosen:
    ddir = Path("data/forecast")/chosen
    f = ddir/"llm_agents.json"
    if not f.exists():
        st.info("Aucun résultat LLM pour cette date (llm_agents.json manquant).")
    else:
        try:
            obj = json.loads(f.read_text(encoding="utf-8"))
        except Exception as e:
            st.warning(f"Lecture impossible: {e}")
            obj = None
        if obj:
            items = obj.get("tickers") or []
            # Summary table
            rows = []
            for it in items:
                ens = it.get('ensemble') or {}
                rows.append({
                    'ticker': it.get('ticker'),
                    'models': ", ".join(ens.get('models') or []),
                    'avg_agreement': ens.get('avg_agreement'),
                    'has_judge': 'adjudication' in ens,
                })
            import pandas as pd
            st.subheader("Résumé par ticker")
            st.dataframe(pd.DataFrame(rows), use_container_width=True)
            # Details
            for it in items:
                t = it.get('ticker')
                st.subheader(t)
                ens = it.get('ensemble') or {}
                if not ens:
                    st.caption("(pas de réponse)")
                    continue
                st.write({
                    'models': ens.get('models'),
                    'avg_agreement': ens.get('avg_agreement'),
                })
                if ens.get('pairwise_agreement'):
                    st.caption("Accord pair-à-pair entre modèles")
                    st.dataframe(pd.DataFrame(ens['pairwise_agreement']), use_container_width=True)
                if ens.get('consensus'):
                    st.caption("Consensus (points communs)")
                    st.json(ens.get('consensus'))
                if ens.get('divergences'):
                    st.caption("Divergences par modèle")
                    st.json(ens.get('divergences'))
                adj = ens.get('adjudication')
                if adj:
                    st.caption(f"Arbitre: {adj.get('judge_model')}")
                    st.write(adj.get('decision'))
                with st.expander("Brut (JSON)"):
                    st.json(ens)
else:
    st.info("Sélectionnez un dossier date pour consulter les résultats des agents LLM.")


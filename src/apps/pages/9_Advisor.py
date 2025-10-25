from pathlib import Path
import sys as _sys
import streamlit as st
import json

SRC = Path(__file__).resolve().parents[2]
if str(SRC) not in _sys.path:
    _sys.path.insert(0, str(SRC))

from analytics.market_intel import build_snapshot
from analytics.econ_llm_agent import EconomicAnalyst, EconomicInput
from core.prompt_context import load_role_briefs, load_next_steps

st.set_page_config(page_title="Conseiller — Finance Agent", layout="wide")
st.title("🧭 Conseiller — Explications en mots simples")

with st.sidebar:
    st.header("Paramètres")
    mode = st.selectbox("Niveau", ["Novice", "Avancé"], index=0)
    locale = st.selectbox("Langue", ["fr-FR", "en-US"], index=0)
    kind = st.selectbox("Sujet", ["Ticker", "Gold miners", "Marché global"], index=0)
    ticker = st.text_input("Ticker (si 'Ticker')", value="NGD.TO")
    horizon = st.selectbox("Horizon", ["1w","1m","1y"], index=1)
    run = st.button("Expliquer simplement")

def _question_text(kind: str, ticker: str, horizon: str, mode: str) -> str:
    if kind == "Ticker":
        return (
            f"Explique simplement la situation de {ticker} pour l’horizon {horizon}. "
            "Utilise des mots du quotidien, évite les abréviations, donne 3 points clés, 3 risques, et 3 signaux à suivre. "
            "Propose des actions génériques (pas de conseil personnalisé)."
        )
    if kind == "Gold miners":
        return (
            f"Explique simplement la situation du secteur des mines d’or pour l’horizon {horizon}. "
            "Évite les abréviations, 3 points clés, 3 risques, 3 signaux; actions génériques."
        )
    return (
        f"Explique simplement la situation du marché global pour l’horizon {horizon}. "
        "Évite les abréviations, 3 points clés, 3 risques, 3 signaux; actions génériques."
    )

if run:
    try:
        if kind == "Ticker":
            snap = build_snapshot(regions=["US","INTL"], window="last_week", ticker=ticker.strip().upper(), limit=150)
        elif kind == "Gold miners":
            # Use a representative ticker set as context (lightweight)
            snap = build_snapshot(regions=["US","INTL"], window="last_week", query="gold OR mining", limit=200)
        else:
            snap = build_snapshot(regions=["US","INTL"], window="last_week", limit=200)

        features = snap.get("features") or {}
        news = snap.get("news") or []
        role_briefs = load_role_briefs()
        next_steps = load_next_steps()

        q = _question_text(kind, ticker, horizon, mode)
        if locale.startswith("fr"):
            sys_prompt = (
                "Tu es un conseiller pédagogique. Évite les abréviations, et si elles sont inévitables, explique-les. "
                "Structure la réponse en sections courtes (points clés, risques, signaux, actions génériques)."
            )
        else:
            sys_prompt = (
                "You are an educational advisor. Avoid abbreviations, or define them simply. "
                "Use short sections: key points, risks, signals, generic actions."
            )

        agent = EconomicAnalyst()
        data = EconomicInput(
            question=q,
            features={"bundle": features, "horizon": horizon},
            news=news[:100],  # cap for speed
            attachments=[{"role_briefs": role_briefs}, {"next_steps": next_steps}],
            locale=locale,
            meta={"kind": "advisor", "simple_mode": mode == "Novice", "horizon": horizon, "topic": kind},
        )
        res = agent.analyze(data)
        st.subheader("Explication")
        st.write(res.get("answer") or "(vide)")
        if st.checkbox("Voir le JSON final"):
            st.json(res.get("parsed"))
    except Exception as e:
        st.error(f"Échec de l’explication: {e}")


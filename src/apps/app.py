# src/apps/app.py
# --- sys.path bootstrap (CRITIQUE) ---
from pathlib import Path
import sys as _sys
_SRC_ROOT = Path(__file__).resolve().parents[1]   # .../analyse-financiere/src
if str(_SRC_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_SRC_ROOT))
# -------------------------------------

import traceback, importlib, sys, json
import datetime as dt
import streamlit as st

st.set_page_config(page_title="Analyse Financière — Hub", layout="wide")
st.title("📈 Analyse Financière — Hub IA")

_DEBUG = st.sidebar.checkbox("Afficher DEBUG", value=True)

def log_debug(msg: str):
    if _DEBUG:
        st.sidebar.write(f"DEBUG: {msg}")
    # toujours log en console
    print(msg, flush=True)

def safe_import(path: str, attr: str | None = None):
    """
    Import robuste : retourne (objet | None, erreur | None)
    - path: ex. 'analytics.econ_llm_agent'
    - attr: ex. 'EconomicAnalyst' ou None pour renvoyer le module
    """
    try:
        mod = importlib.import_module(path)
        if attr is None:
            return mod, None
        if not hasattr(mod, attr):
            return None, f"module '{path}' has no attribute '{attr}'"
        return getattr(mod, attr), None
    except Exception as e:
        return None, f"{e.__class__.__name__}: {e}"

# ===== Imports UI pages =====
render_macro, err = safe_import("apps.macro_sector_app", "render_macro")
if err: log_debug(f"Failed to import apps.macro_sector_app.render_macro: {err}")

render_stock, err = safe_import("apps.stock_analysis_app", "render_stock")
if err: log_debug(f"Failed to import apps.stock_analysis_app.render_stock: {err}")

# ===== Feature providers / analytics =====
# Peers
find_peers, err = safe_import("research.peers_finder", "find_peers")
if err: log_debug(f"Failed to import research.peers_finder.find_peers: {err}")

# News
load_news, err = safe_import("ingestion.finnews", "run_pipeline")
# Wrapper pour load_news
if load_news:
    def _load_news_wrapper(window_days=7, regions=None, sectors=None, tickers=None):
        return load_news(
            regions=regions or ["US", "CA", "INTL", "GEO"],
            window=max(1, window_days),
            query=" ".join(tickers or []) if tickers else "",
            limit=50
        )
    load_news = _load_news_wrapper
if err: log_debug(f"Failed to import ingestion.finnews.run_pipeline: {err}")

# Tech/Funda/Macro features
compute_technical_features, err = safe_import("analytics.phase2_technical", "compute_technical_features")
if err: log_debug(f"Failed to import analytics.phase2_technical.compute_technical_features: {err}")

load_fundamentals, err = safe_import("analytics.phase1_fundamental", "load_fundamentals")
if err: log_debug(f"Failed to import analytics.phase1_fundamental.load_fundamentals: {err}")

get_macro_features, err = safe_import("analytics.phase3_macro", "get_macro_features")
if err: log_debug(f"Failed to import analytics.phase3_macro.get_macro_features: {err}")

# ===== NLP enrich (plusieurs noms possibles) =====
def _resolve_ask_model():
    # 1) research.nlp_enrich.ask_model
    fn, err = safe_import("research.nlp_enrich", "ask_model")
    if not err and fn: return fn
    log_debug(f"Failed to import research.nlp_enrich.ask_model: {err}")
    # 2) research.nlp_enrich.query_model
    fn, err = safe_import("research.nlp_enrich", "query_model")
    if not err and fn: return fn
    log_debug(f"Failed to import research.nlp_enrich.query_model: {err}")
    # 3) analytics.nlp_enrich.ask_model (au cas où)
    fn, err = safe_import("analytics.nlp_enrich", "ask_model")
    if not err and fn: return fn
    log_debug(f"Failed to import analytics.nlp_enrich.ask_model: {err}")
    return None

ask_model = _resolve_ask_model()

# ===== Arbitre (réutilise econ_llm_agent) =====
def _resolve_arbitre():
    """
    On essaie dans l’ordre :
      - fonction module-level 'arbitre' (rare)
      - fonction 'arbitrage' (rare)
      - classe EconomicAnalyst().analyze(context)  ← le plus probable
    """
    # 1) fonctions directes
    fn, err = safe_import("analytics.econ_llm_agent", "arbitre")
    if not err and fn:
        return lambda ctx: fn(ctx)
    log_debug(f"Failed to import analytics.econ_llm_agent.arbitre: {err}")

    fn, err = safe_import("analytics.econ_llm_agent", "arbitrage")
    if not err and fn:
        return lambda ctx: fn(ctx)
    log_debug(f"Failed to import analytics.econ_llm_agent.arbitrage: {err}")

    # 2) classe + méthode
    Cls, err = safe_import("analytics.econ_llm_agent", "EconomicAnalyst")
    InputCls, err_input = safe_import("analytics.econ_llm_agent", "EconomicInput")
    if not err and Cls and not err_input and InputCls:
        def _call(ctx: dict):
            try:
                # lazy import pour éviter erreurs au import-time (ex. g4f)
                analyst = Cls()
                if hasattr(analyst, "analyze"):
                    # Créer un objet EconomicInput avec les bons attributs
                    enriched_ctx = ctx.copy()
                    question = enriched_ctx.get("question", f"Analyse {ctx.get('scope', 'macro')}")
                    features = enriched_ctx.get("macro_features") or enriched_ctx.get("tech_features") or enriched_ctx.get("fundamentals")
                    news = enriched_ctx.get("news")

                    # Créer l'objet EconomicInput
                    input_obj = InputCls(
                        question=question,
                        features=features,
                        news=news,
                        attachments=enriched_ctx.get("attachments"),
                        locale=enriched_ctx.get("locale", "fr"),
                        meta=enriched_ctx
                    )
                    return analyst.analyze(input_obj)
                # fallback : cherche une méthode "arbitre"/"arbitrage"/"judge"/"aggregate"/"decide"
                for cand in ("arbitre", "arbitrage", "judge", "aggregate", "decide"):
                    if hasattr(analyst, cand):
                        return getattr(analyst, cand)(ctx)
                raise RuntimeError("Aucune méthode d'arbitrage trouvée sur EconomicAnalyst")
            except Exception as e:
                traceback.print_exc()
                return {"error": f"arbitre failed: {e.__class__.__name__}: {e}"}
        return _call
    log_debug(f"Failed to import analytics.econ_llm_agent.EconomicAnalyst: {err}")
    return None

arbitre = _resolve_arbitre()

# ===== UI =====
tabs = st.tabs(["💰 Économie", "📊 Action", "📰 Actu"])

# ---- Tab 1: Macro ----
with tabs[0]:
    if render_macro:
        try:
            render_macro()
        except Exception as e:
            st.error(f"Erreur render_macro: {e}")
            st.code(traceback.format_exc())
    else:
        st.warning("Module macro_sector_app indisponible")

    with st.expander("🤖 Analyse IA (NLP_enrich)"):
        if not ask_model:
            st.info("NLP_enrich indisponible (ask_model non trouvé).")
        else:
            q = st.text_input("Question au modèle", placeholder="Que prévois-tu sur l'inflation à 6 mois ?")
            context = {}
            if get_macro_features:
                try:
                    mf = get_macro_features()
                    # jsonifiable si nécessaire
                    context["macro_features"] = mf.to_dict() if hasattr(mf, "to_dict") else mf
                except Exception as e:
                    st.warning(f"get_macro_features() a échoué: {e}")
            if st.button("Poser la question (macro)"):
                try:
                    ans = ask_model(q, context=context)
                    st.write(ans)
                except Exception as e:
                    st.error(f"ask_model a échoué: {e}")
                    st.code(traceback.format_exc())

    with st.expander("⚖️ Arbitre (signaux macro)"):
        if not arbitre:
            st.info("Arbitre indisponible.")
        else:
            try:
                ctx = {"scope": "macro"}
                if get_macro_features:
                    ctx["macro_features"] = get_macro_features()
                decision = arbitre(ctx)
                st.json(decision)
            except Exception as e:
                st.error(f"arbitre() a échoué: {e}")
                st.code(traceback.format_exc())

# ---- Tab 2: Stock ----
with tabs[1]:
    default_ticker = st.session_state.get("ticker", "AAPL")
    if render_stock:
        try:
            render_stock(default_ticker=default_ticker)
        except Exception as e:
            st.error(f"Erreur render_stock: {e}")
            st.code(traceback.format_exc())
    else:
        st.warning("Module stock_analysis_app indisponible")

    with st.expander("🧩 Peers (comparables)"):
        ticker = st.text_input("Ticker (peers)", value=default_ticker, key="peers_ticker").upper()
        k = st.slider("Nombre de comparables", 3, 20, 8)
        if not find_peers:
            st.info("Peers finder indisponible.")
        else:
            try:
                peers = find_peers(ticker, k=k)
                if isinstance(peers, dict) and "peers" in peers:
                    peers = peers["peers"]
                st.write(peers if peers else "Aucun peer trouvé.")
            except Exception as e:
                st.error(f"find_peers a échoué: {e}")
                st.code(traceback.format_exc())

    with st.expander("🤖 Analyse IA (NLP_enrich)"):
        if not ask_model:
            st.info("NLP_enrich indisponible.")
        else:
            q2 = st.text_input("Question au modèle", placeholder="Le momentum de MSFT est-il soutenable 3 mois ?", key="stock_q")
            ticker2 = st.text_input("Ticker contexte", value=default_ticker, key="stock_ctx_ticker").upper()
            ctx2 = {"scope": "stock", "ticker": ticker2}
            if compute_technical_features:
                try:
                    tf = compute_technical_features(ticker2, window=180)
                    ctx2["tech_features"] = tf.to_dict() if hasattr(tf, "to_dict") else tf
                except Exception as e:
                    st.warning(f"compute_technical_features a échoué: {e}")
            if load_fundamentals:
                try:
                    ctx2["fundamentals"] = load_fundamentals(ticker2)
                except Exception as e:
                    st.warning(f"load_fundamentals a échoué: {e}")
            if load_news:
                try:
                    ctx2["news"] = load_news(window_days=14, tickers=[ticker2])
                except Exception as e:
                    st.warning(f"load_news a échoué: {e}")
            if st.button("Poser la question (stock)"):
                try:
                    ans2 = ask_model(q2, context=ctx2)
                    st.write(ans2)
                except Exception as e:
                    st.error(f"ask_model a échoué: {e}")
                    st.code(traceback.format_exc())

    with st.expander("⚖️ Arbitre (signaux action)"):
        if not arbitre:
            st.info("Arbitre indisponible.")
        else:
            try:
                ctx3 = {"scope": "stock", "ticker": default_ticker}
                if compute_technical_features:
                    ctx3["tech_features"] = compute_technical_features(default_ticker, window=180)
                if load_fundamentals:
                    ctx3["fundamentals"] = load_fundamentals(default_ticker)
                decision2 = arbitre(ctx3)
                st.json(decision2)
            except Exception as e:
                st.error(f"arbitre() a échoué: {e}")
                st.code(traceback.format_exc())

# ---- Tab 3: News ----
with tabs[2]:
    st.subheader("🗞️ Actu économique")
    if not load_news:
        st.info("Module news indisponible.")
    else:
        window = st.slider("Fenêtre (jours)", 3, 60, 14)
        regions = st.multiselect("Régions", ["US","EU","FR","WORLD"], default=["US","EU"])
        try:
            items = load_news(window_days=window, regions=regions)
            if not items:
                st.write("Aucune news.")
            else:
                for it in items[:50]:
                    title = f"{it.get('date','?')} — {it.get('title','(sans titre)')}"
                    with st.expander(title):
                        st.write(it.get("summary") or it.get("content") or "")
                        meta = {k: it.get(k) for k in ("ticker","region","source","sentiment")}
                        st.caption(str(meta))
        except Exception as e:
            st.error(f"load_news a échoué: {e}")
            st.code(traceback.format_exc())

st.caption("Hub d'analyse financière — Modules intégrés : Macro, Stock, NLP_enrich, Arbitre, Peers, News")

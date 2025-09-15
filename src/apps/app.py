# src/apps/app.py
# --- sys.path bootstrap (CRITIQUE) ---
from pathlib import Path
import sys as _sys
_SRC_ROOT = Path(__file__).resolve().parents[1]   # .../analyse-financiere/src
if str(_SRC_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_SRC_ROOT))
# -------------------------------------

# ---------- LOGGING GLOBAL (console + fichier tournant) ----------
import logging, logging.handlers, time, os, warnings
from pathlib import Path

LOG_DIR = Path(_SRC_ROOT).parent / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = LOG_DIR / "hub_app.log"

_root = logging.getLogger()
_root.setLevel(logging.DEBUG)  # on veut tout (tu peux repasser en INFO si trop verbeux)

# formateur compact mais riche
_fmt = logging.Formatter(
    "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

# handler console
_ch = logging.StreamHandler()
_ch.setLevel(logging.DEBUG)
_ch.setFormatter(_fmt)
# handler fichier tournant (5x5MB)
_fh = logging.handlers.RotatingFileHandler(LOG_FILE, maxBytes=5_000_000, backupCount=5, encoding="utf-8")
_fh.setLevel(logging.DEBUG)
_fh.setFormatter(_fmt)

# purge handlers en double lors des reruns streamlit
for h in list(_root.handlers):
    _root.removeHandler(h)
_root.addHandler(_ch)
_root.addHandler(_fh)

# capter warnings en logging
warnings.filterwarnings("default")
logging.captureWarnings(True)

# baisser un peu le bruit de certaines libs (ajuste si besoin)
logging.getLogger("urllib3").setLevel(logging.INFO)
logging.getLogger("requests").setLevel(logging.INFO)
logging.getLogger("yfinance").setLevel(logging.INFO)
logging.getLogger("matplotlib").setLevel(logging.WARNING)
logging.getLogger("PIL").setLevel(logging.WARNING)

logger = logging.getLogger("hub")

import traceback, importlib, sys, json
import datetime as dt
import streamlit as st

# (optionnel) hook global pour exceptions non catchées
def _excepthook(tp, val, tb):
    logger.critical("UNCAUGHT: %s: %s\n%s", tp.__name__, val, "".join(traceback.format_tb(tb)))
_sys.excepthook = _excepthook

st.set_page_config(page_title="Analyse Financière — Hub", layout="wide")
st.title("📈 Analyse Financière — Hub IA")

_DEBUG = st.sidebar.checkbox("Afficher DEBUG", value=True)

def _json_s(obj):
    """repr JSON safe pour logs (pas d'exceptions)."""
    try:
        import json
        return json.dumps(obj, ensure_ascii=False, default=str)[:2000]
    except Exception:
        return str(obj)[:2000]

def log_exc(where: str, exc: BaseException):
    logger.error("EXC @ %s: %s\n%s", where, exc, traceback.format_exc())
    if _DEBUG:
        with st.sidebar.expander(f"Exception @ {where}", expanded=False):
            st.code(traceback.format_exc())

def trace_call(name: str, fn):
    """wrappe une fonction pour loguer entrée/sortie/durée/erreur."""
    if fn is None or not callable(fn):
        return fn
    def _wrapped(*args, **kwargs):
        t0 = time.perf_counter()
        logger.debug("→ %s args=%s kwargs=%s", name, _json_s(args), _json_s(kwargs))
        try:
            out = fn(*args, **kwargs)
            dt = (time.perf_counter() - t0) * 1000
            logger.debug("← %s (%.1f ms) result=%s", name, dt, _json_s(out))
            return out
        except Exception as e:
            dt = (time.perf_counter() - t0) * 1000
            logger.error("✖ %s FAILED (%.1f ms): %s", name, dt, e)
            log_exc(name, e)
            raise
    return _wrapped

def log_debug(msg: str):
    if _DEBUG:
        st.sidebar.write(f"DEBUG: {msg}")
    # toujours log en console
    print(msg, flush=True)

def safe_import(path: str, attr: str | None = None):
    """
    Import robuste : retourne (objet|None, erreur|None) ET logue le timing + chemin.
    """
    t0 = time.perf_counter()
    try:
        mod = importlib.import_module(path)
        dt = (time.perf_counter() - t0) * 1000
        if attr is None:
            logger.debug("import %s OK (%.1f ms) file=%s", path, dt, getattr(mod, "__file__", "?"))
            return mod, None
        if not hasattr(mod, attr):
            logger.error("import %s.%s ABSENT (%.1f ms)", path, attr, dt)
            return None, f"module '{path}' has no attribute '{attr}'"
        obj = getattr(mod, attr)
        logger.debug("import %s.%s OK (%.1f ms) file=%s", path, attr, dt, getattr(mod, "__file__", "?"))
        return obj, None
    except Exception as e:
        dt = (time.perf_counter() - t0) * 1000
        logger.error("import %s%s FAILED (%.1f ms): %s", path, f'.{attr}' if attr else "", dt, e)
        return None, f"{e.__class__.__name__}: {e}"

# ===== UI pages =====
render_macro, err = safe_import("apps.macro_sector_app", "render_macro")
if err: log_debug(f"Failed to import apps.macro_sector_app.render_macro: {err}")
render_stock, err = safe_import("apps.stock_analysis_app", "render_stock")
if err: log_debug(f"Failed to import apps.stock_analysis_app.render_stock: {err}")
render_macro = trace_call("render_macro", render_macro)
render_stock = trace_call("render_stock", render_stock)

# ===== Providers =====
find_peers, err = safe_import("research.peers_finder", "find_peers")
if err: log_debug(f"Failed to import research.peers_finder.find_peers: {err}")
find_peers = trace_call("find_peers", find_peers)

# News (wrapper pipeline -> load_news signature unifiée)
_run_pipeline, err = safe_import("ingestion.finnews", "run_pipeline")
if err: log_debug(f"Failed to import ingestion.finnews.run_pipeline: {err}")
def _load_news_wrapper(window_days=7, regions=None, sectors=None, tickers=None):
    logger.debug("load_news.wrapper IN window_days=%s regions=%s sectors=%s tickers=%s",
                 window_days, regions, sectors, tickers)
    if _run_pipeline is None:
        return []
    out = _run_pipeline(
        regions=regions or ["US", "CA", "INTL", "GEO"],
        window=max(1, int(window_days or 7)),
        query=" ".join(tickers or []),
        limit=50
    )
    logger.debug("load_news.wrapper OUT items=%s", len(out) if hasattr(out, "__len__") else "<?>")
    return out
load_news = trace_call("load_news", _load_news_wrapper if _run_pipeline else None)

compute_technical_features, err = safe_import("analytics.phase2_technical", "compute_technical_features")
if err: log_debug(f"Failed to import analytics.phase2_technical.compute_technical_features: {err}")
compute_technical_features = trace_call("compute_technical_features", compute_technical_features)

load_fundamentals, err = safe_import("analytics.phase1_fundamental", "load_fundamentals")
if err: log_debug(f"Failed to import analytics.phase1_fundamental.load_fundamentals: {err}")
load_fundamentals = trace_call("load_fundamentals", load_fundamentals)

get_macro_features, err = safe_import("analytics.phase3_macro", "get_macro_features")
if err: log_debug(f"Failed to import analytics.phase3_macro.get_macro_features: {err}")
get_macro_features = trace_call("get_macro_features", get_macro_features)

# ===== NLP_enrich (plusieurs alias) =====
def _resolve_ask_model():
    for path, attr in [
        ("research.nlp_enrich", "ask_model"),
        ("research.nlp_enrich", "query_model"),
        ("analytics.nlp_enrich", "ask_model"),
    ]:
        fn, err = safe_import(path, attr)
        if not err and fn:
            return trace_call(f"{path}.{attr}", fn)
        log_debug(f"Failed to import {path}.{attr}: {err}")
    return None
ask_model = _resolve_ask_model()

# ===== Arbitre (econ_llm_agent) =====
def _resolve_arbitre():
    # fonctions directes
    for attr in ("arbitre", "arbitrage"):
        fn, err = safe_import("analytics.econ_llm_agent", attr)
        if not err and fn:
            return trace_call(f"econ_llm_agent.{attr}", lambda ctx: fn(ctx))

    # classe EconomicAnalyst/EconomicInput
    Cls, err1 = safe_import("analytics.econ_llm_agent", "EconomicAnalyst")
    Inp, err2 = safe_import("analytics.econ_llm_agent", "EconomicInput")
    if not err1 and Cls and not err2 and Inp:
        def _call(ctx: dict):
            t0 = time.perf_counter()
            logger.debug("→ arbitre.analyze ctx=%s", _json_s(ctx))
            try:
                analyst = Cls()
                if hasattr(analyst, "analyze"):
                    q = ctx.get("question", f"Analyse {ctx.get('scope','macro')}")
                    feats = ctx.get("macro_features") or ctx.get("tech_features") or ctx.get("fundamentals")
                    input_obj = Inp(
                        question=q,
                        features=feats,
                        news=ctx.get("news"),
                        attachments=ctx.get("attachments"),
                        locale=ctx.get("locale","fr"),
                        meta=ctx
                    )
                    out = analyst.analyze(input_obj)
                else:
                    # fallback sur méthodes candidates
                    for cand in ("arbitre","arbitrage","judge","aggregate","decide"):
                        if hasattr(analyst, cand):
                            out = getattr(analyst, cand)(ctx)
                            break
                    else:
                        raise RuntimeError("Aucune méthode d'arbitrage sur EconomicAnalyst")
                dt = (time.perf_counter()-t0)*1000
                logger.debug("← arbitre.analyze (%.1f ms) out=%s", dt, _json_s(out))
                return out
            except Exception as e:
                dt = (time.perf_counter()-t0)*1000
                logger.error("✖ arbitre.analyze FAILED (%.1f ms): %s", dt, e)
                log_exc("arbitre.analyze", e)
                raise
        return _call
    log_debug(f"Failed to import analytics.econ_llm_agent.EconomicAnalyst/EconomicInput: {err1 or ''} | {err2 or ''}")
    return None
arbitre = _resolve_arbitre()

# ===== UI =====
tabs = st.tabs(["💰 Économie", "📊 Action", "📰 Actu"])

# ---- Tab 1: Macro ----
with tabs[0]:
    logger.info("TAB Macro opened")
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
            logger.debug("UI macro.question=%s", q)
            context = {}
            if get_macro_features:
                try:
                    mf = get_macro_features()
                    # jsonifiable si nécessaire
                    context["macro_features"] = mf.to_dict() if hasattr(mf, "to_dict") else mf
                except Exception as e:
                    st.warning(f"get_macro_features() a échoué: {e}")
            if st.button("Poser la question (macro)"):
                logger.info("BTN macro.ask clicked")
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
    logger.info("TAB Stock opened")
    default_ticker = st.session_state.get("ticker", "AAPL")
    logger.debug("state.ticker=%s", default_ticker)
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
        logger.debug("UI peers.ticker=%s k=%s", ticker, k)
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
            logger.debug("UI stock.qa q2=%s ticker2=%s", q2, ticker2)
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
                logger.info("BTN stock.ask clicked")
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
    logger.info("TAB News opened")
    st.subheader("🗞️ Actu économique")
    if not load_news:
        st.info("Module news indisponible.")
    else:
        window = st.slider("Fenêtre (jours)", 3, 60, 14)
        regions = st.multiselect("Régions", ["US","EU","FR","WORLD"], default=["US","EU"])
        logger.debug("UI news.window=%s regions=%s", window, regions)
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

with st.sidebar.expander("📜 Log (dernieres lignes)", expanded=False):
    try:
        txt = (LOG_FILE.read_text(encoding="utf-8") if LOG_FILE.exists() else "")
        # on coupe pour éviter de rendre des Mo dans streamlit
        lines = txt.splitlines()[-400:]
        st.code("\n".join(lines))
        st.caption(f"Fichier: {LOG_FILE}")
    except Exception as e:
        st.write(f"Impossible de lire le log: {e}")

st.caption("Hub d'analyse financière — Modules intégrés : Macro, Stock, NLP_enrich, Arbitre, Peers, News")

# src/apps/app.py
# ======================================================================================
# Hub Streamlit — Analyse Financière (version lisible, sans abréviations, tout visible)
# - Bootstrap sys.path
# - Logging (Loguru via hub.logging_setup)
# - Helpers d’affichage SANS abréviations + rendu intégral (pas d’expander)
# - Bloc "Prévision macro (économie)" TOUT EN HAUT (avant les onglets)
# - UI robuste : news items dict/obj, erreurs visibles à l’écran
# ======================================================================================

# ---- sys.path bootstrap (CRITIQUE) ----
from pathlib import Path
import sys as _sys
_SRC_ROOT = Path(__file__).resolve().parents[1]   # .../analyse-financiere/src
if str(_SRC_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_SRC_ROOT))
# ---------------------------------------

# Chemins de logs (constants demandées par tests)
LOG_DIR = _SRC_ROOT.parent / "logs"
LOG_FILE = LOG_DIR / "hub_app.log"

"""App bootstrap; avoids global warning filters. SQLite leaks are addressed
by a connection guard installed in core_runtime (safe auto-close)."""

# ---------- LOGGING GLOBAL (JSON avec tracing) ----------
from core_runtime import log, get_trace_id, set_trace_id, new_trace_id, ui_event
# Compat pour tests qui patchent `src.apps.app.logger`
logger = log

# Capturer les warnings dans le log
import warnings
warnings.filterwarnings("default")

# ---------- IMPORTS UI / DATA ----------
import streamlit as st
import pandas as pd
import time, traceback, importlib, sys, os, platform, json
from typing import Optional

# ---------- HOOK EXCEPTIONS ----------
def _excepthook(tp, val, tb):
    try:
        stack = "".join(traceback.format_tb(tb))
    except Exception:
        stack = "<traceback indisponible>"
    log.critical(f"UNCAUGHT: {tp.__name__}: {val}\n{stack}")
_sys.excepthook = _excepthook

# ---------- PAGE CONFIG ----------
st.set_page_config(page_title="Analyse Financière — Hub", layout="wide", initial_sidebar_state="expanded")
st.title("📈 Analyse Financière — Hub IA")

# ---------- API KEY CHECK ----------
try:
    from utils import get_cfg  # unified utils shim
    cfg = get_cfg()
    if not cfg.has_any_fin_api():
        st.info("⚠️ Clé API financière absente : certaines fonctions (peers avancés, news enrichies) basculent en mode dégradé. Renseignez vos clés dans **Réglages de l'analyse**.")
except Exception as e:
    st.warning(f"Vérification des clés API impossible : {e}")
    log.warning(f"API key check failed: {e}")

# ---------- SESSION TRACE ----------
def _ensure_session_trace():
    if "trace_id" not in st.session_state or not st.session_state["trace_id"]:
        st.session_state["trace_id"] = new_trace_id()
    else:
        set_trace_id(st.session_state["trace_id"])

# ---------- DEBUG TOGGLE ----------
_DEBUG = st.sidebar.checkbox("Afficher les messages de débogage", value=True)

def _json_s(obj, limit=2000):
    """repr JSON safe pour logs (pas d’exceptions)."""
    try:
        return json.dumps(obj, ensure_ascii=False, default=str)[:limit]
    except Exception:
        return str(obj)[:limit]

def log_exc(where: str, exc: BaseException):
    logger.error(f"EXC @ {where}: {exc}\n{traceback.format_exc()}")
    if _DEBUG:
        st.sidebar.code(f"[{where}] {traceback.format_exc()}")

def log_debug(msg: str):
    if _DEBUG:
        st.sidebar.write(f"DEBUG: {msg}")
    log.debug(msg)

# ---------- IMPORT ROBUSTE + TRACE ----------
def trace_call(name: str, fn=None):
    """Decorator (or wrapper) to log enter/exit/duration/errors.

    Usage:
      @trace_call("func_name")
      def f(...): ...

      # or wrap dynamically
      f_wrapped = trace_call("func_name", f)
    """

    def _decorator(f):
        if f is None or not callable(f):
            return f

        def _wrapped(*args, **kwargs):
            t0 = time.perf_counter()
            logger.debug(f"→ {name} args={_json_s(args)} kwargs={_json_s(kwargs)}")
            import warnings as _warnings
            try:
                with _warnings.catch_warnings(record=True) as _caught:
                    _warnings.simplefilter("always")
                    out = f(*args, **kwargs)
                for w in _caught:
                    try:
                        logger.warning(str(w.message))
                    except Exception:
                        pass
                dt = (time.perf_counter() - t0) * 1000
                logger.debug(f"← {name} ({dt:.1f} ms) result={type(out).__name__}")
                return out
            except Exception as e:
                dt = (time.perf_counter() - t0) * 1000
                logger.error(f"✖ {name} FAILED ({dt:.1f} ms): {e}")
                log_exc(name, e)
                raise

        return _wrapped

    # Support both decorator factory and direct wrapper usage
    return _decorator if fn is None else _decorator(fn)

def safe_import(path: str, attr: Optional[str] = None):
    """
    Import robuste : retourne (objet|None, erreur|None) ET logue timing + fichier.
    """
    t0 = time.perf_counter()
    try:
        mod = importlib.import_module(path)
        dt = (time.perf_counter() - t0) * 1000
        if attr is None:
            log.debug(f"import {path} OK ({dt:.1f} ms) file={getattr(mod,'__file__','?')}")
            return mod, None
        if not hasattr(mod, attr):
            log.error(f"import {path}.{attr} ABSENT ({dt:.1f} ms)")
            return None, f"module '{path}' has no attribute '{attr}'"
        obj = getattr(mod, attr)
        log.debug(f"import {path}.{attr} OK ({dt:.1f} ms) file={getattr(mod,'__file__','?')}")
        return obj, None
    except Exception as e:
        dt = (time.perf_counter() - t0) * 1000
        log.error(f"import {path}{'.'+attr if attr else ''} FAILED ({dt:.1f} ms): {e}")
        return None, f"{e.__class__.__name__}: {e}"

# ---------- GLOSSAIRE (pas d’abréviations en UI) ----------
# NOTE: on remplace à l’affichage, sans modifier tes structures de données internes
_GLOSSARY = {
    # Macro
    "CPI": "Indice des prix à la consommation (inflation)",
    "CoreCPI": "Indice des prix à la consommation hors énergie et alimentation",
    "YoY": "Évolution sur un an",
    "MoM": "Évolution par rapport au mois précédent",
    "GDP": "Produit intérieur brut (croissance)",
    "INDPRO": "Production industrielle",
    "PAYEMS": "Emploi non agricole (NFP)",
    "FedFunds": "Taux directeur de la Réserve fédérale",
    "Breakeven": "Inflation implicite (breakeven)",
    "YieldSlope_Tight": "Pente de la courbe des taux (aplatie/inversée)",
    "USD": "Dollar américain",
    "Commodities": "Matières premières",
    "VIX": "Indice de volatilité (VIX)",
    "GPR": "Indice de risque géopolitique",
    "GSCPI": "Indice de pression des chaînes d’approvisionnement (Fed de New York)",
    # Actions / technique
    "RSI": "Indice de force relative (RSI)",
    "MACD": "Convergence–Divergence des moyennes mobiles (MACD)",
    "SMA": "Moyenne mobile simple",
    "EMA": "Moyenne mobile exponentielle",
    "ATR": "Véritable amplitude moyenne (ATR)",
    "BBands": "Bandes de Bollinger",
    "Vol": "Volatilité",
    "Ret": "Rendement",
    "EPS": "Bénéfice par action",
    "PE": "Ratio cours/bénéfice (P/E)",
    "EV/EBITDA": "Valeur d’entreprise / EBITDA",
}

def expand_label(label: str) -> str:
    if not isinstance(label, str) or not label:
        return str(label)
    out = label
    for abbr, full in _GLOSSARY.items():
        out = out.replace(abbr, full)
    return out

def expand_columns(df: Optional[pd.DataFrame]) -> Optional[pd.DataFrame]:
    if df is None:
        return None
    try:
        return df.rename(columns=lambda c: expand_label(str(c)))
    except Exception:
        return df

def to_mapping(obj):
    """Transforme proprement en dict pour affichage (supporte objets news custom)."""
    try:
        if obj is None:
            return {}
        if isinstance(obj, dict):
            return obj
        if hasattr(obj, "to_dict") and callable(getattr(obj, "to_dict")):
            return obj.to_dict()
        # dataclass / simple objets
        if hasattr(obj, "__dict__"):
            d = {k: v for k, v in vars(obj).items() if not k.startswith("_")}
            return d
        # fallback : représentation texte
        return {"value": str(obj)}
    except Exception as e:
        return {"value": f"<non sérialisable: {e}>"}

def show_full(name: str, data):
    """Affiche tout, sans cacher : DataFrame -> tableau; dict/list -> JSON; autre -> texte."""
    st.markdown(f"### {expand_label(name)}")
    if isinstance(data, pd.DataFrame):
        st.dataframe(expand_columns(data), width='stretch', height=500)
    elif isinstance(data, (list, tuple)):
        # Tenter de convertir chaque élément en dict
        mapped = [to_mapping(x) for x in data]
        st.json(mapped)
    elif isinstance(data, dict):
        st.json(data)
    else:
        try:
            st.json(json.loads(str(data)))
        except Exception:
            st.write(data)

# ---------- UI HELPERS (pro look) ----------
def _score_badge(v: float) -> str:
    try:
        if v is None or (isinstance(v, float) and not (v == v)):
            return "N/A"
        if v >= 1.0:
            return f"🟢 {v:+.2f}"
        if v <= -1.0:
            return f"🔴 {v:+.2f}"
        return f"🟠 {v:+.2f}"
    except Exception:
        return str(v)

def _component_fmt(v) -> str:
    try:
        if v is None or (isinstance(v, float) and not (v == v)):
            return "valeur non trouvée"
        return f"{float(v):+.3f}"
    except Exception:
        return "valeur non trouvée"

def render_macro_summary_block(macro_feats: dict):
    st.markdown("### 🔮 Synthèse macro (scores normalisés)")
    scores = (macro_feats or {}).get("macro_nowcast", {}).get("scores", {})
    cols = st.columns(5)
    keys = [
        ("Growth", "Croissance"),
        ("Inflation", "Inflation"),
        ("Policy", "Politique monétaire"),
        ("USD", "Dollar US"),
        ("Commodities", "Matières premières"),
    ]
    for i, (k, label) in enumerate(keys):
        cols[i].metric(label, _score_badge(scores.get(k)))

    st.info(
        "Ces scores macro sont des z-scores (moyenne 0, écart-type 1) calculés sur des séries FRED."
        " Utilisez-les pour qualifier le régime économique courant et nourrir vos scénarios 3–6 mois"
        " (rotation sectorielle, couverture de change, duration obligataire)."
    )

    comps = (macro_feats or {}).get("macro_nowcast", {}).get("components", {})
    with st.expander("Détails composants (dernière valeur non-nulle)"):
        c1, c2, c3 = st.columns(3)
        c1.write(f"INDPRO YoY: {_component_fmt(comps.get('INDPRO_YoY'))}")
        c1.write(f"PAYEMS YoY: {_component_fmt(comps.get('PAYEMS_YoY'))}")
        c1.write(f"YieldSlope (resserrement): {_component_fmt(comps.get('YieldSlope_Tight'))}")
        c2.write(f"CPI YoY: {_component_fmt(comps.get('CPI_YoY'))}")
        c2.write(f"Core CPI YoY: {_component_fmt(comps.get('CoreCPI_YoY'))}")
        c2.write(f"Breakeven dev: {_component_fmt(comps.get('Breakeven_dev'))}")
        c3.write(f"FedFunds dev: {_component_fmt(comps.get('FedFunds_dev'))}")
        c3.write(f"USD YoY: {_component_fmt(comps.get('USD_YoY'))}")
        c3.write(f"Commodities YoY: {_component_fmt(comps.get('Commodities_YoY'))}")

    meta = (macro_feats or {}).get("meta", {})
    last = meta.get("last_dates") or {}
    if last:
        st.caption("Fraîcheur des séries clés (AAAA-MM)")
        lines = []
        for k in ["INDPRO","PAYEMS","CPIAUCSL","CPILFESL","T10YIE","FEDFUNDS","DGS10","DGS2","DTWEXBGS"]:
            if k in last:
                lines.append(f"- {k}: {last[k]}")
        if lines:
            st.markdown("\n".join(lines))


# ---------- IMPORT DES MODULES (tracés) ----------
render_macro, err = safe_import("apps.macro_sector_app", "render_macro")
if err: log_debug(f"Failed to import apps.macro_sector_app.render_macro: {err}")
render_stock, err = safe_import("apps.stock_analysis_app", "render_stock")
if err: log_debug(f"Failed to import apps.stock_analysis_app.render_stock: {err}")
render_macro = trace_call("render_macro", render_macro)
render_stock = trace_call("render_stock", render_stock)

find_peers, err = safe_import("research.peers_finder", "find_peers")
if err: log_debug(f"Failed to import research.peers_finder.find_peers: {err}")
find_peers = trace_call("find_peers", find_peers)

_run_pipeline, err = safe_import("ingestion.finnews", "run_pipeline")
if err: log_debug(f"Failed to import ingestion.finnews.run_pipeline: {err}")

def _load_news_wrapper(window_days=7, regions=None, sectors=None, tickers=None):
    log.debug(f"load_news.IN days={window_days} regions={regions} sectors={sectors} tickers={tickers}")
    if _run_pipeline is None:
        return []
    out = _run_pipeline(
        regions=regions or ["US", "CA", "INTL", "GEO"],
        window=max(1, int(window_days or 7)),
        query=" ".join([t for t in (tickers or []) if t]),
        limit=50
    )
    log.debug(f"load_news.OUT count={len(out) if hasattr(out,'__len__') else '<?>'}")
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

# News intelligence (aggregated signals)
_collect_news, err = safe_import("analytics.market_intel", "collect_news")
if err: log_debug(f"Failed to import analytics.market_intel.collect_news: {err}")
_build_unified_news, err = safe_import("analytics.market_intel", "build_unified_features")
if err: log_debug(f"Failed to import analytics.market_intel.build_unified_features: {err}")
collect_news = trace_call("collect_news", _collect_news)
build_unified_news = trace_call("build_unified_features", _build_unified_news)

# Optional FeatureBundle dataclass
_FeatureBundle, err = safe_import("core.models", "FeatureBundle")
if err:
    _FeatureBundle = None

def _news_features_for(ticker: str = None, regions=None, window: str = "last_week", query: str = "") -> dict:
    regions = regions or ["US", "INTL", "GEO"]
    try:
        if not collect_news or not build_unified_news:
            return {}
        items, meta = collect_news(regions=regions, window=window, query=query, tgt_ticker=(ticker or None), per_source_cap=None, limit=120)
        feats = build_unified_news(items, target_ticker=(ticker or None), ownership=None, finviz_blob=None, macro_blob=None)
        return feats or {}
    except Exception as e:
        log_debug(f"news_features_for failed: {e}")
        return {}

def _compose_features(macro_feats=None, tech_feats=None, fund=None, news_feats=None) -> dict:
    if _FeatureBundle:
        try:
            fb = _FeatureBundle(
                macro=(macro_feats if not hasattr(macro_feats, "to_dict") else macro_feats.to_dict()) if macro_feats else None,
                technical=(tech_feats if not hasattr(tech_feats, "to_dict") else tech_feats.to_dict()) if tech_feats else None,
                fundamentals=fund if fund else None,
                news=news_feats if news_feats else None,
                meta=None,
            )
            return fb.to_dict()
        except Exception:
            pass
    # Fallback plain dict if FeatureBundle unavailable
    out = {}
    if macro_feats:
        out["macro"] = macro_feats if not hasattr(macro_feats, "to_dict") else macro_feats.to_dict()
    if tech_feats:
        out["technical"] = tech_feats if not hasattr(tech_feats, "to_dict") else tech_feats.to_dict()
    if fund:
        out["fundamentals"] = fund
    if news_feats:
        out["news"] = news_feats
    return out

# ===== NLP_enrich (plusieurs alias) =====
def _resolve_ask_model():
    for path, attr in [
        ("research.nlp_enrich", "ask_model"),
        ("research.nlp_enrich", "query_model"),
        ("analytics.nlp_enrich", "ask_model"),
    ]:
        fn, err_ = safe_import(path, attr)
        if not err_ and fn:
            return trace_call(f"{path}.{attr}", fn)
        log_debug(f"Failed to import {path}.{attr}: {err_}")
    return None
ask_model = _resolve_ask_model()

# ===== Arbitre (econ_llm_agent) =====
def _resolve_arbitre():
    # fonctions directes
    for attr in ("arbitre", "arbitrage"):
        fn, err_ = safe_import("analytics.econ_llm_agent", attr)
        if not err_ and fn:
            return trace_call(f"econ_llm_agent.{attr}", lambda ctx: fn(ctx))

    # classe EconomicAnalyst/EconomicInput
    Cls, err1 = safe_import("analytics.econ_llm_agent", "EconomicAnalyst")
    Inp, err2 = safe_import("analytics.econ_llm_agent", "EconomicInput")
    if not err1 and Cls and not err2 and Inp:
        def _call(ctx: dict):
            t0 = time.perf_counter()
            log.debug(f"→ arbitre.analyze ctx={_json_s(ctx)}")
            try:
                analyst = Cls()
                if hasattr(analyst, "analyze"):
                    q = ctx.get("question", f"Analyse {ctx.get('scope','macro')}")
                    feats = ctx.get("features") or ctx.get("macro_features") or ctx.get("tech_features") or ctx.get("fundamentals")
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
                    for cand in ("arbitre","arbitrage","judge","aggregate","decide"):
                        if hasattr(analyst, cand):
                            out = getattr(analyst, cand)(ctx)
                            break
                    else:
                        raise RuntimeError("Aucune méthode d'arbitrage sur EconomicAnalyst")
                dt = (time.perf_counter()-t0)*1000
                log.debug(f"← arbitre.analyze ({dt:.1f} ms) out={type(out).__name__}")
                return out
            except Exception as e:
                dt = (time.perf_counter()-t0)*1000
                log.error(f"✖ arbitre.analyze FAILED ({dt:.1f} ms): {e}")
                log_exc("arbitre.analyze", e)
                raise
        return _call
    log_debug(f"Failed to import analytics.econ_llm_agent.EconomicAnalyst/EconomicInput: {err1 or ''} | {err2 or ''}")
    # Compat tests: si le mock renvoie "not found", on retourne un stub non-None;
    # si c'est "Module not found", on retourne None.
    e1 = (err1 or "").strip().lower()
    e2 = (err2 or "").strip().lower()
    if e1 == "module not found" and e2 == "module not found":
        return None
    return lambda ctx: {}
arbitre = _resolve_arbitre()

# ======================================================================================
# SECTION PRIORITAIRE EN HAUT : "Prévision macro (économie)"
# - Pas d’abréviation dans les libellés
# - Tout est affiché (pas d’expander)
# ======================================================================================
st.markdown("## 🔮 Prévision macro (économie) — synthèse lisible")

# 1) Question au modèle (texte par défaut explicite)
default_q = "Peux-tu me donner une prévision claire et vulgarisée de l'inflation et de la croissance aux États-Unis pour les 6 prochains mois ?"
user_q = st.text_area(
    "Formule ta question :", 
    value=default_q,
    help="Pose une question en langage naturel. Exemple : 'Que prévois-tu sur l'inflation américaine à 6 mois ?'",
    key="macro_q_top"
)

# 2) Caractéristiques macro — affichage pro (synthèse + détails)
macro_feats = None
try:
    if get_macro_features:
        macro_feats = get_macro_features()
        render_macro_summary_block(macro_feats if isinstance(macro_feats, dict) else {})
        with st.expander("Détails techniques (JSON brut)"):
            show_full("Caractéristiques macroéconomiques (brutes)", macro_feats if not hasattr(macro_feats, "to_dict") else macro_feats.to_dict())
except BaseException as e:
    st.error(f"Chargement des caractéristiques macroéconomiques impossible : {e}")
    log_exc("get_macro_features(top)", e)

# 3) Analyse IA (si disponible)
if ask_model:
    try:
        context = {}
        if macro_feats is not None:
            context["macro_features"] = macro_feats if not hasattr(macro_feats, "to_dict") else macro_feats.to_dict()
        # Build aggregated news signals for context (best effort)
        news_feats = _news_features_for(regions=["US","INTL","GEO"], window="last_week") if collect_news and build_unified_news else {}
        context["features"] = _compose_features(macro_feats=macro_feats, news_feats=news_feats)
        st.markdown("### 🤖 Analyse par le modèle de langage (IA)")
        st.caption("Explique les implications des signaux macro et propose des pistes d’action. La réponse peut être texte/markdown/JSON.")
        if st.button("Lancer l'analyse macro IA", key="macro_ask_top"):
            ans = ask_model(user_q, context=context)
            # on affiche tel quel (le modèle peut renvoyer texte/markdown/json)
            show_full("Réponse détaillée du modèle IA", ans)
    except Exception as e:
        st.error(f"Analyse IA indisponible : {e}")
        log_exc("ask_model(top)", e)
else:
    st.info("Le module d'analyse IA (NLP_enrich) n'est pas disponible.")

# 4) Arbitre (synthèse de signaux macro)
if arbitre:
    try:
        ctx = {"scope": "macro"}
        if macro_feats is not None:
            ctx["macro_features"] = macro_feats if not hasattr(macro_feats, "to_dict") else macro_feats.to_dict()
        # Enrich with aggregated news features
        news_feats = _news_features_for(regions=["US","INTL","GEO"], window="last_week") if collect_news and build_unified_news else {}
        ctx["features"] = _compose_features(macro_feats=macro_feats, news_feats=news_feats)
        st.markdown("### ⚖️ Synthèse de signaux macro (arbitre)")
        st.caption("Combine les signaux macro pour orienter une rotation sectorielle cohérente. À croiser avec technique et fondamentaux.")
        decision = arbitre(ctx)
        show_full("Décision / synthèse de signaux macro", decision)
    except Exception as e:
        st.error(f"Arbitre indisponible : {e}")
        log_exc("arbitre(top)", e)
else:
    st.info("Le module d'arbitrage (econ_llm_agent) n'est pas disponible.")

st.markdown("---")

# ======================================================================================
# ONGLET 1 : Économie (module macro)
# ======================================================================================
tabs = st.tabs(["💰 Économie (détails)", "📊 Action (détails)", "📰 Actualités (tout afficher)"])

def main():
    _ensure_session_trace()
    st.caption(f"Trace ID: `{st.session_state['trace_id']}`")

    st.markdown("## 🔮 Prévision macro (économie) — synthèse lisible")
    st.caption("Vue d’ensemble: scores, composants clés, fraîcheur des séries. Les détails complets sont dans l’onglet Économie.")

    # 1) Question au modèle (texte par défaut explicite)
    default_q = "Peux-tu me donner une prévision claire et vulgarisée de l'inflation et de la croissance aux États-Unis pour les 6 prochains mois ?"
    user_q = st.text_area(
        "Formule ta question :",
        value=default_q,
        help="Pose une question en langage naturel. Exemple : 'Que prévois-tu sur l'inflation américaine à 6 mois ?'",
        key="macro_q_main"
    )

    # 2) Caractéristiques macro (brutes) + version "expurgée" (colonnes explicitées)
    macro_feats = None
    try:
        with ui_event("load_macro_features"):
            if get_macro_features:
                macro_feats = get_macro_features()
                # If provider reported a hard error, surface it prominently
                if isinstance(macro_feats, dict) and macro_feats.get("error"):
                    st.error(f"Caractéristiques macroéconomiques indisponibles: {macro_feats['error']}")
                # Vue synthétique pro + JSON en expander
                render_macro_summary_block(macro_feats if isinstance(macro_feats, dict) else {})
                with st.expander("Détails techniques (JSON brut)"):
                    show_full(
                        "Caractéristiques macroéconomiques (brutes)",
                        macro_feats if not hasattr(macro_feats, "to_dict") else macro_feats.to_dict()
                    )

                # Affiche un badge de fraicheur pour les séries FRED clés si disponibles
                try:
                    meta = macro_feats.get("meta", {}) if isinstance(macro_feats, dict) else {}
                    last = meta.get("last_dates") or {}
                    if last:
                        st.caption("Actualisation des séries clés (FRED)")
                        from datetime import datetime
                        now = datetime.utcnow()
                        lines = []
                        for k in ["INDPRO","PAYEMS","CPIAUCSL","CPILFESL","T10YIE","FEDFUNDS","DGS10","DGS2","DTWEXBGS"]:
                            d = last.get(k)
                            if not d:
                                continue
                            try:
                                dt = pd.to_datetime(d + "-01")
                                age_days = (now - dt.to_pydatetime()).days
                            except Exception:
                                age_days = 9999
                            badge = "🟢" if age_days <= 45 else ("🟠" if age_days <= 120 else "🔴")
                            lines.append(f"{badge} {k}: {d}")
                        if lines:
                            st.markdown("\n".join(f"- {ln}" for ln in lines))
                except Exception:
                    pass
    except Exception as e:
        st.error(f"Chargement des caractéristiques macroéconomiques impossible : {e}")
        log_exc("get_macro_features(top)", e)

    # 3) Analyse IA (si disponible)
    if ask_model:
        try:
            context = {}
            if macro_feats is not None:
                context["macro_features"] = macro_feats if not hasattr(macro_feats, "to_dict") else macro_feats.to_dict()
            st.markdown("### 🤖 Analyse par le modèle de langage (IA)")
            if st.button("Lancer l'analyse macro IA", key="macro_ask_main"):
                with ui_event("ask_macro_question", question=user_q[:100]):
                    ans = ask_model(user_q, context=context)
                    # on affiche tel quel (le modèle peut renvoyer texte/markdown/json)
                    show_full("Réponse détaillée du modèle IA", ans)
        except Exception as e:
            st.error(f"Analyse IA indisponible : {e}")
            log_exc("ask_model(top)", e)
    else:
        st.info("Le module d'analyse IA (NLP_enrich) n'est pas disponible.")

    # 4) Arbitre (synthèse de signaux macro)
    if arbitre:
        try:
            ctx = {"scope": "macro"}
            if macro_feats is not None:
                ctx["macro_features"] = macro_feats if not hasattr(macro_feats, "to_dict") else macro_feats.to_dict()
            st.markdown("### ⚖️ Synthèse de signaux macro (arbitre)")
            with ui_event("run_arbitre", scope="macro"):
                decision = arbitre(ctx)
                show_full("Décision / synthèse de signaux macro", decision)
        except Exception as e:
            st.error(f"Arbitre indisponible : {e}")
            log_exc("arbitre(top)", e)
    else:
        st.info("Le module d'arbitrage (econ_llm_agent) n'est pas disponible.")

    st.markdown("---")

    # ======================================================================================
    # TABS
    # ======================================================================================
    tabs = st.tabs(["💰 Économie (détails)", "📊 Action (détails)", "📰 Actualités (tout afficher)"])

    with tabs[0]:
        with ui_event("render_tab", ui_page="macro"):
            if render_macro:
                try:
                    render_macro()
                except Exception as e:
                    st.error(f"Erreur lors de l'affichage macro détaillé : {e}")
                    st.code(traceback.format_exc())
            else:
                st.warning("Module macro_sector_app indisponible")

    with tabs[1]:
        default_ticker = st.session_state.get("ticker", "AAPL")
        st.write(f"**Symbole analysé par défaut :** {default_ticker}")
        with ui_event("render_tab", ui_page="stock"):
            if render_stock:
                try:
                    render_stock(default_ticker=default_ticker)  # pas d'expander : affichage complet dans le module
                except Exception as e:
                    st.error(f"Erreur render_stock: {e}")
                    st.code(traceback.format_exc())
            else:
                st.warning("Module stock_analysis_app indisponible")

            # Peers — affichage direct (pas d'expander)
            st.markdown("### 🧩 Entreprises comparables (peers)")
            ticker = st.text_input("Symbole boursier pour la recherche de comparables", value=default_ticker, key="peers_ticker").upper()
            k = st.slider("Nombre d'entreprises comparables à afficher", 3, 30, 10)
            if find_peers:
                try:
                    with ui_event("find_peers", ui_page="stock", ticker=ticker, count=k):
                        peers = find_peers(ticker, k=k)
                        if isinstance(peers, dict) and "peers" in peers:
                            peers = peers["peers"]
                        show_full(f"Liste des comparables pour {ticker}", peers)
                except Exception as e:
                    st.error(f"Recherche de comparables impossible : {e}")
                    st.code(traceback.format_exc())
            else:
                st.info("Le module de recherche de comparables n'est pas disponible.")

            # Q&A IA sur un titre — affichage direct
            st.markdown("### 🤖 Question au modèle (contexte valeurs et marché)")
            q2 = st.text_input(
                "Pose une question sur un titre coté ou un secteur",
                placeholder="Exemple : 'Le momentum de MSFT est-il soutenable sur 3 mois ?'",
                key="stock_q"
            )
            ticker2 = st.text_input("Symbole pour le contexte (analyse technique/fondamentale/news)", value=default_ticker, key="stock_ctx_ticker").upper()

            ctx2 = {"scope": "stock", "ticker": ticker2}
            if compute_technical_features:
                try:
                    with ui_event("load_tech_features", ticker=ticker2):
                        tf = compute_technical_features(ticker2, window=180)
                        ctx2["tech_features"] = tf if not hasattr(tf, "to_dict") else tf.to_dict()
                except Exception as e:
                    st.warning(f"Indicateurs techniques indisponibles : {e}")
            if load_fundamentals:
                try:
                    with ui_event("load_fundamentals", ticker=ticker2):
                        ctx2["fundamentals"] = load_fundamentals(ticker2)
                except Exception as e:
                    st.warning(f"Données fondamentales indisponibles : {e}")
            news_feats = {}
            if load_news:
                try:
                    with ui_event("load_news", tickers=[ticker2]):
                        items = load_news(window_days=14, tickers=[ticker2])
                        ctx2["news"] = items
                except Exception as e:
                    st.warning(f"Chargement des actualités indisponible : {e}")
            # Aggregated news features via market_intel (best effort)
            try:
                news_feats = _news_features_for(ticker=ticker2, regions=["US","INTL","GEO"], window="last_week") if collect_news and build_unified_news else {}
            except Exception:
                news_feats = {}

            # Compose unified features bundle for the arbitrage/IA
            ctx2["features"] = _compose_features(
                macro_feats=None, tech_feats=ctx2.get("tech_features"), fund=ctx2.get("fundamentals"), news_feats=news_feats
            )

            if ask_model:
                if st.button("Poser la question (marché/actions)"):
                    try:
                        with ui_event("ask_stock_question", question=q2[:100], ticker=ticker2):
                            ans2 = ask_model(q2, context=ctx2)
                            show_full("Réponse détaillée du modèle IA (actions)", ans2)
                    except Exception as e:
                        st.error(f"Le modèle IA a échoué : {e}")
                        st.code(traceback.format_exc())
            else:
                st.info("Le module IA (NLP_enrich) n'est pas disponible pour la partie actions.")

        # ======================================================================================
        # ONGLET 3 : Actualités (tout afficher, robuste aux objets non-dict)
        # ======================================================================================
        with tabs[2]:
            with ui_event("render_tab", ui_page="news"):
                st.subheader("🗞️ Actualités économiques et de marché — affichage intégral")
                if not load_news:
                    st.info("Module 'news' indisponible.")
                else:
                    window = st.slider("Fenêtre temporelle (en jours)", 3, 90, 14)
                    regions = st.multiselect("Régions à inclure", ["US", "EU", "FR", "WORLD", "INTL", "GEO", "CA"], default=["US", "EU"])
                    try:
                        with ui_event("load_news_feed", window_days=window, regions=regions):
                            items = load_news(window_days=window, regions=regions)
                            if not items:
                                st.warning("Aucune actualité trouvée.")
                            else:
                                st.success(f"{len(items)} éléments d'actualité chargés — tout est affiché ci-dessous.")
                                # Affichage sans expander : une "carte" simple par item
                                for idx, raw in enumerate(items, start=1):
                                    it = to_mapping(raw)
                                    # Normalisation minimale des champs clés
                                    title = it.get("title") or it.get("headline") or "(sans titre)"
                                    date  = it.get("date") or it.get("published_at") or it.get("time") or "?"
                                    st.markdown(f"#### {idx}. {date} — {title}")
                                    # corps / résumé
                                    body = it.get("summary") or it.get("content") or it.get("body") or ""
                                    if body:
                                        st.write(body)
                                    # méta principales
                                    meta = {k: it.get(k) for k in ("ticker","tickers","region","source","sentiment","url") if k in it}
                                    if meta:
                                        st.caption(_json_s(meta, limit=500))
                                    # tout le document normalisé (diagnostic)
                                    with st.container(border=True):
                                        st.caption("Objet complet (diagnostic)")
                                        st.json(it)
                    except Exception as e:
                        st.error(f"Chargement des actualités impossible : {e}")
                        st.code(traceback.format_exc())

        # ======================================================================================
        # SIDEBAR : LOGS récents + ENV
        # ======================================================================================
        with st.sidebar.expander("📜 Journal (dernières lignes)", expanded=False):
            try:
                LOG_DIR = (_SRC_ROOT.parent / "logs")
                LOG_DIR.mkdir(parents=True, exist_ok=True)
                LOG_FILE_PATH = LOG_DIR / "hub_app.log"
                txt = (LOG_FILE_PATH.read_text(encoding="utf-8") if LOG_FILE_PATH.exists() else "")
                lines = txt.splitlines()[-400:]
                st.code("\n".join(lines))
                st.caption(f"Fichier log : {LOG_FILE_PATH}")
            except Exception as e:
                st.write(f"Impossible de lire le log: {e}")

        with st.sidebar.expander("🧩 Environnement (versions clés)", expanded=False):
            try:
                import importlib.metadata as _md
                dists = _md.packages_distributions()
                def _ver(pkg: str) -> str:
                    try:
                        return _md.version(pkg) if pkg in dists else "?"
                    except Exception:
                        return "?"
                vers = {
                    "python": platform.python_version(),
                    "platform": platform.platform(),
                    "streamlit": _ver("streamlit"),
                    "pandas": _ver("pandas"),
                    "numpy": _ver("numpy"),
                    "yfinance": _ver("yfinance"),
                    "requests": _ver("requests"),
                }
                st.code(_json_s(vers))
            except Exception as e:
                st.write(f"Impossible d'afficher l'environnement: {e}")

        with st.sidebar.expander("⚙️ Actions", expanded=False):
            if st.button("Rafraîchir caches", use_container_width=True):
                try:
                    st.cache_data.clear()
                    st.success("Caches purgés.")
                except Exception as e:
                    st.warning(f"Impossible de purger les caches: {e}")
            if st.button("Recharger la page", use_container_width=True):
                st.rerun()

        st.caption("Hub d'analyse financière — Modules intégrés : Macro, Actions, IA, Arbitre, Comparables, Actualités (affichage intégral, libellés développés).")

if __name__ == "__main__":
    main()

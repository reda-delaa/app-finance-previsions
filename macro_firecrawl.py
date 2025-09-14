# macro_firecrawl.py
# Collecte macro/market/news via Firecrawl SDK uniquement (clé intégrée).
# Requis: pip install firecrawl-py

import os
import time
import json
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple

# Streamlit est optionnel
try:
    import streamlit as st
    HAS_ST = True
except Exception:
    HAS_ST = False

import numpy as np
import pandas as pd

# =========================
# Firecrawl – SDK only
# =========================
# Resolve Firecrawl API key: prefer environment variable, then local secrets file
try:
    from src.secrets_local import get_key  # type: ignore
    FIRECRAWL_API_KEY: str = get_key("FIRECRAWL_API_KEY") or ""
except Exception:
    FIRECRAWL_API_KEY = os.getenv("FIRECRAWL_API_KEY") or ""

_FC_AVAILABLE = False
_app = None
try:
    from firecrawl import FirecrawlApp  # type: ignore
    if FIRECRAWL_API_KEY:
        _app = FirecrawlApp(api_key=FIRECRAWL_API_KEY)
        _FC_AVAILABLE = True
    else:
        _app = None
        _FC_AVAILABLE = False
except Exception as _init_err:
    _app = None
    _FC_AVAILABLE = False

def _log(msg: str, level: str = "info"):
    if HAS_ST:
        if "macro_fc_logs" not in st.session_state:
            st.session_state.macro_fc_logs = []
        ts = datetime.utcnow().strftime("%H:%M:%S")
        st.session_state.macro_fc_logs.append(f"[{ts}] {level.upper()}: {msg}")
    else:
        print(f"[{level}] {msg}")

def _coerce_number(x: Any) -> Optional[float]:
    if x is None:
        return None
    if isinstance(x, (int, float, np.integer, np.floating)):
        try:
            return float(x)
        except Exception:
            return None
    if isinstance(x, str):
        s = x.strip()
        if not s:
            return None
        s = s.replace(" ", "").replace("\u00A0", "")
        s = s.replace(",", ".")
        for ch in ["%", "$", "€", "£", "bp"]:
            s = s.replace(ch, "")
        try:
            return float(s)
        except Exception:
            return None
    return None

def _json_ready(obj: Any) -> Any:
    if isinstance(obj, (np.integer, np.floating)):
        return float(obj)
    if isinstance(obj, (np.ndarray,)):
        return obj.tolist()
    if isinstance(obj, (pd.Timestamp, datetime)):
        return obj.isoformat()
    if isinstance(obj, (set, tuple)):
        return list(obj)
    if isinstance(obj, dict):
        return {k: _json_ready(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_json_ready(v) for v in obj]
    # Try to coerce unknown/SDK objects (like ExtractResponse) to dicts
    try:
        # If object provides conversion helpers, prefer them
        if hasattr(obj, "to_dict") and callable(getattr(obj, "to_dict")):
            return _json_ready(obj.to_dict())
        if hasattr(obj, "dict") and callable(getattr(obj, "dict")):
            return _json_ready(obj.dict())
        # Fallback: use vars() to read __dict__ for simple objects
        if hasattr(obj, "__dict__"):
            return {k: _json_ready(v) for k, v in vars(obj).items() if not k.startswith("_")}
    except Exception:
        pass
    # As last resort, stringify
    try:
        return str(obj)
    except Exception:
        return None

def _require_fc() -> None:
    if not _FC_AVAILABLE or _app is None:
        msg = ("Firecrawl SDK indisponible ou clé API manquante. Installez le SDK `pip install firecrawl-py` "
               "et définissez la variable d'environnement FIRECRAWL_API_KEY ou ajoutez-la dans src/secrets_local.py pour le dev.")
        _log(msg, "error")
        raise RuntimeError(msg)

# ===================================================
# Abstraction Firecrawl (SDK)
# ===================================================
def fc_extract(urls: List[str], prompt: str, schema: Dict, retries: int = 2, timeout_s: int = 40) -> Dict[str, Any]:
    """Extrait des structures JSON à partir de pages web via Firecrawl SDK."""
    _require_fc()
    t0 = time.time()
    urls = list(dict.fromkeys(urls))  # dé-dupe
    last_err: Optional[str] = None
    for attempt in range(retries + 1):
        try:
            payload = _app.extract(urls=urls, prompt=prompt, schema=schema, timeout=timeout_s)  # type: ignore
            _log(f"fc_extract urls={len(urls)} in {(time.time()-t0)*1000:.0f}ms", "info")
            # Convert SDK response objects to plain Python structures
            return _json_ready(payload or {})
        except Exception as e:
            last_err = str(e)
            time.sleep(0.75 * (attempt + 1))
    _log(f"fc_extract failed after retries: {last_err}", "warning")
    return {}

def fc_search(query: str, limit: int = 10, retries: int = 1) -> Dict[str, Any]:
    """Recherche web via Firecrawl SDK."""
    _require_fc()
    last_err = None
    for attempt in range(retries + 1):
        try:
            res = _app.search(query=query, limit=limit)  # type: ignore
            return _json_ready(res or {"web": []})
        except Exception as e:
            last_err = str(e)
            time.sleep(0.5 * (attempt + 1))
    _log(f"fc_search failed: {last_err}", "warning")
    return {"web": []}

# =======================
# Dataclass résultat
# =======================
@dataclass
class MacroData:
    economic_indicators: Dict[str, Any]
    market_data: Dict[str, Any]
    news_impact: Dict[str, Any]
    geopolitical_risks: Dict[str, Any]
    commodity_prices: Dict[str, Any]
    central_bank_activities: Dict[str, Any]
    economic_calendar: Dict[str, Any]
    leading_indicators: Dict[str, Any]
    sentiment_indicators: Dict[str, Any]
    credit_conditions: Dict[str, Any]
    supply_chain_metrics: Dict[str, Any]
    business_cycle_indicators: Dict[str, Any]
    forecast_scenarios: Dict[str, Any]
    risk_metrics: Dict[str, Any]
    monetary_conditions: Dict[str, Any]
    meta: Dict[str, Any]

# ======================================
# Blocs de collecte — sources robustes
# ======================================
def _filter_workable(urls: List[str]) -> List[str]:
    blocked = ("wsj.com", "bloomberg.com", "ft.com")  # éviter paywalls durs
    return [u for u in urls if not any(b in u for b in blocked)]

def get_economic_indicators() -> Dict[str, Any]:
    urls = _filter_workable([
        "https://www.bea.gov/news/releases",
        "https://www.bls.gov/cpi/",
        "https://www.bls.gov/news.release/empsit.nr0.htm",
        "https://www.federalreserve.gov/releases/g17/current.htm"
    ])
    schema = {
        "type": "object",
        "properties": {
            "gdp": {"type": "object", "properties": {
                "latest_qoq_annualized": {"type": "number"},
                "latest_yoy": {"type": "number"},
                "previous": {"type": "number"},
                "consensus": {"type": "number"}
            }},
            "inflation": {"type": "object", "properties": {
                "cpi_yoy": {"type": "number"},
                "core_cpi_yoy": {"type": "number"},
                "pce_yoy": {"type": "number"}
            }},
            "employment": {"type": "object", "properties": {
                "unemployment_rate": {"type": "number"},
                "nonfarm_payrolls_k": {"type": "number"}
            }},
            "production": {"type": "object", "properties": {
                "industrial_production_yoy": {"type": "number"},
                "capacity_utilization": {"type": "number"}
            }}
        }
    }
    out = fc_extract(urls, prompt="Extract latest headline numbers where present.", schema=schema)
    for k in ("gdp", "inflation", "employment", "production"):
        if isinstance(out.get(k), dict):
            for kk, vv in list(out[k].items()):
                out[k][kk] = _coerce_number(vv)
    return out

def get_market_data() -> Dict[str, Any]:
    urls = _filter_workable([
        "https://www.marketwatch.com/markets",
        "https://www.reuters.com/markets",
        "https://finance.yahoo.com/world-indices"
    ])
    schema = {
        "type": "object",
        "properties": {
            "indices": {"type": "object", "properties": {
                "sp500_pct": {"type": "number"},
                "nasdaq_pct": {"type": "number"},
                "dow_pct": {"type": "number"},
            }},
            "forex": {"type": "object", "properties": {
                "eurusd": {"type": "number"},
                "usdjpy": {"type": "number"},
                "usdcad": {"type": "number"}
            }},
            "bonds": {"type": "object", "properties": {
                "us10y_yield": {"type": "number"},
                "us2y_yield": {"type": "number"}
            }},
            "vix": {"type": "number"}
        }
    }
    out = fc_extract(urls, prompt="Extract latest changes/levels.", schema=schema)
    for section in out.values():
        if isinstance(section, dict):
            for kk, vv in list(section.items()):
                section[kk] = _coerce_number(vv)
    if "vix" in out:
        out["vix"] = _coerce_number(out["vix"])
    return out

def get_news_impact() -> Dict[str, Any]:
    sr = fc_search("major economic news markets inflation monetary policy recession", limit=15)
    urls = [it.get("url") for it in sr.get("web", []) if it.get("url")][:12]
    schema = {
        "type": "object",
        "properties": {
            "headlines": {"type": "array", "items": {
                "type": "object",
                "properties": {
                    "title": {"type": "string"},
                    "sentiment": {"type": "string"},
                    "impact": {"type": "string"},
                    "assets": {"type": "array", "items": {"type": "string"}}
                }
            }},
            "themes": {"type": "array", "items": {"type": "string"}}
        }
    }
    return fc_extract(urls, prompt="Summarize in bullet points with sentiment and affected assets.", schema=schema)

def get_geopolitical_risks() -> Dict[str, Any]:
    urls = _filter_workable([
        "https://www.cfr.org/global-conflict-tracker",
        "https://www.crisisgroup.org/crisiswatch",
        "https://www.worldbank.org/en/topic/fragilityconflictviolence"
    ])
    schema = {
        "type": "object",
        "properties": {
            "conflicts": {"type": "array", "items": {
                "type": "object",
                "properties": {"region": {"type": "string"},
                               "severity": {"type": "string"},
                               "economic_impact": {"type": "string"}}
            }},
            "trade_tensions": {"type": "array", "items": {"type": "string"}}
        }
    }
    return fc_extract(urls, prompt="Extract list of conflicts & tensions with qualitative severity.", schema=schema)

def get_commodity_prices() -> Dict[str, Any]:
    urls = _filter_workable([
        "https://www.investing.com/commodities/",
        "https://www.reuters.com/markets/commodities/"
    ])
    schema = {
        "type": "object",
        "properties": {
            "energy": {"type": "object", "properties": {
                "wti": {"type": "number"}, "brent": {"type": "number"}, "natgas": {"type": "number"}
            }},
            "metals": {"type": "object", "properties": {
                "gold": {"type": "number"}, "silver": {"type": "number"}, "copper": {"type": "number"}
            }}
        }
    }
    out = fc_extract(urls, prompt="Numbers in USD where applicable.", schema=schema)
    for sec in out.values():
        if isinstance(sec, dict):
            for k, v in list(sec.items()):
                sec[k] = _coerce_number(v)
    return out

def get_central_bank_activities() -> Dict[str, Any]:
    urls = _filter_workable([
        "https://www.federalreserve.gov/monetarypolicy.htm",
        "https://www.ecb.europa.eu/press/pressconf",
        "https://www.boj.or.jp/en/announcements/",
        "https://www.bankofcanada.ca/press/"
    ])
    schema = {
        "type": "object",
        "properties": {
            "fed": {"type": "object", "properties": {
                "policy_rate": {"type": "number"}, "stance": {"type": "string"}, "next_meeting": {"type": "string"}
            }},
            "ecb": {"type": "object", "properties": {
                "policy_rate": {"type": "number"}, "stance": {"type": "string"}, "next_meeting": {"type": "string"}
            }},
            "boj": {"type": "object", "properties": {
                "policy_rate": {"type": "number"}, "stance": {"type": "string"}, "next_meeting": {"type": "string"}
            }},
            "boc": {"type": "object", "properties": {
                "policy_rate": {"type": "number"}, "stance": {"type": "string"}, "next_meeting": {"type": "string"}
            }}
        }
    }
    out = fc_extract(urls, prompt="Extract most current policy details.", schema=schema)
    for bank in out.values():
        if isinstance(bank, dict):
            for k, v in list(bank.items()):
                if "rate" in k:
                    bank[k] = _coerce_number(v)
    return out

def get_economic_calendar() -> Dict[str, Any]:
    urls = _filter_workable([
        "https://www.forexfactory.com/calendar",
        "https://www.investing.com/economic-calendar/"
    ])
    schema = {
        "type": "object",
        "properties": {
            "events": {"type": "array", "items": {
                "type": "object",
                "properties": {
                    "date": {"type": "string"}, "time": {"type": "string"},
                    "country": {"type": "string"}, "event": {"type": "string"},
                    "impact": {"type": "string"}, "forecast": {"type": "string"}, "previous": {"type": "string"}
                }
            }}
        }
    }
    return fc_extract(urls, prompt="List next 10 business days of key events.", schema=schema)

def get_leading_indicators() -> Dict[str, Any]:
    urls = _filter_workable([
        "https://www.conference-board.org/topics/us-leading-indicators",
        "https://www.ismworld.org/supply-management-news-and-reports/reports/ism-report-on-business/",
        "https://www.philadelphiafed.org/surveys-and-data"
    ])
    schema = {
        "type": "object",
        "properties": {
            "composite_leading_index": {"type": "number"},
            "pmi_manufacturing": {"type": "number"},
            "regional_surveys": {"type": "array", "items": {"type": "string"}}
        }
    }
    out = fc_extract(urls, prompt="Extract headline numbers and trend sign (+/-).", schema=schema)
    for k in list(out.keys()):
        out[k] = _coerce_number(out[k]) if isinstance(out.get(k), (int, float, str)) else out[k]
    return out

def get_sentiment_indicators() -> Dict[str, Any]:
    urls = _filter_workable([
        "https://data.sca.isr.umich.edu/",
        "https://www.conference-board.org/topics/consumer-confidence"
    ])
    schema = {
        "type": "object",
        "properties": {
            "umich_sentiment": {"type": "number"},
            "conference_board_confidence": {"type": "number"}
        }
    }
    out = fc_extract(urls, prompt="Extract latest index levels.", schema=schema)
    for k in list(out.keys()):
        out[k] = _coerce_number(out[k])
    return out

def get_credit_conditions() -> Dict[str, Any]:
    urls = _filter_workable([
        "https://www.chicagofed.org/publications/nfci/index",
        "https://www.federalreserve.gov/data/sloos.htm"
    ])
    schema = {
        "type": "object",
        "properties": {
            "nfci": {"type": "number"},
            "sloos_tightening": {"type": "string"}
        }
    }
    out = fc_extract(urls, prompt="Latest NFCI level and whether SLOOS indicates tightening/easing.", schema=schema)
    out["nfci"] = _coerce_number(out.get("nfci"))
    return out

def get_supply_chain_metrics() -> Dict[str, Any]:
    urls = _filter_workable([
        "https://www.ismworld.org/supply-management-news-and-reports/reports/ism-report-on-business/",
        "https://www.nyfed.org/research/policy/gscpi"
    ])
    schema = {
        "type": "object",
        "properties": {
            "supplier_deliveries_index": {"type": "number"},
            "gscpi": {"type": "number"}
        }
    }
    out = fc_extract(urls, prompt="Extract latest supplier deliveries and GSCPI.", schema=schema)
    for k in list(out.keys()):
        out[k] = _coerce_number(out[k])
    return out

def get_business_cycle_indicators() -> Dict[str, Any]:
    urls = _filter_workable([
        "https://fred.stlouisfed.org/series/USREC",
        "https://www.nber.org/research/business-cycle-dating"
    ])
    schema = {
        "type": "object",
        "properties": {
            "recession_flag": {"type": "string"},
            "nber_note": {"type": "string"}
        }
    }
    return fc_extract(urls, prompt="Are we in recession? Any note from NBER dates.", schema=schema)

def get_forecast_scenarios() -> Dict[str, Any]:
    urls = _filter_workable([
        "https://www.imf.org/en/Publications/WEO",
        "https://www.worldbank.org/en/publication/global-economic-prospects"
    ])
    schema = {
        "type": "object",
        "properties": {
            "baseline": {"type": "object", "properties": {
                "gdp_growth": {"type": "number"}, "inflation": {"type": "number"}
            }},
            "downside": {"type": "object", "properties": {
                "gdp_growth": {"type": "number"}, "prob": {"type": "number"}
            }}
        }
    }
    out = fc_extract(urls, prompt="Extract baseline and downside scenario headline numbers.", schema=schema)
    for sect in out.values():
        if isinstance(sect, dict):
            for k, v in list(sect.items()):
                sect[k] = _coerce_number(v)
    return out

def get_risk_metrics() -> Dict[str, Any]:
    urls = _filter_workable([
        "https://www.cboe.com/tradable_products/vix/",
        "https://www.policyuncertainty.com/"
    ])
    schema = {
        "type": "object",
        "properties": {
            "vix_spot": {"type": "number"},
            "epu_us": {"type": "number"}
        }
    }
    out = fc_extract(urls, prompt="Extract latest VIX spot and US EPU index if available.", schema=schema)
    out["vix_spot"] = _coerce_number(out.get("vix_spot"))
    out["epu_us"] = _coerce_number(out.get("epu_us"))
    return out

def get_monetary_conditions() -> Dict[str, Any]:
    urls = _filter_workable([
        "https://www.federalreserve.gov/monetarypolicy.htm",
        "https://www.ecb.europa.eu/mopo/html/index.en.html",
        "https://www.bis.org/statistics/index.htm"
    ])
    schema = {
        "type": "object",
        "properties": {
            "policy_rates": {"type": "object", "properties": {
                "fed_funds_target": {"type": "number"},
                "ecb_deposit_rate": {"type": "number"}
            }},
            "liquidity": {"type": "string"}
        }
    }
    out = fc_extract(urls, prompt="Policy rates & qualitative liquidity stance.", schema=schema)
    if "policy_rates" in out and isinstance(out["policy_rates"], dict):
        for k, v in list(out["policy_rates"].items()):
            out["policy_rates"][k] = _coerce_number(v)
    return out

# ==========================
# Agrégateur principal
# ==========================
def get_macro_data_firecrawl() -> MacroData:
    t0 = time.time()
    blocks: List[Tuple[str, Dict[str, Any]]] = []
    for name, fn in [
        ("economic_indicators", get_economic_indicators),
        ("market_data", get_market_data),
        ("news_impact", get_news_impact),
        ("geopolitical_risks", get_geopolitical_risks),
        ("commodity_prices", get_commodity_prices),
        ("central_bank_activities", get_central_bank_activities),
        ("economic_calendar", get_economic_calendar),
        ("leading_indicators", get_leading_indicators),
        ("sentiment_indicators", get_sentiment_indicators),
        ("credit_conditions", get_credit_conditions),
        ("supply_chain_metrics", get_supply_chain_metrics),
        ("business_cycle_indicators", get_business_cycle_indicators),
        ("forecast_scenarios", get_forecast_scenarios),
        ("risk_metrics", get_risk_metrics),
        ("monetary_conditions", get_monetary_conditions),
    ]:
        try:
            data = fn()
        except Exception as e:
            _log(f"{name} failed: {e}", "warning")
            data = {}
        blocks.append((name, data))

    meta = {
        "generated_at_utc": datetime.utcnow().isoformat(),
        "firecrawl_available": _FC_AVAILABLE,
        "sdk": True,
        "elapsed_ms": round((time.time() - t0) * 1000, 1),
    }
    payload = {k: v for k, v in blocks}
    return MacroData(meta=meta, **payload)

def save_macro_data(data: MacroData, filename: str = "macro_data.json") -> None:
    try:
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(_json_ready(asdict(data)), f, ensure_ascii=False, indent=2)
        _log(f"Saved {filename}", "info")
    except Exception as e:
        _log(f"Save failed: {e}", "error")
        if HAS_ST:
            st.error(f"Erreur lors de la sauvegarde des données: {str(e)}")

# ==========================
# Demo Streamlit (optionnel)
# ==========================
def _render_streamlit(macro: MacroData):
    st.title("Analyse Macroéconomique & Prévisions (Firecrawl)")
    st.caption(f"Généré: {macro.meta.get('generated_at_utc')} — Firecrawl: {macro.meta.get('firecrawl_available')}")
    tabs = st.tabs([
        "Indicateurs", "Marchés", "News Impact", "Géopolitique", "Commodities",
        "Banques centrales", "Calendrier", "Leading", "Sentiment",
        "Crédit", "Supply Chain", "Cycle", "Scénarios", "Risque", "Monétaire", "Logs"
    ])
    sections = [
        ("Indicateurs", macro.economic_indicators),
        ("Marchés", macro.market_data),
        ("News Impact", macro.news_impact),
        ("Géopolitique", macro.geopolitical_risks),
        ("Commodities", macro.commodity_prices),
        ("Banques centrales", macro.central_bank_activities),
        ("Calendrier", macro.economic_calendar),
        ("Leading", macro.leading_indicators),
        ("Sentiment", macro.sentiment_indicators),
        ("Crédit", macro.credit_conditions),
        ("Supply Chain", macro.supply_chain_metrics),
        ("Cycle", macro.business_cycle_indicators),
        ("Scénarios", macro.forecast_scenarios),
        ("Risque", macro.risk_metrics),
        ("Monétaire", macro.monetary_conditions),
    ]
    for (label, data), tab in zip(sections, tabs[:-1]):
        with tab:
            if isinstance(data, dict) and data:
                st.json(data)
            else:
                st.info("Aucune donnée disponible.")

    with tabs[-1]:
        logs = "\n".join(st.session_state.get("macro_fc_logs", []))
        st.code(logs or "No logs yet.", language="text")

if __name__ == "__main__":
    data = get_macro_data_firecrawl()
    save_macro_data(data, "macro_data.json")
    if HAS_ST:
        _render_streamlit(data)
    else:
        print(json.dumps(_json_ready(asdict(data))[:2000], indent=2))
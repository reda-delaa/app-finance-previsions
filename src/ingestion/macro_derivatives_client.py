# src/ingestion/macro_derivatives_client.py
# -*- coding: utf-8 -*-
"""
Macro & Derivatives client (FRED, TradingEconomics, CBOE, CFTC)
- Télécharge, met en cache, normalise et agrège des indicateurs macro et de dérivés.
- Conçu pour cohabiter avec finnews / finviz_client; schemas stables et JSONL-friendly.

Fonctions principales:
- fred_series(series_ids, start=None, end=None) -> dict[str, list[{date,value}]]
- tradingeconomics_calendar(countries=None, start=None, end=None, importance=None) -> list[...]
- cboe_indexes(which=['VIX','VVIX','SKEW']) -> dict[str, {value, change, ts}]
- cftc_cot(symbols=['SPX','NDX',...]) -> dict[symbol, {long, short, net, ts}]
- build_macro_snapshot(...) -> dict avec "macro", "risk", "surprises", "sources_used"

Notes:
- Clés API facultatives (FRED, TradingEconomics) via variables d'env:
  FRED_API_KEY, TE_API_KEY, TE_CLIENT, TE_SECRET
- Tous les appels utilisent un cache local (cache/macro/...) + retries, backoff.

Auteur: toi
"""
from __future__ import annotations

import os, re, io, csv, sys, json, math, time, enum, random, hashlib, datetime as dt
from dataclasses import dataclass, asdict, field
from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path

# --------- HTTP / Parsing
try:
    import requests
except Exception as e:
    raise RuntimeError("macro_derivatives_client requires `requests` (pip install requests)") from e

try:
    import pandas as pd
except Exception:
    pd = None

try:
    from tqdm import tqdm
except Exception:
    def tqdm(it, **kw): 
        return it

# =========================
# Config, cache & utilities
# =========================

CACHE_DIR = Path("cache") / "macro"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_TIMEOUT = float(os.getenv("MACRO_TIMEOUT", "20"))
RETRIES = int(os.getenv("MACRO_RETRIES", "2"))
BACKOFF = float(os.getenv("MACRO_BACKOFF", "1.5"))
UA_ROT = [
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/128 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:128.0) Gecko/20100101 Firefox/128.0",
]
def _ua(): return random.choice(UA_ROT)
def _now(): return dt.datetime.now(dt.timezone.utc)
def _iso(d: dt.datetime) -> str: return d.replace(tzinfo=dt.timezone.utc).isoformat().replace("+00:00","Z")

def _sha1(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8", errors="ignore")).hexdigest()

def _cache_path(url: str) -> Path:
    return CACHE_DIR / f"{_sha1(url)}.cache"

def _get(url: str, params: Optional[Dict[str, Any]] = None, use_cache=True, as_json=False, as_text=True) -> Any:
    params = params or {}
    # build cache key
    key = url
    if params:
        # freeze params order
        parts = "&".join(f"{k}={requests.utils.quote(str(params[k]))}" for k in sorted(params))
        key = f"{url}?{parts}"
    cpath = _cache_path(key)
    if use_cache and cpath.exists():
        try:
            raw = cpath.read_bytes()
            if as_json:
                return json.loads(raw.decode("utf-8", errors="ignore"))
            if as_text:
                return raw.decode("utf-8", errors="ignore")
            return raw
        except Exception:
            pass

    last_err = None
    wait = 0.25
    for att in range(1, RETRIES + 2):
        try:
            r = requests.get(url, params=params, headers={"User-Agent": _ua()}, timeout=DEFAULT_TIMEOUT)
            if r.status_code == 200:
                if as_json:
                    data = r.json()
                    try: cpath.write_text(json.dumps(data), encoding="utf-8")
                    except Exception: pass
                    return data
                elif as_text:
                    txt = r.text
                    try: cpath.write_text(txt, encoding="utf-8")
                    except Exception: pass
                    return txt
                else:
                    raw = r.content
                    try: cpath.write_bytes(raw)
                    except Exception: pass
                    return raw
            last_err = f"HTTP {r.status_code}"
        except Exception as e:
            last_err = f"{type(e).__name__}: {e}"
        time.sleep(wait); wait *= BACKOFF
    raise RuntimeError(f"GET failed {url} | last_err={last_err}")

def _to_float(x: Any) -> Optional[float]:
    if x is None: return None
    if isinstance(x, (int, float)): return float(x)
    s = str(x).replace(",","").strip()
    if s in ("","-","N/A","NaN","null"): return None
    try: return float(s)
    except Exception: return None

def _pct(a: Optional[float], b: Optional[float]) -> Optional[float]:
    if a is None or b is None or b == 0: return None
    return (a - b) / abs(b) * 100.0

def _diff(a: Optional[float], b: Optional[float]) -> Optional[float]:
    if a is None or b is None: return None
    return a - b

# ==============
# FRED ENDPOINTS
# ==============
FRED_API = "https://api.stlouisfed.org/fred"
try:
    from src.secrets_local import get_key  # type: ignore
    FRED_KEY = (get_key("FRED_API_KEY") or "").strip() or None
except Exception:
    FRED_KEY = os.getenv("FRED_API_KEY", "").strip() or None

def fred_series(series_ids: List[str], start: Optional[str] = None, end: Optional[str] = None,
                use_cache=True) -> Dict[str, List[Dict[str, Any]]]:
    """
    Retourne {series_id: [{date:'YYYY-MM-DD', value:float}, ...]}
    Si pas de clé API FRED, on bascule vers CSV public 'fredgraph' (moins fiable mais utile).
    """
    out: Dict[str, List[Dict[str, Any]]] = {}
    ids = [s.strip() for s in series_ids if s and s.strip()]
    if not ids:
        return out

    if FRED_KEY:
        for sid in ids:
            params = {
                "series_id": sid,
                "api_key": FRED_KEY,
                "file_type": "json",
            }
            if start: params["observation_start"] = start
            if end: params["observation_end"] = end
            try:
                data = _get(f"{FRED_API}/series/observations", params, use_cache=use_cache, as_json=True)
                obs = data.get("observations", [])
                out[sid] = [{"date": o.get("date"), "value": _to_float(o.get("value"))} for o in obs if o.get("date")]
            except Exception:
                out[sid] = []
        return out

    # fallback CSV (fredgraph)
    for sid in ids:
        params = {"id": sid}
        try:
            txt = _get("https://fred.stlouisfed.org/graph/fredgraph.csv", params, use_cache=use_cache, as_text=True)
            rows = []
            for i, row in enumerate(csv.reader(io.StringIO(txt))):
                if i == 0: 
                    continue
                if len(row) < 2: 
                    continue
                rows.append({"date": row[0], "value": _to_float(row[1])})
            out[sid] = rows
        except Exception:
            out[sid] = []
    return out

# Séries utiles par défaut (US)
DEFAULT_FRED = {
    "CPI": "CPIAUCSL",         # CPI All Urban Consumers
    "CPI_CORE": "CPILFESL",    # Core CPI
    "PCE": "PCEPI",            # PCE price index
    "UNEMP": "UNRATE",         # Unemployment rate
    "GDP": "GDPC1",            # Real GDP
    "ISM_MFG": "NAPM",         # ISM Manufacturing PMI (proxy older)
    "ISM_SERV": "NAPMNOI",     # ISM Non-mfg (proxy)
    "UST2Y": "DGS2",           # 2-Year Treasury
    "UST10Y": "DGS10",         # 10-Year Treasury
    "HY_OAS": "BAMLH0A0HYM2",  # HY option-adjusted spread
}

# ==================================
# TradingEconomics calendar endpoints
# ==================================
TE_KEY = (os.getenv("TE_API_KEY") or "").strip() or None
TE_CLIENT = os.getenv("TE_CLIENT", "").strip() or None
TE_SECRET = os.getenv("TE_SECRET", "").strip() or None
# Two auth modes: single API key (basic x-api-key) or client/secret (token)

def _te_headers():
    if TE_KEY:
        return {"accept": "application/json", "x-api-key": TE_KEY}
    return {"accept": "application/json"}

def tradingeconomics_calendar(countries: Optional[List[str]] = None,
                              start: Optional[str] = None,
                              end: Optional[str] = None,
                              importance: Optional[int] = None,
                              use_cache=True, limit: int = 1000) -> List[Dict[str, Any]]:
    """
    Calendar with fields: {country, category, actual, previous, forecast, date, importance, unit}
    Docs: https://developer.tradingeconomics.com
    """
    base = "https://api.tradingeconomics.com/calendar"
    params: Dict[str, Any] = {}
    if countries:
        params["country"] = ",".join(countries)
    if start: params["from"] = start
    if end: params["to"] = end
    if importance is not None:
        params["importance"] = int(importance)

    try:
        data = _get(base, params, use_cache=use_cache, as_json=True)
        rows = []
        for d in data[:limit]:
            rows.append({
                "country": d.get("Country"),
                "category": d.get("Category"),
                "event": d.get("Event"),
                "actual": _to_float(d.get("Actual")),
                "previous": _to_float(d.get("Previous")),
                "forecast": _to_float(d.get("Forecast")),
                "unit": d.get("Unit"),
                "date": d.get("Date"),
                "importance": d.get("Importance"),
                "source": "TradingEconomics",
            })
        return rows
    except Exception:
        return []

# ==============
# CBOE endpoints
# ==============
def cboe_indexes(which: List[str] = ["VIX","VVIX","SKEW"], use_cache=True) -> Dict[str, Dict[str, Any]]:
    """
    Récupère VIX, VVIX, SKEW depuis endpoints publics (CSV / JSON)
    - VIX CSV example: https://cdn.cboe.com/api/global/us_indices/daily_prices/VIX_History.csv
    - SKEW CSV:       https://cdn.cboe.com/api/global/us_indices/daily_prices/SKEW_History.csv
    """
    mapping = {
        "VIX":  "https://cdn.cboe.com/api/global/us_indices/daily_prices/VIX_History.csv",
        "VVIX": "https://cdn.cboe.com/api/global/us_indices/daily_prices/VVIX_History.csv",
        "SKEW": "https://cdn.cboe.com/api/global/us_indices/daily_prices/SKEW_History.csv",
    }
    out: Dict[str, Dict[str, Any]] = {}
    for key in which:
        url = mapping.get(key.upper())
        if not url:
            continue
        try:
            txt = _get(url, None, use_cache=use_cache, as_text=True)
            last = None
            prev = None
            for i, row in enumerate(csv.DictReader(io.StringIO(txt))):
                val = _to_float(row.get("CLOSE") or row.get("Close") or row.get("close"))
                date = row.get("DATE") or row.get("Date") or row.get("date")
                if val is None or not date: 
                    continue
                prev = last
                last = {"ts": date, "value": val}
            chg = _diff(last["value"] if last else None, prev["value"] if prev else None)
            out[key.upper()] = {"value": last["value"] if last else None,
                                "change": chg, "ts": last["ts"] if last else None}
        except Exception:
            out[key.upper()] = {"value": None, "change": None, "ts": None}
    return out

# ===========
# CFTC - COT
# ===========
"""
CFTC publie des CSV hebdo. Pour simplifier, nous utilisons le rapport 'legacy futures only' par marché.
Ports utiles: S&P500 (E-mini), Nasdaq, Crude, Gold, Copper, EUR, JPY, 10Y…
Le mapping ci-dessous pointe vers les 'Futures Only, Positions of Traders' codes (market code)
via un CSV agrégé officieux largement mirroré. En cas d'échec → retour vide.
"""
CFTC_SOURCES = {
    # Mirrors stables (si indisponible, le client renverra vide; tu peux swap pour endpoints officiels)
    "LEGACY_FUT_ONLY": "https://tradingcharts.com/resources/cot-data/futures_only.csv"
    # NB: si ce miroir devient indispo, remplacer par https://www.cftc.gov/MarketReports/CommitmentsofTraders/HistoricalCompressed/index.htm (zip)
}

# Heuristique mapping (marchés de référence → regex de nom pour retrouver la ligne)
COT_MARKET_REGEX = {
    "SPX": r"S&P 500.*Mini.*",
    "NDX": r"NASDAQ-100.*Mini.*",
    "WTI": r"Crude Oil.*New York Mercantile Exchange",
    "GOLD": r"Gold.*COMEX",
    "COPPER": r"Copper.*COMEX",
    "EUR": r"Euro FX.*",
    "JPY": r"Japanese Yen.*",
    "UST10Y": r"U\.S\. Treasury Bonds|10-Year",
}

def cftc_cot(symbols: List[str] = ["SPX","NDX","WTI","GOLD","COPPER","EUR","JPY"],
             use_cache=True) -> Dict[str, Dict[str, Any]]:
    """
    Retourne {symbol: {long, short, net, ts, market}}
    Données agrégées Non-commercials si disponibles.
    """
    url = CFTC_SOURCES["LEGACY_FUT_ONLY"]
    txt: Optional[str] = None
    try:
        txt = _get(url, None, use_cache=use_cache, as_text=True)
    except Exception:
        return {}

    out: Dict[str, Dict[str, Any]] = {}
    try:
        rdr = csv.DictReader(io.StringIO(txt))
        # colonnes typiques: Market_and_Exchange_Names, Report_Date_as_YYYY-MM-DD, NonComm_Long_All, NonComm_Short_All
        rows = list(rdr)
        for sym in symbols:
            pat = re.compile(COT_MARKET_REGEX.get(sym, sym), flags=re.I)
            # garder la dernière occurrence temporelle
            best = None
            for r in rows:
                name = r.get("Market_and_Exchange_Names") or ""
                if pat.search(name):
                    if not best or r.get("Report_Date_as_YYYY-MM-DD","") > best.get("Report_Date_as_YYYY-MM-DD",""):
                        best = r
            if best:
                long_ = _to_float(best.get("NonComm_Long_All"))
                short_ = _to_float(best.get("NonComm_Short_All"))
                net = _diff(long_, short_)
                ts = best.get("Report_Date_as_YYYY-MM-DD")
                out[sym] = {
                    "market": best.get("Market_and_Exchange_Names"),
                    "long": long_, "short": short_, "net": net, "ts": ts
                }
            else:
                out[sym] = {"market": None, "long": None, "short": None, "net": None, "ts": None}
    except Exception:
        return {}
    return out

# =====================
# Helper: series window
# =====================

def _series_tail(series: List[Dict[str, Any]], n: int = 3) -> List[Dict[str, Any]]:
    return [x for x in series if x.get("value") is not None][-n:]

def _mom(series: List[Dict[str, Any]]) -> Optional[float]:
    ts = _series_tail(series, 2)
    if len(ts) < 2: return None
    return _pct(ts[-1]["value"], ts[-2]["value"])

def _yoy(series: List[Dict[str, Any]]) -> Optional[float]:
    # cherche valeur courante vs. N=12 mois avant (si mensuel), fallback 4 trimestres pour GDP
    data = [x for x in series if _to_float(x.get("value")) is not None]
    if not data: return None
    last = data[-1]
    last_date = last.get("date")
    if not last_date: return None
    # find approx match -12 months
    target_year = int(last_date.split("-")[0]) - 1
    target_month = int(last_date.split("-")[1]) if "-" in last_date else None
    prev = None
    for x in reversed(data[:-1]):
        d = x.get("date","")
        if not d: 
            continue
        try:
            y = int(d.split("-")[0]); m = int(d.split("-")[1]) if "-" in d else None
            if y == target_year and (target_month is None or m == target_month):
                prev = x; break
        except Exception:
            continue
    if prev is None:
        # fallback: 4 obs back
        prev = data[-5] if len(data) >= 5 else None
    if prev is None:
        return None
    return _pct(last["value"], prev["value"])

# ===========================
# Macro snapshot construction
# ===========================

def build_macro_snapshot(
    fred_ids: Dict[str, str] = DEFAULT_FRED,
    include_te: bool = True,
    include_cboe: bool = True,
    include_cot: bool = True,
    use_cache: bool = True,
    lookback_months: int = 18
) -> Dict[str, Any]:
    """
    Construit un snapshot macro unifié:
    - macro: dict de séries clés (last, MoM/YoY)
    - rates & spreads: ust2y, ust10y, 2s10s
    - risk: vix, vvix, skew, hy_oas
    - surprises: événements TE récents (actual-forecast, signe)
    - positioning (COT): net non-commercials sur marchés clés
    """
    sources = []
    # 1) FRED series
    start = None
    if lookback_months:
        start_date = (_now().date().replace(day=1) - dt.timedelta(days=lookback_months*30))
        start = start_date.isoformat()
    fred_data = fred_series(list(fred_ids.values()), start=start, end=None, use_cache=use_cache)
    sources.append("FRED")

    # build macro dict
    macro = {}
    def _last(series):
        s = [x for x in series if x.get("value") is not None]
        return s[-1]["value"] if s else None
    for k, sid in fred_ids.items():
        s = fred_data.get(sid, [])
        macro[k] = {
            "series_id": sid,
            "last": _last(s),
            "mom_pct": _mom(s),
            "yoy_pct": _yoy(s),
            "last_date": s[-1]["date"] if s else None
        }

    # rates and spreads
    ust2 = macro.get("UST2Y", {}).get("last")
    ust10 = macro.get("UST10Y", {}).get("last")
    two_ten = _diff(ust10, ust2)
    spreads = {"UST2Y": ust2, "UST10Y": ust10, "2s10s": two_ten}

    # risk gauges
    risk = {}
    if include_cboe:
        risk.update(cboe_indexes(["VIX","VVIX","SKEW"], use_cache=use_cache))
        sources.append("CBOE")
    # HY OAS (déjà dans FRED macro)
    risk["HY_OAS"] = {"value": macro.get("HY_OAS",{}).get("last"), "ts": macro.get("HY_OAS",{}).get("last_date")}

    # TradingEconomics surprises (sur fenêtre récente, importance élevée)
    surprises: List[Dict[str, Any]] = []
    if include_te:
        te = tradingeconomics_calendar(countries=["United States"], importance=3, use_cache=use_cache, limit=400)
        for ev in te:
            actual = ev.get("actual"); forecast = ev.get("forecast")
            if actual is None or forecast is None:
                continue
            surprises.append({
                "country": ev["country"],
                "event": ev["event"] or ev["category"],
                "date": ev["date"],
                "surprise": _diff(actual, forecast),
                "surprise_pct": _pct(actual, forecast),
                "actual": actual, "forecast": forecast, "previous": ev.get("previous"),
                "unit": ev.get("unit"),
            })
        sources.append("TradingEconomics")

    # COT positioning
    positioning = {}
    if include_cot:
        positioning = cftc_cot(use_cache=use_cache)
        sources.append("CFTC (mirror)")

    # Momentum composite simple
    # (+) si CPI YoY en baisse, UNEMP stable, ISM > 50, 2s10s se pentifie, HY_OAS en baisse
    comp = 0.0; weights = 0.0
    def _add(score, w): 
        nonlocal comp, weights; 
        if score is None: return
        comp += score * w; weights += w
    # CPI YoY -> score inversé (baisse = bon)
    cpi_yoy = macro.get("CPI",{}).get("yoy_pct")
    if cpi_yoy is not None:
        _add(max(-cpi_yoy/5.0, -2), 1.0)  # crude scaling
    # UNEMP MoM (+ c'est mauvais)
    unemp_mom = macro.get("UNEMP",{}).get("mom_pct")
    if unemp_mom is not None:
        _add(-unemp_mom/1.0, 0.6)
    # ISM mfg last (distance 50)
    ism = macro.get("ISM_MFG",{}).get("last")
    if ism is not None:
        _add((ism-50.0)/10.0, 0.8)
    # 2s10s pente (plus haut = mieux)
    if two_ten is not None:
        _add(two_ten/100.0, 0.8)
    # HY OAS (plus bas = mieux); approx change YoY
    hy_yoy = macro.get("HY_OAS",{}).get("yoy_pct")
    if hy_yoy is not None:
        _add(-hy_yoy/50.0, 0.7)
    momentum_score = comp/weights if weights>0 else None

    return {
        "ok": True,
        "macro": macro,
        "spreads": spreads,
        "risk": risk,
        "surprises": surprises[:200],
        "positioning": positioning,
        "momentum_score": momentum_score,
        "asof_utc": _iso(_now()),
        "sources_used": sources,
        "notes": "All endpoints best-effort; missing APIs gracefully return empty sections."
    }

# =========
# CLI tools
# =========

def _print(obj: Any):
    print(json.dumps(obj, ensure_ascii=False, indent=2))

def main():
    import argparse
    ap = argparse.ArgumentParser(description="Macro & Derivatives client (FRED/TE/CBOE/CFTC)")
    sub = ap.add_subparsers(dest="cmd")

    # fred
    ap_fred = sub.add_parser("fred", help="Fetch FRED series")
    ap_fred.add_argument("--ids", type=str, required=False, default=",".join(DEFAULT_FRED.values()),
                         help="Comma-separated series ids")
    ap_fred.add_argument("--start", type=str, default=None)
    ap_fred.add_argument("--end", type=str, default=None)
    ap_fred.add_argument("--no_cache", action="store_true")

    # TE calendar
    ap_te = sub.add_parser("te", help="TradingEconomics calendar")
    ap_te.add_argument("--countries", type=str, default="United States")
    ap_te.add_argument("--start", type=str, default=None)
    ap_te.add_argument("--end", type=str, default=None)
    ap_te.add_argument("--importance", type=int, default=3)
    ap_te.add_argument("--no_cache", action="store_true")
    ap_te.add_argument("--limit", type=int, default=500)

    # CBOE
    ap_cboe = sub.add_parser("cboe", help="CBOE indexes (VIX,VVIX,SKEW)")
    ap_cboe.add_argument("--which", type=str, default="VIX,VVIX,SKEW")
    ap_cboe.add_argument("--no_cache", action="store_true")

    # CFTC
    ap_cftc = sub.add_parser("cftc", help="CFTC COT (legacy futures only mirror)")
    ap_cftc.add_argument("--symbols", type=str, default="SPX,NDX,WTI,GOLD,COPPER,EUR,JPY")
    ap_cftc.add_argument("--no_cache", action="store_true")

    # Snapshot
    ap_snap = sub.add_parser("snapshot", help="Build macro snapshot")
    ap_snap.add_argument("--no_te", action="store_true")
    ap_snap.add_argument("--no_cboe", action="store_true")
    ap_snap.add_argument("--no_cot", action="store_true")
    ap_snap.add_argument("--lookback_months", type=int, default=18)
    ap_snap.add_argument("--no_cache", action="store_true")

    args = ap.parse_args()
    if not args.cmd:
        ap.print_help(); sys.exit(0)

    if args.cmd == "fred":
        ids = [s.strip() for s in args.ids.split(",") if s.strip()]
        out = fred_series(ids, start=args.start, end=args.end, use_cache=not args.no_cache)
        _print(out); return

    if args.cmd == "te":
        c = [x.strip() for x in (args.countries.split(",") if args.countries else []) if x.strip()]
        out = tradingeconomics_calendar(countries=c or None, start=args.start, end=args.end,
                                        importance=args.importance, use_cache=not args.no_cache, limit=args.limit)
        _print(out); return

    if args.cmd == "cboe":
        which = [w.strip() for w in args.which.split(",") if w.strip()]
        out = cboe_indexes(which=which, use_cache=not args.no_cache)
        _print(out); return

    if args.cmd == "cftc":
        syms = [s.strip() for s in args.symbols.split(",") if s.strip()]
        out = cftc_cot(symbols=syms, use_cache=not args.no_cache)
        _print(out); return

    if args.cmd == "snapshot":
        out = build_macro_snapshot(include_te=not args.no_te, include_cboe=not args.no_cboe,
                                   include_cot=not args.no_cot, use_cache=not args.no_cache,
                                   lookback_months=args.lookback_months)
        _print(out); return

if __name__ == "__main__":
    main()
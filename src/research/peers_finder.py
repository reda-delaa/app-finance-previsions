# peers_finder.py
# -*- coding: utf-8 -*-
"""
Peers finder basé Finnhub (+ validation yfinance) — général US/CA.

Entrée: ticker Yahoo OU nom de société (ex: "NGD.TO" ou "New Gold" ou "HUT" ou "Hut 8")
Sortie: liste de tickers Yahoo US/CA plausibles (scores simples + filtres secteur/industrie/keywords)

Dépendances: yfinance, requests, numpy
Secrets: FINNHUB_API_KEY (via secrets_local.py ou variable d'env)

Exemples:
  python peers_finder.py NGD.TO --min 5 --max 10 --log DEBUG
  python peers_finder.py "Hut 8" --min 5 --max 10 --log INFO
"""

from __future__ import annotations
import os, re, time, math, json, argparse, logging
from typing import List, Tuple, Dict, Any, Optional
from datetime import datetime, timezone

import requests
import numpy as np
import yfinance as yf

# ------------------------------------------------------------------------------
# Logging
# ------------------------------------------------------------------------------
logger = logging.getLogger("peers_finder")
logger.propagate = False  # <<< Prevent propagation to root logger
if not logger.handlers:
    h = logging.StreamHandler()
    h.setFormatter(logging.Formatter("%(asctime)s | %(levelname)-7s | %(name)s | %(message)s", "%H:%M:%S"))
    logger.addHandler(h)
logger.setLevel(logging.INFO)

YF_SLEEP = 0.10
HTTP_TIMEOUT = (8, 20)

# ------------------------------------------------------------------------------
# Finnhub helper
# ------------------------------------------------------------------------------
# Load Finnhub API key: prefer environment, fall back to local secrets file for dev
try:
    from src.secrets_local import get_key  # type: ignore
    _FINNHUB_KEY = get_key("FINNHUB_API_KEY") or get_key("FINNHUB_KEY") or None
except Exception:
    _FINNHUB_KEY = os.getenv("FINNHUB_API_KEY") or os.getenv("FINNHUB_KEY") or None

def _load_finnhub_key() -> str:
    """Return the Finnhub API key from env or local secrets fallback."""
    return (str(_FINNHUB_KEY).strip() if _FINNHUB_KEY else "")

def _fh_get(path: str, params: Dict[str, Any]) -> Dict[str, Any] | List[Any] | None:
    token = _load_finnhub_key()
    if not token:
        raise RuntimeError("FINNHUB_API_KEY manquante (env ou secrets_local).")
    url = f"https://finnhub.io/api/v1{path}"
    p = dict(params or {})
    p["token"] = token
    r = requests.get(url, params=p, timeout=HTTP_TIMEOUT)
    r.raise_for_status()
    ctype = (r.headers.get("Content-Type") or "").lower()
    if "application/json" not in ctype:
        raise RuntimeError(f"Finnhub non-JSON (Content-Type={ctype or 'unknown'})")
    return r.json()

# ------------------------------------------------------------------------------
# Yahoo <-> échange mapping et suffix utils
# ------------------------------------------------------------------------------
_YF_SUFFIXES = (
    ".TO", ".V", ".NE", ".CN", ".U",  # Canada
    ".N", ".O", ".NY", ".AS", ".AM",  # US (quelques suffixes legacy)
    ".L", ".SW", ".MI", ".PA", ".F", ".DE", ".BR", ".HK", ".AX", ".NZ", ".SI", ".KS", ".KQ",
)
_EX_TO_YF = {
    # Canada
    "TSX": ".TO", "TSE": ".TO", "TSXV": ".V", "V": ".V", "NEO": ".NE", "CSE": ".CN",
    # US (Yahoo le plus souvent sans suffixe)
    "NASDAQ": "", "NYSE": "", "NYSE ARCA": "", "BATS": "", "AMEX": "",
}

def _strip_yahoo_suffix(sym: str) -> str:
    s = sym.split(":")[0]
    u = s.upper()
    for suf in _YF_SUFFIXES:
        if u.endswith(suf):
            return s[: -len(suf)]
    return s

def _to_yahoo_symbol(base: str, exchange: str) -> str:
    """
    Mappe un symbole “base” (sans suffixe Yahoo) + nom d'échange Finnhub vers un ticker Yahoo probable.
    Ex: ("NGD", "TSX") -> "NGD.TO" ; ("HUT", "NASDAQ") -> "HUT"
    """
    suf = _EX_TO_YF.get((exchange or "").upper(), "")
    return base.upper() + suf

# ------------------------------------------------------------------------------
# yfinance helpers
# ------------------------------------------------------------------------------
def _yf_info(t: yf.Ticker) -> Dict[str, Any]:
    info = {}
    # API récente
    if hasattr(t, "get_info"):
        try:
            info = t.get_info() or {}
        except Exception:
            info = {}
    if not info:
        try:
            info = t.info or {}
        except Exception:
            info = {}
    # fast fields
    try:
        fi = getattr(t, "fast_info", None)
        if fi:
            info.setdefault("marketCap", getattr(fi, "market_cap", None))
            info.setdefault("regularMarketPrice", getattr(fi, "last_price", None))
            info.setdefault("currency", getattr(fi, "currency", None))
    except Exception:
        pass
    return info or {}

def _safe_float(x, default=np.nan) -> float:
    try:
        if x is None: return default
        v = float(x)
        return v if math.isfinite(v) else default
    except Exception:
        return default

def _is_us_ca(info: Dict[str, Any]) -> bool:
    c = (info.get("country") or info.get("countryName") or "").upper()
    return ("CANADA" in c) or ("UNITED STATES" in c) or (c in {"CA", "US", "USA"})

def _looks_active(ticker: str, max_days_gap: int = 60) -> bool:
    try:
        h = yf.Ticker(ticker).history(period="1y", interval="1d", auto_adjust=True)
        if h.empty: return False
        last_ts = h.index.max()
        if getattr(last_ts, "tz", None) is not None:
            last_ts = last_ts.tz_localize(None)
        delta_days = (datetime.now(timezone.utc) - last_ts.replace(tzinfo=timezone.utc)).days
        if delta_days > max_days_gap: return False
        price = float(h["Close"].dropna().iloc[-1])
        return math.isfinite(price) and price > 0
    except Exception:
        return False

def _has_useful_multiples(info: Dict[str, Any]) -> bool:
    fields = [
        _safe_float(info.get("trailingPE"), np.nan),
        _safe_float(info.get("forwardPE"), np.nan),
        _safe_float(info.get("priceToSalesTrailing12Months"), np.nan),
        _safe_float(info.get("enterpriseToEbitda"), np.nan),
    ]
    return any(math.isfinite(x) and x > 0 for x in fields)

# ------------------------------------------------------------------------------
# Résolution d’entrée (ticker Yahoo ou nom)
# ------------------------------------------------------------------------------
def _resolve_symbol(user_input: str) -> Tuple[str, str, Dict[str, Any]]:
    """
    Retourne: (yahoo_symbol, base_symbol_sans_suffixe, my_info_yf)
    - Si l'entrée ressemble déjà à un ticker Yahoo, on teste directement.
    - Sinon: utilise Finnhub /search pour prendre le 1er meilleur résultat, puis map exchange->Yahoo.
    """
    s = user_input.strip()
    # cas ticker direct (heuristique: pas d'espace, <= 10 chars)
    looks_ticker = ((" " not in s) and (len(s) <= 10))
    if looks_ticker:
        # tester tel quel
        info = _yf_info(yf.Ticker(s))
        if info and _looks_active(s):
            base = _strip_yahoo_suffix(s)
            return (s.upper(), base.upper(), info)

    # Sinon: Finnhub symbol lookup
    try:
        data = _fh_get("/search", {"q": s}) or {}
    except Exception as e:
        logger.debug(f"Finnhub /search error: {e}")
        data = {}

    items = (data.get("result") or []) if isinstance(data, dict) else []
    # priorité aux actions (type='Common Stock'/'EQS' etc.)
    def _score_item(it: Dict[str, Any]) -> int:
        t = (it.get("type") or "").lower()
        ex = (it.get("exchange") or "").upper()
        score = 0
        if "stock" in t or t in {"eqa", "eqs"}: score += 3
        if ex in {"NASDAQ", "NYSE", "TSX", "TSXV"}: score += 2
        if (it.get("symbol") or "").upper() == s.upper(): score += 4
        name = (it.get("description") or "").lower()
        if s.lower() in name: score += 1
        return score

    items = sorted([it for it in items if isinstance(it, dict)], key=_score_item, reverse=True)
    if not items:
        # dernier recours: considérer l'entrée comme ticker Yahoo et tenter quand même
        info = _yf_info(yf.Ticker(s))
        return (s.upper(), _strip_yahoo_suffix(s).upper(), info)

    best = items[0]
    raw_sym = (best.get("symbol") or "")
    base = _strip_yahoo_suffix(raw_sym)
    ex = (best.get("exchange") or "")
    yahoo = _to_yahoo_symbol(base, ex)
    info = _yf_info(yf.Ticker(yahoo))
    # si map pas terrible, retomber sur base sans suffixe
    if not _looks_active(yahoo):
        alt = base.upper()
        info2 = _yf_info(yf.Ticker(alt))
        if _looks_active(alt):
            return (alt, base.upper(), info2)
    return (yahoo, base.upper(), info)

# ------------------------------------------------------------------------------
# Peers Finnhub -> mapping Yahoo -> filtres -> score
# ------------------------------------------------------------------------------
def _finnhub_peers(base_symbol: str) -> List[str]:
    base = _strip_yahoo_suffix(base_symbol)
    tried = [base, f"{base}:TSX", f"{base}:TO"]
    logger.debug(f"Finnhub: variants essayés pour '{base_symbol}': {tried}")
    for variant in tried:
        try:
            out = _fh_get("/stock/peers", {"symbol": variant}) or []
            if isinstance(out, list) and out:
                logger.debug(f"Finnhub peers({variant}): {out}")
                return [str(x).upper() for x in out if isinstance(x, str) and x]
        except Exception as e:
            logger.debug(f"Finnhub peers({variant}) error: {e}")
    return []

def _text_keywords(text: str) -> set[str]:
    txt = (text or "").lower()
    # très simple: tokens alphanum >2 chars
    toks = re.findall(r"[a-z0-9][a-z0-9\-]{2,}", txt)
    return set(toks)

# Mots-clés “infra/compute/énergie/crypto” (pour couvrir HUT pivot IA/énergie tout en restant général)
_INFRA_KEYWORDS = {
    "datacenter","data-center","data centers","colocation","hpc","compute","ai","gpu",
    "accelerator","nvidia","inference","training","cloud","baremetal","power","energy",
    "electricity","generation","renewable","grid","immersion","cooling","thermal",
    "blockchain","bitcoin","crypto","mining","asic","hashrate",
}

def _same_bucket(my_info: Dict[str, Any], peer_info: Dict[str, Any]) -> bool:
    """Filtrage flexible: même secteur OU industrie recoupée OU forte similarité mots-clés infra/HPC/énergie/crypto."""
    my_sec = (my_info.get("sector") or "").lower()
    my_ind = (my_info.get("industry") or "").lower()
    pe_sec = (peer_info.get("sector") or "").lower()
    pe_ind = (peer_info.get("industry") or "").lower()

    # 1) même secteur strict
    if my_sec and pe_sec and my_sec == pe_sec:
        return True

    # 2) industrie qui se recoupe (substring)
    if my_ind and pe_ind and (my_ind in pe_ind or pe_ind in my_ind):
        return True

    # 3) mots-clés infra/HPC/énergie/crypto (utile pour HUT pivot)
    my_sum = (my_info.get("longBusinessSummary") or "")
    pe_sum = (peer_info.get("longBusinessSummary") or "")
    ks = _text_keywords(my_sum) | set(my_ind.split()) | set(my_sec.split())
    kp = _text_keywords(pe_sum) | set(pe_ind.split()) | set(pe_sec.split())

    if ks & kp:
        return True
    # cas “infra” plus flou: si l’un contient des keywords infra et l’autre aussi
    if (ks & _INFRA_KEYWORDS) and (kp & _INFRA_KEYWORDS):
        return True

    # fallback: si on n’a quasiment aucune info, ne pas filtrer trop fort
    if not (my_sec or my_ind or my_sum) or not (pe_sec or pe_ind or pe_sum):
        return True

    return False

def _score_candidate(my_info: Dict[str, Any], peer_info: Dict[str, Any]) -> float:
    score = 0.0
    my_mcap = _safe_float(my_info.get("marketCap"), np.nan)
    pe_mcap = _safe_float(peer_info.get("marketCap"), np.nan)
    if math.isfinite(my_mcap) and my_mcap > 0 and math.isfinite(pe_mcap) and pe_mcap > 0:
        ratio = max(my_mcap, pe_mcap) / max(1.0, min(my_mcap, pe_mcap))
        score += 1.0 / ratio
    if (peer_info.get("currency") or "").upper() == (my_info.get("currency") or "").upper():
        score += 0.2
    if (peer_info.get("country") or "") == (my_info.get("country") or ""):
        score += 0.2
    return score

def _map_and_validate(peers_base: List[str], my_info: Dict[str, Any]) -> List[Tuple[str, float]]:
    validated: List[Tuple[str, float]] = []
    seen_yahoo: set[str] = set()

    for b in peers_base:
        b_clean = _strip_yahoo_suffix(b)
        # profil Finnhub pour récupérer l'échange (mapping Yahoo)
        try:
            prof = _fh_get("/stock/profile2", {"symbol": b_clean}) or {}
        except Exception:
            prof = {}
        ex = (prof.get("exchange") or "")

        yahoo = _to_yahoo_symbol(b_clean, ex)
        pi = _yf_info(yf.Ticker(yahoo))
        # fallback si pas actif: tenter base sans suffixe
        if not _looks_active(yahoo):
            alt = b_clean.upper()
            pi2 = _yf_info(yf.Ticker(alt))
            if _looks_active(alt):
                yahoo, pi = alt, pi2

        if not pi or not _is_us_ca(pi):    # limiter à US/CA
            continue
        if not _has_useful_multiples(pi):  # éviter shells / données trop pauvres
            continue
        if not _same_bucket(my_info, pi):  # filtre secteur/industrie/keywords (souple)
            continue

        sc = _score_candidate(my_info, pi)
        if yahoo not in seen_yahoo:
            seen_yahoo.add(yahoo)
            validated.append((yahoo, sc))
            time.sleep(YF_SLEEP)

    validated.sort(key=lambda x: x[1], reverse=True)
    return validated

# ------------------------------------------------------------------------------
# Orchestrateur
# ------------------------------------------------------------------------------
def get_peers_auto(user_input: str, min_peers: int = 5, max_peers: int = 15, logger_=None) -> List[str]:
    log = logger_ or logger

    yahoo_sym, base_sym, my_info = _resolve_symbol(user_input)
    input_label = user_input
    # sécurité: si aucune info récupérée, quand même créer un my_info minimal
    if not my_info:
        my_info = _yf_info(yf.Ticker(yahoo_sym))

    # candidats via Finnhub
    peers_base = _finnhub_peers(base_sym)
    if not peers_base:
        log.warning("Aucun peer Finnhub — résultat vide.")
        return []

    log.debug(f"candidats bruts: {peers_base}")

    validated_scored = _map_and_validate(peers_base, my_info)
    peers: List[str] = []
    for s, _sc in validated_scored:
        if s.upper() != yahoo_sym.upper() and s not in peers:
            peers.append(s)
        if len(peers) >= max_peers:
            break

    if len(peers) < min_peers:
        log.warning(f"Pairs trouvés {len(peers)}/{min_peers} — élargis la recherche ou augmente --max.")
    log.info(f"[{input_label}] peers retenus: {peers}")
    return peers

def find_peers(ticker: str, k: int = 10):
    """Fonction d'interface avec fallback corrélation Yahoo si pas d'API Finnhub"""
    # Try Finnhub first
    try:
        peers = get_peers_auto(ticker, min_peers=k, max_peers=k)
        if peers:
            return peers
    except Exception as e:
        logger.debug(f"Finnhub peers failed, trying correlation fallback: {e}")

    # Fallback: corrélation sur un univers réduit d'ETFs/large caps
    try:
        universe = ["AAPL","MSFT","GOOGL","AMZN","META","NVDA","TSLA",
                   "XOM","JNJ","JPM","V","MA","UNH","HD","PG","KO","PEP"]
        if ticker not in universe:
            universe.insert(0, ticker)

        import pandas as pd
        px = yf.download(universe, period="1y", interval="1d", auto_adjust=True)["Close"]
        ret = px.pct_change().dropna()
        if ticker not in ret.columns:
            return []
        corr = ret.corr()[ticker].drop(ticker).sort_values(ascending=False)
        correlation_peers = corr.head(k).index.tolist()
        logger.info(f"[{ticker}] Using correlation-based peers fallback: {correlation_peers}")
        return correlation_peers

    except Exception as e:
        logger.warning(f"Correlation peers fallback failed: {e}")
        return []

# ------------------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Peers finder (Finnhub + yfinance, US/CA, secteur-général)")
    ap.add_argument("input", help="Ticker Yahoo OU nom de société (ex: NGD.TO ou 'New Gold' ou HUT)")
    ap.add_argument("--min", type=int, default=5)
    ap.add_argument("--max", type=int, default=15)
    ap.add_argument("--log", default="INFO", choices=["DEBUG","INFO","WARNING","ERROR","CRITICAL"])
    args = ap.parse_args()
    logger.setLevel(getattr(logging, args.log))

    peers = []
    try:
        peers = get_peers_auto(args.input, min_peers=args.min, max_peers=args.max, logger_=logger)
    except Exception as e:
        logger.error(f"Erreur: {e}")

    print(json.dumps({"input": args.input, "peers": peers}, indent=2))

if __name__ == "__main__":
    main()

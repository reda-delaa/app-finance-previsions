# peers_firecrawl.py
# -*- coding: utf-8 -*-
"""
Moteur de pairs (US/CA) via Firecrawl (+ validation yfinance) — STRICT (pas de fallback).
Requis: pip install firecrawl-py yfinance pandas numpy

Usage CLI:
    # 1) mettre la clé dans secrets_local.py (voir modèle plus bas) OU dans l'env
    # 2) lancer avec debug complet + dump
    python peers_firecrawl.py NGD.TO --min 5 --max 12 --log DEBUG --refresh --why --dump artifacts_phase1
"""

from __future__ import annotations

import os
import re
import time
import json
import math
import argparse
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import yfinance as yf

# =========================
# Chargement clé Firecrawl (ENV > secrets_local.py)
# =========================
FIRECRAWL_API_KEY = os.getenv("FIRECRAWL_API_KEY", "").strip()
if not FIRECRAWL_API_KEY:
    try:
        import secrets_local  # type: ignore
        FIRECRAWL_API_KEY = getattr(secrets_local, "FIRECRAWL_API_KEY", "").strip()
    except Exception:
        FIRECRAWL_API_KEY = ""

_FC_AVAILABLE = False
_app = None
_fc_err: Optional[str] = None
try:
    from firecrawl import FirecrawlApp  # type: ignore
    if not FIRECRAWL_API_KEY:
        raise RuntimeError("FIRECRAWL_API_KEY non défini (ENV ou secrets_local.py).")
    _app = FirecrawlApp(api_key=FIRECRAWL_API_KEY)
    _FC_AVAILABLE = True
except Exception as _e:
    _fc_err = str(_e)
    _app = None
    _FC_AVAILABLE = False

# =========================
# Logging
# =========================
logger = logging.getLogger("peers_firecrawl")
if not logger.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter("%(asctime)s | %(levelname)-7s | %(name)s | %(message)s", "%H:%M:%S"))
    logger.addHandler(_h)
logger.setLevel(logging.INFO)

# =========================
# Constantes & helpers généraux
# =========================
YF_SLEEP = 0.10
MAX_PEERS = 15
CACHE_FILE = "artifacts_phase1/peers_cache_firecrawl.json"

TICKER_RE = re.compile(r"\b([A-Z]{1,4}\.?[A-Z]{1,3})\b")  # rough: KGC, NEM, ABX.TO, AEM.TO, MSFT
YF_OK_SUFFIX = {".TO", ".V", ".NE", ".CN"}  # suffixes Yahoo pour le CA
US_EXCH_HINTS = {"NYSE", "Nasdaq", "NASDAQ", "AMEX"}

@dataclass
class PeerCandidate:
    symbol: str
    name: Optional[str] = None
    score: float = 0.0
    reason: Optional[str] = None


def _ensure_dir(path: str):
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)


def _load_cache() -> Dict[str, List[str]]:
    try:
        if os.path.exists(CACHE_FILE):
            with open(CACHE_FILE, "r") as f:
                return json.load(f)
    except Exception:
        pass
    return {}


def _save_cache(cache: Dict[str, List[str]]):
    try:
        _ensure_dir(CACHE_FILE)
        with open(CACHE_FILE, "w") as f:
            json.dump(cache, f, indent=2)
    except Exception:
        pass


def _yf_info(t: yf.Ticker) -> Dict[str, Any]:
    info = {}
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
    # enrichir avec fast_info quand utile
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
        if x is None:
            return default
        v = float(x)
        return v if math.isfinite(v) else default
    except Exception:
        return default


def _is_us_ca(info: Dict[str, Any]) -> bool:
    c = (info.get("country") or info.get("countryName") or "").upper()
    return ("CANADA" in c) or ("UNITED STATES" in c) or (c in {"CA", "US", "USA"})


def _same_bucket(ind_a: str, ind_b: str) -> bool:
    if not ind_a or not ind_b:
        return True
    a, b = ind_a.lower(), ind_b.lower()
    return a in b or b in a


def _looks_active(ticker: str, max_days_gap: int = 60) -> bool:
    try:
        h = yf.Ticker(ticker).history(period="1y", interval="1d", auto_adjust=True)
        if h.empty:
            return False
        last_ts = h.index.max()
        if getattr(last_ts, "tz", None) is not None:
            last_ts = last_ts.tz_localize(None)
        delta_days = (datetime.now(timezone.utc) - last_ts.replace(tzinfo=timezone.utc)).days
        if delta_days > max_days_gap:
            return False
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


def _normalize_symbol(sym: str) -> str:
    sym = sym.strip().upper()
    sym = re.sub(r"[^A-Z0-9\.\-]", "", sym)
    return sym


def _extract_symbols_from_text(text: str) -> List[str]:
    raw = set(m.group(1) for m in TICKER_RE.finditer(text or ""))
    cleaned = []
    for s in raw:
        s = _normalize_symbol(s)
        if 1 <= len(s) <= 10:
            cleaned.append(s)
    return cleaned


# =========================
# Firecrawl – compat SDK + backoff
# =========================

def _fc_search(query: str, *, retries: int = 2, pause: float = 0.5) -> List[Dict[str, str]]:
    """
    Retourne toujours une liste de {title, url}. Gère:
      - SDK qui renvoie un objet (pydantic) type SearchData
      - dict classique {'results': [...]}
      - Rate limit avec petit backoff
    """
    assert _app is not None
    last_err = None
    for attempt in range(retries + 1):
        try:
            res = _app.search(query)  # type: ignore
            # cas 1: dict-like
            if isinstance(res, dict):
                items = res.get("results") or []
                return [{"title": i.get("title", ""), "url": i.get("url", "")} for i in items if i.get("url")]
            # cas 2: objet avec .results
            results = getattr(res, "results", None)
            if results is not None:
                out = []
                for i in results:
                    # i peut être dict-like ou objet avec attrs
                    title = i.get("title") if isinstance(i, dict) else getattr(i, "title", "")
                    url = i.get("url") if isinstance(i, dict) else getattr(i, "url", "")
                    if url:
                        out.append({"title": title or "", "url": url})
                return out
            # cas 3: objet avec .dict() / .model_dump()
            if hasattr(res, "dict"):
                d = res.dict()
                items = d.get("results") or []
                return [{"title": i.get("title", ""), "url": i.get("url", "")} for i in items if i.get("url")]
            if hasattr(res, "model_dump"):
                d = res.model_dump()
                items = d.get("results") or []
                return [{"title": i.get("title", ""), "url": i.get("url", "")} for i in items if i.get("url")]
            # défaut – rien
            return []
        except Exception as e:
            msg = str(e)
            last_err = msg
            if "Rate limit" in msg or "Rate Limit" in msg:
                # backoff doux
                sleep_s = pause * (2 ** attempt)
                logger.debug(f"Firecrawl rate-limited, retry in {sleep_s:.1f}s …")
                time.sleep(sleep_s)
                continue
            # autre erreur → remonter
            raise
    # si on sort de la boucle
    if last_err:
        logger.debug(f"Firecrawl error (search): {last_err}")
    return []


def _fc_scrape(url: str, *, retries: int = 1, pause: float = 0.5) -> str:
    """
    Retourne du texte (markdown ou texte brut) depuis Firecrawl.
    Gère dict / objets pydantic.
    """
    assert _app is not None
    last_err = None
    for attempt in range(retries + 1):
        try:
            data = _app.scrape_url(url)  # type: ignore
            # dict-like
            if isinstance(data, dict):
                return (data.get("markdown") or data.get("text") or data.get("content") or "") or ""
            # objet avec attributs
            for attr in ("markdown", "text", "content", "html"):
                val = getattr(data, attr, None)
                if isinstance(val, str) and val.strip():
                    return val
            # objet avec .dict()/model_dump()
            if hasattr(data, "dict"):
                d = data.dict()
                return d.get("markdown") or d.get("text") or d.get("content") or ""
            if hasattr(data, "model_dump"):
                d = data.model_dump()
                return d.get("markdown") or d.get("text") or d.get("content") or ""
            return ""
        except Exception as e:
            msg = str(e)
            last_err = msg
            if "Rate limit" in msg or "Rate Limit" in msg:
                time.sleep(pause * (2 ** attempt))
                continue
            raise
    if last_err:
        logger.debug(f"Firecrawl error (scrape): {last_err}")
    return ""


def _firecrawl_search_queries(company: str, symbol: str, industry: str) -> List[str]:
    base = f"{company} {symbol}"
    kin = "competitors OR similar companies OR peers"
    lists = "list OR overview OR companies"
    return [
        f"{base} {kin}",
        f"{symbol} {kin}",
        f"{company} competitors {lists}",
        f"{industry} competitors public companies US Canada",
        f"{company} peer group site:wikipedia.org",
        f"{company} similar companies site:finance.yahoo.com",
    ]


def _firecrawl_collect_candidates(company: str, symbol: str, industry: str, limit_pages: int = 6, timeout: int = 30) -> Dict[str, str]:
    """
    Retourne {symbol: name?} parsé depuis les pages retournées par Firecrawl.
    Compatible avec SDK objet + backoff.
    """
    if not _FC_AVAILABLE or _app is None:
        raise RuntimeError("Firecrawl indisponible (clé absente, import KO, ou init échoué).")

    found: Dict[str, str] = {}
    queries = _firecrawl_search_queries(company, symbol, industry)
    t0 = time.time()
    for q in queries:
        if time.time() - t0 > timeout:
            break
        logger.debug(f"Firecrawl.search: {q}")
        results = _fc_search(q)
        if not results:
            continue
        for item in results[:limit_pages]:
            url = item.get("url")
            title = item.get("title") or ""
            if not url:
                continue
            logger.debug(f"Firecrawl.scrape: {url}")
            text = _fc_scrape(url)
            if not text:
                continue
            for s in _extract_symbols_from_text(text):
                # filtres rapides
                if s in {"USA", "CAD", "ETF", "NYSE", "NASDAQ"}:
                    continue
                found.setdefault(s, title)
        # throttle doux pour éviter rate limit (6 req/min sur free)
        time.sleep(0.2)
    return found


# =========================
# Scoring & validation YF
# =========================

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


def _coerce_us_ca_symbol(sym: str) -> Optional[str]:
    """
    Tente sym, puis sym.TO, puis sym.V — retourne la première variante
    (US/CA) active avec multiples utiles.
    """
    candidates = [sym]
    if "." not in sym:
        candidates += [f"{sym}.TO", f"{sym}.V"]

    for c in candidates:
        try:
            yt = yf.Ticker(c)
            pi = _yf_info(yt)
            if not _is_us_ca(pi):
                continue
            if not _looks_active(c):
                continue
            if not _has_useful_multiples(pi):
                continue
            return c
        except Exception:
            continue
    return None


# =========================
# Debug helpers
# =========================
def _debug_log_candidate(sym: str, pi: Dict[str, Any], *,
                         ok_us_ca: bool, ok_bucket: bool,
                         ok_active: bool, ok_mult: bool,
                         score: Optional[float] = None):
    logger.debug(
        f"[cand] {sym} | "
        f"us_ca={ok_us_ca} bucket={ok_bucket} active={ok_active} mult={ok_mult} "
        f"mcap={_safe_float(pi.get('marketCap'))} "
        f"peT={_safe_float(pi.get('trailingPE'))} peF={_safe_float(pi.get('forwardPE'))} "
        f"ps={_safe_float(pi.get('priceToSalesTrailing12Months'))} "
        f"evE={_safe_float(pi.get('enterpriseToEbitda'))} "
        f"score={score if score is not None else 'NA'} "
        f"country={pi.get('country')} industry={pi.get('industry')}"
    )


def _dump_debug_json(path: str, payload: Dict[str, Any]):
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump(payload, f, indent=2, default=str)
        logger.info(f"[dump] écrit → {path}")
    except Exception as e:
        logger.warning(f"[dump] impossible d’écrire {path}: {e}")


# =========================
# Core
# =========================
def get_peers_firecrawl(
    ticker: str,
    min_peers: int = 5,
    max_peers: int = MAX_PEERS,
    use_cache: bool = True,
    logger_: Optional[logging.Logger] = None,
    why: bool = False,
    dump_dir: Optional[str] = None,
) -> List[str]:
    """
    Renvoie une liste de pairs (tickers Yahoo) pour `ticker`, validés & triés.
    STRICT: pas de fallback si insuffisant.
    """
    if not _FC_AVAILABLE or _app is None:
        raise RuntimeError(f"Firecrawl indisponible: {_fc_err or 'installez `pip install firecrawl-py` et fournissez FIRECRAWL_API_KEY.'}")

    log = logger_ or logger
    debug_bundle: Dict[str, Any] = {"ticker": ticker, "steps": {}}

    cache = _load_cache() if use_cache else {}
    if use_cache and ticker in cache and cache[ticker]:
        log.info(f"[{ticker}] peers (cache): {cache[ticker]}")
        peers_cached = cache[ticker][:max_peers]
        debug_bundle["steps"]["cache_hit"] = {"peers": peers_cached}
        if dump_dir:
            _dump_debug_json(os.path.join(dump_dir, f"peers_debug_{ticker}.json"), debug_bundle)
        if len(peers_cached) >= min_peers:
            return peers_cached
        # sinon: on continue pour compléter via Firecrawl

    target = yf.Ticker(ticker)
    my_info = _yf_info(target)
    company = my_info.get("shortName") or my_info.get("longName") or my_info.get("symbol") or ticker
    industry = my_info.get("industry") or my_info.get("sector") or ""
    log.debug(f"[{ticker}] company={company} | industry={industry}")

    # 1) Firecrawl collect
    raw = _firecrawl_collect_candidates(company, ticker, industry, limit_pages=6, timeout=30)
    if not raw:
        raise RuntimeError(f"[{ticker}] Aucun candidat trouvé via Firecrawl.")
    log.info(f"[{ticker}] candidats Firecrawl: {list(raw.keys())[:20]}")
    debug_bundle["steps"]["firecrawl_candidates"] = {"symbols": list(raw.keys())}

    # 2) Validation yfinance (+ coercion .TO/.V)
    validated: List[Tuple[str, float]] = []
    cand_details: List[Dict[str, Any]] = []

    for sym, _title in raw.items():
        original = _normalize_symbol(sym)
        coerced = _coerce_us_ca_symbol(original)
        detail = {"original": original, "coerced": coerced, "accepted": False, "reasons": []}

        if not coerced:
            detail["reasons"].append("coercion_failed_or_inactive_or_no_multiples")
            cand_details.append(detail)
            if why:
                logger.debug(f"[cand] {original} rejeté → coercion/actif/multiples KO")
            continue

        try:
            yt = yf.Ticker(coerced)
            pi = _yf_info(yt)

            ok_us_ca = _is_us_ca(pi)
            ok_bucket = _same_bucket(industry, pi.get("industry", ""))
            ok_active = _looks_active(coerced)
            ok_mult = _has_useful_multiples(pi)

            if not ok_us_ca:  detail["reasons"].append("not_us_ca")
            if not ok_bucket: detail["reasons"].append("industry_mismatch")
            if not ok_active: detail["reasons"].append("inactive_or_stale")
            if not ok_mult:   detail["reasons"].append("no_useful_multiples")

            if ok_us_ca and ok_bucket and ok_active and ok_mult:
                score = _score_candidate(my_info, pi)
                validated.append((coerced, score))
                detail["accepted"] = True
                detail["score"] = score
                if why:
                    _debug_log_candidate(coerced, pi,
                                         ok_us_ca=True, ok_bucket=True,
                                         ok_active=True, ok_mult=True,
                                         score=score)
            else:
                if why:
                    _debug_log_candidate(coerced, pi,
                                         ok_us_ca=ok_us_ca, ok_bucket=ok_bucket,
                                         ok_active=ok_active, ok_mult=ok_mult,
                                         score=None)

            cand_details.append(detail)
            time.sleep(YF_SLEEP)
        except Exception as e:
            detail["reasons"].append(f"exception:{type(e).__name__}")
            cand_details.append(detail)
            if why:
                logger.debug(f"[cand] {coerced or original} exception: {e}")
            continue

    debug_bundle["steps"]["validation"] = {"candidates": cand_details}

    # 3) Tri (sans complétion)
    validated.sort(key=lambda x: x[1], reverse=True)
    peers: List[str] = []
    for s, _sc in validated:
        if s.upper() != ticker.upper() and s not in peers:
            peers.append(s)
        if len(peers) >= max_peers:
            break

    # 4) Strict: si insuffisant -> erreur explicite
    if len(peers) < min_peers:
        debug_bundle["steps"]["final"] = {"peers": peers, "n": len(peers)}
        if dump_dir:
            _dump_debug_json(os.path.join(dump_dir, f"peers_debug_{ticker}.json"), debug_bundle)
        raise RuntimeError(
            f"[{ticker}] Nombre de pairs valides insuffisant via Firecrawl uniquement "
            f"({len(peers)}/{min_peers}). Ré-essaye avec --max plus grand, --why pour le détail, "
            f"ou élargis les requêtes."
        )

    log.info(f"[{ticker}] peers (final): {peers} (n={len(peers)})")
    debug_bundle["steps"]["final"] = {"peers": peers, "n": len(peers)}
    if dump_dir:
        _dump_debug_json(os.path.join(dump_dir, f"peers_debug_{ticker}.json"), debug_bundle)

    if use_cache and peers:
        cache = _load_cache()
        cache[ticker] = peers
        _save_cache(cache)

    return peers[:max_peers]


# =========================
# CLI
# =========================
def main():
    parser = argparse.ArgumentParser(description="Peer engine via Firecrawl + yfinance (STRICT)")
    parser.add_argument("ticker", help="Ticker Yahoo (ex: NGD.TO)")
    parser.add_argument("--min", type=int, default=5, dest="min_peers", help="Nombre minimum de pairs")
    parser.add_argument("--max", type=int, default=MAX_PEERS, dest="max_peers", help="Nombre maximum de pairs")
    parser.add_argument("--no-cache", action="store_true", help="Désactiver le cache local")
    parser.add_argument("--refresh", action="store_true", help="Ignore le cache (équiv. --no-cache)")
    parser.add_argument("--why", action="store_true", help="Log détaillé par candidat (raison accept/rejet)")
    parser.add_argument("--dump", type=str, default=None, help="Répertoire où dumper un JSON debug complet")
    parser.add_argument("--log", default="INFO", choices=["DEBUG","INFO","WARNING","ERROR","CRITICAL"])
    args = parser.parse_args()

    logger.setLevel(getattr(logging, args.log))
    print(f"Firecrawl available: {_FC_AVAILABLE} | API key present: {bool(FIRECRAWL_API_KEY)}")

    if not _FC_AVAILABLE or _app is None:
        logger.error(f"Firecrawl indisponible: {_fc_err or 'installez `pip install firecrawl-py` et exportez FIRECRAWL_API_KEY ou créez secrets_local.py.'}")
        raise SystemExit(2)

    try:
        peers = get_peers_firecrawl(
            args.ticker,
            min_peers=args.min_peers,
            max_peers=args.max_peers,
            use_cache=not (args.no_cache or args.refresh),
            logger_=logger,
            why=args.why,
            dump_dir=args.dump,
        )
        print(json.dumps({"ticker": args.ticker, "peers": peers}, indent=2))
    except Exception as e:
        logger.error(str(e))
        raise SystemExit(3)


if __name__ == "__main__":
    main()
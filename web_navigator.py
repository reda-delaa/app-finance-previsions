# web_navigator.py — finance-first web search (SearXNG + optional Serper/Tavily)

import os
import re
import json
import time
import random
from urllib.parse import urlencode
from typing import Iterable, List, Dict
import logging
import argparse
import requests

# -----------------------------------------------------------------------------
# Logger (module-level)
# -----------------------------------------------------------------------------
logger = logging.getLogger("web_navigator")
if not logger.handlers:
    logger.addHandler(logging.NullHandler())

# -----------------------------------------------------------------------------
# Config SearXNG
# -----------------------------------------------------------------------------
SEARXNG_FETCH_URL = "https://searx.space/data/instances.json"
SEARXNG_TIMEOUT = (8, 20)
SEARXNG_DEFAULT_ENGINES = ["duckduckgo", "bing", "google"]
SEARXNG_KNOWN_JSON_OK = [
    "https://search.buddyverse.net",
    "https://search.inetol.net",
    "https://search.bus-hit.me",
]

# Catégorie par défaut et “news” pour finance
SEARXNG_BASE_PARAMS = {
    "format": "json",
    "language": "en",
    "safesearch": 0,
    "categories": "general",
}
SEARXNG_NEWS_PARAMS = {
    **SEARXNG_BASE_PARAMS,
    "categories": "news",
    # SearXNG supporte souvent “time_range” ∈ {day, week, month, year}
    # Certaines instances l’ignorent => safe.
    "time_range": "week",
}

SEARXNG_FALLBACK_INSTANCES = [
    "https://searx.be",
    "https://searx.tiekoetter.com",
    "https://search.inetol.net",
    "https://xo.wtf",
    "https://searx.headpat.exchange",
    "https://searxng.site",
    "https://searxng.brihx.fr",
    "https://search.bus-hit.me",
]

# -----------------------------------------------------------------------------
# Domains / heuristics
# -----------------------------------------------------------------------------
TRUSTED_FINANCE = (
    "reuters.com", "bloomberg.com", "wsj.com", "ft.com",
    "sec.gov", "investor.", "ir.", "sedar", "canada.ca",
    "nasdaq.com", "marketwatch.com", "seekingalpha.com",
    "morningstar.", "theglobeandmail.com", "yahoo.com",
    "barrons.com", "investopedia.com",
)
LOW_VALUE = ("pinterest.", "reddit.com/r/", "/ads?", "utm_", "doubleclick.net")
HARD_BLOCK = ("github.com", "stackoverflow.com", "stackexchange.com",
              "superuser.com", "serverfault.com")

FINANCE_KEYWORDS = {
    # résultats / guidance
    "earnings","results","guidance","outlook","quarter","q1","q2","q3","q4","fy",
    "revenue","sales","profit","margin","ebitda","eps","forecast","beat","miss",
    # corporate / equity
    "offering","secondary","placement","buyback","dividend","downgrade","upgrade",
    "merger","acquisition","m&a","spin-off","spinoff","ipo","debt","notes",
    # marchés / macro
    "inflation","cpi","ppi","jobs","fomc","rate","yields","housing","manufacturing",
}

# -----------------------------------------------------------------------------
# HTTP helpers
# -----------------------------------------------------------------------------
def _random_ua() -> str:
    v = ".".join(str(random.randint(60, 125)) for _ in range(3))
    chrome = f"Chrome/{v}"
    safari = f"Safari/{random.randint(500, 600)}.{random.randint(1, 50)}"
    return f"Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) {chrome} {safari}"

def _request_json(url: str, params: dict | None = None, timeout=SEARXNG_TIMEOUT) -> dict:
    headers = {
        "User-Agent": _random_ua(),
        "Accept": "application/json,text/*;q=0.4,*/*;q=0.1",
        "Accept-Language": "en-US,en;q=0.7,fr;q=0.5",
        "Cache-Control": "no-cache",
        "Pragma": "no-cache",
    }
    r = requests.get(url, params=params, headers=headers, timeout=timeout, allow_redirects=True)
    r.raise_for_status()
    ctype = (r.headers.get("Content-Type") or "").lower()
    if "application/json" not in ctype:
        raise ValueError(f"non-json response (Content-Type={ctype or 'unknown'})")
    return r.json()

# -----------------------------------------------------------------------------
# SearXNG instance discovery
# -----------------------------------------------------------------------------
def fetch_searxng_instances(logger_=None) -> list[str]:
    if logger_ is None:
        logger_ = logger
    try:
        data = _request_json(SEARXNG_FETCH_URL, timeout=(6, 12))
        inst = []
        if isinstance(data, dict):
            if "instances" in data and isinstance(data["instances"], dict):
                inst = list(data["instances"].keys())
            elif "urls" in data and isinstance(data["urls"], list):
                inst = [u for u in data["urls"] if isinstance(u, str)]
        inst = [u.rstrip("/") for u in inst if isinstance(u, str) and u.startswith("http")]
        inst = list(dict.fromkeys(inst))
        random.shuffle(inst)
        inst = [
            u for u in inst
            if ".onion" not in u
            and u.startswith("https://")
            and ".within.website" not in u
        ]
        prime = [u for u in SEARXNG_KNOWN_JSON_OK if u in inst]
        rest = [u for u in inst if u not in prime]
        inst = prime + rest
        if inst:
            logger_.debug(f"SearXNG public instances fetched: {len(inst)}")
            logger_.debug(f"Sample: {inst[:5]}")
            return inst
    except Exception as e:
        logger_.debug(f"searx.space fetch error: {e}")

    fallback = [u.rstrip("/") for u in (SEARXNG_KNOWN_JSON_OK + SEARXNG_FALLBACK_INSTANCES)]
    random.shuffle(fallback)
    logger_.info(f"Using fallback SearXNG instances: {fallback[:5]}...")
    return fallback

# -----------------------------------------------------------------------------
# SearXNG search core
# -----------------------------------------------------------------------------
def _search_on_instances(base_params: dict, query: str, num: int, engines: Iterable[str], logger_) -> list[dict]:
    engines = list(engines)
    engines_param = ",".join(engines)
    instances = fetch_searxng_instances(logger_=logger_)

    for base in instances:
        url = f"{base}/search"
        # try with engines
        try:
            params = dict(base_params, q=query, count=min(max(num, 1), 50), engines=engines_param)
            logger_.debug(f"SearXNG query on {base}: {query} | engines={engines_param}")
            data = _request_json(url, params=params)
            items = _extract_items(data)
            if items:
                return items
        except Exception as e:
            logger_.debug(f"SearXNG error on {base} (with engines): {e}")

        # retry without engines
        try:
            params = dict(base_params, q=query, count=min(max(num, 1), 50))
            data = _request_json(url, params=params)
            items = _extract_items(data)
            if items:
                logger_.debug(f"Succeeded on {base} without engines param.")
                return items
        except Exception as e2:
            logger_.debug(f"SearXNG error on {base} (no engines): {e2}")
            continue

    logger_.warning("All SearXNG instances failed.")
    return []

def _extract_items(data: dict) -> list[dict]:
    items = []
    for r in (data.get("results") or []):
        title = (r.get("title") or "").strip()
        url_ = (r.get("url") or "").strip()
        snippet = (r.get("content") or r.get("snippet") or "").strip()
        if not url_ or not title:
            continue
        items.append({"title": title, "url": url_, "snippet": snippet})
    return items

def search_searxng(query: str, num: int = 10, engines: Iterable[str] = SEARXNG_DEFAULT_ENGINES, logger_=logger) -> dict:
    items = _search_on_instances(SEARXNG_BASE_PARAMS, query, num, engines, logger_)
    return {"results": _rank_results(items, query, top=num)}

# -----------------------------------------------------------------------------
# Ranking / filtering
# -----------------------------------------------------------------------------
def _domain(url: str) -> str:
    try:
        from urllib.parse import urlparse
        h = urlparse(url).netloc.lower()
        for p in ("www.", "m.", "amp."):
            if h.startswith(p):
                h = h[len(p):]
        return h
    except Exception:
        return ""

def _kw_density_score(title: str, snippet: str, q: str) -> float:
    q_tokens = [t for t in re.split(r"\W+", q.lower()) if t]
    hay = f"{title} {snippet}".lower()
    return sum(hay.count(t) for t in q_tokens) / max(len(q_tokens), 1)

def _host_score(host: str) -> float:
    if any(bad in host for bad in HARD_BLOCK):
        return -9.0
    s = 0.0
    if any(k in host for k in TRUSTED_FINANCE):
        s += 1.2
    if any(k in host for k in LOW_VALUE):
        s -= 0.8
    return s

def _finance_text_score(title: str, snippet: str) -> float:
    text = f"{title} {snippet}".lower()
    # Bonus si parenthèses type "XYZ (NASDAQ: XYZ)" / "XYZ (TSX: ...)"
    paren_bonus = 0.5 if re.search(r"\(([a-z]{2,6}[:\s\-])?[a-z]{1,6}\)", text) else 0.0
    base = sum(1.0 for k in FINANCE_KEYWORDS if k in text)
    return base * 0.2 + paren_bonus

def _rank_results(items: list[dict], q: str, top: int | None = None, finance_mode: bool | None = None) -> list[dict]:
    if finance_mode is None:
        finance_mode = bool(re.search(r"\b(competitors?|peers?|similar\s+stocks?|stock|earnings|results)\b", q, re.I))

    seen = set()
    ranked = []
    for it in items:
        url = it["url"]
        host = _domain(url)
        title = it["title"].strip()
        snippet = it.get("snippet", "").strip()

        # hard block
        if any(host.endswith(b) or b in host for b in HARD_BLOCK):
            continue

        key = (host, title.lower())
        if key in seen:
            continue
        seen.add(key)

        score = 0.0
        score += _kw_density_score(title, snippet, q)
        score += _host_score(host)
        if finance_mode:
            score += _finance_text_score(title, snippet)

        ranked.append((score, it))

    ranked.sort(key=lambda x: x[0], reverse=True)
    out = [it for _, it in ranked]
    return out[:top] if top else out

# -----------------------------------------------------------------------------
# Finance-first orchestrator
# -----------------------------------------------------------------------------
def _finance_queries(symbol: str | None, company: str | None, topic: str | None) -> list[str]:
    s = (symbol or "").strip()
    c = (company or "").strip()
    t = (topic or "").strip()

    qs = []
    # très direct autour d’un titre
    if c:
        qs += [
            f'{c} stock news',
            f'{c} earnings OR results OR guidance',
            f'{c} investor relations news',
            f'{c} SEC filing OR 6-K OR 8-K OR press release',
        ]
    if s:
        qs += [
            f'{s} stock news',
            f'{s} earnings OR results OR guidance',
            f'{s} peers OR competitors',
        ]
    # macro / économie générique
    if t:
        qs += [t, f'{t} latest news', f'{t} market news']
    return qs or [t or "markets news US"]

def finance_search(
    symbol: str | None = None,
    company: str | None = None,
    topic: str | None = None,
    num: int = 12,
    logger_=logger
) -> dict:
    """
    Cherche des articles liés à un ticker, une société ou un sujet économique.
    Priorise “news”, booste domaines finance, filtre bruit, période ~1 semaine si supportée.
    """
    queries = _finance_queries(symbol, company, topic)
    agg: List[dict] = []

    # 1) SearXNG (news category + engines)
    engines = SEARXNG_DEFAULT_ENGINES
    for q in queries:
        items = _search_on_instances(SEARXNG_NEWS_PARAMS, q, num=min(10, num), engines=engines, logger_=logger_)
        agg.extend(items)
    ranked = _rank_results(agg, " ".join(queries), top=num, finance_mode=True)

    # 2) si trop peu, wikipedia + general
    if len(ranked) < max(3, num // 3):
        logger_.debug("Few finance results; trying wikipedia engine + general category.")
        wiki_items = _search_on_instances({**SEARXNG_BASE_PARAMS, "categories": "general"},
                                          f"{company or symbol} company", num=min(10, num),
                                          engines=["wikipedia"], logger_=logger_)
        agg.extend(wiki_items)
        ranked = _rank_results(agg, " ".join(queries), top=num, finance_mode=True)

    # 3) fallback Serper/Tavily si dispo
    if len(ranked) < max(3, num // 2):
        try:
            from secrets_local import SERPER_API_KEY as _SERPER  # type: ignore
        except Exception:
            _SERPER = os.getenv("SERPER_API_KEY", "")
        try:
            from secrets_local import TAVILY_API_KEY as _TAVILY  # type: ignore
        except Exception:
            _TAVILY = os.getenv("TAVILY_API_KEY", "")

        if _SERPER:
            try:
                agg.extend(_search_serper(queries, num=min(10, num), api_key=_SERPER, logger_=logger_))
            except Exception as e:
                logger_.debug(f"Serper error: {e}")

        if len(agg) < max(3, num // 2) and _TAVILY:
            try:
                agg.extend(_search_tavily(queries, num=min(10, num), api_key=_TAVILY, logger_=logger_))
            except Exception as e:
                logger_.debug(f"Tavily error: {e}")

        ranked = _rank_results(agg, " ".join(queries), top=num, finance_mode=True)

    return {"results": ranked}

# -----------------------------------------------------------------------------
# Optional: Serper / Tavily light clients
# -----------------------------------------------------------------------------
def _search_serper(queries: list[str], num: int, api_key: str, logger_=logger) -> list[dict]:
    headers = {"X-API-KEY": api_key, "Content-Type": "application/json"}
    out = []
    for q in queries:
        try:
            r = requests.post(
                "https://google.serper.dev/search",
                headers=headers,
                json={"q": q, "num": min(10, num), "gl": "us", "hl": "en"},
                timeout=(8, 20),
            )
            r.raise_for_status()
            d = r.json()
            for it in (d.get("organic") or []):
                title = it.get("title", "")
                url = it.get("link", "")
                snippet = it.get("snippet", "")
                if title and url:
                    out.append({"title": title, "url": url, "snippet": snippet})
        except Exception as e:
            logger_.debug(f"Serper request failed: {e}")
    return out

def _search_tavily(queries: list[str], num: int, api_key: str, logger_=logger) -> list[dict]:
    out = []
    for q in queries:
        try:
            r = requests.post(
                "https://api.tavily.com/search",
                headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"},
                json={"api_key": api_key, "query": q, "max_results": min(10, num), "search_depth": "basic"},
                timeout=(8, 20),
            )
            r.raise_for_status()
            d = r.json()
            for it in (d.get("results") or []):
                title = it.get("title", "")
                url = it.get("url", "")
                snippet = it.get("content", "")
                if title and url:
                    out.append({"title": title, "url": url, "snippet": snippet})
        except Exception as e:
            logger_.debug(f"Tavily request failed: {e}")
    return out

# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
def _set_log_level(level_str: str):
    level = getattr(logging, level_str.upper(), logging.INFO)
    root = logging.getLogger()
    if not root.handlers:
        logging.basicConfig(
            level=level,
            format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
            datefmt="%H:%M:%S",
        )
    else:
        root.setLevel(level)
    logger.setLevel(level)

def main():
    parser = argparse.ArgumentParser(description="Finance-first web navigator (SearXNG + optional Serper/Tavily).")
    sub = parser.add_subparsers(dest="cmd", required=True)

    # raw searxng
    p_raw = sub.add_parser("searx", help="Recherche brute SearXNG")
    p_raw.add_argument("--q", required=True, help="Query")
    p_raw.add_argument("--n", type=int, default=10, help="Nombre de résultats")
    p_raw.add_argument("--log", default="INFO", choices=["DEBUG","INFO","WARNING","ERROR","CRITICAL"])

    # finance
    p_fin = sub.add_parser("finance", help="Recherche finance/news (ticker / société / sujet)")
    p_fin.add_argument("--symbol", help="Ticker (ex: AAPL, NGD.TO)")
    p_fin.add_argument("--company", help="Nom société (ex: New Gold)")
    p_fin.add_argument("--topic", help="Sujet macro (ex: US inflation)")
    p_fin.add_argument("--n", type=int, default=12)
    p_fin.add_argument("--log", default="INFO", choices=["DEBUG","INFO","WARNING","ERROR","CRITICAL"])

    args = parser.parse_args()
    _set_log_level(getattr(args, "log", "INFO"))

    if args.cmd == "searx":
        out = search_searxng(query=args.q, num=args.n, engines=SEARXNG_DEFAULT_ENGINES, logger_=logger)
        print(json.dumps(out, indent=2, ensure_ascii=False))
        return

    if args.cmd == "finance":
        out = finance_search(symbol=args.symbol, company=args.company, topic=args.topic, num=args.n, logger_=logger)
        print(json.dumps(out, indent=2, ensure_ascii=False))
        return

if __name__ == "__main__":
    main()
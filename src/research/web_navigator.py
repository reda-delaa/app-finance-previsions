from __future__ import annotations

# web_navigator.py — finance-first web search (SearXNG + optional Serper/Tavily)
# ✅ Fixes: no-redirect JSON-only requests, engines optional, POST fallback,
#           instance preflight+health cache, 429 cooldown (Retry-After), blacklist on 30x/403,
#           disque: cache des instances "bonnes".

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
    "time_range": "week",  # ignoré par certaines instances → ok
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
    "earnings","results","guidance","outlook","quarter","q1","q2","q3","q4","fy",
    "revenue","sales","profit","margin","ebitda","eps","forecast","beat","miss",
    "offering","secondary","placement","buyback","dividend","downgrade","upgrade",
    "merger","acquisition","m&a","spin-off","spinoff","ipo","debt","notes",
    "inflation","cpi","ppi","jobs","fomc","rate","yields","housing","manufacturing",
}

# -----------------------------------------------------------------------------
# Instance health cache / blacklist / cooldown (in-memory + disque)
# -----------------------------------------------------------------------------
_INSTANCE_HEALTH: Dict[str, Dict[str, float]] = {}  # {base: {"score": float, "last": ts}}
_BLACKLIST_UNTIL: Dict[str, float] = {}             # {base: ts_until}
BLACKLIST_TTL = 60 * 30  # 30 min
HEALTH_DECAY_SEC = 60 * 60  # 1h

# Cooldown 429
_COOLDOWN_UNTIL: Dict[str, float] = {}              # {base: ts_until}
RATE_LIMIT_COOLDOWN_DEFAULT = 180                   # 3 min

# Cache disque des instances qui ont déjà bien répondu
CACHE_PATH = os.path.expanduser("~/.cache/web_navigator_searx.json")

def _load_cached_good_instances() -> list[str]:
    try:
        with open(CACHE_PATH, "r") as f:
            data = json.load(f)
        arr = data.get("good", [])
        return [u.rstrip("/") for u in arr if isinstance(u, str)]
    except Exception:
        return []

def _remember_good_instance(base: str):
    try:
        os.makedirs(os.path.dirname(CACHE_PATH), exist_ok=True)
        cur = _load_cached_good_instances()
        if base in cur:
            cur.remove(base)
        cur.insert(0, base)
        cur = cur[:8]
        with open(CACHE_PATH, "w") as f:
            json.dump({"good": cur}, f)
    except Exception:
        pass

def _merge_cached_first(instances: list[str]) -> list[str]:
    seen, out = set(), []
    for u in _load_cached_good_instances() + instances:
        u = u.rstrip("/")
        if u and u not in seen:
            out.append(u)
            seen.add(u)
    return out

def _in_cooldown(base: str) -> bool:
    until = _COOLDOWN_UNTIL.get(base, 0)
    if until and until > time.time():
        return True
    if until and until <= time.time():
        _COOLDOWN_UNTIL.pop(base, None)
    return False

def _cooldown(base: str, seconds: float | None):
    _COOLDOWN_UNTIL[base] = time.time() + (seconds or RATE_LIMIT_COOLDOWN_DEFAULT)
    _boost_health(base, -0.2)

def _health_score(base: str) -> float:
    h = _INSTANCE_HEALTH.get(base)
    if not h:
        return 0.0
    # léger decay dans le temps pour éviter le sur-apprentissage
    age = max(time.time() - h.get("last", 0), 0)
    decay = max(0.5, 1.0 - age / (HEALTH_DECAY_SEC * 2))
    return h.get("score", 0.0) * decay

def _boost_health(base: str, delta: float):
    cur = _INSTANCE_HEALTH.get(base, {"score": 0.0, "last": 0.0})
    cur["score"] = min(5.0, max(-5.0, cur.get("score", 0.0) + delta))
    cur["last"] = time.time()
    _INSTANCE_HEALTH[base] = cur

def _blacklisted(base: str) -> bool:
    until = _BLACKLIST_UNTIL.get(base, 0)
    if until and until > time.time():
        return True
    if until and until <= time.time():
        _BLACKLIST_UNTIL.pop(base, None)
    return False

def _blacklist(base: str, ttl: float = BLACKLIST_TTL):
    _BLACKLIST_UNTIL[base] = time.time() + ttl
    _boost_health(base, -0.8)

# -----------------------------------------------------------------------------
# HTTP helpers (fixed)
# -----------------------------------------------------------------------------
def _random_ua() -> str:
    v = ".".join(str(random.randint(100, 125)) for _ in range(3))
    chrome = f"Chrome/{v}"
    safari = f"Safari/{random.randint(600, 620)}.{random.randint(1, 50)}"
    return f"Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) {chrome} {safari}"

_JSON_HEADERS = {
    "User-Agent": _random_ua(),
    "Accept": "application/json",
    "Accept-Language": "en-US,en;q=0.7,fr;q=0.5",
    "Cache-Control": "no-cache",
    "Pragma": "no-cache",
}

class RedirectError(Exception): ...
class NonJSONError(Exception): ...
class ForbiddenError(Exception): ...
class TooManyRequestsError(Exception):
    def __init__(self, retry_after: int | None = None):
        super().__init__("429 Too Many Requests")
        self.retry_after = retry_after or 0

def _ensure_json_response(r: requests.Response) -> dict:
    if r.status_code in (301, 302, 303, 307, 308):
        raise RedirectError(f"{r.status_code} redirect to {r.headers.get('Location')}")
    if r.status_code == 403:
        raise ForbiddenError("403 Forbidden")
    if r.status_code == 429:
        ra = r.headers.get("Retry-After")
        try:
            ra_val = int(ra) if ra is not None else 0
        except Exception:
            ra_val = 0
        raise TooManyRequestsError(retry_after=ra_val)

    r.raise_for_status()
    ctype = (r.headers.get("Content-Type") or "").lower()
    # Certaines instances ne mettent pas le bon header mais renvoient du JSON valide
    txt = (r.text or "").lstrip()
    if "application/json" not in ctype and not (txt.startswith("{") or txt.startswith("[")):
        raise NonJSONError(f"non-json response (Content-Type={ctype or 'unknown'})")
    return json.loads(txt) if "application/json" not in ctype else r.json()

def _request_json_get(url: str, params: dict | None = None, timeout=SEARXNG_TIMEOUT) -> dict:
    r = requests.get(url, params=params, headers=_JSON_HEADERS, timeout=timeout, allow_redirects=False)
    return _ensure_json_response(r)

def _request_json_post(url: str, data: dict | None = None, timeout=SEARXNG_TIMEOUT) -> dict:
    headers = dict(_JSON_HEADERS)
    headers["Content-Type"] = "application/x-www-form-urlencoded"
    r = requests.post(url, data=data or {}, headers=headers, timeout=timeout, allow_redirects=False)
    return _ensure_json_response(r)

# -----------------------------------------------------------------------------
# SearXNG instance discovery (+preflight)
# -----------------------------------------------------------------------------
def fetch_searxng_instances(logger_=None) -> list[str]:
    if logger_ is None:
        logger_ = logger
    # Prefer local instance if provided via env
    local = os.getenv("SEARXNG_LOCAL_URL", "").strip().rstrip('/')
    prefer_local: list[str] = [local] if local.startswith("http") else []

    try:
        data = _request_json_get(SEARXNG_FETCH_URL, timeout=(6, 12))
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
        inst = prefer_local + prime + rest
        if inst:
            logger_.debug(f"SearXNG public instances fetched: {len(inst)}")
            logger_.debug(f"Sample: {inst[:5]}")
    except Exception as e:
        logger_.debug(f"searx.space fetch error: {e}")
        inst = []

    if not inst:
        fallback = [u.rstrip("/") for u in (prefer_local + SEARXNG_KNOWN_JSON_OK + SEARXNG_FALLBACK_INSTANCES)]
        random.shuffle(fallback)
        logger_.info(f"Using fallback SearXNG instances: {fallback[:5]}...")
        inst = fallback

    # Ordonner par santé + préférer celles du cache
    inst_scored = sorted(inst, key=lambda u: _health_score(u), reverse=True)
    inst_scored = _merge_cached_first(inst_scored)

    tested: list[str] = []
    budget = 10  # nombre max d'instances à ping
    for base in inst_scored:
        if len(tested) >= 2:  # 1–2 OK suffisent
            break
        if _blacklisted(base) or _in_cooldown(base):
            continue
        if _preflight_instance(base, logger_):
            tested.append(base)
        budget -= 1
        if budget <= 0:
            break

    # Si rien, renvoyer un petit lot brut non testés (hors cooldown/blacklist)
    if not tested:
        fallback_pick = [u for u in inst_scored if not _blacklisted(u) and not _in_cooldown(u)][:6]
        return fallback_pick
    return tested

def _preflight_instance(base: str, logger_) -> bool:
    """GET minimal /search?format=json&q=q&count=1, sans redirect & JSON only.
       429 => cooldown; 403/redirect => blacklist; non-JSON => demi-blacklist."""
    url = f"{base.rstrip('/')}/search"
    try:
        params = {"format": "json", "q": "q", "count": 1, "language": "en"}
        r = requests.get(url, params=params, headers=_JSON_HEADERS, timeout=(4, 8), allow_redirects=False)
        data = _ensure_json_response(r)
        if isinstance(data, dict) and "results" in data:
            _boost_health(base, +0.6)
            return True
    except TooManyRequestsError as e:
        logger_.debug(f"Preflight 429 on {base} (Retry-After={e.retry_after})")
        _cooldown(base, e.retry_after or RATE_LIMIT_COOLDOWN_DEFAULT)
        return False
    except RedirectError as e:
        logger_.debug(f"Preflight redirect on {base}: {e}")
        _blacklist(base, ttl=BLACKLIST_TTL)
    except ForbiddenError:
        logger_.debug(f"Preflight 403 on {base}")
        _blacklist(base, ttl=BLACKLIST_TTL)
    except NonJSONError as e:
        logger_.debug(f"Preflight non-JSON on {base}: {e}")
        _blacklist(base, ttl=BLACKLIST_TTL / 2)
    except Exception as e:
        logger_.debug(f"Preflight error on {base}: {e}")
        _boost_health(base, -0.3)
    return False

# -----------------------------------------------------------------------------
# SearXNG search core (fixed ordering, POST fallback, cooldown/blacklist)
# -----------------------------------------------------------------------------
def _search_on_instances(base_params: dict, query: str, num: int, engines: Iterable[str], logger_) -> list[dict]:
    engines = list(engines or [])
    engines_param = ",".join(engines) if engines else ""
    instances = _merge_cached_first(fetch_searxng_instances(logger_=logger_))

    for base in instances:
        if _blacklisted(base) or _in_cooldown(base):
            logger_.debug(f"Skipping unavailable instance: {base}")
            continue

        url = f"{base}/search"
        modes = (("GET", False), ("GET", True), ("POST", False))
        for method, with_engines in modes:
            try:
                params = dict(base_params, q=query, count=min(max(num, 1), 50))
                if with_engines and engines_param:
                    params["engines"] = engines_param

                if method == "GET":
                    data = _request_json_get(url, params=params)
                else:
                    data = _request_json_post(url, data=params)

                items = _extract_items(data)
                if items:
                    _boost_health(base, +0.4)
                    _remember_good_instance(base)
                    if method == "POST":
                        logger_.debug(f"Succeeded on {base} via POST.")
                    return items

            except TooManyRequestsError as e:
                logger_.debug(f"SearXNG 429 on {base} ({'with engines' if with_engines else 'no engines'}; Retry-After={e.retry_after})")
                _cooldown(base, e.retry_after or RATE_LIMIT_COOLDOWN_DEFAULT)

            except RedirectError as e:
                logger_.debug(f"SearXNG redirect on {base}: {e}")
                _blacklist(base)

            except ForbiddenError:
                logger_.debug(f"SearXNG 403 on {base}")
                _blacklist(base)

            except NonJSONError as e:
                logger_.debug(f"SearXNG non-JSON on {base}: {e}")
                _blacklist(base, ttl=BLACKLIST_TTL / 2)

            except Exception as e:
                logger_.debug(f"SearXNG error on {base}: {e}")
                _boost_health(base, -0.1)

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

    # 1) SearXNG (news category, essai GET sans engines puis avec, puis POST)
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
            from src.secrets_local import get_key  # type: ignore
            _SERPER = get_key("SERPER_API_KEY") or ""
            _TAVILY = get_key("TAVILY_API_KEY") or ""
        except Exception:
            _SERPER = os.getenv("SERPER_API_KEY", "")
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

# src/ingestion/finviz_client.py
# -*- coding: utf-8 -*-
"""
Finviz client (scrape + normalize) — conçu pour fonctionner conjointement avec finnews:
- Company snapshot (ratios, ownership, short interest, profile…)
- Insider trades / Latest filings
- Options chain (list view, by expiry)
- News (global and per ticker)
- Futures dashboards (quotes/performance/charts) across categories

Robustesse:
- Cache HTML local (cache/finviz/{sha1}.html)
- User-Agent rotation, backoff, retries, timeout
- Parse tolérant (BS4), défensif (None-safe), schéma JSON stable

Sorties (schemas JSON-friendly):
- Company: {ticker, name, sector, industry, country, market, metrics{...}, ownership{...}, short{...}, links{...}}
- Insider: [{insider, relation, date, transaction, shares, price, value, link}]
- Filings: [{title, date, form, link}]
- Options: {ticker, expiry, calls:[...], puts:[...]}
- News: [{ts, source, title, link, tickers:[], tags:{...}, summary:...}]
- Futures: {category, timeframe, rows: [{symbol, name, price, change, pct, vol, oi, link}]}

Auteur: toi
"""

from __future__ import annotations
import os, re, sys, json, time, random, hashlib, datetime as dt
from dataclasses import dataclass, asdict, field
from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path

# --- HTTP / Parsing
try:
    import requests
except Exception as e:
    raise RuntimeError("finviz_client requires `requests`. pip install requests") from e

try:
    from bs4 import BeautifulSoup
except Exception as e:
    raise RuntimeError("finviz_client requires `beautifulsoup4`. pip install beautifulsoup4 lxml") from e

# --- Optional
try:
    from tqdm import tqdm
except Exception:
    def tqdm(it, **kw): return it

# =========================
# Config & helper utilities
# =========================

BASE = "https://finviz.com"
CACHE_DIR = Path("cache") / "finviz"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

UA_ROTATION = [
    # Quelques UA plausibles
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:128.0) Gecko/20100101 Firefox/128.0",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/127.0.6533.72 Safari/537.36",
]

DEFAULT_TIMEOUT = float(os.getenv("FINVIZ_TIMEOUT", "15"))
RETRIES = int(os.getenv("FINVIZ_RETRIES", "2"))
BACKOFF = float(os.getenv("FINVIZ_BACKOFF", "1.4"))  # multiplicatif
SLEEP_JITTER = (0.3, 1.1)

def _sha1(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8", errors="ignore")).hexdigest()

def _cache_path(url: str) -> Path:
    return CACHE_DIR / f"{_sha1(url)}.html"

def _ua() -> str:
    return random.choice(UA_ROTATION)

def _sleep_jitter():
    time.sleep(random.uniform(*SLEEP_JITTER))

def _get(url: str, params: Optional[Dict[str, Any]] = None, use_cache=True) -> str:
    """GET with cache+retry. Returns HTML text (str)."""
    params = params or {}
    # Build full URL with querystring for cache key
    if params:
        qs = "&".join(f"{k}={requests.utils.quote(str(v))}" for k,v in params.items())
        full = f"{url}?{qs}"
    else:
        full = url

    cpath = _cache_path(full)
    if use_cache and cpath.exists():
        try:
            return cpath.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            pass

    last_err = None
    wait = 0.2
    for att in range(1, RETRIES + 2):
        try:
            resp = requests.get(full, headers={"User-Agent": _ua(), "Referer": BASE},
                                timeout=DEFAULT_TIMEOUT)
            if resp.status_code == 200 and ("captcha" not in resp.text.lower()):
                html = resp.text
                try:
                    cpath.write_text(html, encoding="utf-8")
                except Exception:
                    pass
                _sleep_jitter()
                return html
            # soft-block or 403/429
            last_err = f"HTTP {resp.status_code}"
        except Exception as e:
            last_err = f"{type(e).__name__}: {e}"
        time.sleep(wait)
        wait *= BACKOFF
    raise RuntimeError(f"Finviz GET failed: {full} | last_err={last_err}")

def _soup(html: str) -> BeautifulSoup:
    return BeautifulSoup(html, "lxml") if "lxml" in sys.modules else BeautifulSoup(html, "html.parser")

def _text(el) -> str:
    try:
        return re.sub(r"\s+", " ", el.get_text(" ", strip=True)).strip()
    except Exception:
        return ""

def _to_float(x: Optional[str]) -> Optional[float]:
    if x is None: return None
    s = x.replace(",", "").replace("%", "").strip()
    if s in ("", "-", "N/A"): return None
    try:
        return float(s)
    except Exception:
        # handle K/M/B suffix
        m = re.match(r"^([0-9\.]+)\s*([KMB])$", s, flags=re.I)
        if not m: return None
        v = float(m.group(1))
        suf = m.group(2).upper()
        mult = {"K":1e3,"M":1e6,"B":1e9}.get(suf,1)
        return v*mult

def _parse_iso(ts: str) -> Optional[str]:
    ts = (ts or "").strip()
    if not ts: return None
    # finviz often uses like "Sep 12, 2025" or "09:45AM"
    try:
        for fmt in ("%b %d, %Y", "%Y-%m-%d", "%m/%d/%Y", "%H:%M%p", "%I:%M%p"):
            try:
                d = dt.datetime.strptime(ts, fmt)
                if d.year < 1975:
                    d = d.replace(year=dt.datetime.now(dt.timezone.utc).year)
                return d.replace(tzinfo=dt.timezone.utc).isoformat().replace("+00:00", "Z")
            except Exception:
                continue
    except Exception:
        pass
    return None


# ======================
# Company snapshot block
# ======================

def company_snapshot(ticker: str, use_cache=True) -> Dict[str, Any]:
    """
    Parse finviz quote main page for snapshot ratios/ownership/short/links etc.
    Example URL: https://finviz.com/quote.ashx?t=AAPL&p=w
    """
    t = ticker.upper().strip()
    url = f"{BASE}/quote.ashx"
    html = _get(url, {"t": t, "p": "w"}, use_cache=use_cache)
    soup = _soup(html)

    out: Dict[str, Any] = {
        "ticker": t, "name": None, "sector": None, "industry": None, "country": None, "market": None,
    "metrics": {}, "ownership": {}, "short": {}, "links": {}, "profile": {}, "asof_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
    }

    # Header name + breadcrumb-ish
    try:
        h = soup.select_one("div.fullview-title")
        if h:
            out["name"] = _text(h)
    except Exception:
        pass

    try:
        crumbs = soup.select("div.fullview-links a")
        # contains 'Industry', 'Sector', 'Country' links
        for a in crumbs:
            txt = _text(a)
            href = (a.get("href") or "").lower()
            if "sector" in href:
                out["sector"] = txt
            elif "industry" in href:
                out["industry"] = txt
            elif "country" in href:
                out["country"] = txt
    except Exception:
        pass

    # Left “snapshot” table of key metrics
    def _grab_metric(key: str, val: str):
        if key and val:
            out["metrics"][key] = val

    try:
        # snapshot table has cells with <td class="snapshot-td2">Key</td><td class="snapshot-td2">Val</td>
        tds = soup.select("table.snapshot-table2 td")
        for i in range(0, len(tds), 2):
            k = _text(tds[i]) if i < len(tds) else ""
            v = _text(tds[i+1]) if i+1 < len(tds) else ""
            _grab_metric(k, v)
    except Exception:
        pass

    # Ownership/short metrics are in the same snapshot set but store numeric conversions
    # Common keys: "Shs Outstand", "Shs Float", "Insider Own", "Inst Own", "Short Float", "Short Ratio"
    def _maybe_num(key: str) -> Optional[float]:
        val = out["metrics"].get(key)
        return _to_float(val) if isinstance(val, str) else None

    out["ownership"] = {
        "insider_own_pct": _maybe_num("Insider Own"),
        "inst_own_pct": _maybe_num("Inst Own"),
        "insider_trans_pct": _maybe_num("Insider Trans"),
        "inst_trans_pct": _maybe_num("Inst Trans"),
        "shs_outstand": _maybe_num("Shs Outstand"),
        "shs_float": _maybe_num("Shs Float"),
    }
    out["short"] = {
        "short_float_pct": _maybe_num("Short Float"),
        "short_ratio": _maybe_num("Short Ratio"),
    }

    # Links (Latest filings, Insider, Options…)
    try:
        linkbar = soup.select("a.tab-link")
        for a in linkbar:
            txt = _text(a).lower()
            href = a.get("href") or ""
            if "latest filings" in txt:
                out["links"]["filings"] = BASE + "/" + href.lstrip("/")
            elif txt.startswith("options") or "ty=oc" in href:
                out["links"]["options"] = BASE + "/" + href.lstrip("/")
            elif "insider" in txt:
                out["links"]["insider"] = BASE + "/" + href.lstrip("/")
            elif "short interest" in txt:
                out["links"]["short_interest"] = BASE + "/" + href.lstrip("/")
    except Exception:
        pass

    # Basic profile text block (if shown)
    try:
        prof = soup.select_one("td.fullview-profile")
        if prof:
            out["profile"]["text"] = _text(prof)
    except Exception:
        pass

    return out


# ==================
# Insider & Filings
# ==================

def insider_trades(ticker: str, use_cache=True) -> List[Dict[str, Any]]:
    """Parse insider trades table (if present)."""
    t = ticker.upper().strip()
    url = f"{BASE}/quote.ashx"
    html = _get(url, {"t": t, "p": "i"}, use_cache=use_cache)  # p=i tab often shows insider
    soup = _soup(html)

    rows: List[Dict[str, Any]] = []
    table = soup.select_one("table.body-table") or soup.select_one("table#insider")
    if not table:
        return rows

    for tr in table.select("tr")[1:]:
        tds = tr.select("td")
        if len(tds) < 8:
            continue
        try:
            link = ""
            a = tds[0].select_one("a")
            if a:
                link = BASE + "/" + (a.get("href") or "").lstrip("/")
            rows.append({
                "insider": _text(tds[0]),
                "relation": _text(tds[1]),
                "last_date": _text(tds[2]),
                "transaction": _text(tds[3]),
                "shares": _to_float(_text(tds[4])),
                "price": _to_float(_text(tds[5])),
                "value": _to_float(_text(tds[6])),
                "link": link,
            })
        except Exception:
            continue
    return rows

def latest_filings(ticker: str, use_cache=True) -> List[Dict[str, Any]]:
    """Parse 'Latest Filings' tab list."""
    t = ticker.upper().strip()
    url = f"{BASE}/quote.ashx"
    html = _get(url, {"t": t, "p": "f"}, use_cache=use_cache)  # p=f filings tab
    soup = _soup(html)

    out: List[Dict[str, Any]] = []
    table = soup.select_one("table.body-table")
    if not table:
        return out
    for tr in table.select("tr")[1:]:
        tds = tr.select("td")
        if len(tds) < 4:
            continue
        title = _text(tds[0])
        form = _text(tds[1])
        date = _text(tds[2])
        a = tds[0].select_one("a")
        link = BASE + "/" + (a.get("href") or "").lstrip("/") if a else ""
        out.append({"title": title, "form": form, "date": date, "link": link})
    return out


# =============
# Options chain
# =============

def options_chain(ticker: str, expiry: Optional[str] = None, use_cache=True) -> Dict[str, Any]:
    """
    Fetch options list view (calls/puts) for a given expiry if provided.
    - expiry format like '2025-09-19' (as seen in finviz)
    - returns {ticker, expiry, calls:[...], puts:[...]}
    """
    t = ticker.upper().strip()
    params = {"t": t, "p": "w", "ty": "oc"}
    if expiry:
        params["e"] = expiry
        params["ov"] = "list_date"
    html = _get(f"{BASE}/quote.ashx", params, use_cache=use_cache)
    soup = _soup(html)

    def _parse_side(hdr: str) -> List[Dict[str, Any]]:
        # locate table under "Calls" / "Puts" heading
        out: List[Dict[str, Any]] = []
        h = None
        for el in soup.select("div.table-top, h2, h3, span"):
            if _text(el).lower() == hdr:
                h = el
                break
        if not h:
            return out
        table = h.find_next("table")
        if not table:
            return out
        # expected header names
        head = [ _text(th).lower() for th in table.select("tr th") ]
        idx = {name:i for i,name in enumerate(head)}
        for tr in table.select("tr")[1:]:
            tds = tr.select("td")
            if not tds: continue
            def g(k): 
                j = idx.get(k, -1)
                return _text(tds[j]) if 0 <= j < len(tds) else ""
            try:
                out.append({
                    "contract": g("contract name") or g("name"),
                    "strike": _to_float(g("strike")),
                    "last": _to_float(g("last close") or g("last")),
                    "bid": _to_float(g("bid")),
                    "ask": _to_float(g("ask")),
                    "change": _to_float(g("change $") or g("change")),
                    "pct": _to_float(g("change %").replace("%","")) if g("change %") else None,
                    "volume": _to_float(g("volume")),
                    "open_interest": _to_float(g("open int.") or g("open interest")),
                })
            except Exception:
                continue
        return out

    calls = _parse_side("calls")
    puts  = _parse_side("puts")

    # try detect expiry label on page
    found_expiry = expiry
    try:
        sel = soup.select_one("select#expiry") or soup.find("select")
        if sel:
            opt = sel.find("option", selected=True) or sel.find("option")
            if opt:
                found_expiry = (opt.get("value") or _text(opt)) or expiry
    except Exception:
        pass

    return {"ticker": t, "expiry": found_expiry, "calls": calls, "puts": puts}


# =====
# News
# =====

def news(ticker: Optional[str] = None, use_cache=True, limit: int = 200) -> List[Dict[str, Any]]:
    """
    Finviz news feed (global or per ticker)
    - Global: https://finviz.com/news.ashx?v=2
    - Per ticker: https://finviz.com/quote.ashx?t=AAPL (news panel)
    """
    if ticker:
        html = _get(f"{BASE}/quote.ashx", {"t": ticker.upper()}, use_cache=use_cache)
        soup = _soup(html)
        blk = soup.select_one("#news-table") or soup.find("table", {"id": "news-table"})
        rows: List[Dict[str, Any]] = []
        if not blk:
            return rows
        for a in blk.select("a")[:limit]:
            href = a.get("href") or ""
            title = _text(a)
            ts = a.find_previous("td").get_text(strip=True) if a.find_previous("td") else ""
            dom = re.sub(r"^https?://(www\.)?", "", (href or "").split("/")[2]) if "://" in href else ""
            rows.append({
                "ts": ts, "source": dom, "title": title, "link": href,
                "sent": 0.0, "tickers": [ticker.upper()], "summary": None, "tags": {}
            })
        return rows

    # global news page
    html = _get(f"{BASE}/news.ashx", {"v": "2"}, use_cache=use_cache)
    soup = _soup(html)
    rows: List[Dict[str, Any]] = []
    for tr in soup.select("table.news tr")[:limit]:
        a = tr.find("a")
        if not a: continue
        link = a.get("href") or ""
        title = _text(a)
        tds = tr.select("td")
        ts  = _text(tds[0]) if tds else ""
        src = ""
        if len(tds) > 1:
            src = _text(tds[1]).split(" ",1)[0]
        rows.append({"ts": ts, "source": src, "title": title, "link": link, "sent": 0.0, "tickers": [], "summary": None, "tags": {}})
    return rows


# =========
# Futures
# =========

FUTURES_CATEGORIES = [
    "Indices","Energy","Metals","Bonds","Currencies","Softs","Meats","Grains","Crypto"
]

def futures(category: Optional[str] = None, timeframe: str = "w", tab: str = "quotes",
            use_cache=True) -> Dict[str, Any]:
    """
    Scrape futures dashboard by category:
      https://finviz.com/futures_{tab}.ashx?p={timeframe}
    Then pick the tab for category (Metals, Energy, ...). We normalize each tile row.

    Returns: {category, timeframe, tab, rows:[{symbol,name,price,change,pct,vol,oi,link}]}
    """
    assert tab in ("quotes","performance","charts","maps"), "tab invalid"
    url = f"{BASE}/futures_{tab}.ashx"
    html = _get(url, {"p": timeframe}, use_cache=use_cache)
    soup = _soup(html)

    target_cat = (category or "").strip().lower()
    best_block = None
    blocks = soup.select("div#futures") or soup.select("div.content")
    # fallback: page groups are often marked with h3/h4 category headers
    groups = []
    for h in soup.find_all(["h2","h3","h4","div"], string=True):
        txt = _text(h).strip()
        if any(c.lower() == txt.lower() for c in FUTURES_CATEGORIES):
            groups.append((txt, h))
    if groups:
        if target_cat:
            for txt, h in groups:
                if txt.lower() == target_cat:
                    best_block = h.find_next("table") or h.find_next("div")
                    target_cat = txt
                    break
        else:
            # take first group if category unspecified
            target_cat = groups[0][0]
            best_block = groups[0][1].find_next("table") or groups[0][1].find_next("div")

    rows: List[Dict[str, Any]] = []
    if best_block:
        # table tiles
        table = best_block if best_block.name == "table" else best_block.find("table")
        if table:
            for tr in table.select("tr"):
                tds = tr.select("td")
                if len(tds) < 2:
                    continue
                try:
                    a = tds[0].select_one("a")
                    link = BASE + "/" + (a.get("href") or "").lstrip("/") if a else ""
                    name = _text(tds[0])
                    price = _to_float(_text(tds[1]))
                    chg = _to_float(_text(tds[2])) if len(tds) > 2 else None
                    pct = _to_float(_text(tds[3])) if len(tds) > 3 else None
                    vol = _to_float(_text(tds[4])) if len(tds) > 4 else None
                    oi  = _to_float(_text(tds[5])) if len(tds) > 5 else None
                    sym = re.findall(r"\(([A-Z0-9=^/]+)\)", name)
                    symbol = sym[-1] if sym else None
                    clean_name = re.sub(r"\s*\([^)]+\)\s*$", "", name).strip()
                    rows.append({
                        "symbol": symbol, "name": clean_name, "price": price,
                        "change": chg, "pct": pct, "vol": vol, "oi": oi, "link": link
                    })
                except Exception:
                    continue

    return {"category": target_cat or category, "timeframe": timeframe, "tab": tab, "rows": rows}


# ==========================
# High-level convenience API
# ==========================

@dataclass
class FinvizCompany:
    ticker: str
    name: Optional[str] = None
    sector: Optional[str] = None
    industry: Optional[str] = None
    country: Optional[str] = None
    market: Optional[str] = None
    metrics: Dict[str, Any] = field(default_factory=dict)
    ownership: Dict[str, Any] = field(default_factory=dict)
    short: Dict[str, Any] = field(default_factory=dict)
    profile: Dict[str, Any] = field(default_factory=dict)
    links: Dict[str, str] = field(default_factory=dict)
    insiders: List[Dict[str, Any]] = field(default_factory=list)
    filings: List[Dict[str, Any]] = field(default_factory=list)
    options: Optional[Dict[str, Any]] = None
    news: List[Dict[str, Any]] = field(default_factory=list)
    asof_utc: str = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

def fetch_company_all(ticker: str, expiry: Optional[str] = None,
                      include_news: bool = True, use_cache=True) -> FinvizCompany:
    snap = company_snapshot(ticker, use_cache=use_cache)
    ins  = insider_trades(ticker, use_cache=use_cache)
    fil  = latest_filings(ticker, use_cache=use_cache)
    opt  = options_chain(ticker, expiry=expiry, use_cache=use_cache) if expiry else None
    nws  = news(ticker=ticker, use_cache=use_cache, limit=100) if include_news else []

    return FinvizCompany(
        ticker=ticker.upper(),
        name=snap.get("name"),
        sector=snap.get("sector"),
        industry=snap.get("industry"),
        country=snap.get("country"),
        market=snap.get("market"),
        metrics=snap.get("metrics", {}),
        ownership=snap.get("ownership", {}),
        short=snap.get("short", {}),
        profile=snap.get("profile", {}),
        links=snap.get("links", {}),
        insiders=ins,
        filings=fil,
        options=opt,
        news=nws,
    )


# =====
# CLI
# =====

def _print_json(obj: Any):
    print(json.dumps(obj, ensure_ascii=False, indent=2))

def main():
    import argparse
    ap = argparse.ArgumentParser(description="Finviz scraper client")
    ap.add_argument("--ticker", type=str, help="Ticker for company mode")
    ap.add_argument("--expiry", type=str, default=None, help="Options expiry e.g. 2025-09-19")
    ap.add_argument("--futures", type=str, default=None, help="Futures category (Energy, Metals, ...). If set, scrape futures dashboard.")
    ap.add_argument("--timeframe", type=str, default="w", help="Futures timeframe (1,5,15,m,30,1H,D,W,M) — Finviz p= param (use 'w' for weekly)")
    ap.add_argument("--tab", type=str, default="quotes", help="Futures tab: quotes|performance|charts|maps")
    ap.add_argument("--global_news", action="store_true", help="Scrape global news page v=2")
    ap.add_argument("--no_cache", action="store_true", help="Bypass cache")
    args = ap.parse_args()

    use_cache = not args.no_cache

    if args.futures:
        out = futures(category=args.futures, timeframe=args.timeframe, tab=args.tab, use_cache=use_cache)
        _print_json(out); return

    if args.global_news:
        out = news(ticker=None, use_cache=use_cache)
        _print_json(out); return

    if args.ticker:
        out = fetch_company_all(args.ticker, expiry=args.expiry, include_news=True, use_cache=use_cache)
        _print_json(out.to_dict()); return

    ap.print_help()

if __name__ == "__main__":
    main()
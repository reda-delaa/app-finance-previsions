#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Monolithic Financial News Module (personal/pro-grade)
- Unified schema
- Multi-region & sector sources (FR/DE/US/CA/INTL/GEO)
- RSS/Atom ingest + normalize + dedup
- Enrichment: lang detect -> translate (noop fallback) -> summarize -> entities -> event/sector tags -> sentiment
- Search: full text + boolean (AND/OR/NOT) + filters (date/window/region/source/lang/sector/event/ticker)
- Signals: aggregated features per ticker/sector for modeling (phase4/phase5)
- Quick backtest: align daily features with returns (needs price loader hook)
- CLI: regions, window, query, company/aliases/ticker, per_source_cap, limit, jsonl/pretty

Author: you
"""

from __future__ import annotations
import os, re, sys, json, hashlib, argparse, datetime as dt
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional
from collections import defaultdict, Counter

from taxonomy.news_taxonomy import tag_sectors, classify_event, tag_geopolitics
from core.io_utils import write_jsonl

# ---- Optional external deps (graceful fallback) ----
try:
    import feedparser
except Exception as e:
    print("ERROR: feedparser is required. pip install feedparser", file=sys.stderr); raise

try:
    from bs4 import BeautifulSoup
except Exception:
    BeautifulSoup = None

try:
    from langdetect import detect as lang_detect
except Exception:
    def lang_detect(_text:str)->str:
        # naive fallback
        return "en"

try:
    from tqdm import tqdm
except Exception:
    def tqdm(it, **kw): return it

try:
    import numpy as np
    import pandas as pd
except Exception:
    np = None
    pd = None

# Optional sentiment fallback
try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    _VADER = SentimentIntensityAnalyzer()
except Exception:
    _VADER = None

# ---- Optional internal modules (use if present) ----
# nlp_enrich: expected to offer summarize(text), translate(text, target_lang), sentiment(text), ner(text) ...
try:
    from research.nlp_enrich import summarize, translate, sentiment, ner
    nlp_enrich = True
except Exception:
    summarize = None
    translate = None
    sentiment = None
    ner = None
    nlp_enrich = None

# news_taxonomy: expected to provide sector/event lexicons or tagger
try:
    from taxonomy.news_taxonomy import tag_sectors, classify_event, tag_geopolitics
    news_taxonomy = True
except Exception:
    tag_sectors = None
    classify_event = None
    tag_geopolitics = None
    news_taxonomy = None

# stock utilities (ticker mapping) if available
try:
    from core.stock_utils import guess_ticker  # assuming this exists or create it
    stock = True
except Exception:
    guess_ticker = None
    stock = None


# =========================
# Configuration & Taxonomy
# =========================

CACHE_DIR = os.path.join("cache", "news")
os.makedirs(CACHE_DIR, exist_ok=True)

SECTORS_LEX = {
    "banks": ["bank", "lender", "credit", "loan", "Basel", "capital ratio", "deposit"],
    "healthcare": ["pharma", "drug", "clinical", "FDA", "EMA", "hospital", "biotech"],
    "defense": ["defense", "missile", "NATO", "contractor", "aircraft", "frigate", "Rafale", "F-35"],
    "energy": ["oil", "gas", "OPEC", "Brent", "WTI", "pipeline", "refinery", "IEA"],
    "tech": ["AI", "semiconductor", "chip", "cloud", "SaaS", "cybersecurity", "GPU", "5G"],
    "materials": ["mine", "copper", "iron ore", "steel", "lithium", "nickel"],
    "industrials": ["logistics", "manufacturing", "aerospace", "rail", "shipping"],
    "consumer": ["retail", "e-commerce", "grocer", "apparel", "CPG", "consumer"],
    "utilities": ["grid", "electricity", "capacity", "storage", "renewables"],
    "real_estate": ["REIT", "housing", "mortgage", "occupancy", "rent"],
    "telecom": ["telecom", "5G", "spectrum", "carrier", "fiber"],
    "financials": ["asset manager", "broker", "exchange", "insurer", "mutual fund"],
}

EVENTS_LEX = {
    "earnings": ["earnings", "EPS", "guidance", "revenue", "buyback", "dividend"],
    "mna": ["acquire", "merger", "M&A", "takeover", "stake", "spin-off"],
    "sanctions": ["sanction", "embargo", "export control", "blacklist"],
    "geopolitics": ["war", "conflict", "strike", "drone", "border", "BRICS", "NATO", "EU Council"],
    "regulation": ["regulator", "SEC", "SEBI", "ACCC", "DoJ", "antitrust", "GDPR"],
    "macro": ["inflation", "CPI", "GDP", "unemployment", "PMI", "rate hike", "ECB", "Fed"],
    "product": ["launch", "product", "device", "service"],
    "energy_shock": ["oil price", "Brent", "WTI", "OPEC+", "supply cut", "price cap"],
}

# Regions and curated sources (kept concise; expand freely)
SOURCES: Dict[str, List[str]] = {
    # US / Business
    "US": [
        "https://www.cnbc.com/id/10001147/device/rss/rss.html",
        "https://www.marketwatch.com/rss/topstories",
        "https://www.reuters.com/markets/us/rss",
        "https://feeds.a.dj.com/rss/RSSMarketsMain.xml",
    ],
    # Canada
    "CA": [
        "https://www.theglobeandmail.com/feeds/business/",
        "https://financialpost.com/feed/",
        "https://www.bnnbloomberg.ca/polopoly_fs/BNNBloombergBusinessNews.xml",
        "https://www.reuters.com/markets/americas/rss",
    ],
    # France (presse financière)
    "FR": [
        "https://www.lesechos.fr/rss/finance-marches.xml",
        "https://www.boursorama.com/rss/flux-actus-boursorama.xml",
        "https://www.zonebourse.com/rss/flash/",
        "https://www.bfmtv.com/economie/rss/",
    ],
    # Germany (Fin/eco)
    "DE": [
        "https://www.handelsblatt.com/contentexport/feed/meistgelesen",
        "https://www.faz.net/rss/aktuell/wirtschaft/",
        "https://www.boersen-zeitung.de/rss",
        "https://www.manager-magazin.de/rss",
    ],
    # International & Geo
    "INTL": [
        "https://feeds.bbci.co.uk/news/business/rss.xml",
        "https://www.aljazeera.com/xml/rss/all.xml",
        "https://foreignpolicy.com/feed/",
        "https://www.economist.com/business/rss.xml",
        "https://www.oilprice.com/rss/main",
    ],
    # Africa Business slice
    "AFRICA": [
        "https://allafrica.com/tools/headlines/rdf/business/headlines.rdf",
        "https://allafrica.com/tools/headlines/rdf/eastafrica/headlines.rdf",
    ],
    # GEO = Geopolitics (alias)
    "GEO": [
        "https://feeds.bbci.co.uk/news/world/rss.xml",
        "https://www.aljazeera.com/xml/rss/all.xml",
        "https://foreignpolicy.com/feed/",
    ],
}

# Default language per source domain (heuristic)
DOMAIN_LANG_HINT = {
    "lesechos.fr": "fr",
    "boursorama.com": "fr",
    "zonebourse.com": "fr",
    "bfmtv.com": "fr",
    "handelsblatt.com": "de",
    "faz.net": "de",
    "boersen-zeitung.de": "de",
    "manager-magazin.de": "de",
}


# ===============
# Utility helpers
# ===============

def now_utc() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)

def parse_date(d: Any) -> Optional[dt.datetime]:
    if not d: return None
    # feedparser already returns parsed if possible
    if isinstance(d, dt.datetime):
        return d if d.tzinfo else d.replace(tzinfo=dt.timezone.utc)
    return None

def strip_html(txt: str) -> str:
    if not txt: return ""
    if BeautifulSoup:
        return BeautifulSoup(txt, "html.parser").get_text(" ", strip=True)
    return re.sub("<[^>]*>", " ", txt)

def sha256(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8", errors="ignore")).hexdigest()

def domain_of(url: str) -> str:
    m = re.match(r"^https?://([^/]+)/?", url or "")
    return m.group(1).lower() if m else ""

def guess_lang(text: str, url: str = "") -> str:
    dom = domain_of(url)
    if dom in DOMAIN_LANG_HINT:
        return DOMAIN_LANG_HINT[dom]
    try:
        return lang_detect(text[:5000] or "en")
    except Exception:
        return "en"

def bool_query_match(q: str, text: str) -> bool:
    """
    Simple boolean parser for queries like:
    "BRICS OR oil", "Ukraine AND sanctions", "chip NOT Nvidia"
    """
    if not q: return True
    # tokenize by AND/OR/NOT (uppercase words)
    # naive: split by OR first, then AND within, then NOT terms exclude
    expr = q
    # Normalize spacing
    text_low = text.lower()
    # Split by OR
    ors = [x.strip() for x in re.split(r"\bOR\b", expr, flags=re.I) if x.strip()]
    if not ors: ors = [expr]
    for part in ors:
        # AND segments
        ands = [y.strip() for y in re.split(r"\bAND\b", part, flags=re.I) if y.strip()]
        ok_and = True
        for a in ands:
            # Handle NOT within segment: "foo NOT bar"
            toks = [z.strip() for z in re.split(r"\bNOT\b", a, flags=re.I)]
            must = toks[0]
            nots = toks[1:] if len(toks) > 1 else []
            if must and must.lower() not in text_low:
                ok_and = False
                break
            neg_hit = any(n and n.lower() in text_low for n in nots)
            if neg_hit:
                ok_and = False
                break
        if ok_and:
            return True
    return False

def _matches_tickers(item, tickers):
    if not tickers:
        return True
    tset = {t.upper() for t in tickers}
    # 1) champ structuré s'il existe
    for t in getattr(item, "tickers", []) or []:
        if t and t.upper() in tset:
            return True
    # 2) fallback sur texte
    hay = " ".join([
        getattr(item, "title", "") or "",
        getattr(item, "summary", "") or "",
        getattr(item, "raw_text", "") or "",
    ]).upper()
    return any(re.search(rf"\b{re.escape(t)}\b", hay) for t in tset)


# ============
# News Schema
# ============

@dataclass
class NewsItem:
    id: str
    source: str
    title: str
    link: str
    published: str   # ISO
    summary: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    region: Optional[str] = None
    language: Optional[str] = None
    raw_text: Optional[str] = None

    # enrichments
    sentiment: Optional[float] = None
    entities: List[str] = field(default_factory=list)
    sectors: List[str] = field(default_factory=list)
    event_types: List[str] = field(default_factory=list)
    tickers: List[str] = field(default_factory=list)

    importance: Optional[float] = None
    freshness: Optional[float] = None
    relevance: Optional[float] = None

    meta: Dict[str, Any] = field(default_factory=dict)


# =====================
# Ingest & Normalizers
# =====================

def list_sources(regions: List[str]) -> List[str]:
    final = []
    for r in regions:
        r = r.strip().upper()
        if r in SOURCES:
            final.extend(SOURCES[r])
    return sorted(set(final))

def fetch_feed(url: str, per_source_cap: Optional[int] = None, timeout: int = 30) -> List[Dict[str, Any]]:
    """Enhanced RSS feed fetching with robust error handling and headers."""
    import requests

    headers = {
        "User-Agent": "Mozilla/5.0 (NewsFetcher/1.0)",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.5",
        "Accept-Encoding": "gzip, deflate",
        "Connection": "keep-alive",
    }

    try:
        # First try direct feedparser parsing
        d = feedparser.parse(url, agent=headers.get("User-Agent"))
        if d and hasattr(d, 'entries') and len(d.entries) > 0:
            return _parse_feed_entries(d, per_source_cap)
    except Exception as e:
        pass

    # Fallback: use requests first to handle encoding issues
    try:
        response = requests.get(
            url,
            headers=headers,
            timeout=timeout,
            allow_redirects=True
        )
        response.raise_for_status()

        # Try to detect encoding from content
        if response.apparent_encoding:
            content = response.content.decode(response.apparent_encoding, errors='ignore')
        else:
            content = response.text

        # Re-parse with feedparser
        d = feedparser.parse(content)
        if d and hasattr(d, 'entries') and len(d.entries) > 0:
            return _parse_feed_entries(d, per_source_cap)
        else:
            raise ValueError("No entries found after re-parsing")

    except requests.RequestException as e:
        pass  # Fall through to return empty
    except Exception as e:
        pass

    # Return empty if all attempts fail
    return []

def _parse_feed_entries(d, per_source_cap: Optional[int]) -> List[Dict[str, Any]]:
    """Parse feed entries into standardized format."""
    items = []
    cap = per_source_cap or len(d.entries)
    for e in d.entries[:cap]:
        title = (e.get("title") or "").strip()
        link = (e.get("link") or "").strip()
        if not title or not link:
            continue
        pub = None
        if "published_parsed" in e and e.published_parsed:
            pub = dt.datetime(*e.published_parsed[:6], tzinfo=dt.timezone.utc)
        elif "updated_parsed" in e and e.updated_parsed:
            pub = dt.datetime(*e.updated_parsed[:6], tzinfo=dt.timezone.utc)
        pub_iso = pub.isoformat().replace("+00:00", "Z") if pub else now_utc().isoformat().replace("+00:00", "Z")
        summary = strip_html(e.get("summary", "") or e.get("description", ""))
        content = summary
        # optional content fields
        if "content" in e and e.content:
            blobs = [strip_html(c.get("value", "")) for c in e.content if isinstance(c, dict)]
            content = max(blobs + [summary], key=len) if blobs else summary

        items.append(dict(
            title=title, link=link, published=pub_iso, summary=summary, raw_text=content
        ))
    neg = len(re.findall(r"\b(down|loss|drop|probe|sanction|war|strike|glut)\b", text, re.I))
    return (pos - neg) / (pos + neg + 1)

def _entities(text: str) -> List[str]:
    if not text: return []
    if nlp_enrich and hasattr(nlp_enrich, "ner"):
        try:
            ents = nlp_enrich.ner(text) or []
            # expect list[str]
            return list(dict.fromkeys([str(e) for e in ents]))
        except Exception:
            pass
    # naive: capitalized multi-words
    cands = re.findall(r"\b([A-Z][a-zA-Z0-9&\-.]+(?:\s+[A-Z][a-zA-Z0-9&\-.]+){0,3})\b", text)
    return list(dict.fromkeys(cands))[:20]

def _tag_sectors(text: str) -> List[str]:
    tags = []
    low = text.lower()
    src = SECTORS_LEX
    if news_taxonomy and hasattr(news_taxonomy, "tag_sectors"):
        try:
            t = news_taxonomy.tag_sectors(text) or []
            return list(dict.fromkeys(t))
        except Exception:
            pass
    for k, words in src.items():
        if any(w.lower() in low for w in words):
            tags.append(k)
    return list(dict.fromkeys(tags))

def _tag_events(text: str) -> List[str]:
    tags = []
    low = text.lower()
    src = EVENTS_LEX
    if news_taxonomy and hasattr(news_taxonomy, "tag_events"):
        try:
            t = news_taxonomy.tag_events(text) or []
            return list(dict.fromkeys(t))
        except Exception:
            pass
    for k, words in src.items():
        if any(w.lower() in low for w in words):
            tags.append(k)
    return list(dict.fromkeys(tags))

def _map_tickers(ents: List[str], aliases: List[str], tgt_ticker: Optional[str]) -> List[str]:
    # If you have stock.ticker_from_name or a mapping DB, use it.
    out = []
    if stock and hasattr(stock, "guess_ticker"):
        for e in ents + aliases:
            try:
                tk = stock.guess_ticker(e)
                if tk: out.append(tk)
            except Exception:
                pass
    # Add explicit tgt_ticker if provided
    if tgt_ticker:
        out.append(tgt_ticker.upper())
    # Naive: uppercase tokens that look like tickers (2-5 chars)
    for e in ents:
        for tok in re.findall(r"\b[A-Z]{2,5}\b", e):
            if tok not in out:
                out.append(tok)
    return list(dict.fromkeys(out))[:10]

def _score_importance(item: NewsItem) -> float:
    # simple heuristics: title length, presence of event tags, source domain weight
    w_source = 1.0
    dom = domain_of(item.link)
    if "reuters" in dom or "ft.com" in dom or "bloomberg" in dom:
        w_source = 1.2
    w_event = 1.0 + 0.2 * len(item.event_types)
    w_sector = 1.0 + 0.1 * len(item.sectors)
    base = 0.3 + min(len(item.title) / 120.0, 0.7)
    return round(base * w_event * w_sector * w_source, 4)

def _score_freshness(published_iso: str) -> float:
    try:
        t = dt.datetime.fromisoformat(published_iso.replace("Z", "+00:00"))
        age_h = max((now_utc() - t).total_seconds() / 3600.0, 0.0)
        return round(1.0 / (1.0 + age_h / 12.0), 4)  # 12h half-life
    except Exception:
        return 0.5

def _score_relevance(text: str, query: str, company: Optional[str], tickers: List[str]) -> float:
    s = 0.0
    low = text.lower()
    if query:
        # reward matches of query terms (split by spaces)
        for term in re.findall(r"[A-Za-z0-9\-\+_]{2,}", query):
            if term.lower() in low:
                s += 0.2
    if company and company.lower() in low:
        s += 0.5
    s += 0.15 * len(tickers)
    return round(min(1.5, s), 4)


# ================
# Search & Filters
# ================

def in_window(published_iso: str, window: str) -> bool:
    """
    window: '1h','6h','12h','24h','48h','last_day','last_week','last_month', or 'all'
    """
    if window in ("all", None, ""): 
        return True
    try:
        t = dt.datetime.fromisoformat(published_iso.replace("Z", "+00:00"))
        delta = now_utc() - t
        wmap = {
            "1h": 1, "6h": 6, "12h": 12, "24h": 24, "48h": 48,
            "last_day": 24, "last_week": 24*7, "last_month": 24*30
        }
        hours = wmap.get(window, 24)
        return delta.total_seconds() <= hours * 3600
    except Exception:
        return True

def filter_items(items: List[NewsItem],
                 query: str = "",
                 window: str = "last_week",
                 regions: Optional[List[str]] = None,
                 sources_substr: Optional[List[str]] = None,
                 languages: Optional[List[str]] = None,
                 sectors: Optional[List[str]] = None,
                 events: Optional[List[str]] = None,
                 tickers: Optional[List[str]] = None) -> List[NewsItem]:
    out = []
    regions = [r.upper() for r in (regions or [])]
    src_sub = [s.lower() for s in (sources_substr or [])]
    languages = [l.lower() for l in (languages or [])]
    sectors = [s.lower() for s in (sectors or [])]
    events = [e.lower() for e in (events or [])]
    tickers = [t.upper() for t in (tickers or [])]

    for it in items:
        if not in_window(it.published, window):
            continue
        hay = " ".join([it.title or "", it.summary or "", it.raw_text or ""])
        if query and not bool_query_match(query, hay):
            continue
        if regions and (it.region or "").upper() not in regions:
            continue
        if src_sub and not any(ss in (it.source or "").lower() for ss in src_sub):
            continue
        if languages and (it.language or "en").lower() not in languages:
            continue
        if sectors and not set(s.lower() for s in it.sectors).intersection(sectors):
            continue
        if events and not set(e.lower() for e in it.event_types).intersection(events):
            continue
        if tickers and not _matches_tickers(it, tickers):
            continue
        out.append(it)
    return out


# ======================
# Main pipeline (fetch)
# ======================

def run_pipeline(regions: List[str],
                 window: str,
                 query: str = "",
                 company: Optional[str] = None,
                 aliases: Optional[List[str]] = None,
                 tgt_ticker: Optional[str] = None,
                 per_source_cap: Optional[int] = None,
                 limit: int = 100) -> List[NewsItem]:

    srcs = list_sources(regions)
    all_items: List[NewsItem] = []

    for u in tqdm(srcs, desc="Fetching feeds"):
        try:
            raw_items = fetch_feed(u, per_source_cap=per_source_cap)
            raw_items = dedup_items(raw_items, source=u)
        except Exception:
            # continue on errors
            continue

        for r in raw_items:
            title = r["title"]; link = r["link"]; published = r["published"]
            raw_text = (r.get("raw_text") or r.get("summary") or "").strip()
            lang = guess_lang((title + " " + raw_text)[:2000], url=link)
            text_for_enrich = raw_text

            # translate to EN (fallback noop)
            if lang != "en":
                text_for_enrich = _translate(raw_text, target_lang="en")

            # summarization
            short_sum = _summarize(text_for_enrich, max_sent=3)
            # entities
            ents = _entities(text_for_enrich)
            # sectors & events
            sects = _tag_sectors(text_for_enrich + " " + title)
            evts = _tag_events(text_for_enrich + " " + title)
            # map tickers
            aliases_list = [company] if company else []
            if aliases:
                aliases_list.extend([a.strip() for a in aliases if a.strip()])
            tks = _map_tickers(ents, aliases_list, tgt_ticker)
            # sentiment
            sent = _sentiment(text_for_enrich)
            # scores
            imp = _score_importance(NewsItem(
                id=r["_id"], source=u, title=title, link=link, published=published,
                summary=short_sum, region=None, language=lang, raw_text=raw_text,
                sentiment=sent, entities=ents, sectors=sects, event_types=evts, tickers=tks
            ))
            fresh = _score_freshness(published)
            rel = _score_relevance(title + " " + short_sum, query, company, tks)

            ni = NewsItem(
                id=r["_id"], source=u, title=title, link=link, published=published,
                summary=short_sum, tags=[], region=_region_guess(u),
                language=lang, raw_text=raw_text,
                sentiment=sent, entities=ents, sectors=sects, event_types=evts, tickers=tks,
                importance=imp, freshness=fresh, relevance=rel,
                meta={"domain": domain_of(link)}
            )
            all_items.append(ni)

    # global filtering (window + query)
    filtered = filter_items(all_items, query=query, window=window)
    # order by combined score: importance * freshness + relevance
    filtered.sort(key=lambda x: (x.importance or 0)* (x.freshness or 0) + (x.relevance or 0), reverse=True)
    if limit:
        filtered = filtered[:limit]
    return filtered


def _region_guess(source_url: str) -> str:
    url = source_url.lower()
    if "lesechos" in url or "boursorama" in url or "bfmtv" in url or "zonebourse" in url: return "FR"
    if "handelsblatt" in url or "faz.net" in url or "boersen-zeitung" in url or "manager-magazin" in url: return "DE"
    if "reuters" in url and "/americas/" in url: return "CA"
    if "globeandmail" in url or "bnnbloomberg" in url or "financialpost" in url: return "CA"
    if "bbc" in url or "economist.com" in url or "foreignpolicy" in url or "aljazeera" in url: return "INTL"
    if "oilprice" in url: return "INTL"
    return "US"


# ===========================
# Signals (ticker-level feats)
# ===========================

def build_news_features(items: List[NewsItem], target_ticker: Optional[str] = None,
                        window: str = "last_week") -> Dict[str, Dict[str, float]]:
    """
    Aggregate simple features per ticker for modeling stage:
      - count, mean_sentiment, pos_ratio, neg_ratio, event flags, sector counts, novelty (unique sources)
    Return: {ticker: {feature: value}}
    """
    # Helper to read fields from either NewsItem objects or plain dicts
    def _get_attr(obj, name, default=None):
        """Fetch attribute from dataclass-like object or dict-like item."""
        if obj is None:
            return default
        if isinstance(obj, dict):
            return obj.get(name, default)
        return getattr(obj, name, default)

    by_tk: Dict[str, List[NewsItem]] = defaultdict(list)
    for it in items:
        tks = _get_attr(it, 'tickers') or (["GEN"] if not target_ticker else [])
        # normalize single-string tickers to list
        if isinstance(tks, str):
            tks = [tks]
        # if a target_ticker is specified, keep only items touching it
        if target_ticker:
            if target_ticker.upper() in [t.upper() for t in (tks or [])]:
                by_tk[target_ticker.upper()].append(it)
        else:
            for tk in (tks or []):
                try:
                    by_tk[tk.upper()].append(it)
                except Exception:
                    # skip malformed ticker entries
                    continue

    feats: Dict[str, Dict[str, float]] = {}
    for tk, arr in by_tk.items():
        if not arr:
            continue
        sents = [(_get_attr(it, 'sentiment') or _get_attr(it, 'sent') or 0.0) for it in arr]
        pos = sum(1 for x in sents if x > 0.15)
        neg = sum(1 for x in sents if x < -0.15)
        zero = max(1, len(arr))
        sectors = Counter([s for it in arr for s in (_get_attr(it, 'sectors') or [])])
        events = Counter([e for it in arr for e in (_get_attr(it, 'event_types') or [])])
        srcs = set((_get_attr(it, 'source') or "") for it in arr)

        feats[tk] = {
            "news_count": float(len(arr)),
            "mean_sentiment": round(sum(sents)/zero, 4),
            "pos_ratio": round(pos/zero, 4),
            "neg_ratio": round(neg/zero, 4),
            "unique_sources": float(len(srcs)),
            # event flags
            "flag_earnings": float(events.get("earnings", 0) > 0),
            "flag_mna": float(events.get("mna", 0) > 0),
            "flag_sanctions": float(events.get("sanctions", 0) > 0),
            "flag_geopolitics": float(events.get("geopolitics", 0) > 0),
            "flag_energy_shock": float(events.get("energy_shock", 0) > 0),
            # top sector hints
            "sector_energy": float(sectors.get("energy", 0)),
            "sector_banks": float(sectors.get("banks", 0)),
            "sector_defense": float(sectors.get("defense", 0)),
            "sector_tech": float(sectors.get("tech", 0)),
        }
    return feats


# ==================
# Quick backtesting
# ==================

def price_loader_stub(ticker: str, start: str, end: str) -> Optional[Any]:
    """
    Replace this by your real price loader (e.g., yfinance, polygon, local DB).
    Must return DataFrame with columns: ['close'] indexed by date (UTC naive ok).
    """
    if pd is None:
        print("Pandas not available for backtest.", file=sys.stderr)
        return None
    # Minimal stub: return None to skip backtest gracefully.
    return None

def align_news_with_returns(items: List[NewsItem], ticker: str, horizon_days: int = 1,
                            loader=price_loader_stub) -> Optional[Any]:
    if pd is None: 
        return None
    if not items:
        return None
    dates = []
    for it in items:
        try:
            t = dt.datetime.fromisoformat(it.published.replace("Z", "+00:00")).date()
            dates.append(t)
        except Exception:
            continue
    if not dates:
        return None
    start = (min(dates) - dt.timedelta(days=2)).isoformat()
    end = (max(dates) + dt.timedelta(days=horizon_days+2)).isoformat()
    px = loader(ticker, start, end)
    if px is None or px.empty:
        return None
    # daily features: group by day
    day_sent = defaultdict(list)
    for it in items:
        try:
            d = dt.datetime.fromisoformat(it.published.replace("Z", "+00:00")).date()
            day_sent[d].append(it.sentiment or 0.0)
        except Exception:
            pass
    rows = []
    for d, arr in sorted(day_sent.items()):
        if d not in px.index: 
            continue
        s = sum(arr)/max(1, len(arr))
        # forward return
        d2 = d + dt.timedelta(days=horizon_days)
        if d2 in px.index:
            ret = (px.loc[d2, "close"] / px.loc[d, "close"]) - 1.0
            rows.append({"date": d, "mean_sentiment": s, f"ret_{horizon_days}d": ret})
    if not rows:
        return None
    df = pd.DataFrame(rows).set_index("date").sort_index()
    return df


# =========
# IO utils
# =========

def save_jsonl(items: List[NewsItem], path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(asdict(it), ensure_ascii=False) + "\n")


# =====
# CLI
# =====

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--regions", type=str, default="US,CA,INTL,GEO",
                    help="Comma list among: US,CA,FR,DE,INTL,AFRICA,GEO")
    ap.add_argument("--window", type=str, default="last_week",
                    help="1h,6h,12h,24h,48h,last_day,last_week,last_month,all")
    ap.add_argument("--query", type=str, default="", help="Boolean query e.g. 'BRICS OR oil'")
    ap.add_argument("--limit", type=int, default=100)
    ap.add_argument("--per_source_cap", type=int, default=None, help="Max items per source feed")
    ap.add_argument("--jsonl", action="store_true", help="Output JSONL lines")
    ap.add_argument("--pretty", action="store_true", help="Pretty print records")
    ap.add_argument("--company", type=str, default=None, help="Company name for relevance boosting")
    ap.add_argument("--aliases", type=str, default=None, help="Comma separated aliases for entity linking")
    ap.add_argument("--ticker", type=str, default=None, help="Target ticker (to boost mapping & features)")
    ap.add_argument("--features_for", type=str, default=None, help="Ticker to aggregate features for")
    ap.add_argument("--backtest_for", type=str, default=None, help="Ticker to backtest news vs returns")
    ap.add_argument("--backtest_horizon", type=int, default=1, help="Return horizon (days)")

    args = ap.parse_args()

    regions = [r.strip() for r in args.regions.split(",") if r.strip()]
    aliases = [a.strip() for a in (args.aliases.split(",") if args.aliases else []) if a.strip()]

    items = run_pipeline(
        regions=regions,
        window=args.window,
        query=args.query,
        company=args.company,
        aliases=aliases,
        tgt_ticker=args.ticker,
        per_source_cap=args.per_source_cap,
        limit=args.limit
    )

    # Output
    if args.jsonl:
        for it in items:
            print(json.dumps(asdict(it), ensure_ascii=False))
    elif args.pretty:
        for it in items:
            print(f"[{it.region or 'NA'}] {it.published} | {it.source} | {it.title}")
            print(f"  link: {it.link}")
            print(f"  lang: {it.language}  sent: {it.sentiment}  imp*fresh+rel: {round((it.importance or 0)*(it.freshness or 0)+(it.relevance or 0),3)}")
            print(f"  sectors: {it.sectors}  events: {it.event_types}  tickers: {it.tickers}")
            print(f"  sum: {it.summary}\n")
    else:
        # minimal JSON (array)
        print(json.dumps([asdict(x) for x in items], ensure_ascii=False, indent=2))

    # Features aggregation (optional quick view)
    if args.features_for:
        feats = build_news_features(items, target_ticker=args.features_for.upper())
        print(json.dumps({"features": feats.get(args.features_for.upper(), {})}, ensure_ascii=False, indent=2))

    # Backtest (needs price loader)
    if args.backtest_for:
        if pd is None:
            print("Pandas not available, skip backtest.", file=sys.stderr)
        else:
            df = align_news_with_returns(items, args.backtest_for.upper(), horizon_days=args.backtest_horizon)
            if df is None:
                print("Backtest unavailable (no prices loader or no overlap).", file=sys.stderr)
            else:
                # show simple correlation
                corr = float(df["mean_sentiment"].corr(df[f"ret_{args.backtest_horizon}d"]))
                print(json.dumps({
                    "backtest": {
                        "rows": len(df),
                        "horizon_days": args.backtest_horizon,
                        "corr_sentiment_vs_return": round(corr, 4)
                    }
                }, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()

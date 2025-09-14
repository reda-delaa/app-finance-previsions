#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
finnews.py — Agrégateur RSS/Atom orienté news financières & géopolitiques
Caractéristiques :
- Régions : US, CA, INTL, GEO, EU (extensibles)
- Fenêtres temporelles : 1h, 6h, 24h, 48h, 7d ou alias last_hour, last_day, last_week, last_month
- Requête booléenne simple (OR/AND/NOT), filtre entreprise (company + aliases + ticker)
- Cap par source (--per_source_cap) et limite globale (--limit)
- Enrichissements : tags thématiques, score de “sentiment” rudimentaire, détection d’entités clés
- Sortie JSONL (--jsonl) ou tableau lisible

Dépendances :
- Standard lib uniquement. Si `feedparser` est installé, il sera utilisé automatiquement (recommandé).
"""

import argparse
import sys
import json
import re
import time
from datetime import datetime, timedelta, timezone
from html import unescape
from typing import List, Dict, Any, Tuple, Optional
from urllib.request import Request, urlopen
from urllib.error import URLError, HTTPError
import xml.etree.ElementTree as ET
from email.utils import parsedate_to_datetime

# --- SOURCES PAR RÉGION -------------------------------------------------------

SOURCES: Dict[str, List[Tuple[str, str]]] = {
    # États-Unis (business/markets/énergie/tech)
    "US": [
        ("cnbc_business", "https://www.cnbc.com/id/100003114/device/rss/rss.html"),  # Business
        ("marketwatch", "https://feeds.marketwatch.com/marketwatch/topstories/"),
        ("barrons_markets", "https://www.barrons.com/feeds/rss/markets"),
        ("oilprice", "https://feeds.feedburner.com/oilpricecom"),
        ("seekingalpha_market", "https://seekingalpha.com/market_currents.xml"),
    ],
    # Canada
    "CA": [
        ("globeandmail_business", "https://www.theglobeandmail.com/arc/outboundfeeds/rss/category/business/"),
        ("financialpost", "https://financialpost.com/feed/"),
        ("bnnbloomberg", "https://www.bnnbloomberg.ca/polopoly_fs/2.2259!/menu/editorial/rss.rss"),
    ],
    # International (éco/énergie/tech)
    "INTL": [
        ("economist_business", "https://www.economist.com/business/rss.xml"),
        ("ft_companies", "https://www.ft.com/companies?format=rss"),
        ("reuters_business", "https://feeds.reuters.com/reuters/businessNews"),
        ("oilprice", "https://feeds.feedburner.com/oilpricecom"),
    ],
    # Géopolitique monde
    "GEO": [
        ("bbc_world", "https://feeds.bbci.co.uk/news/world/rss.xml"),
        ("aljazeera", "https://www.aljazeera.com/xml/rss/all.xml"),
        ("foreign_policy", "https://foreignpolicy.com/feed/"),
        ("africa_business", "https://allafrica.com/tools/headlines/rdf/business/headlines.rdf"),
        ("hindu_business", "https://www.thehindubusinessline.com/feeder/default.rss"),
    ],
    # Europe (UE/banques centrales/éco EU)
    "EU": [
        ("ecb_press", "https://www.ecb.europa.eu/press/pressconf/pressconf.rss"),
        ("europa_news", "https://ec.europa.eu/commission/presscorner/home/en/rss"),
        ("ft_europe", "https://www.ft.com/world/europe?format=rss"),
    ],
}

# --- UTILS --------------------------------------------------------------------

BOOL_TOK = re.compile(r'\b(AND|OR|NOT)\b', re.I)

def now_utc() -> datetime:
    return datetime.now(timezone.utc)

def parse_window(s: str) -> timedelta:
    s = s.strip().lower()
    aliases = {
        "last_hour": "1h",
        "last_day": "24h",
        "last_week": "7d",
        "last_month": "30d",
    }
    s = aliases.get(s, s)
    m = re.fullmatch(r'(\d+)\s*(h|d)', s)
    if not m:
        # fallback
        return timedelta(days=7)
    n, unit = int(m.group(1)), m.group(2)
    return timedelta(hours=n) if unit == 'h' else timedelta(days=n)

def http_get(url: str, timeout: float = 15.0) -> Optional[bytes]:
    try:
        req = Request(url, headers={"User-Agent": "finnews/1.0 (+https://localhost)"})
        with urlopen(req, timeout=timeout) as r:
            return r.read()
    except (HTTPError, URLError, TimeoutError):
        return None

def text_or_none(x: Optional[str]) -> Optional[str]:
    if not x:
        return None
    t = unescape(x).strip()
    return t or None

def safe_iso(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")

# --- PARSE FEEDS --------------------------------------------------------------

def parse_with_feedparser(content: bytes) -> List[Dict[str, Any]]:
    try:
        import feedparser  # type: ignore
    except Exception:
        return []
    d = feedparser.parse(content)
    out = []
    for e in d.entries:
        title = text_or_none(getattr(e, "title", None))
        link = text_or_none(getattr(e, "link", None))
        summary = text_or_none(getattr(e, "summary", None) or getattr(e, "description", None))
        # dates
        pub: Optional[datetime] = None
        if getattr(e, "published_parsed", None):
            pub = datetime.fromtimestamp(time.mktime(e.published_parsed), tz=timezone.utc)
        elif getattr(e, "updated_parsed", None):
            pub = datetime.fromtimestamp(time.mktime(e.updated_parsed), tz=timezone.utc)
        out.append({"title": title, "link": link, "summary": summary, "published": pub})
    return out

def parse_xml_fallback(content: bytes) -> List[Dict[str, Any]]:
    out = []
    try:
        root = ET.fromstring(content)
    except ET.ParseError:
        return out
    # Try RSS
    for item in root.findall(".//item"):
        title = text_or_none((item.findtext("title") or ""))
        link = text_or_none((item.findtext("link") or ""))
        summary = text_or_none(item.findtext("description"))
        pub_s = item.findtext("pubDate") or item.findtext("{http://purl.org/dc/elements/1.1/}date")
        pub = None
        if pub_s:
            try:
                pub = parsedate_to_datetime(pub_s)
                if pub.tzinfo is None:
                    pub = pub.replace(tzinfo=timezone.utc)
                else:
                    pub = pub.astimezone(timezone.utc)
            except Exception:
                pub = None
        out.append({"title": title, "link": link, "summary": summary, "published": pub})
    # Try Atom
    if not out:
        ns = {"a": "http://www.w3.org/2005/Atom"}
        for e in root.findall(".//a:entry", ns):
            title = text_or_none(e.findtext("a:title", default="", namespaces=ns))
            link_el = e.find("a:link", ns)
            href = link_el.get("href") if link_el is not None else None
            link = text_or_none(href)
            summary = text_or_none(e.findtext("a:summary", default="", namespaces=ns) or e.findtext("a:content", default="", namespaces=ns))
            updated = e.findtext("a:updated", default="", namespaces=ns) or e.findtext("a:published", default="", namespaces=ns)
            pub = None
            if updated:
                try:
                    # naive ISO parse
                    pub = datetime.fromisoformat(updated.replace("Z", "+00:00")).astimezone(timezone.utc)
                except Exception:
                    pub = None
            out.append({"title": title, "link": link, "summary": summary, "published": pub})
    return out

def fetch_feed_entries(url: str) -> List[Dict[str, Any]]:
    raw = http_get(url)
    if not raw:
        return []
    parsed = parse_with_feedparser(raw)
    if parsed:
        return parsed
    return parse_xml_fallback(raw)

# --- FILTRES & ENRICHISSEMENT -------------------------------------------------

def within_window(pub: Optional[datetime], earliest: datetime) -> bool:
    if pub is None:
        # Si pas de date, on garde par défaut (tu peux changer ce choix)
        return True
    return pub >= earliest

def compile_bool_query(q: str):
    """
    Compilateur OR/AND/NOT minimaliste.
    - Mots/phrases non qualifiés -> terme “contains” (casefold)
    - Support guillemets "exact phrase"
    """
    q = q.strip()
    if not q:
        return lambda text: True

    # Tokenisation très simple : "exact phrase" ou mots, + bools
    terms = []
    for token in re.finditer(r'"([^"]+)"|(\S+)', q):
        if token.group(1):
            terms.append(('TERM', token.group(1).casefold()))
        else:
            t = token.group(2)
            if t.upper() in ("AND", "OR", "NOT"):
                terms.append((t.upper(), t.upper()))
            else:
                terms.append(('TERM', t.casefold()))

    def eval_text(text: str) -> bool:
        hay = text.casefold()
        # Shunting-yard minimal -> on évalue gauche à droite avec priorité NOT > AND > OR
        # Pour simplicité, on transforme en liste de bools + ops puis on applique priorités
        vals, ops = [], []
        for kind, val in terms:
            if kind == 'TERM':
                vals.append(val in hay)
            else:
                ops.append(val)

        # Appliquer NOT sur place (unaires) : NOT s'applique au terme précédent dans cette implémentation simple
        i = 0
        while i < len(ops):
            if ops[i] == 'NOT':
                # NOT s'applique au prochain bool (si présent)
                if i < len(vals):
                    vals[i] = (not vals[i])
                ops.pop(i)
            else:
                i += 1

        # AND
        i = 0
        while i < len(ops):
            if ops[i] == 'AND':
                if i + 1 < len(vals):
                    vals[i] = vals[i] and vals[i+1]
                    vals.pop(i+1)
                ops.pop(i)
            else:
                i += 1

        # OR (reste)
        res = False
        for v in vals:
            res = res or v
        return res

    return eval_text

def company_filter_factory(company: str, aliases: List[str], ticker: str):
    keys = []
    if company:
        keys.append(company.casefold())
    for a in aliases:
        a = a.strip()
        if a:
            keys.append(a.casefold())
    if ticker:
        keys.append(ticker.casefold())

    if not keys:
        return lambda title, summary: True

    def f(title: Optional[str], summary: Optional[str]) -> bool:
        blob = " ".join([x for x in [title or "", summary or ""]]).casefold()
        return any(k in blob for k in keys)

    return f

# Tagging simple par mots-clés
TAG_RULES = [
    (re.compile(r'\b(brent|wti|opec\+?|rig|shale|refinery|lng|pipeline|sanction|embargo|oil|gas|diesel|crude)\b', re.I), "Energy"),
    (re.compile(r'\b(fed|ecb|rate hike|interest rate|inflation|cpi|ppi|gdp|jobs|unemployment|yield|bond)\b', re.I), "Macro"),
    (re.compile(r'\b(ceasefire|nato|russia|ukraine|israel|gaza|china|taiwan|houthi|missile|drone|strike|sanction)\b', re.I), "Geopolitics"),
    (re.compile(r'\b(earnings|guidance|buyback|dividend|ipo|m&a|merger|acquisition|downgrade|upgrade)\b', re.I), "Markets"),
    (re.compile(r'\b(ai|semiconductor|chip|fab|cloud|iphone|android|nvidia|apple|google|meta|microsoft)\b', re.I), "Tech"),
]

POS_WORDS = {"beats", "surge", "rally", "record", "growth", "win", "strong", "optimism", "bull", "soar"}
NEG_WORDS = {"miss", "plunge", "slump", "cut", "layoff", "sanction", "probe", "fine", "bear", "war", "strike", "attack"}

ENT_PATTERNS = [
    (re.compile(r'\b[A-Z]{1,5}\b'), "Ticker?"),
    (re.compile(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b'), "Org?"),
]

def enrich_item(text: str) -> Tuple[List[str], float, List[str]]:
    tags = set()
    for pat, tag in TAG_RULES:
        if pat.search(text):
            tags.add(tag)

    score = 0.0
    low = text.casefold()
    for w in POS_WORDS:
        if w in low:
            score += 1.0
    for w in NEG_WORDS:
        if w in low:
            score -= 1.0

    ents = set()
    for pat, _label in ENT_PATTERNS:
        for m in pat.findall(text):
            ents.add(m if isinstance(m, str) else " ".join(m))
    return sorted(tags), score, sorted(ents)

# --- PIPELINE -----------------------------------------------------------------

def harvest(
    regions: List[str],
    window: str,
    query: str,
    limit: int,
    per_source_cap: int,
    company: str,
    aliases: List[str],
    ticker: str
) -> List[Dict[str, Any]]:

    earliest = now_utc() - parse_window(window)
    qmatch = compile_bool_query(query or "")
    cmatch = company_filter_factory(company, aliases, ticker)

    # réunir les URLs selon régions
    wanted = []
    for r in regions:
        r = r.strip().upper()
        if r in SOURCES:
            wanted.extend(SOURCES[r])
        else:
            # région inconnue ignorée
            pass

    results = []
    for source_name, url in wanted:
        entries = fetch_feed_entries(url)
        count = 0
        for e in entries:
            # base fields
            title = e.get("title")
            link = e.get("link")
            summary = e.get("summary")
            pub = e.get("published")
            if not title and not summary:
                continue

            # filtres
            if not within_window(pub, earliest):
                continue

            textblob = " ".join([x for x in [title or "", summary or ""]])
            if not qmatch(textblob):
                continue

            if not cmatch(title, summary):
                continue

            tags, sent, ents = enrich_item(textblob)

            results.append({
                "source": source_name,
                "title": title,
                "link": link,
                "published": safe_iso(pub) if pub else None,
                "summary": summary,
                "tags": tags,
                "sentiment": sent,
                "entities": ents,
                "region": infer_region_from_source(source_name),
            })
            count += 1
            if per_source_cap > 0 and count >= per_source_cap:
                break

    # tri par date desc, puis coupe à limit
    def key_dt(it):
        p = it.get("published")
        try:
            return datetime.fromisoformat((p or "1970-01-01T00:00:00+00:00").replace("Z", "+00:00"))
        except Exception:
            return datetime(1970, 1, 1, tzinfo=timezone.utc)

    results.sort(key=key_dt, reverse=True)
    if limit > 0:
        results = results[:limit]
    return results

def infer_region_from_source(src: str) -> str:
    src = src.lower()
    if "bbc" in src or "aljazeera" in src or "foreign_policy" in src or "africa" in src or "hindu" in src or "oilprice" in src:
        return "GEO"
    if "economist" in src or "ft_" in src or "reuters" in src:
        return "INTL"
    if "globe" in src or "financialpost" in src or "bnnbloomberg" in src:
        return "CA"
    if "ecb" in src or "europa" in src or "ft_europe" in src:
        return "EU"
    return "US"

# --- SORTIE -------------------------------------------------------------------

def print_jsonl(items: List[Dict[str, Any]]) -> None:
    for it in items:
        print(json.dumps(it, ensure_ascii=False))

def print_table(items: List[Dict[str, Any]]) -> None:
    from textwrap import shorten
    print(f"{'SOURCE':<18} {'PUBLISHED':<20} {'SENT':>4}  TITLE")
    print("-" * 110)
    for it in items:
        src = it["source"][:18]
        pub = (it["published"] or "")[:19]
        sent = f"{it['sentiment']:+.0f}"
        title = shorten(it["title"] or "", width=75, placeholder="…")
        print(f"{src:<18} {pub:<20} {sent:>4}  {title}")
    print("\nTotal:", len(items))

# --- CLI ----------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description="Financial & Geopolitical News Aggregator")
    p.add_argument("--regions", type=str, default="US,CA,INTL,GEO",
                   help="Régions comma-séparées (US,CA,INTL,GEO,EU)")
    p.add_argument("--window", type=str, default="last_week",
                   help="Fenêtre ex: 1h, 24h, 48h, 7d, last_day, last_week, last_month")
    p.add_argument("--query", type=str, default="",
                   help='Requête booléenne simple (ex: "oil AND Ukraine", "BRICS OR sanctions")')
    p.add_argument("--limit", type=int, default=100, help="Limite globale d’items")
    p.add_argument("--jsonl", action="store_true", help="Sortie JSON Lines")
    p.add_argument("--company", type=str, default="", help="Nom d’entreprise à filtrer")
    p.add_argument("--aliases", type=str, default="", help="Alias supplémentaires, séparés par des virgules")
    # nouveaux flags
    p.add_argument("--per_source_cap", type=int, default=10, help="Max d’items par source (0 = illimité)")
    p.add_argument("--ticker", type=str, default="", help="Ticker boursier (ex: AAPL, MSFT, ORA)")
    args = p.parse_args()

    regions = [r.strip() for r in args.regions.split(",") if r.strip()]
    aliases = [a.strip() for a in (args.aliases or "").split(",") if a.strip()]

    items = harvest(
        regions=regions,
        window=args.window,
        query=args.query,
        limit=args.limit,
        per_source_cap=args.per_source_cap,
        company=args.company.strip(),
        aliases=aliases,
        ticker=args.ticker.strip()
    )

    if args.jsonl:
        print_jsonl(items)
    else:
        print_table(items)

if __name__ == "__main__":
    main()
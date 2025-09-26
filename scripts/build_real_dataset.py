# scripts/build_real_dataset.py
"""
Build a REAL dataset from live sources:
- News: prefers src.ingestion.finnews if available, else RSS via feedparser
- Market: via yfinance (robust fallbacks)
- Tags & sentiment: lightweight keyword rules (no heavy deps)

Outputs:
  data/real/<prefix>_features.json
  data/real/<prefix>_news.jsonl
  data/real/<prefix>_meta.json

Usage:
  python scripts/build_real_dataset.py --limit 160 --hours 96 --outdir data/real
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
import math
import datetime as dt
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

# ---------- Optional imports from your repo ----------
FINNEWS = None
MARKET_DATA = None
try:
    from src.ingestion import finnews as FINNEWS  # type: ignore
except Exception:
    FINNEWS = None

try:
    from src.core import stock_utils as MARKET_DATA  # optional
except Exception:
    MARKET_DATA = None

# ---------- Third-party, kept minimal ----------
try:
    import feedparser  # RSS fallback
except Exception:
    feedparser = None

try:
    import yfinance as yf  # market data
except Exception:
    yf = None


# -------------------- Helpers: time & paths --------------------

def utcnow() -> dt.datetime:
    # timezone-aware UTC (fixes DeprecationWarning)
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0)


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


# -------------------- News acquisition --------------------

DEFAULT_RSS = [
    "https://www.reuters.com/finance/markets/rss",
    "https://feeds.a.dj.com/rss/RSSMarketsMain.xml",  # WSJ markets
    "https://www.ft.com/markets?format=rss",
    "https://www.economist.com/finance-and-economics/rss.xml",
    "https://www.cnbc.com/id/100003114/device/rss/rss.html",
]

_POS_WORDS = {
    "record high", "beats", "surge", "rally", "growth", "upbeat",
    "win", "approval", "optimism", "rebound", "expands", "soars",
}
_NEG_WORDS = {
    "miss", "downgrade", "cuts", "plunge", "selloff", "default",
    "sanction", "war", "conflict", "bankruptcy", "strike",
    "recession", "inflation surge", "tariff", "probe",
}

_TAG_RULES = [
    ("earnings", r"\b(earnings|results|profit|loss|EPS)\b"),
    ("sanctions", r"\b(sanction|tariff|embargo|ban|blacklist)\b"),
    ("geopolitics", r"\b(war|conflict|border|tension|missile|NATO|UN)\b"),
    ("energy", r"\b(oil|gas|brent|wti|opec|refinery|offshore wind|power)\b"),
    ("banks", r"\b(bank|lender|credit|loan|FDIC|ECB|Fed)\b"),
    ("defense", r"\b(army|defense|missile|boeing|lockheed|raytheon)\b"),
    ("tech", r"\b(chip|ai|semiconductor|iphone|android|cloud|data center)\b"),
]

_tag_compiled = [(k, re.compile(pat, re.I)) for (k, pat) in _TAG_RULES]


def _score_sentiment(text: str) -> float:
    t = text.lower()
    score = 0
    for w in _POS_WORDS:
        if w in t:
            score += 1
    for w in _NEG_WORDS:
        if w in t:
            score -= 1
    # normalize lightly
    return max(-1.0, min(1.0, score / 3.0))


def _extract_tags(text: str) -> Dict[str, bool]:
    t = text.lower()
    tags: Dict[str, bool] = {k: False for (k, _) in _TAG_RULES}
    for k, rx in _tag_compiled:
        if rx.search(t):
            tags[k] = True
    return tags


def fetch_news(limit: int, hours: int, extra_rss: List[str]) -> List[Dict[str, Any]]:
    deadline = utcnow() - dt.timedelta(hours=hours)
    items: List[Dict[str, Any]] = []

    # 1) prefer project’s finnews if available
    if FINNEWS is not None and hasattr(FINNEWS, "fetch_latest"):
        try:
            raw = FINNEWS.fetch_latest(limit=limit * 2)  # slightly larger, will time-filter
            for r in raw:
                ts = r.get("ts") or r.get("timestamp") or r.get("published")
                if not ts:
                    continue
                try:
                    tdt = dt.datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
                except Exception:
                    continue
                if tdt < deadline:
                    continue
                title = r.get("title") or r.get("headline") or ""
                source = r.get("source") or ""
                link = r.get("link") or r.get("url") or ""
                summary = r.get("summary") or r.get("description") or ""
                sent = r.get("sent")
                if sent is None:
                    sent = _score_sentiment(f"{title}. {summary}")
                tags = r.get("tags")
                if not isinstance(tags, dict):
                    tags = _extract_tags(f"{title}. {summary}")
                items.append({
                    "ts": tdt.isoformat().replace("+00:00", "Z"),
                    "source": source,
                    "title": title,
                    "link": link,
                    "sent": float(sent),
                    "tickers": r.get("tickers") or [],
                    "summary": summary,
                    "tags": tags,
                })
        except Exception as e:
            print(f"⚠ finnews.fetch_latest failed: {e}", file=sys.stderr)

    # 2) fallback: RSS via feedparser
    if len(items) < limit and feedparser is not None:
        feeds = list(dict.fromkeys(DEFAULT_RSS + (extra_rss or [])))
        for url in feeds:
            try:
                parsed = feedparser.parse(url)
                for e in parsed.entries[:limit * 2]:
                    # parse time if present
                    pub = None
                    for key in ("published_parsed", "updated_parsed"):
                        if getattr(e, key, None):
                            pub = dt.datetime(*getattr(e, key)[:6], tzinfo=dt.timezone.utc)
                            break
                    if pub is None:
                        pub = utcnow()
                    if pub < deadline:
                        continue
                    title = getattr(e, "title", "")
                    link = getattr(e, "link", "")
                    summ = getattr(e, "summary", "")
                    source = parsed.feed.get("title", "").split(" - ")[0] if getattr(parsed, "feed", None) else ""
                    sent = _score_sentiment(f"{title}. {summ}")
                    tags = _extract_tags(f"{title}. {summ}")
                    items.append({
                        "ts": pub.isoformat().replace("+00:00", "Z"),
                        "source": source,
                        "title": title,
                        "link": link,
                        "sent": float(sent),
                        "tickers": [],
                        "summary": summ,
                        "tags": tags,
                    })
            except Exception as e:
                print(f"⚠ RSS parse failed for {url}: {e}", file=sys.stderr)

    # dedupe by link/title
    seen = set()
    deduped: List[Dict[str, Any]] = []
    for it in sorted(items, key=lambda x: x["ts"], reverse=True):
        key = (it.get("link") or "", it.get("title") or "")
        if key in seen:
            continue
        seen.add(key)
        deduped.append(it)

    return deduped[:limit]


# -------------------- Market acquisition --------------------

# We try ^DXY first (indices sometimes blocked), then fallback to UUP (ETF proxy)
TICKERS = {
    "DXY": "^DXY",
    "DXY_ALT": "UUP",
    "BRENT": "BZ=F",
    "WTI": "CL=F",
    "UST10Y": "^TNX",
    "EUROSTOXX": "^STOXX50E",
    "EUROSTOXX_ALT": "FEZ",
}

def _pct(a: float, b: float) -> Optional[float]:
    try:
        if b == 0 or a is None or b is None:
            return None
        return round(100.0 * (a - b) / b, 2)
    except Exception:
        return None

def _bp(a: float, b: float) -> Optional[float]:
    try:
        if a is None or b is None:
            return None
        # ^TNX is yield*100 in Yahoo (i.e., 10y yield in % * 100)
        return round((a - b), 1)
    except Exception:
        return None


def _fetch_yf_series(symbol: str, lookback_days: int = 30) -> List[Tuple[dt.date, float]]:
    if yf is None:
        return []
    try:
        df = yf.download(symbol, period=f"{max(lookback_days,18)}d", interval="1d", progress=False, auto_adjust=False)
        if df is None or len(df) == 0 or "Close" not in df:
            return []
        out: List[Tuple[dt.date, float]] = []
        for idx, row in df.iterrows():
            try:
                val = float(row["Close"])
                d = idx.date()
                out.append((d, val))
            except Exception:
                continue
        return out
    except Exception:
        return []


def _last_and_week_ago(series: List[Tuple[dt.date, float]]) -> Tuple[Optional[float], Optional[float]]:
    """Return last close and ~one-week-ago close using trading days (approx 5)"""
    if not series:
        return None, None
    closes = [v for (_d, v) in series]
    last = closes[-1]
    week_ago = closes[-6] if len(closes) >= 6 else (closes[0] if closes else None)
    return last, week_ago


def fetch_market_snapshot() -> Dict[str, Any]:
    # DXY (or UUP fallback), Brent, EuroStoxx, UST10Y (bps)
    # Prefer project MARKET_DATA if available and working
    used = dict(TICKERS)

    def series_for(sym: str) -> List[Tuple[dt.date, float]]:
        return _fetch_yf_series(sym, lookback_days=40)

    # DXY
    dxy_series = series_for(TICKERS["DXY"])
    dxy_symbol_used = TICKERS["DXY"]
    if not dxy_series:
        dxy_series = series_for(TICKERS["DXY_ALT"])
        dxy_symbol_used = TICKERS["DXY_ALT"]

    brent_series = series_for(TICKERS["BRENT"])
    stoxx_series = series_for(TICKERS["EUROSTOXX"]) or series_for(TICKERS["EUROSTOXX_ALT"])
    ust10_series = series_for(TICKERS["UST10Y"])

    dxy_last, dxy_week = _last_and_week_ago(dxy_series)
    brent_last, brent_week = _last_and_week_ago(brent_series)
    stoxx_last, stoxx_week = _last_and_week_ago(stoxx_series)
    ust10_last, ust10_week = _last_and_week_ago(ust10_series)

    # compute %
    dxy_wow = _pct(dxy_last, dxy_week)
    brent_wow = _pct(brent_last, brent_week)
    stoxx_wow = _pct(stoxx_last, stoxx_week)

    # ^TNX is already bps of % yield; difference is in "points" which equal bps.
    ust10y_bp = _bp(ust10_last, ust10_week)

    # small cleanups for None
    def nz(x, fallback=0.0):
        return float(x) if isinstance(x, (int, float)) else fallback

    return {
        "dxy_symbol": dxy_symbol_used,
        "dxy_wow": nz(dxy_wow, 0.0),
        "brent_symbol": TICKERS["BRENT"],
        "brent_wow": nz(brent_wow, 0.0),
        "eurostoxx_symbol": TICKERS["EUROSTOXX"] if stoxx_series else TICKERS["EUROSTOXX_ALT"],
        "eurostoxx_wow": nz(stoxx_wow, 0.0),
        "ust10y_bp": nz(ust10y_bp, 0.0),
        "asof_utc": utcnow().isoformat().replace("+00:00", "Z"),
    }


# -------------------- Feature engineering --------------------

SECTOR_KEYS = ["energy", "banks", "defense", "tech"]

def build_features(news: List[Dict[str, Any]], market: Dict[str, Any]) -> Dict[str, Any]:
    n = len(news)
    if n == 0:
        return {
            "news_count": 0,
            "mean_sentiment": 0.0,
            "pos_ratio": 0.0,
            "neg_ratio": 0.0,
            "unique_sources": 0,
            "flag_earnings": 0,
            "flag_mna": 0,
            "flag_sanctions": 0,
            "flag_geopolitics": 0,
            "flag_energy_shock": 0,
            **{f"sector_{s}": 0 for s in SECTOR_KEYS},
        }

    sents = [float(x.get("sent", 0.0)) for x in news]
    mean_sent = round(sum(sents) / max(1, n), 4)
    pos_ratio = round(sum(1 for s in sents if s > 0.25) / n, 3)
    neg_ratio = round(sum(1 for s in sents if s < -0.25) / n, 3)
    sources = len({x.get("source", "") for x in news})

    def has_kw(kw: str) -> bool:
        kw = kw.lower()
        for x in news:
            blob = f"{x.get('title','')}. {x.get('summary','')}".lower()
            if kw in blob:
                return True
        return False

    flags = {
        "flag_earnings": int(has_kw("earnings") or has_kw("results")),
        "flag_mna": int(has_kw("merger") or has_kw("acquisition") or " m&a" in " ".join(
            (x.get("title","") + " " + x.get("summary","")).lower() for x in news)),
        "flag_sanctions": int(has_kw("sanction") or has_kw("tariff") or has_kw("embargo")),
        "flag_geopolitics": int(has_kw("war") or has_kw("conflict") or has_kw("border")),
        "flag_energy_shock": int(has_kw("opec") or has_kw("brent") or has_kw("wti") or has_kw("offshore wind")),
    }

    sectors = {f"sector_{s}": 0 for s in SECTOR_KEYS}
    for x in news:
        t = x.get("tags", {})
        for s in SECTOR_KEYS:
            if t.get(s):
                sectors[f"sector_{s}"] += 1

    feats = {
        "news_count": n,
        "mean_sentiment": mean_sent,
        "pos_ratio": pos_ratio,
        "neg_ratio": neg_ratio,
        "unique_sources": sources,
        **flags,
        **sectors,
    }
    return feats


# -------------------- CLI --------------------

def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Build real dataset (news + market snapshot)")
    ap.add_argument("--limit", type=int, default=200, help="Max number of news to keep")
    ap.add_argument("--hours", type=int, default=120, help="Lookback hours for news")
    ap.add_argument("--outdir", type=str, default="data/real", help="Output directory")
    ap.add_argument("--prefix", type=str, default=None, help="Custom file prefix (default: UTC YYYYMMDD_HHMM)")
    ap.add_argument("--rss", action="append", default=[], help="Extra RSS URLs (can be specified multiple times)")
    args = ap.parse_args(argv)

    outdir = Path(args.outdir)
    ensure_dir(outdir)

    prefix = args.prefix or utcnow().strftime("%Y%m%d_%H%M")

    print("▶ Fetching news…")
    news = fetch_news(limit=args.limit, hours=args.hours, extra_rss=args.rss)

    print("▶ Fetching market…")
    market = fetch_market_snapshot()

    print("▶ Aggregating features…")
    features = build_features(news, market)

    agg = {
        "features": features,
        "market": market,
    }

    f_json = outdir / f"{prefix}_features.json"
    f_jsonl = outdir / f"{prefix}_news.jsonl"
    f_meta = outdir / f"{prefix}_meta.json"

    with f_json.open("w", encoding="utf-8") as f:
        json.dump(agg, f, ensure_ascii=False, indent=2)

    with f_jsonl.open("w", encoding="utf-8") as f:
        for it in news:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")

    with f_meta.open("w", encoding="utf-8") as f:
        json.dump({
            "attachments": [
                f"FX: DXY {agg['market']['dxy_wow']}% WoW; Brent {agg['market']['brent_wow']}% WoW; "
                f"UST10Y {agg['market']['ust10y_bp']}bp; EuroStoxx {agg['market']['eurostoxx_wow']}% WoW."
            ],
            "notes": "Real dataset built from live RSS/Yahoo. Tags & sentiment are heuristic.",
            "tickers_used": {k: v for k, v in TICKERS.items()},
            "sources_used": list(dict.fromkeys(DEFAULT_RSS + args.rss)),
        }, f, ensure_ascii=False, indent=2)

    print("✓ Wrote:")
    print(" ", f_json)
    print(" ", f_jsonl)
    print(" ", f_meta)
    return 0


if __name__ == "__main__":
    sys.exit(main())

# scripts/build_real_dataset.py
"""
Build a REAL dataset from live sources:
- News: uses src.ingestion.finnews if available, else RSS via feedparser
- Market: uses src.market_data if available, else yfinance
- Tags & sentiment: lightweight keyword rules (no heavy deps)
Outputs:
  data/real/<prefix>_features.json
  data/real/<prefix>_news.jsonl
  data/real/<prefix>_meta.json
"""

import argparse, json, re, sys, time, datetime as dt
from pathlib import Path

# ---------- Optional imports from your repo ----------
def _try_import(path, name):
    try:
        mod = __import__(path, fromlist=[name])
        return getattr(mod, name)
    except Exception:
        return None

FinNews = _try_import("src.ingestion.finnews", "FinNews") or _try_import("searxng-local.finnews", "FinNews")
get_prices = _try_import("src.market_data", "get_prices")  # expected signature(symbol, start, end, interval)
# ---------- Fallback deps (no-auth) ----------
try:
    import feedparser  # pip install feedparser
except Exception:
    feedparser = None

try:
    import yfinance as yf  # pip install yfinance
except Exception:
    yf = None

# ---------- Default sources / tickers ----------
DEFAULT_RSS = [
    "https://www.reuters.com/finance/markets/rss",
    "https://feeds.a.dj.com/rss/RSSMarketsMain.xml",
    "https://www.ft.com/markets?format=rss",
    "https://www.economist.com/finance-and-economics/rss.xml",
    "https://www.cnbc.com/id/100003114/device/rss/rss.html",
]
# Yahoo Finance symbols (robustes)
TICKERS = {
    "DXY": "^DXY",            # Dollar Index (parfois sparse)
    "DXY_ALT": "UUP",         # ETF USD (fallback)
    "BRENT": "BZ=F",          # Brent futures
    "WTI": "CL=F",            # WTI (fallback)
    "UST10Y": "^TNX",         # 10y yield * 10 (i.e. 45.0 = 4.5%)
    "EUROSTOXX": "^STOXX50E", # EURO STOXX 50
    "EUROSTOXX_ALT": "FEZ",   # ETF fallback
}

# ---------- Lightweight dictionaries ----------
KW = {
    "earnings": r"\b(earnings|profits?|EPS|guidance|quarterly|results|outlook|profit warning)\b",
    "sanctions": r"\b(sanction|export control|embargo|tariff|restriction|blacklist)\b",
    "geopolitics": r"\b(war|conflict|tension|escalation|missile|drone|border|invasion|ceasefire)\b",
    "energy": r"\b(oil|brent|wti|opec\+?|gas|refinery|diesel|petroleum|crude)\b",
    "banks": r"\b(bank|lender|credit|tier ?1|capital ratio|depositor|liquidity)\b",
    "defense": r"\b(defense|missile|aerospace|military|nato|procurement)\b",
    "tech": r"\b(chip|semiconductor|ai|cloud|software|hardware|gpu|foundry)\b",
}
KW_RE = {k: re.compile(v, re.I) for k, v in KW.items()}

SENT_POS = re.compile(r"\b(beat|surge|soar|record|resilient|strong|robust|improve|ease|cool|expand|grow)\b", re.I)
SENT_NEG = re.compile(r"\b(miss|fall|drop|slump|weak|fragile|contract|worsen|tighten|spike|shortage|ban|sanction)\b", re.I)

def _now_utc():
    return dt.datetime.utcnow().replace(microsecond=0)

def _iso_z(t: dt.datetime):
    return t.replace(microsecond=0).isoformat() + "Z"

# ---------- NEWS ----------
def fetch_news(limit: int, hours_back: int, sources: list[str]) -> list[dict]:
    cut = _now_utc() - dt.timedelta(hours=hours_back)

    # Try your FinNews collector first
    if FinNews:
        client = FinNews()
        items = client.fetch(limit=limit, hours_back=hours_back)  # expected to exist; otherwise fallback below
        out = []
        for it in items:
            ts = it.get("ts") or it.get("published") or it.get("time")
            try:
                ts_dt = dt.datetime.fromisoformat(ts.replace("Z",""))
            except Exception:
                ts_dt = _now_utc()
            if ts_dt < cut:
                continue
            out.append({
                "ts": _iso_z(ts_dt),
                "source": it.get("source") or it.get("feed") or "",
                "title": it.get("title") or it.get("headline") or "",
                "link": it.get("link") or it.get("url") or "",
                "sent": it.get("sentiment") if isinstance(it.get("sentiment"), (int,float)) else None,
                "tickers": it.get("tickers") or it.get("symbols") or [],
                "summary": it.get("summary") or it.get("snippet") or "",
            })
        return out[:limit]

    # Fallback: RSS via feedparser
    if not feedparser:
        print("ERROR: neither FinNews nor feedparser available.", file=sys.stderr)
        return []
    out = []
    for url in sources:
        try:
            feed = feedparser.parse(url)
            for e in feed.entries:
                # parse time
                ts_dt = _now_utc()
                if hasattr(e, "published_parsed") and e.published_parsed:
                    ts_dt = dt.datetime.fromtimestamp(time.mktime(e.published_parsed))
                if ts_dt < cut:
                    continue
                out.append({
                    "ts": _iso_z(ts_dt),
                    "source": feed.feed.get("title", ""),
                    "title": e.get("title", ""),
                    "link": e.get("link", ""),
                    "sent": None,
                    "tickers": [],
                    "summary": e.get("summary", "")[:500],
                })
        except Exception as ex:
            print(f"WARN: RSS error for {url}: {ex}", file=sys.stderr)
    # De-dup by title+link
    seen = set()
    uniq = []
    for it in out:
        key = (it["title"], it["link"])
        if key in seen: 
            continue
        seen.add(key)
        uniq.append(it)
    uniq.sort(key=lambda x: x["ts"], reverse=True)
    return uniq[:limit]

# ---------- SENTIMENT + TAGS ----------
def score_sentiment(txt: str) -> float:
    """Tiny heuristic: (#pos - #neg) / (#pos + #neg + 1). Range ~[-1,1]."""
    if not txt:
        return 0.0
    pos = len(SENT_POS.findall(txt))
    neg = len(SENT_NEG.findall(txt))
    return (pos - neg) / max(1.0, (pos + neg))

def bool_kw(txt: str, pat: re.Pattern) -> bool:
    return bool(pat.search(txt or ""))

def enrich_news(news: list[dict]) -> list[dict]:
    out = []
    for it in news:
        blob = " ".join([it.get("title",""), it.get("summary","")])
        sent = it.get("sent")
        if sent is None:
            sent = round(score_sentiment(blob), 4)
        tags = {
            "earnings": bool_kw(blob, KW_RE["earnings"]),
            "sanctions": bool_kw(blob, KW_RE["sanctions"]),
            "geopolitics": bool_kw(blob, KW_RE["geopolitics"]),
            "energy": bool_kw(blob, KW_RE["energy"]),
            "banks": bool_kw(blob, KW_RE["banks"]),
            "defense": bool_kw(blob, KW_RE["defense"]),
            "tech": bool_kw(blob, KW_RE["tech"]),
        }
        it = {**it, "sent": sent, "tags": tags}
        out.append(it)
    return out

# ---------- MARKET DATA ----------
def _yf_series(symbol: str, days: int = 14):
    if not yf:
        return []
    try:
        # '1h' can be rate limited; daily is fine for WoW
        hist = yf.Ticker(symbol).history(period=f"{days}d", interval="1d")
        # hist index is datetime; we need close series
        return [float(x) for x in hist["Close"].dropna().tolist()]
    except Exception:
        return []

def _series(symbol: str, start: dt.datetime, end: dt.datetime):
    # Prefer your repo's helper if available
    if get_prices:
        try:
            df = get_prices(symbol, start=start.date().isoformat(), end=end.date().isoformat(), interval="1d")
            closes = df["close"] if "close" in df else df.iloc[:, -1]
            return [float(x) for x in closes.dropna().tolist()]
        except Exception:
            pass
    # Fallback yfinance
    return _yf_series(symbol, (end - start).days + 2)

def pct_wow(symbols: list[str]) -> tuple[str, float | None]:
    end = _now_utc()
    start = end - dt.timedelta(days=16)
    for sym in symbols:
        ser = _series(sym, start, end)
        if len(ser) >= 7:
            last = ser[-1]
            prev = ser[-6]  # ~1 week ago (trading days)
            if prev != 0:
                return sym, round(100.0 * (last - prev) / prev, 2)
    return symbols[0], None

def ust10y_change_bp() -> float | None:
    end = _now_utc()
    start = end - dt.timedelta(days=16)
    ser = _series(TICKERS["UST10Y"], start, end)
    if len(ser) >= 7:
        last = ser[-1]
        prev = ser[-6]
        # ^TNX is yield*10 (e.g., 45.0 -> 4.5%)
        return round((last - prev), 1)  # already in bp-equivalent units
    return None

# ---------- FEATURE AGGREGATION ----------
def aggregate_features(news: list[dict], market_extra: dict) -> dict:
    n = len(news)
    mean_sent = round(sum(x.get("sent", 0.0) for x in news) / n, 4) if n else 0.0
    pos_ratio = round(sum(1 for x in news if x.get("sent",0) > 0.05) / max(1, n), 3)
    neg_ratio = round(sum(1 for x in news if x.get("sent",0) < -0.05) / max(1, n), 3)
    sources = len(set(x.get("source","") for x in news if x.get("source")))
    # flags (any mention)
    flag_earnings = int(any(x["tags"]["earnings"] for x in news))
    flag_sanctions = int(any(x["tags"]["sanctions"] for x in news))
    flag_geo = int(any(x["tags"]["geopolitics"] for x in news))
    flag_energy = int(any(x["tags"]["energy"] for x in news))
    # sectors: crude counts -> simple scale
    sector_energy = min(30, sum(1 for x in news if x["tags"]["energy"]))
    sector_banks = min(30, sum(1 for x in news if x["tags"]["banks"]))
    sector_defense = min(30, sum(1 for x in news if x["tags"]["defense"]))
    sector_tech = min(30, sum(1 for x in news if x["tags"]["tech"]))

    return {
        "features": {
            "news_count": n,
            "mean_sentiment": mean_sent,
            "pos_ratio": pos_ratio,
            "neg_ratio": neg_ratio,
            "unique_sources": sources,
            "flag_earnings": flag_earnings,
            "flag_mna": 0,
            "flag_sanctions": flag_sanctions,
            "flag_geopolitics": flag_geo,
            "flag_energy_shock": int(flag_energy or (market_extra.get("brent_wow") or 0) >= 2.0),
            "sector_energy": sector_energy,
            "sector_banks": sector_banks,
            "sector_defense": sector_defense,
            "sector_tech": sector_tech,
        },
        "market": market_extra,
    }

# ---------- MAIN ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=160, help="max number of news")
    ap.add_argument("--hours", type=int, default=96, help="lookback hours for news")
    ap.add_argument("--rss", action="append", default=[], help="extra RSS urls")
    ap.add_argument("--outdir", default="data/real", help="output directory")
    ap.add_argument("--prefix", default=None, help="file prefix; default=YYYYMMDD_HHMM")
    args = ap.parse_args()

    Path(args.outdir).mkdir(parents=True, exist_ok=True)
    prefix = args.prefix or dt.datetime.utcnow().strftime("%Y%m%d_%H%M")

    print("▶ Fetching news…")
    news = fetch_news(limit=args.limit, hours_back=args.hours, sources=(DEFAULT_RSS + args.rss))
    news = enrich_news(news)

    print("▶ Fetching market…")
    dxy_sym, dxy = pct_wow([TICKERS["DXY"], TICKERS["DXY_ALT"]])
    br_sym, brent = pct_wow([TICKERS["BRENT"], TICKERS["WTI"]])
    es_sym, estoxx = pct_wow([TICKERS["EUROSTOXX"], TICKERS["EUROSTOXX_ALT"]])
    ust_bp = ust10y_change_bp()

    market = {
        "dxy_symbol": dxy_sym, "dxy_wow": dxy,
        "brent_symbol": br_sym, "brent_wow": brent,
        "eurostoxx_symbol": es_sym, "eurostoxx_wow": estoxx,
        "ust10y_bp": ust_bp,
        "asof_utc": _iso_z(_now_utc()),
    }

    print("▶ Aggregating features…")
    agg = aggregate_features(news, market)

    f_json  = Path(args.outdir) / f"{prefix}_features.json"
    f_jsonl = Path(args.outdir) / f"{prefix}_news.jsonl"
    f_meta  = Path(args.outdir) / f"{prefix}_meta.json"

    with open(f_json, "w", encoding="utf-8") as f:
        json.dump(agg, f, ensure_ascii=False, indent=2)

    with open(f_jsonl, "w", encoding="utf-8") as f:
        for it in news:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")

    with open(f_meta, "w", encoding="utf-8") as f:
        json.dump({
            "attachments": [
                f"FX: DXY {agg['market']['dxy_wow']}% WoW; Brent {agg['market']['brent_wow']}% WoW; "
                f"UST10Y {agg['market']['ust10y_bp']}bp; EuroStoxx {agg['market']['eurostoxx_wow']}% WoW."
            ],
            "notes": "Real dataset built from live RSS/Yahoo. Tags & sentiment are heuristic.",
            "tickers_used": {k:v for k,v in TICKERS.items()},
            "sources_used": DEFAULT_RSS + args.rss,
        }, f, ensure_ascii=False, indent=2)

    print("✓ Wrote:")
    print(" ", f_json)
    print(" ", f_jsonl)
    print(" ", f_meta)

if __name__ == "__main__":
    main()
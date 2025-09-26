#!/usr/bin/env python3
"""
Backtest: couple news to price moves (event study)

What it does
------------
- Loads your enriched JSONL news (from finnews.py + nlp_enrich.py)
- Extracts (timestamp, ticker(s), sentiment, event_class, relevance)
- Fetches historical OHLCV via yfinance (with a tiny on-disk cache)
- Computes abnormal returns around each article date using a market model baseline
- Aggregates CARs by sentiment bucket / event class / source / region
- Outputs:
  * per_event.csv         : one row per (article,ticker) with AR/CAR
  * aggregates.csv        : grouped stats (mean/median t-stats)
  * diagnostics.json      : counts, data coverage, warnings
  * (optional) PNG plots  : average CAR curves by bucket

CLI Examples
------------
python backtest_news_impact.py \
  --news data/news_enriched.jsonl \
  --index SPY \
  --window -1,1 --post 5 \
  --bucket sentiment --min_relevance 0.3

python backtest_news_impact.py \
  --news data/news_enriched.jsonl \
  --index ^GSPTSE --window -1,1 --post 3 --region CA --sector_map data/ticker_sector.csv

Notes
-----
- Requires: pandas, numpy, yfinance, scipy, matplotlib (optional for plots)
- Internet access is needed at runtime for price fetches unless cached.
- Timestamps are assumed UTC in the input JSONL (finnews.py already outputs ISO UTC).
- Trading-day alignment: we compute D0 as the first market session at/after the article time
  (using daily bars). For higher precision you can later switch to intraday data.
"""

from __future__ import annotations
import argparse
import json
import os
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple
from pathlib import Path

import numpy as np
import pandas as pd

from core.io_utils import read_jsonl, write_jsonl, Cache, get_artifacts_dir
from core.stock_utils import fetch_price_history
from ingestion.finnews import Article  # Import the Article class from finnews.py

try:
    import yfinance as yf
except Exception:
    yf = None

try:
    from scipy import stats
except Exception:
    stats = None

# ---------------------------
# Utilities
# ---------------------------

def parse_window(s: str) -> Tuple[int, int]:
    """Parse window like "-1,1" -> (-1, 1)"""
    a, b = s.split(",")
    return int(a), int(b)


def ensure_tz_utc(ts: datetime) -> datetime:
    if ts.tzinfo is None:
        return ts.replace(tzinfo=timezone.utc)
    return ts.astimezone(timezone.utc)


# Article class imported from ingestion.finnews at top of file


# ---------------------------
# Load news
# ---------------------------

def load_news(jsonl_path: str,
              region_filter: Optional[str] = None,
              min_relevance: Optional[float] = None,
              min_abs_sent: Optional[float] = None,
              max_age_days: Optional[int] = None) -> List[Article]:
    rows: List[Article] = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            # Expected enriched fields
            ts = obj.get("published") or obj.get("time") or obj.get("timestamp")
            if ts is None:
                continue
            try:
                dt = ensure_tz_utc(datetime.fromisoformat(ts.replace("Z", "+00:00")))
            except Exception:
                continue

            if max_age_days is not None:
                if (datetime.now(timezone.utc) - dt).days > max_age_days:
                    continue

            tickers = obj.get("tickers") or obj.get("entities", {}).get("tickers") or []
            # normalize tickers (upper, strip)
            tickers = sorted({str(t).upper().strip() for t in tickers if str(t).strip()})
            if not tickers:
                # fallback: if entities has a single company with ticker-like key
                pass

            sent = float(obj.get("sentiment", 0.0))
            if min_abs_sent is not None and abs(sent) < min_abs_sent:
                continue

            rel = obj.get("relevance")
            if rel is not None:
                try:
                    rel = float(rel)
                except Exception:
                    rel = None
            if min_relevance is not None and (rel is None or rel < min_relevance):
                continue

            src = obj.get("source")
            reg = obj.get("region")
            if region_filter and reg and region_filter.upper() != str(reg).upper():
                continue

            art = Article(
                ts=dt,
                tickers=tickers,
                sentiment=sent,
                event_class=obj.get("event_class"),
                relevance=rel,
                source=src,
                region=reg,
                id=obj.get("id") or obj.get("link")
            )
            if art.tickers:
                rows.append(art)
    return rows


# ---------------------------
# Market data with light cache
# ---------------------------

class PriceCache:
    def __init__(self, root: str = ".cache_prices"):
        self.root = root
        os.makedirs(self.root, exist_ok=True)

    def path(self, ticker: str) -> str:
        return os.path.join(self.root, f"{ticker.upper()}.parquet")

    def get(self, ticker: str) -> Optional[pd.DataFrame]:
        p = self.path(ticker)
        if os.path.exists(p):
            try:
                df = pd.read_parquet(p)
                # Ensure tz-aware index
                if df.index.tz is None:
                    df.index = df.index.tz_localize("UTC")
                return df
            except Exception:
                return None
        return None

    def set(self, ticker: str, df: pd.DataFrame) -> None:
        p = self.path(ticker)
        try:
            df.to_parquet(p, index=True)
        except Exception:
            pass


def fetch_daily_prices(ticker: str, start: datetime, end: datetime, cache: PriceCache) -> pd.DataFrame:
    # Expand window for estimation period
    start_ = (start - timedelta(days=400)).date()
    end_ = (end + timedelta(days=10)).date()

    df = cache.get(ticker)
    if df is None or df.index.min().date() > start_ or df.index.max().date() < end_:
        if yf is None:
            raise RuntimeError("yfinance not available; please install it.")
        t = yf.Ticker(ticker)
        hist = t.history(start=str(start_), end=str(end_ + timedelta(days=1)), interval="1d", auto_adjust=True)
        if hist.empty:
            raise RuntimeError(f"No price data for {ticker}")
        hist.index = pd.to_datetime(hist.index, utc=True)
        hist = hist.rename(columns={"Close": "close", "Open": "open", "High": "high", "Low": "low", "Volume": "volume"})
        cache.set(ticker, hist)
        df = hist
    # Ensure proper slice
    return df


# ---------------------------
# Event study
# ---------------------------

@dataclass
class EventConfig:
    pre_days: int  # e.g., -1 (D-1 to D-1)
    post_days: int  # e.g., +1, +3, +5
    est_days: int = 120  # market model estimation window length
    align_to_open: bool = True  # place event at next trading day open if article came after market close


def nearest_trading_day(df: pd.DataFrame, ts: datetime, align_to_open: bool = True) -> pd.Timestamp:
    # df index is trading days (UTC). If ts date exists, choose that; else pick next index >= ts
    idx = df.index
    # move to date boundary for daily bars
    # If article after 20:00 UTC we can align next day; keep simple: next index >= ts
    locs = idx.searchsorted(pd.Timestamp(ts))
    if locs >= len(idx):
        return idx[-1]
    return idx[locs]


def compute_market_model_AR(stock: pd.Series, market: pd.Series, event_loc: int, pre: int, post: int, est_len: int) -> Tuple[pd.DataFrame, Dict[str, float]]:
    # stock & market: aligned daily close returns series
    # estimation window: [event_loc - est_len - 1, event_loc - 1]
    # event window: [event_loc + pre, event_loc + post]
    est_start = max(0, event_loc - est_len - 1)
    est_end = max(0, event_loc - 1)
    est_X = market.iloc[est_start:est_end]
    est_y = stock.iloc[est_start:est_end]
    if len(est_X) < est_len // 2:
        raise RuntimeError("Not enough estimation data")
    X = np.vstack([np.ones(len(est_X)), est_X.values]).T
    beta = np.linalg.lstsq(X, est_y.values, rcond=None)[0]  # alpha, beta

    # expected returns over event window
    win = market.iloc[event_loc + pre: event_loc + post + 1]
    s_win = stock.iloc[event_loc + pre: event_loc + post + 1]
    exp = beta[0] + beta[1] * win
    ar = s_win - exp
    car = ar.cumsum()

    df = pd.DataFrame({
        "ret_stock": s_win.values,
        "ret_mkt": win.values,
        "ar": ar.values,
        "car": car.values,
    }, index=range(pre, post + 1))

    stats_out = {}
    if stats is not None and len(ar) > 1:
        t, p = stats.ttest_1samp(ar.values, 0.0, nan_policy='omit')
        stats_out.update({"t_ar": float(t), "p_ar": float(p)})
        t2, p2 = stats.ttest_1samp(car.values, 0.0, nan_policy='omit')
        stats_out.update({"t_car": float(t2), "p_car": float(p2)})

    stats_out.update({"alpha": float(beta[0]), "beta": float(beta[1])})
    return df, stats_out


def make_daily_returns(df: pd.DataFrame) -> pd.Series:
    close = df["close"].astype(float)
    rets = close.pct_change().dropna()
    rets.index = pd.to_datetime(rets.index, utc=True)
    return rets


# ---------------------------
# Main pipeline
# ---------------------------

def run_backtest(news_path: str,
                 index_ticker: str,
                 out_dir: str,
                 window_pre: int,
                 window_post: int,
                 est_days: int,
                 min_relevance: Optional[float],
                 min_abs_sent: Optional[float],
                 region: Optional[str],
                 plot: bool,
                 cache_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    cache = PriceCache(cache_dir)

    arts = load_news(news_path, region_filter=region, min_relevance=min_relevance, min_abs_sent=min_abs_sent)
    if not arts:
        raise SystemExit("No eligible articles after filters.")

    # Collect tickers universe
    tickers = sorted({t for a in arts for t in a.tickers})
    tickers = [t for t in tickers if t.upper() != index_ticker.upper()]

    # Determine date range
    ts_min = min(a.ts for a in arts) - timedelta(days=est_days + 10)
    ts_max = max(a.ts for a in arts) + timedelta(days=window_post + 5)

    # Fetch market & returns
    mkt_df = fetch_daily_prices(index_ticker, ts_min, ts_max, cache)
    mkt_ret = make_daily_returns(mkt_df)

    per_event_rows = []
    diagnostics = {"articles": len(arts), "tickers": len(tickers), "errors": 0}

    for tkr in tickers:
        try:
            df = fetch_daily_prices(tkr, ts_min, ts_max, cache)
            s_ret = make_daily_returns(df)
            # align indexes
            pair = pd.concat([s_ret, mkt_ret], axis=1, join="inner")
            pair.columns = ["s", "m"]
            if pair.empty:
                continue
        except Exception:
            diagnostics["errors"] = diagnostics.get("errors", 0) + 1
            continue

        for a in [a for a in arts if tkr in a.tickers]:
            # find event day index into pair
            d0 = nearest_trading_day(pair, a.ts)
            # event_loc is index position of d0 in pair
            try:
                event_loc = pair.index.get_indexer_for([d0])[0]
            except Exception:
                continue
            try:
                df_ev, stat = compute_market_model_AR(
                    stock=pair["s"], market=pair["m"], event_loc=event_loc,
                    pre=window_pre, post=window_post, est_len=est_days,
                )
            except Exception:
                diagnostics["errors"] = diagnostics.get("errors", 0) + 1
                continue

            car_last = float(df_ev["car"].iloc[-1]) if not df_ev.empty else np.nan
            row = {
                "id": a.id,
                "ticker": tkr,
                "source": a.source,
                "region": a.region,
                "timestamp": a.ts.isoformat(),
                "sentiment": a.sentiment,
                "event_class": a.event_class,
                "relevance": a.relevance,
                "alpha": stat.get("alpha"),
                "beta": stat.get("beta"),
                "car_end": car_last,
                "t_ar": stat.get("t_ar"),
                "p_ar": stat.get("p_ar"),
                "t_car": stat.get("t_car"),
                "p_car": stat.get("p_car"),
            }
            # attach per-day CAR/AR columns with k=-1..+post
            for k, r in df_ev.iterrows():
                row[f"ar_{k}"] = float(r["ar"]) if pd.notna(r["ar"]) else np.nan
                row[f"car_{k}"] = float(r["car"]) if pd.notna(r["car"]) else np.nan
            per_event_rows.append(row)

    if not per_event_rows:
        raise SystemExit("No per-event results computed.")

    per_event = pd.DataFrame(per_event_rows)
    per_event.to_csv(os.path.join(out_dir, "per_event.csv"), index=False)

    # Aggregates
    def bucket_sent(s: float) -> str:
        if s is None:
            return "unknown"
        if s >= 0.25:
            return "+"
        if s <= -0.25:
            return "-"
        return "neutral"

    per_event["bucket_sentiment"] = per_event["sentiment"].apply(lambda x: bucket_sent(float(x) if x is not None else 0.0))
    grp_cols = ["bucket_sentiment", "event_class", "region", "source"]
    aggs = per_event.groupby(grp_cols).agg(
        n=("car_end", "count"),
        mean_car=("car_end", "mean"),
        median_car=("car_end", "median"),
    ).reset_index()

    # simple t-stats on CAR per group
    tstats = []
    if stats is not None:
        for keys, sub in per_event.groupby(grp_cols):
            t, p = stats.ttest_1samp(sub["car_end"].dropna().values, 0.0) if len(sub) > 1 else (np.nan, np.nan)
            tstats.append({"keys": keys, "t": float(t) if not np.isnan(t) else np.nan, "p": float(p) if not np.isnan(p) else np.nan})
    aggs.to_csv(os.path.join(out_dir, "aggregates.csv"), index=False)

    diag = {
        **diagnostics,
        "groups": len(aggs),
        "news_path": news_path,
        "index": index_ticker,
        "window_pre": window_pre,
        "window_post": window_post,
        "est_days": est_days,
    }
    with open(os.path.join(out_dir, "diagnostics.json"), "w", encoding="utf-8") as f:
        json.dump(diag, f, indent=2)

    # Optional plots
    if plot:
        try:
            import matplotlib.pyplot as plt
            # average CAR curve by sentiment bucket
            ks = sorted({int(c.split("_")[1]) for c in per_event.columns if c.startswith("car_") and c != "car_end"})
            idx = pd.Index(ks, name="day")
            for bucket, sub in per_event.groupby("bucket_sentiment"):
                mat = []
                for _, r in sub.iterrows():
                    row = [r.get(f"car_{k}", np.nan) for k in ks]
                    mat.append(row)
                M = np.array(mat, dtype=float)
                avg = np.nanmean(M, axis=0)
                plt.figure()
                plt.plot(idx, avg)
                plt.axhline(0, linestyle="--", linewidth=1)
                plt.title(f"Avg CAR by day — bucket={bucket} (n={len(sub)})")
                plt.xlabel("Event day k")
                plt.ylabel("CAR")
                plt.tight_layout()
                plt.savefig(os.path.join(out_dir, f"avg_car_{bucket}.png"), dpi=144)
                plt.close()
        except Exception:
            pass


# ---------------------------
# CLI
# ---------------------------

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Backtest news impact via event study (daily)")
    p.add_argument("--news", required=True, help="Path to enriched JSONL news (from finnews.py + nlp_enrich.py)")
    p.add_argument("--index", default="SPY", help="Market index ticker for baseline (e.g., SPY, ^GSPTSE, ^GDAXI)")
    p.add_argument("--out", default="out_backtest", help="Output directory")
    p.add_argument("--window", nargs="+", default=["-1","1"],help="pre post window in days. Examples: --window -1 1   or   --window -1,1")
    p.add_argument("--post", type=int, default=None, help="Alternative way to set post days (keeps pre from --window)")
    p.add_argument("--est_days", type=int, default=120, help="Estimation window length for market model")
    p.add_argument("--min_relevance", type=float, default=None, help="Drop articles with relevance < x")
    p.add_argument("--min_abs_sent", type=float, default=None, help="Drop articles with |sentiment| < x")
    p.add_argument("--region", type=str, default=None, help="Filter by region tag (US, CA, INTL, GEO)")
    p.add_argument("--plot", action="store_true", help="Save PNG plots of average CAR")
    p.add_argument("--cache", default=".cache_prices", help="Directory for on-disk price cache")

    import re
    args = p.parse_args()
    
    def _parse_window(ws):
        if len(ws) == 1 and ("," in ws[0] or ":" in ws[0]):
            a, b = re.split(r"[,:]", ws[0])
            return int(a), int(b)
        if len(ws) == 2:
            return int(ws[0]), int(ws[1])
        raise ValueError("Invalid --window. Use: --window -1 1  or  --window -1,1")
    
    pre, post = _parse_window(args.window)
    if args.post is not None:
        post = args.post

    run_backtest(
        news_path=args.news,
        index_ticker=args.index,
        out_dir=args.out,
        window_pre=pre,
        window_post=post,
        est_days=args.est_days,
        min_relevance=args.min_relevance,
        min_abs_sent=args.min_abs_sent,
        region=args.region,
        plot=args.plot,
        cache_dir=args.cache,
    )

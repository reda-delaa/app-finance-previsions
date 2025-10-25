"""
Thin data access layer for prices, fundamentals, and macro series.

Prefers API keys from environment when available:
- FINNHUB_API_KEY for quotes/fundamentals via Finnhub
- FRED_API_KEY for macro via FRED JSON API

Falls back to yfinance for prices/fundamentals and FRED CSV for macro.
All functions are best-effort and return None/empty on failure.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional
from datetime import datetime
import os
import io

import pandas as pd
import requests


def _env(name: str) -> Optional[str]:
    v = os.getenv(name)
    return v.strip() if isinstance(v, str) and v.strip() else None


# ================= Prices =================
def get_price_history(ticker: str, start: Optional[str] = None, end: Optional[str] = None, interval: str = "1d") -> Optional[pd.DataFrame]:
    """Fetch OHLCV history using yfinance. Returns DataFrame or None."""
    try:
        import yfinance as yf
        stock = yf.Ticker(ticker)
        kw: Dict[str, Any] = {"interval": interval, "auto_adjust": True}
        if start:
            kw["start"] = start
        if end:
            kw["end"] = end
        df = stock.history(**kw)
        if df is None or df.empty:
            return None
        if getattr(df.index, "tz", None) is not None:
            df.index = df.index.tz_localize(None)
        # ensure expected cols
        for col in ["Open", "High", "Low", "Close", "Volume"]:
            if col not in df.columns:
                df[col] = pd.NA
        return df
    except Exception:
        return None


# ================= Fundamentals =================
def get_fundamentals(symbol: str) -> Dict[str, Any]:
    """Return a minimal fundamentals dict. Prefer Finnhub, fallback to yfinance."""
    # Finnhub first
    api_key = _env("FINNHUB_API_KEY")
    if api_key:
        try:
            base = "https://finnhub.io/api/v1"
            hdr = {"User-Agent": "analyse-financiere/1.0"}
            quote = requests.get(f"{base}/quote", params={"symbol": symbol, "token": api_key}, timeout=15, headers=hdr).json()
            prof = requests.get(f"{base}/stock/profile2", params={"symbol": symbol, "token": api_key}, timeout=15, headers=hdr).json()
            metrics = requests.get(f"{base}/stock/metric", params={"symbol": symbol, "metric": "all", "token": api_key}, timeout=20, headers=hdr).json()
            data = {
                "price": (quote or {}).get("c"),
                "market_cap": (metrics or {}).get("metric", {}).get("marketCapitalization"),
                "pe": (metrics or {}).get("metric", {}).get("peNormalizedAnnual"),
                "beta": (metrics or {}).get("metric", {}).get("beta"),
                "dividend_yield": (metrics or {}).get("metric", {}).get("dividendYieldIndicatedAnnual"),
                "name": (prof or {}).get("name"),
                "exchange": (prof or {}).get("exchange"),
                "country": (prof or {}).get("country"),
                "currency": (prof or {}).get("currency"),
                "source": "finnhub",
            }
            return {k: v for k, v in data.items() if v is not None}
        except Exception:
            pass
    # yfinance fallback
    try:
        import yfinance as yf
        it = yf.Ticker(symbol)
        info = it.info if hasattr(it, "info") else {}
        price = None
        try:
            q = it.fast_info  # faster path if available
            price = getattr(q, "last_price", None)
        except Exception:
            pass
        data = {
            "price": price,
            "market_cap": info.get("marketCap"),
            "pe": info.get("trailingPE") or info.get("forwardPE"),
            "beta": info.get("beta"),
            "dividend_yield": info.get("dividendYield"),
            "name": info.get("longName") or info.get("shortName"),
            "exchange": info.get("exchange"),
            "country": info.get("country"),
            "currency": info.get("currency"),
            "source": "yfinance",
        }
        return {k: v for k, v in data.items() if v is not None}
    except Exception:
        return {}


# ================= Macro (FRED) =================
def _normalize_fred_key(key: Optional[str]) -> Optional[str]:
    if not key:
        return None
    k = key.strip()
    if len(k) == 32 and k.isalnum() and k == k.lower():
        return k
    return None


def get_fred_series(series_id: str, start: Optional[str] = None) -> pd.DataFrame:
    """Return a single-column DataFrame for a FRED series. Best-effort."""
    key = _normalize_fred_key(_env("FRED_API_KEY"))
    # 1) JSON API
    if key:
        try:
            params = {"series_id": series_id, "api_key": key, "file_type": "json"}
            if start:
                params["observation_start"] = pd.to_datetime(start).strftime("%Y-%m-%d")
            r = requests.get("https://api.stlouisfed.org/fred/series/observations", params=params, timeout=30)
            r.raise_for_status()
            js = r.json()
            if "observations" in js:
                df = pd.DataFrame(js["observations"])
                if not df.empty and {"date", "value"}.issubset(df.columns):
                    df["date"] = pd.to_datetime(df["date"], errors="coerce")
                    df["value"] = pd.to_numeric(df["value"].replace(".", pd.NA), errors="coerce")
                    df = df.set_index("date")[["value"]].rename(columns={"value": series_id}).sort_index()
                    return df
        except Exception:
            pass
    # 2) CSV fallback
    try:
        url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
        resp = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=30)
        resp.raise_for_status()
        text = resp.text.strip()
        if text[:1] in ("<", "{"):
            raise RuntimeError("FRED returned non-CSV content")
        df = pd.read_csv(io.StringIO(text))
        # date col
        date_col = None
        for cand in ("DATE", "date", "observation_date"):
            if cand in df.columns:
                date_col = cand; break
        if date_col is None:
            raise KeyError("No DATE column")
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
        df = df.set_index(date_col)
        # value col
        if series_id in df.columns:
            val_col = series_id
        elif "VALUE" in df.columns:
            val_col = "VALUE"
        elif "value" in df.columns:
            val_col = "value"
        else:
            candidates = [c for c in df.columns if c.lower() not in ("date", "observation_date")]
            if len(candidates) == 1:
                val_col = candidates[0]
            else:
                raise KeyError("Cannot identify value column")
        df[val_col] = pd.to_numeric(df[val_col].replace(".", pd.NA), errors="coerce")
        out = df[[val_col]].rename(columns={val_col: series_id}).sort_index()
        if start:
            out = out[out.index >= pd.to_datetime(start)]
        return out
    except Exception:
        return pd.DataFrame(columns=[series_id])


@dataclass
class SnapshotInputs:
    ticker: Optional[str] = None
    window: str = "last_week"
    regions: str = "US,INTL"
    query: str = ""


"""
Backfill 5y prices for WATCHLIST tickers (or data/watchlist.json) using yfinance.
Writes to data/prices/ticker=XYZ/prices.parquet
"""

from __future__ import annotations

import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import List
import pandas as pd


def _load_watchlist() -> List[str]:
    wl = [x.strip().upper() for x in (os.getenv('WATCHLIST') or '').split(',') if x.strip()]
    if wl:
        return wl
    try:
        import json
        obj = json.loads(Path('data/watchlist.json').read_text(encoding='utf-8'))
        return [x.strip().upper() for x in (obj.get('watchlist') or []) if isinstance(x, str) and x.strip()]
    except Exception:
        return []


def fetch_prices(ticker: str, years: int = 5) -> pd.DataFrame | None:
    try:
        import yfinance as yf
        stock = yf.Ticker(ticker)
        start = (datetime.utcnow() - timedelta(days=365*years+30)).strftime('%Y-%m-%d')
        df = stock.history(start=start, interval='1d', auto_adjust=True)
        if df is None or df.empty:
            return None
        if getattr(df.index, 'tz', None) is not None:
            df.index = df.index.tz_localize(None)
        for col in ['Open','High','Low','Close','Volume']:
            if col not in df.columns:
                df[col] = pd.NA
        df = df.reset_index().rename(columns={'Date':'date'})
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        return df
    except Exception:
        return None


def main() -> int:
    wl = _load_watchlist()
    if not wl:
        print({'ok': False, 'error': 'empty watchlist'})
        return 1
    out = []
    for t in wl:
        df = fetch_prices(t, years=int(os.getenv('BACKFILL_YEARS','5')))
        if df is not None and not df.empty:
            p = Path('data/prices')/f'ticker={t}'/'prices.parquet'
            p.parent.mkdir(parents=True, exist_ok=True)
            try:
                df.to_parquet(p, index=False)
                out.append(str(p))
            except Exception:
                continue
    print({'ok': True, 'written': len(out)})
    return 0


if __name__ == '__main__':
    raise SystemExit(main())


"""
Earnings Calendar Agent — collects upcoming earnings dates for watchlist tickers (best‑effort via yfinance).

Output: data/earnings/dt=YYYYMMDD/earnings.json with a list of events {ticker, date, info} sorted.
"""

from __future__ import annotations

from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any
import json


def _load_watchlist() -> List[str]:
    import os
    wl = [x.strip().upper() for x in (os.getenv('WATCHLIST') or '').split(',') if x.strip()]
    if wl:
        return wl
    try:
        obj = json.loads(Path('data/watchlist.json').read_text(encoding='utf-8'))
        return [x.strip().upper() for x in (obj.get('watchlist') or []) if isinstance(x, str) and x.strip()]
    except Exception:
        return []


def _earnings_for(ticker: str) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    try:
        import yfinance as yf
        t = yf.Ticker(ticker)
        # Try get_earnings_dates if available
        df = None
        try:
            df = t.get_earnings_dates(limit=12)
        except Exception:
            pass
        if df is not None and not df.empty:
            for idx, row in df.reset_index().iterrows():
                try:
                    d = row.get('Earnings Date') or row.get('index')
                    out.append({
                        'ticker': ticker,
                        'date': str(d)[:10],
                        'info': {k: (None if k=='Earnings Date' else row.get(k)) for k in df.columns},
                    })
        else:
            # Fallback to .calendar
            cal = t.calendar
            if cal is not None and not cal.empty:
                if 'Earnings Date' in cal.index:
                    val = cal.loc['Earnings Date'].values[0]
                    out.append({'ticker': ticker, 'date': str(val)[:10], 'info': {'source': 'calendar'}})
    except Exception:
        pass
    return out


def run() -> Path:
    wl = _load_watchlist()
    events: List[Dict[str, Any]] = []
    for t in wl[:50]:  # safety cap
        events.extend(_earnings_for(t))
    # sort by date
    try:
        events = sorted(events, key=lambda x: x.get('date') or '')
    except Exception:
        pass
    out = {'asof': datetime.utcnow().isoformat()+'Z', 'events': events}
    outdir = Path('data/earnings')/f"dt={datetime.utcnow().strftime('%Y%m%d')}"
    outdir.mkdir(parents=True, exist_ok=True)
    p = outdir/'earnings.json'
    p.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding='utf-8')
    return p


if __name__ == '__main__':
    print(run())


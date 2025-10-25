from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List

import pandas as pd


def _today_dt() -> str:
    return datetime.utcnow().strftime("%Y%m%d")


def _latest_dt_under(base: str, pattern: str) -> str | None:
    try:
        parts = sorted(Path(base).glob(pattern))
        if not parts:
            return None
        p = parts[-1]
        if 'dt=' in p.as_posix():
            return p.as_posix().split('dt=')[-1].split('/')[0]
        # Fallback: parent folder is dt=…
        return p.parent.as_posix().split('dt=')[-1].split('/')[0]
    except Exception:
        return None


def _watchlist() -> List[str]:
    import os
    env = os.getenv("WATCHLIST")
    if env:
        return [t.strip().upper() for t in env.split(',') if t.strip()]
    p = Path('data/watchlist.json')
    if p.exists():
        try:
            obj = json.loads(p.read_text(encoding='utf-8'))
            if isinstance(obj, list):
                return [str(x).upper() for x in obj if str(x).strip()]
        except Exception:
            pass
    return ["AAPL","MSFT","NVDA","SPY"]


def _prices_coverage_ok(tickers: List[str], years_min: int = 5) -> float | None:
    """Return ratio of tickers whose local parquet coverage >= years_min (best-effort)."""
    ok = 0
    seen = 0
    for t in tickers:
        try:
            p = Path('data/prices')/f'ticker={t}'/'prices.parquet'
            if not p.exists():
                continue
            df = pd.read_parquet(p)
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'], errors='coerce')
                df = df.set_index('date')
            if df.empty:
                continue
            age_days = (df.index.max() - df.index.min()).days
            if age_days >= years_min*365 - 5:
                ok += 1
            seen += 1
        except Exception:
            continue
    if seen == 0:
        return None
    return ok/seen


def run_once() -> Dict[str, Any]:
    latest_forecast_dt = _latest_dt_under('data/forecast', 'dt=*/forecasts.parquet')
    latest_final_dt = _latest_dt_under('data/forecast', 'dt=*/final.parquet')
    latest_macro_dt = _latest_dt_under('data/macro/forecast', 'dt=*/macro_forecast.json')
    latest_news_dt = _latest_dt_under('data/news', 'dt=*/*')
    latest_quality_dt = _latest_dt_under('data/quality', 'dt=*/report.json')

    pr_cov = _prices_coverage_ok(_watchlist(), years_min=5)

    rep = {
        "asof": datetime.utcnow().isoformat()+"Z",
        "dt": _today_dt(),
        "latest": {
            "forecast_dt": latest_forecast_dt,
            "final_dt": latest_final_dt,
            "macro_forecast_dt": latest_macro_dt,
            "news_dt": latest_news_dt,
            "quality_dt": latest_quality_dt,
        },
        "checks": {
            "forecasts_today": latest_forecast_dt == _today_dt(),
            "final_today": latest_final_dt == _today_dt(),
            "macro_today": latest_macro_dt == _today_dt(),
            "prices_5y_coverage_ratio": pr_cov,
        }
    }

    outdir = Path('data/quality')/f'dt={_today_dt()}'
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir/'freshness.json').write_text(json.dumps(rep, ensure_ascii=False, indent=2), encoding='utf-8')
    return rep


if __name__ == '__main__':
    print(json.dumps(run_once())[:200] + '…')


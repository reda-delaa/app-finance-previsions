from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Iterable

import numpy as np
import pandas as pd

# Lazy imports to keep CLI simple
try:
    from core.market_data import get_price_history
except Exception:  # pragma: no cover
    import sys as _sys
    _SRC = Path(__file__).resolve().parents[1]
    if str(_SRC) not in _sys.path:
        _sys.path.insert(0, str(_SRC))
    from core.market_data import get_price_history


DT_FMT = "%Y%m%d"


def _today_dt() -> str:
    return datetime.utcnow().strftime(DT_FMT)


def _load_watchlist() -> List[str]:
    # Priority: env WATCHLIST, then data/watchlist.json, fallback basic
    env = os.getenv("WATCHLIST")
    if env:
        return [t.strip().upper() for t in env.split(",") if t.strip()]
    p = Path("data/watchlist.json")
    if p.exists():
        try:
            import json
            obj = json.loads(p.read_text(encoding="utf-8"))
            if isinstance(obj, list):
                return [str(x).upper() for x in obj if str(x).strip()]
        except Exception:
            pass
    return ["AAPL", "MSFT", "NVDA", "SPY"]


def _momentum(series: pd.Series, days: int) -> float | None:
    try:
        if len(series) <= days:
            return None
        return float(series.iloc[-1] / series.iloc[-days] - 1.0)
    except Exception:
        return None


def _forecasts_for_ticker(ticker: str, horizons: Iterable[str]) -> List[dict]:
    # Fetch ~500 calendar days to derive simple features (momentum/vol)
    start = (datetime.utcnow().date() - timedelta(days=500)).isoformat()
    hist = get_price_history(ticker, start=start)
    rows: List[dict] = []
    if hist is None or hist.empty or "Close" not in hist.columns:
        for h in horizons:
            rows.append({
                "ticker": ticker,
                "horizon": h,
                "direction": "flat",
                "confidence": 0.5,
                "expected_return": 0.0,
            })
        return rows

    close = pd.to_numeric(hist["Close"], errors="coerce").dropna()
    mom_5 = _momentum(close, 5)
    mom_21 = _momentum(close, 21)
    mom_63 = _momentum(close, 63)

    # Volatility proxy (std of daily returns)
    try:
        rets = close.pct_change().dropna()
        vol = float(rets.std())
    except Exception:
        vol = 0.02

    for h in horizons:
        # Map horizon to momentum window and scaling
        if h == "1w":
            m = mom_5 if mom_5 is not None else 0.0
            scale = 1.0
        elif h == "1m":
            m = mom_21 if mom_21 is not None else 0.0
            scale = 1.2
        else:  # "1y" as coarse proxy
            m = mom_63 if mom_63 is not None else 0.0
            scale = 1.8

        exp_ret = float(m) * scale
        # Clamp expected return to reasonable bounds
        exp_ret = float(np.clip(exp_ret, -0.25, 0.25))
        direction = "up" if exp_ret > 0.0 else ("down" if exp_ret < 0.0 else "flat")
        # Confidence increases with momentum signal-to-noise; cap between 0.35 and 0.85
        try:
            snr = abs(exp_ret) / max(1e-6, vol)
            conf = float(np.clip(0.35 + 0.2 * snr, 0.35, 0.85))
        except Exception:
            conf = 0.5

        rows.append({
            "ticker": ticker,
            "horizon": h,
            "direction": direction,
            "confidence": round(conf, 3),
            "expected_return": round(exp_ret, 4),
        })
    return rows


def run_once() -> Path:
    tickers = _load_watchlist()
    horizons = ["1w", "1m", "1y"]
    all_rows: List[dict] = []
    for t in tickers:
        all_rows.extend(_forecasts_for_ticker(t, horizons))
    df = pd.DataFrame(all_rows)
    df.insert(0, "dt", pd.to_datetime(_today_dt()))
    outdir = Path("data/forecast") / f"dt={_today_dt()}"
    outdir.mkdir(parents=True, exist_ok=True)
    out = outdir / "forecasts.parquet"
    df.to_parquet(out, index=False)
    return out


if __name__ == "__main__":
    p = run_once()
    print(p)

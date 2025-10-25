"""
Simple ML baseline for next-horizon returns using Ridge regression on
recent price-based features. Designed to run fast on a single ticker.

Features (latest day):
- r_5, r_21, r_63: past returns over 5/21/63 trading days
- vol_21: 21-day realized volatility

Target:
- forward horizon return over H days (H in {5,21,252} for 1w/1m/1y)

Returns predicted return (float) and a crude confidence (0..1) based on
sample size and out-of-sample R^2 if available.
"""

from __future__ import annotations

from typing import Tuple, Optional
import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.linear_model import RidgeCV
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import r2_score

HORIZON_TO_DAYS = {"1w": 5, "1m": 21, "1y": 252}


def _load_prices(ticker: str) -> Optional[pd.DataFrame]:
    """Load cached prices parquet if present; fallback to yfinance."""
    pfile = Path("data/prices") / f"ticker={ticker}" / "prices.parquet"
    if pfile.exists():
        try:
            df = pd.read_parquet(pfile)
            if "date" in df.columns:
                df["date"] = pd.to_datetime(df["date"], errors="coerce")
                df = df.set_index("date")
            return df
        except Exception:
            pass
    # fallback
    try:
        import yfinance as yf
        hist = yf.Ticker(ticker).history(period="5y", auto_adjust=True)
        return hist if hist is not None and not hist.empty else None
    except Exception:
        return None


def _make_features(df: pd.DataFrame) -> pd.DataFrame:
    close = df["Close"].dropna().copy()
    r = close.pct_change()
    feats = pd.DataFrame(index=close.index)
    feats["r_5"] = close.pct_change(5)
    feats["r_21"] = close.pct_change(21)
    feats["r_63"] = close.pct_change(63)
    feats["vol_21"] = r.rolling(21).std() * np.sqrt(252)
    return feats.dropna()


def _forward_return(close: pd.Series, days: int) -> pd.Series:
    fwd = close.shift(-days) / close - 1.0
    return fwd


def ml_predict_next_return(ticker: str, horizon: str = "1m") -> Tuple[Optional[float], float]:
    days = HORIZON_TO_DAYS.get(horizon, 21)
    df = _load_prices(ticker)
    if df is None or df.empty or "Close" not in df.columns:
        return None, 0.2
    feats = _make_features(df)
    # align target
    close = df["Close"].reindex(feats.index)
    y = _forward_return(close, days)
    Xy = pd.concat([feats, y.rename("target")], axis=1).dropna()
    if len(Xy) < 150:
        # insufficient history
        # simple momentum proxy = r_21 * 0.5
        val = float(feats.iloc[-1]["r_21"]) * 0.5 if not feats.empty else 0.0
        return val, 0.3
    X = Xy.drop(columns=["target"]).values
    yv = Xy["target"].values
    # time series CV
    try:
        tscv = TimeSeriesSplit(n_splits=5)
        model = RidgeCV(alphas=(0.1, 1.0, 10.0))
        oof_pred = np.zeros_like(yv)
        for train_idx, test_idx in tscv.split(X):
            model.fit(X[train_idx], yv[train_idx])
            oof_pred[test_idx] = model.predict(X[test_idx])
        r2 = max(-1.0, min(1.0, float(r2_score(yv, oof_pred))))
        # fit on full data and predict next
        model.fit(X, yv)
        x_last = feats.iloc[[-1]].values
        pred = float(model.predict(x_last)[0])
        # crude confidence from sample size and R^2
        n = len(Xy)
        conf = max(0.2, min(0.95, 0.4 + 0.3 * max(0.0, r2) + 0.3 * (n / 1000.0)))
        return pred, conf
    except Exception:
        # fallback momentum
        val = float(feats.iloc[-1]["r_21"]) * 0.5 if not feats.empty else 0.0
        return val, 0.3


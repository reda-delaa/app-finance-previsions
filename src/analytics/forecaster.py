"""
Lightweight forecasting helpers (baseline + sentiment blend) for 1w/1m/1y.
Designed to work without heavy dependencies and to integrate with
analytics.market_intel outputs.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple
from dataclasses import dataclass
import pandas as pd

from core.market_data import get_price_history


HORIZON_TO_DAYS = {
    "1w": 5,
    "1m": 21,
    "1y": 252,
}


@dataclass
class ForecastResult:
    horizon: str
    direction: str  # "up" | "down" | "flat"
    confidence: float
    expected_return: Optional[float] = None
    drivers: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "horizon": self.horizon,
            "direction": self.direction,
            "confidence": round(float(self.confidence), 3),
            "expected_return": None if self.expected_return is None else round(float(self.expected_return), 4),
            "drivers": self.drivers or {},
        }


def _sma_signal(close: pd.Series) -> Tuple[str, float]:
    """Simple SMA(20/50/200) composite signal with confidence in [0,1]."""
    if close is None or len(close) < 210:
        return "flat", 0.3
    sma20 = close.rolling(20).mean()
    sma50 = close.rolling(50).mean()
    sma200 = close.rolling(200).mean()
    last = close.iloc[-1]
    s20, s50, s200 = sma20.iloc[-1], sma50.iloc[-1], sma200.iloc[-1]
    score = 0
    if last > s20: score += 1
    if last > s50: score += 1
    if last > s200: score += 1
    if last < s20: score -= 1
    if last < s50: score -= 1
    if last < s200: score -= 1
    # normalize to [-1,1]
    score = max(-3, min(3, score)) / 3.0
    direction = "up" if score > 0.15 else ("down" if score < -0.15 else "flat")
    confidence = 0.5 + 0.4 * abs(score)  # base confidence
    return direction, float(confidence)


def _sentiment_adjust(conf: float, features: Optional[Dict[str, Any]]) -> float:
    if not features:
        return conf
    mean_sent = float(features.get("mean_sentiment", 0.0) or 0.0)
    pos_ratio = float(features.get("pos_ratio", 0.0) or 0.0)
    neg_ratio = float(features.get("neg_ratio", 0.0) or 0.0)
    # small nudges based on sentiment
    adj = 0.05 * (pos_ratio - neg_ratio) + 0.03 * mean_sent
    return max(0.0, min(1.0, conf + adj))


def forecast_ticker(ticker: str, horizon: str = "1m", features: Optional[Dict[str, Any]] = None) -> ForecastResult:
    """Baseline forecast using SMA structure and light sentiment blend."""
    days = HORIZON_TO_DAYS.get(horizon, 21)
    hist = get_price_history(ticker, start=pd.Timestamp.today().normalize() - pd.Timedelta(days=400))
    if hist is None or hist.empty:
        return ForecastResult(horizon=horizon, direction="flat", confidence=0.3, drivers={"reason": "no_history"})
    close = hist["Close"].dropna()
    direction, base_conf = _sma_signal(close)
    conf = _sentiment_adjust(base_conf, features)
    # rough expected return proxy: recent drift
    try:
        ret_days = min(len(close) - 1, days)
        drift = float((close.iloc[-1] / close.iloc[-1 - ret_days]) - 1.0) if ret_days > 0 else 0.0
    except Exception:
        drift = 0.0
    exp_ret = drift if direction == "up" else (-abs(drift) if direction == "down" else 0.0)
    drivers = {"sma": "20/50/200", "sentiment": {k: features.get(k) for k in ("mean_sentiment","pos_ratio","neg_ratio")} if features else {}}
    return ForecastResult(horizon=horizon, direction=direction, confidence=conf, expected_return=exp_ret, drivers=drivers)


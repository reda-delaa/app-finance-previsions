"""
Simple recommender that ranks tickers using forecast outputs and basic features.
Inputs: list of { ticker, features, forecasts: {1w,1m,1y} }
Outputs: sorted list with score and reasons.
"""

from __future__ import annotations

from typing import Any, Dict, List


def _score_item(item: Dict[str, Any]) -> Dict[str, Any]:
    f = item.get("forecasts", {})
    f1m = f.get("1m", {})
    dir_bonus = {"up": 1.0, "flat": 0.0, "down": -1.0}
    # base: 1m is primary; blend some of 1w/1y if present
    base = dir_bonus.get(f1m.get("direction"), 0.0) * (f1m.get("confidence") or 0.0)
    exp = (f1m.get("expected_return") or 0.0)
    features = item.get("features", {})
    momentum = float(features.get("SMA200_z", 0.0) or 0.0) if isinstance(features.get("SMA200_z", 0.0), (int, float)) else 0.0
    sentiment = float(features.get("mean_sentiment", 0.0) or 0.0)
    # simple weighted sum
    score = 1.0 * base + 0.5 * exp + 0.2 * momentum + 0.1 * sentiment
    reasons = []
    if base > 0.4: reasons.append("1m up w/ high confidence")
    if exp > 0.0: reasons.append("positive expected return")
    if momentum > 0.5: reasons.append("positive momentum")
    if sentiment > 0.1: reasons.append("positive sentiment")
    return {
        "ticker": item.get("ticker"),
        "score": round(float(score), 4),
        "reasons": reasons[:4],
        "raw": item,
    }


def rank(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    scored = [_score_item(x) for x in items]
    return sorted(scored, key=lambda r: r["score"], reverse=True)


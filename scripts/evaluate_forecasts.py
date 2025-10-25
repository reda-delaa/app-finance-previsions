#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import duckdb
import pandas as pd
import numpy as np

HORIZON_TO_DAYS = {"1w": 5, "1m": 21, "1y": 252}


def load_forecasts(horizon: str) -> pd.DataFrame:
    con = duckdb.connect()
    df = con.execute(
        """
        select * from read_parquet('data/forecast/dt=*/forecasts.parquet')
        where horizon = $1
        order by dt
        """,
        [horizon],
    ).fetch_df()
    try:
        df["dt"] = pd.to_datetime(df["dt"], errors="coerce")
    except Exception:
        pass
    return df


def cached_prices(ticker: str) -> pd.DataFrame | None:
    p = Path("data/prices") / f"ticker={ticker}" / "prices.parquet"
    if not p.exists():
        return None
    try:
        df = pd.read_parquet(p)
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
            df = df.set_index("date")
        return df
    except Exception:
        return None


def compute_score(df: pd.DataFrame) -> pd.Series:
    dir_base = df["direction"].map({"up": 1.0, "down": -1.0}).fillna(0.0)
    if "ml_return" in df.columns:
        mlc = df.get("ml_conf", 0.6)
        score = (
            dir_base * df["confidence"].astype(float)
            + 0.4 * df["expected_return"].fillna(0.0).astype(float)
            + 0.3 * df["ml_return"].fillna(0.0).astype(float) * mlc
        )
    else:
        score = dir_base * df["confidence"].astype(float) + 0.5 * df["expected_return"].fillna(0.0).astype(float)
    return score


def realized_return_for(ticker: str, dt: pd.Timestamp, days: int) -> float | None:
    dfp = cached_prices(ticker)
    if dfp is None or dfp.empty or "Close" not in dfp.columns:
        return None
    try:
        idx = dfp.index.get_loc(dt, method="nearest")
    except Exception:
        # align to next available date
        after = dfp[dfp.index >= dt]
        if after.empty:
            return None
        idx = dfp.index.get_loc(after.index[0])
    j = min(len(dfp) - 1, idx + days)
    r = float(dfp["Close"].iloc[j] / dfp["Close"].iloc[idx] - 1.0)
    return r


def evaluate(horizon: str = "1m", top_n: int = 5, days_back: int = 120) -> Dict:
    days = HORIZON_TO_DAYS.get(horizon, 21)
    df = load_forecasts(horizon)
    if df.empty:
        return {"ok": False, "error": "no forecasts"}
    # window
    end = df["dt"].max()
    start = end - pd.Timedelta(days=days_back)
    df = df[(df["dt"] >= start) & (df["dt"] <= end)].copy()
    if df.empty:
        return {"ok": False, "error": "no forecasts in window"}

    df["score"] = compute_score(df)
    details: List[Dict] = []
    basket = []
    for d, sdf in df.groupby(df["dt"].dt.date):
        sdf = sdf.sort_values("score", ascending=False).head(top_n)
        rets = []
        for _, row in sdf.iterrows():
            rr = realized_return_for(str(row["ticker"]), pd.Timestamp(d), days)
            if rr is None:
                continue
            rets.append(rr)
            details.append({
                "dt": str(d),
                "ticker": row["ticker"],
                "horizon": horizon,
                "score": float(row["score"]),
                "direction": row["direction"],
                "confidence": float(row["confidence"]),
                "expected_return": float(row.get("expected_return") or 0.0),
                "ml_return": float(row.get("ml_return") or 0.0) if "ml_return" in row else None,
                "ml_conf": float(row.get("ml_conf") or 0.0) if "ml_conf" in row else None,
                "realized_return": float(rr),
            })
        if rets:
            basket.append(np.mean(rets))

    # metrics
    if basket:
        s = pd.Series(basket)
        sharpe = float(s.mean() / s.std(ddof=1)) if s.std(ddof=1) and len(s) > 1 else 0.0
        summary = {
            "count_days": int(s.count()),
            "avg_basket_return": float(s.mean()),
            "median": float(s.median()),
            "stdev": float(s.std(ddof=1)) if len(s) > 1 else 0.0,
            "sharpe_like": sharpe,
        }
    else:
        summary = {"count_days": 0, "avg_basket_return": 0.0, "median": 0.0, "stdev": 0.0, "sharpe_like": 0.0}

    outdir = Path("data/eval")
    outdir.mkdir(parents=True, exist_ok=True)
    Path(outdir / "eval_details.parquet").write_bytes(pd.DataFrame(details).to_parquet(index=False))
    (outdir / "eval_summary.json").write_text(json.dumps({"horizon": horizon, "top_n": top_n, **summary}, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"ok": True, **summary}


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--horizon", default="1m", choices=list(HORIZON_TO_DAYS.keys()))
    p.add_argument("--top-n", type=int, default=5)
    p.add_argument("--days-back", type=int, default=120)
    args = p.parse_args()
    res = evaluate(horizon=args.horizon, top_n=args.top_n, days_back=args.days_back)
    print(json.dumps(res, ensure_ascii=False))


if __name__ == "__main__":
    main()


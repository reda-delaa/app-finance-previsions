"""
Daily agent runner:
- Builds snapshots via analytics.market_intel
- Runs baseline forecasts for watchlist
- Writes outputs to data/forecast/dt=YYYYMMDD/
"""

from __future__ import annotations

import os
import json
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any

# Ensure src on sys.path for module imports
import sys as _sys
_SRC_ROOT = Path(__file__).resolve().parents[1] / "src"
if str(_SRC_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_SRC_ROOT))

from analytics.market_intel import build_snapshot
from analytics.forecaster import forecast_ticker
from analytics.ml_baseline import ml_predict_next_return
from core.data_store import write_parquet
from core.market_data import get_fred_series, get_price_history
import pandas as pd


WATCHLIST = os.getenv("WATCHLIST", "NGD.TO,AEM.TO,ABX.TO,K.TO,GDX").split(",")
OUTDIR = Path("data/forecast") / f"dt={datetime.utcnow().strftime('%Y%m%d')}"


def main() -> int:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    results: Dict[str, Any] = {"date": datetime.utcnow().isoformat() + "Z", "tickers": []}

    rows = []
    feat_rows = []
    feat_flat_rows = []
    for t in [x.strip().upper() for x in WATCHLIST if x.strip()]:
        try:
            snap = build_snapshot(regions=["US","INTL"], window="last_week", ticker=t, limit=150)
            feats = snap.get("features") or {}
            f_1w = forecast_ticker(t, horizon="1w", features=feats).to_dict()
            f_1m = forecast_ticker(t, horizon="1m", features=feats).to_dict()
            f_1y = forecast_ticker(t, horizon="1y", features=feats).to_dict()
            # ML baseline for 1m (and optionally others)
            ml_1m, mlc_1m = ml_predict_next_return(t, "1m")
            if ml_1m is not None:
                f_1m["ml_return"], f_1m["ml_conf"] = float(ml_1m), float(mlc_1m)
            item = {"ticker": t, "features": feats, "forecasts": {"1w": f_1w, "1m": f_1m, "1y": f_1y}}
            results["tickers"].append(item)
            # write per-ticker file
            with (OUTDIR / f"{t}.json").open("w", encoding="utf-8") as f:
                json.dump(item, f, ensure_ascii=False, indent=2)
            # add parquet rows (one per horizon)
            for h, fh in (("1w", f_1w), ("1m", f_1m), ("1y", f_1y)):
                rows.append({
                    "dt": datetime.utcnow().strftime("%Y-%m-%d"),
                    "ticker": t,
                    "horizon": h,
                    "direction": fh.get("direction"),
                    "confidence": fh.get("confidence"),
                    "expected_return": fh.get("expected_return"),
                    "ml_return": fh.get("ml_return"),
                    "ml_conf": fh.get("ml_conf"),
                    "drivers_json": json.dumps(fh.get("drivers") or {}, ensure_ascii=False),
                })
            # features parquet row (store as JSON for schema stability)
            feat_row = {
                "dt": datetime.utcnow().strftime("%Y-%m-%d"),
                "ticker": t,
                "features_json": json.dumps(feats or {}, ensure_ascii=False),
            }
            feat_rows.append(feat_row)
            # flatten a subset of numeric features into columns for fast filtering
            flat: Dict[str, Any] = {k: v for k, v in (feats or {}).items() if isinstance(v, (int, float))}
            # keep only a manageable subset
            keep_keys = [
                "news_count", "mean_sentiment", "pos_ratio", "neg_ratio",
                "y_price", "y_market_cap", "y_pe", "y_beta", "dividend_yield",
            ] + [k for k in flat.keys() if k.startswith("macro_")]
            flat_selected = {k: float(flat[k]) for k in keep_keys if k in flat}
            flat_selected.update({"dt": feat_row["dt"], "ticker": t})
            feat_flat_rows.append(flat_selected)
        except Exception as e:
            results["tickers"].append({"ticker": t, "error": str(e)})

    # write summary
    with (OUTDIR / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    # write parquet table partition (dt=YYYYMMDD)
    try:
        df = pd.DataFrame(rows)
        pq_path = Path("data/forecast") / f"dt={datetime.utcnow().strftime('%Y%m%d')}" / "forecasts.parquet"
        write_parquet(df, pq_path)
    except Exception:
        pass

    # write features parquet
    try:
        dff = pd.DataFrame(feat_rows)
        pqf_path = Path("data/features") / f"dt={datetime.utcnow().strftime('%Y%m%d')}" / "features.parquet"
        write_parquet(dff, pqf_path)
    except Exception:
        pass
    # write features_flat parquet (selected numeric columns)
    try:
        dff2 = pd.DataFrame(feat_flat_rows)
        pqff_path = Path("data/features") / f"dt={datetime.utcnow().strftime('%Y%m%d')}" / "features_flat.parquet"
        write_parquet(dff2, pqff_path)
    except Exception:
        pass

    # write a simple daily brief JSON for UI (with macro deltas)
    try:
        top_1m = sorted(
            [r for r in rows if r.get("horizon") == "1m"],
            key=lambda r: ( (1.0 if r.get("direction") == "up" else (-1.0 if r.get("direction") == "down" else 0.0)) * float(r.get("confidence") or 0.0) + 0.5 * float(r.get("expected_return") or 0.0) ),
            reverse=True
        )[:5]
        # Macro deltas (best-effort): DXY (DTWEXBGS) d1/ WoW, 10Y (DGS10) bp d1/WoW, Gold futures GC=F d1/WoW
        macro = {}
        changes = {"macro": {}, "watchlist_moves": []}
        try:
            dxy = get_fred_series("DTWEXBGS")
            s = dxy.iloc[:,0].dropna() if dxy is not None and not dxy.empty else pd.Series([], dtype=float)
            if len(s) > 5:
                macro["DXY_wow"] = float((s.iloc[-1] / s.iloc[-5]) - 1.0)
            if len(s) > 1:
                changes["macro"]["DXY_d1"] = float((s.iloc[-1] / s.iloc[-2]) - 1.0)
        except Exception:
            pass
        try:
            dgs10 = get_fred_series("DGS10")
            s10 = dgs10.iloc[:,0].dropna() if dgs10 is not None and not dgs10.empty else pd.Series([], dtype=float)
            if len(s10) > 5:
                macro["UST10Y_bp_wow"] = float((s10.iloc[-1] - s10.iloc[-5]) * 100.0)
            if len(s10) > 1:
                changes["macro"]["UST10Y_bp_d1"] = float((s10.iloc[-1] - s10.iloc[-2]) * 100.0)
        except Exception:
            pass
        try:
            g = get_price_history("GC=F", start=(datetime.utcnow().date().isoformat()))  # may return None; fallback below
            if g is None:
                # broader window
                g = get_price_history("GC=F", start=(datetime.utcnow().date().replace(day=1).isoformat()))
            if g is not None and not g.empty and len(g.index) > 5:
                macro["Gold_wow"] = float((g["Close"].iloc[-1] / g["Close"].iloc[-5]) - 1.0)
            if g is not None and not g.empty and len(g.index) > 1:
                changes["macro"]["Gold_d1"] = float((g["Close"].iloc[-1] / g["Close"].iloc[-2]) - 1.0)
        except Exception:
            pass
        # Watchlist 1‑day moves (best-effort)
        try:
            for t in [x.strip().upper() for x in (os.getenv("WATCHLIST") or "").split(",") if x.strip()]:
                ph = get_price_history(t, start=(datetime.utcnow().date().replace(day=max(1, datetime.utcnow().day-7)).isoformat()))
                if ph is not None and not ph.empty and len(ph.index) > 1:
                    d1 = float((ph["Close"].iloc[-1] / ph["Close"].iloc[-2]) - 1.0)
                    changes["watchlist_moves"].append({"ticker": t, "d1": d1})
            # sort by absolute move and keep top 5
            changes["watchlist_moves"] = sorted(changes["watchlist_moves"], key=lambda x: abs(x.get("d1") or 0.0), reverse=True)[:5]
        except Exception:
            pass
        brief = {
            "date": datetime.utcnow().isoformat() + "Z",
            "universe": [x.strip().upper() for x in os.getenv("WATCHLIST", "").split(",") if x.strip()] or [r.get("ticker") for r in top_1m],
            "focus": "Gold miners & gold sector if present in universe",
            "macro": macro,
            "changes": changes,
            "top_picks_1m": [{"ticker": r.get("ticker"), "direction": r.get("direction"), "confidence": r.get("confidence"), "expected_return": r.get("expected_return")} for r in top_1m],
            "notes": [
                "Scores based on SMA+sentiment baseline; ML+LLM blend to be added.",
                "Use Deep Dive for per-ticker details and news context.",
            ],
        }
        with (OUTDIR / "brief.json").open("w", encoding="utf-8") as f:
            json.dump(brief, f, ensure_ascii=False, indent=2)
    except Exception:
        pass

    print(json.dumps({"ok": True, "outdir": str(OUTDIR), "count": len(results["tickers"])}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

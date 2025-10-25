"""
Recession Risk Agent — estimates recession probability using macro proxies:
- Yield curve inversion (DGS10 - DGS2, bp)
- Unemployment 6m change (UNRATE)
- NFCI (Chicago Fed index)
- HY spread (BAMLH0A0HYM2)

Outputs: data/macro/recession/dt=YYYYMMDD/recession.json
Fields: asof, inputs, scores (normalized), probability, summary_fr
"""

from __future__ import annotations

from pathlib import Path
from datetime import datetime
from typing import Dict, Any
import json
import math
import pandas as pd


def _dropna_series(df: pd.DataFrame) -> pd.Series:
    return df.iloc[:, 0].dropna() if df is not None and not df.empty else pd.Series([], dtype=float)


def _last(s: pd.Series) -> float | None:
    s = s.dropna()
    if s.empty:
        return None
    return float(s.iloc[-1])


def _ch_6m(s: pd.Series) -> float | None:
    s = s.dropna()
    if len(s) < 6:
        return None
    try:
        return float(s.iloc[-1] - s.iloc[-6])
    except Exception:
        return None


def _sigmoid(x: float, k: float = 1.0) -> float:
    try:
        return 1.0 / (1.0 + math.exp(-k * x))
    except Exception:
        return 0.5


def run() -> Path:
    from src.core.market_data import get_fred_series

    d10 = get_fred_series("DGS10")
    d2 = get_fred_series("DGS2")
    un = get_fred_series("UNRATE")
    nfci = get_fred_series("NFCI")
    hy = get_fred_series("BAMLH0A0HYM2")

    s10 = _dropna_series(d10)
    s2 = _dropna_series(d2)
    sun = _dropna_series(un)
    snf = _dropna_series(nfci)
    shy = _dropna_series(hy)

    # Inputs
    yc_bp = None
    if len(s10) and len(s2):
        try:
            yc_bp = float((s10.iloc[-1] - s2.iloc[-1]) * 100.0)
        except Exception:
            yc_bp = None
    un_ch6 = _ch_6m(sun)
    nf = _last(snf)
    hy_last = _last(shy)

    inputs: Dict[str, Any] = {
        "yield_curve_bp": yc_bp,
        "unrate_6m_change": un_ch6,
        "nfci": nf,
        "hy_spread": hy_last,
    }

    # Normalize to risk scores (0..1, increasing with risk)
    # yield curve inversion: more negative → higher risk; map -100bp → ~0.7
    yc_score = 0.0
    if yc_bp is not None:
        yc_score = max(0.0, min(1.0, -yc_bp / 100.0))
    # unemployment 6m ↑ increases risk; +0.5pp ~ 0.5
    un_score = 0.0
    if un_ch6 is not None:
        un_score = max(0.0, min(1.0, un_ch6 / 1.0))
    # NFCI positive → tighter conditions; 0.5 → ~0.65 (saturate)
    nf_score = 0.0
    if nf is not None:
        nf_score = max(0.0, min(1.0, (nf + 0.1) / 0.6))
    # HY spread higher → risk up; 6 → ~0.5 (simple scale)
    hy_score = 0.0
    if hy_last is not None:
        hy_score = max(0.0, min(1.0, (hy_last - 2.5) / 7.5))

    # Composite via smooth logistic: average scores then sigmoid
    comps = [yc_score, un_score, nf_score, hy_score]
    avg = sum(comps) / max(1, len([c for c in comps if c is not None]))
    prob = _sigmoid(2.0 * (avg - 0.5))  # push to 0..1 around 0.5
    prob = float(round(prob, 3))

    summary = (
        f"Probabilité de récession (approx.): {int(prob*100)}%. "
        f"Courbe des taux: {None if yc_bp is None else round(yc_bp,1)} bp; chômage Δ6m: {None if un_ch6 is None else round(un_ch6,2)} pp; "
        f"NFCI: {nf}; HY: {None if hy_last is None else round(hy_last,2)}. "
        f"Note: estimation heuristique, à confronter aux indicateurs officiels."
    )

    out = {
        "asof": datetime.utcnow().isoformat() + "Z",
        "inputs": inputs,
        "scores": {
            "yield_curve": yc_score,
            "unemployment": un_score,
            "nfci": nf_score,
            "hy_spread": hy_score,
            "avg": round(avg, 3),
        },
        "probability": prob,
        "summary_fr": summary,
    }
    outdir = Path("data/macro/recession") / f"dt={datetime.utcnow().strftime('%Y%m%d')}"
    outdir.mkdir(parents=True, exist_ok=True)
    p = outdir / "recession.json"
    p.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    return p


if __name__ == "__main__":
    print(run())


"""
Macro Regime Agent — classifies current macro regime with simple heuristics from FRED.

Outputs JSON at data/macro/regime/dt=YYYYMMDD/regime.json with:
- indicators (yoy CPI, yoy GDP, yield_curve_bp, unrate, t10y_ie)
- regime probabilities for: expansion, slowdown, inflation, deflation
- text summary in FR (brief)
"""

from __future__ import annotations

from pathlib import Path
from datetime import datetime
from typing import Dict, Any
import json
import pandas as pd


def _yoy(series: pd.Series) -> float | None:
    s = series.dropna()
    if len(s) < 13:
        return None
    try:
        return float(s.iloc[-1] / s.iloc[-13] - 1.0)
    except Exception:
        return None


def _last(series: pd.Series) -> float | None:
    s = series.dropna()
    if s.empty:
        return None
    return float(s.iloc[-1])


def _ch_6m(series: pd.Series) -> float | None:
    s = series.dropna()
    if len(s) < 6:
        return None
    try:
        return float(s.iloc[-1] - s.iloc[-6])
    except Exception:
        return None


def classify_regime(ind: Dict[str, float | None]) -> Dict[str, float]:
    # Scoring heuristics
    cpi_y = ind.get('cpi_yoy')
    gdp_y = ind.get('gdp_yoy')
    yc_bp = ind.get('yield_curve_bp')
    un = ind.get('unrate')
    un_6m = ind.get('unrate_ch_6m')

    # defaults
    expansion = 0.0
    slowdown = 0.0
    inflation = 0.0
    deflation = 0.0

    # Expansion: GDP>1%, CPI in [1%,4%], yield curve normal, unemployment not rising
    if gdp_y is not None:
        expansion += 0.7 if gdp_y > 0.01 else 0.0
        slowdown += 0.6 if gdp_y < 0.005 else 0.0
        deflation += 0.6 if gdp_y < 0 else 0.0
    if cpi_y is not None:
        if 0.01 <= cpi_y <= 0.04:
            expansion += 0.5
        if cpi_y > 0.04:
            inflation += 0.8
        if cpi_y < 0.002:
            deflation += 0.5
    if yc_bp is not None:
        expansion += 0.3 if yc_bp > 0 else 0.0
        slowdown += 0.6 if yc_bp < 0 else 0.0
    if un_6m is not None:
        slowdown += 0.5 if un_6m > 0.2 else 0.0
        expansion += 0.2 if un_6m < -0.2 else 0.0

    # Normalize to probabilities
    import math
    vec = [expansion, slowdown, inflation, deflation]
    # ensure non-negative
    vec = [max(0.0, v) for v in vec]
    s = sum(vec) or 1.0
    probs = [v / s for v in vec]
    return {
        'expansion': round(probs[0], 3),
        'slowdown': round(probs[1], 3),
        'inflation': round(probs[2], 3),
        'deflation': round(probs[3], 3),
    }


def run() -> Path:
    # Local import to reuse repo utils
    from src.core.market_data import get_fred_series

    cpi = get_fred_series('CPIAUCSL')
    gdp = get_fred_series('GDPC1')
    d10 = get_fred_series('DGS10')
    d2 = get_fred_series('DGS2')
    un = get_fred_series('UNRATE')
    t10yie = get_fred_series('T10YIE')

    ind: Dict[str, Any] = {}
    if cpi is not None and not cpi.empty:
        ind['cpi_yoy'] = _yoy(cpi.iloc[:, 0])
    if gdp is not None and not gdp.empty:
        ind['gdp_yoy'] = _yoy(gdp.iloc[:, 0])
    if d10 is not None and not d10.empty and d2 is not None and not d2.empty:
        try:
            ind['yield_curve_bp'] = float((d10.iloc[-1, 0] - d2.iloc[-1, 0]) * 100.0)
        except Exception:
            ind['yield_curve_bp'] = None
    if un is not None and not un.empty:
        ind['unrate'] = _last(un.iloc[:, 0])
        ind['unrate_ch_6m'] = _ch_6m(un.iloc[:, 0])
    if t10yie is not None and not t10yie.empty:
        ind['t10y_ie'] = _last(t10yie.iloc[:, 0])

    probs = classify_regime(ind)
    # brief summary in FR
    order = sorted(probs.items(), key=lambda x: -x[1])
    top_regime, top_p = order[0]
    brief = (
        f"Régime dominant: {top_regime} (~{int(top_p*100)}%). "
        f"Indicateurs: CPI YoY={None if ind.get('cpi_yoy') is None else round(ind['cpi_yoy']*100,1)}%, "
        f"GDP YoY={None if ind.get('gdp_yoy') is None else round(ind['gdp_yoy']*100,1)}%, "
        f"Yield curve={None if ind.get('yield_curve_bp') is None else round(ind['yield_curve_bp'],1)} bp, "
        f"Chômage={ind.get('unrate')} (Δ6m={ind.get('unrate_ch_6m')})."
    )

    out = {
        'asof': datetime.utcnow().isoformat() + 'Z',
        'indicators': ind,
        'probs': probs,
        'summary_fr': brief,
    }
    outdir = Path('data/macro/regime')/f"dt={datetime.utcnow().strftime('%Y%m%d')}"
    outdir.mkdir(parents=True, exist_ok=True)
    p = outdir/'regime.json'
    p.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding='utf-8')
    return p


if __name__ == '__main__':
    print(run())


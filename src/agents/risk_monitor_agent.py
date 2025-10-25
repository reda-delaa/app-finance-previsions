"""
Risk Monitor Agent — computes a simple composite risk score using FRED series:
- Yield curve inversion magnitude (DGS10 - DGS2, bp)
- High Yield spread (BAMLH0A0HYM2)
- Chicago NFCI (NFCI)

Writes data/risk/dt=YYYYMMDD/risk.json with normalized components and composite.
"""

from __future__ import annotations

from pathlib import Path
from datetime import datetime
from typing import Dict, Any
import json
import pandas as pd


def _last(s: pd.Series) -> float | None:
    s = s.dropna()
    if s.empty: return None
    return float(s.iloc[-1])


def _z(val: float | None, mean: float, std: float) -> float | None:
    if val is None: return None
    if std <= 0: return 0.0
    return float((val - mean)/std)


def run() -> Path:
    from src.core.market_data import get_fred_series

    d10 = get_fred_series('DGS10')
    d2 = get_fred_series('DGS2')
    hy = get_fred_series('BAMLH0A0HYM2')
    nfci = get_fred_series('NFCI')

    yc_bp = None
    if d10 is not None and not d10.empty and d2 is not None and not d2.empty:
        try:
            yc_bp = float((d10.iloc[-1,0] - d2.iloc[-1,0]) * 100.0)
        except Exception:
            yc_bp = None
    hy_last = _last(hy.iloc[:,0]) if hy is not None and not hy.empty else None
    nfci_last = _last(nfci.iloc[:,0]) if nfci is not None and not nfci.empty else None

    # z-normalize using last 3y window if available
    def _z_series(df: pd.DataFrame) -> tuple[float | None, float, float]:
        if df is None or df.empty: return (None, 0.0, 1.0)
        s = df.iloc[-(252*3):, 0].dropna()
        if s.empty: return (None, 0.0, 1.0)
        return (float(s.iloc[-1]), float(s.mean()), float(s.std(ddof=1) or 1.0))

    hy_val, hy_mu, hy_sd = _z_series(hy)
    nf_val, nf_mu, nf_sd = _z_series(nfci)
    # yield curve: treat inversion (negative) as higher risk; use absolute inversion magnitude
    yc_risk = None
    if yc_bp is not None:
        # approximate z by scaling bp (100bp ~ 1 sd)
        yc_risk = float(max(0.0, -yc_bp/100.0))

    hy_z = _z(hy_val, hy_mu, hy_sd)
    nf_z = _z(nf_val, nf_mu, nf_sd)

    # composite: average of available components after mapping to positive risk
    comps = []
    if yc_risk is not None: comps.append(yc_risk)
    if hy_z is not None: comps.append(max(0.0, hy_z))
    if nf_z is not None: comps.append(max(0.0, nf_z))
    composite = float(sum(comps)/len(comps)) if comps else 0.0
    risk_level = 'low'
    if composite >= 1.5: risk_level = 'high'
    elif composite >= 0.7: risk_level = 'medium'

    out = {
        'asof': datetime.utcnow().isoformat()+'Z',
        'components': {
            'yield_curve_bp': yc_bp,
            'yield_curve_inversion_risk': yc_risk,
            'hy_spread': hy_last,
            'hy_z': hy_z,
            'nfci': nfci_last,
            'nfci_z': nf_z,
        },
        'composite': composite,
        'risk_level': risk_level,
    }
    outdir = Path('data/risk')/f"dt={datetime.utcnow().strftime('%Y%m%d')}"
    outdir.mkdir(parents=True, exist_ok=True)
    p = outdir/'risk.json'
    p.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding='utf-8')
    return p


if __name__ == '__main__':
    print(run())


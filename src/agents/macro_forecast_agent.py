from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any

import numpy as np
import pandas as pd

try:
    from core.market_data import get_fred_series
except Exception:  # pragma: no cover
    import sys as _sys
    _SRC = Path(__file__).resolve().parents[1]
    if str(_SRC) not in _sys.path:
        _sys.path.insert(0, str(_SRC))
    from core.market_data import get_fred_series


def _today_dt() -> str:
    return datetime.utcnow().strftime("%Y%m%d")


def _yoy(series: pd.Series) -> pd.Series:
    try:
        return series.pct_change(12)
    except Exception:
        return pd.Series(dtype=float)


def _zscore(s: pd.Series) -> pd.Series:
    try:
        m, sd = s.mean(skipna=True), s.std(skipna=True)
        return (s - m) / sd if sd and sd > 0 else s * 0
    except Exception:
        return pd.Series(index=s.index, dtype=float)


def run_once() -> Dict[str, Any]:
    # Fetch core macro series (best-effort)
    # CPI (CPIAUCSL), Unemployment (UNRATE), 10Y rate (DGS10), 2Y rate (DGS2)
    cpi = get_fred_series("CPIAUCSL")  # monthly
    unemp = get_fred_series("UNRATE")   # monthly
    d10 = get_fred_series("DGS10")      # daily
    d2 = get_fred_series("DGS2")        # daily

    # Build signals
    slope = None
    if not d10.empty and not d2.empty:
        df_rates = d10.join(d2, how='inner')
        df_rates.columns = ['DGS10', 'DGS2']
        df_rates['slope'] = df_rates['DGS10'] - df_rates['DGS2']
        slope = float(df_rates['slope'].dropna().iloc[-1]) if not df_rates.empty else None

    cpi_yoy = None
    if not cpi.empty:
        c = _yoy(cpi.iloc[:, 0])
        cpi_yoy = float(c.dropna().iloc[-1]) if not c.empty else None

    un = None
    if not unemp.empty:
        un = float(unemp.iloc[:, 0].dropna().iloc[-1])

    # Crude probability of recession based on negative slope and elevated unemployment trend
    prob_rec = None
    try:
        z_slope = 0.0 if slope is None else float(np.clip(-slope / 1.0, -3, 3))  # invert: more negative slope → higher risk
        z_un = 0.0
        if not unemp.empty:
            z_un = float(_zscore(unemp.iloc[:, 0]).dropna().iloc[-1])
        raw = 0.5 + 0.2 * z_slope + 0.1 * z_un
        prob_rec = float(np.clip(raw, 0.0, 1.0))
    except Exception:
        prob_rec = None

    # Construct simple horizon views (1m/3m/12m) using current signals
    horizons = {}
    for h in ("1m", "3m", "12m"):
        horizons[h] = {
            "inflation_yoy": cpi_yoy,
            "yield_curve_slope": slope,
            "unemployment": un,
            "recession_prob": prob_rec,
        }

    out_json = {
        "asof": datetime.utcnow().isoformat() + "Z",
        "dt": _today_dt(),
        "horizons": horizons,
        "notes": "Baseline macro forecast (FRED-derived). Slope=DGS10-DGS2; inflation_yoy=CPIAUCSL YoY.",
    }

    outdir = Path("data/macro/forecast") / f"dt={_today_dt()}"
    outdir.mkdir(parents=True, exist_ok=True)
    jp = outdir / "macro_forecast.json"
    jp.write_text(json.dumps(out_json, ensure_ascii=False, indent=2), encoding='utf-8')

    # Also write a parquet summary for easy joins
    rows = []
    for h, v in horizons.items():
        row = {"dt": pd.to_datetime(_today_dt()), "horizon": h}
        row.update(v)
        rows.append(row)
    pdf = pd.DataFrame(rows)
    pdf.to_parquet(outdir / "macro_forecast.parquet", index=False)

    return out_json


if __name__ == "__main__":
    js = run_once()
    print(json.dumps(js)[:200] + "…")


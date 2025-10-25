from __future__ import annotations

from datetime import datetime
from pathlib import Path
import pandas as pd
import numpy as np

DT_FMT = "%Y%m%d"


def _today_dt() -> str:
    return datetime.utcnow().strftime(DT_FMT)


def _latest_forecasts_path() -> Path | None:
    parts = sorted(Path('data/forecast').glob('dt=*/forecasts.parquet'))
    return parts[-1] if parts else None


def aggregate() -> Path | None:
    p = _latest_forecasts_path()
    if not p:
        return None
    df = pd.read_parquet(p)
    if df.empty:
        return None

    # Compute a simple final score combining direction*confidence and expected_return
    dir_map = {"up": 1.0, "flat": 0.0, "down": -1.0}
    df['dir_base'] = df['direction'].map(dir_map).fillna(0.0)
    df['expected_return'] = pd.to_numeric(df['expected_return'], errors='coerce').fillna(0.0)
    df['confidence'] = pd.to_numeric(df['confidence'], errors='coerce').fillna(0.5)
    base = df['dir_base'] * df['confidence']
    final = base + 0.6 * df['expected_return']
    # Normalize to a bounded range for stability
    try:
        m, s = float(final.mean()), float(final.std())
        if s and s > 0:
            final = (final - m) / s
    except Exception:
        pass
    df_out = df[['ticker','horizon']].copy()
    df_out.insert(0, 'dt', pd.to_datetime(_today_dt()))
    df_out['final_score'] = final.astype(float).round(4)

    outdir = Path("data/forecast") / f"dt={_today_dt()}"
    outdir.mkdir(parents=True, exist_ok=True)
    out = outdir / 'final.parquet'
    df_out.to_parquet(out, index=False)
    return out


if __name__ == "__main__":
    p = aggregate()
    print(p)


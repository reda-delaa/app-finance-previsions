"""
Fuse forecasts: combine rule/expected_return + ML signals (+optional LLM) into final scores.

Writes data/forecast/dt=YYYYMMDD/final.parquet with columns:
ticker, horizon, final_score, direction, confidence, expected_return, ml_return, ml_conf
"""

from __future__ import annotations

from pathlib import Path
from datetime import datetime
import json
import pandas as pd


def latest_dt(root: Path) -> Path | None:
    parts = sorted(root.glob('dt=*'))
    return parts[-1] if parts else None


def main() -> int:
    root = Path('data/forecast')
    ddir = latest_dt(root)
    if not ddir:
        print({'ok': False, 'error': 'no forecast directory'})
        return 1
    pq = ddir/'forecasts.parquet'
    if not pq.exists():
        print({'ok': False, 'error': 'no forecasts.parquet'})
        return 1
    df = pd.read_parquet(pq)
    if df.empty:
        print({'ok': False, 'error': 'empty forecasts'})
        return 1
    # scoring
    dir_map = {'up': 1.0, 'flat': 0.0, 'down': -1.0}
    df['dir_base'] = df['direction'].map(dir_map).fillna(0.0)
    base_score = df['dir_base']*df['confidence'].astype(float) + 0.5*df['expected_return'].fillna(0.0).astype(float)
    if 'ml_return' in df.columns:
        mlc = df.get('ml_conf', 0.6).fillna(0.6)
        ml_part = (df['ml_return'].fillna(0.0).astype(float) * mlc.astype(float))
    else:
        ml_part = 0.0
    df['final_score'] = 0.7*base_score + 0.3*ml_part
    out_cols = ['ticker','horizon','direction','confidence','expected_return','ml_return','ml_conf','final_score']
    out = df[[c for c in out_cols if c in df.columns]].copy()
    # write
    fp = ddir/'final.parquet'
    fp.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(fp, index=False)
    print({'ok': True, 'path': str(fp), 'rows': len(out)})
    return 0


if __name__ == '__main__':
    raise SystemExit(main())


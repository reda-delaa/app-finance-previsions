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
    # optional LLM consensus from llm_agents.json (avg_agreement per ticker)
    llm_map = {}
    try:
        agents = sorted(ddir.glob('llm_agents.json'))
        if agents:
            obj = json.loads(agents[-1].read_text(encoding='utf-8'))
            for it in (obj.get('tickers') or []):
                t = (it or {}).get('ticker')
                ens = (it or {}).get('ensemble') or {}
                aa = ens.get('avg_agreement')
                if t and isinstance(aa, (int, float)):
                    # clip 0..1
                    llm_map[str(t)] = max(0.0, min(1.0, float(aa)))
    except Exception:
        pass
    df['llm_consensus'] = df['ticker'].map(lambda x: llm_map.get(str(x)))
    df['llm_consensus'] = df['llm_consensus'].fillna(0.0).astype(float)
    # final blend: rule 0.65, ml 0.25, llm 0.10
    df['final_score'] = 0.65*base_score + 0.25*ml_part + 0.10*df['llm_consensus']
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

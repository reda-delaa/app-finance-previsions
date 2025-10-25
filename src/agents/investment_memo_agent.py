"""
Investment Memo Agent — per‑ticker concise memo via LLM ensemble (text‑only).

Writes data/memos/dt=YYYYMMDD/<ticker>.json with:
- answer (markdown), parsed (if JSON tail parsed), and metadata
"""

from __future__ import annotations

from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional
import json


def _iso() -> str:
    return datetime.utcnow().isoformat() + "Z"


def memo_for_ticker(ticker: str, locale: str = "fr-FR", top_n: int = 3) -> Path:
    from src.analytics.market_intel import build_snapshot
    from src.analytics.econ_llm_agent import EconomicAnalyst, EconomicInput

    snap = build_snapshot(regions=["US","INTL"], window="last_week", ticker=ticker, limit=180)
    feats = (snap or {}).get("features") or {}
    news = (snap or {}).get("news") or []
    agent = EconomicAnalyst()
    ein = EconomicInput(
        question=(
            f"Rédige une note d'investissement synthétique (≤250 mots) sur {ticker}. "
            "Sections : Thèse (3 puces), Catalyseurs (3–5), Risques (3–5), Valorisation rapide (pairs/ratio), "
            "Technique (supports/résistances), Timeframe (1m/3m). "
            "Ajoute une ligne JSON (summary, risks, actions, confidence)."
        ),
        features=feats,
        news=news,
        attachments=None,
        locale=locale,
        meta={"kind": "memo", "ticker": ticker},
    )
    try:
        res = agent.analyze_ensemble(ein, top_n=top_n, force_power=True, adjudicate=True)
        answer_txt = (res.get('adjudication') or {}).get('decision') or ""
        if not answer_txt:
            for r in (res.get('results') or []):
                if r.get('ok') and r.get('answer'):
                    answer_txt = r['answer']; break
        parsed_obj = None
        for r in (res.get('results') or []):
            if r.get('parsed'):
                parsed_obj = r['parsed']; break
        out = {
            'asof': _iso(),
            'ticker': ticker,
            'answer': answer_txt,
            'parsed': parsed_obj,
            'ensemble': {k: res.get(k) for k in ['models','avg_agreement','adjudication'] if k in res},
        }
    except Exception:
        res = agent.analyze(ein)
        out = {
            'asof': _iso(),
            'ticker': ticker,
            'answer': res.get('answer'),
            'parsed': res.get('parsed'),
            'ensemble': None,
        }
    outdir = Path('data/memos')/f"dt={datetime.utcnow().strftime('%Y%m%d')}"
    outdir.mkdir(parents=True, exist_ok=True)
    p = outdir/f"{ticker}.json"
    p.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding='utf-8')
    return p


def run_all(watchlist: Optional[List[str]] = None, locale: str = "fr-FR", top_n: int = 3) -> List[str]:
    import os
    wl = watchlist
    if not wl:
        raw = os.getenv("WATCHLIST") or "";
        wl = [x.strip().upper() for x in raw.split(',') if x.strip()]
    if not wl:
        # try data/watchlist.json
        try:
            obj = json.loads(Path('data/watchlist.json').read_text(encoding='utf-8'))
            wl = [x.strip().upper() for x in (obj.get('watchlist') or []) if isinstance(x, str) and x.strip()]
        except Exception:
            wl = []
    paths = []
    for t in wl[:10]:  # safety cap
        try:
            paths.append(str(memo_for_ticker(t, locale=locale, top_n=top_n)))
        except Exception:
            continue
    return paths


if __name__ == '__main__':
    print({'ok': True, 'written': run_all()})


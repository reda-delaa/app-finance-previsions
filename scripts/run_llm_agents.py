"""
Run LLM ensemble forecasters per ticker using g4f-backed EconomicAnalyst.

Outputs: data/forecast/dt=YYYYMMDD/llm_agents.json (per-ticker results)
"""

from __future__ import annotations

import os
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any

import sys as _sys
_SRC_ROOT = Path(__file__).resolve().parents[1] / "src"
if str(_SRC_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_SRC_ROOT))

from analytics.market_intel import build_snapshot
from analytics.econ_llm_agent import EconomicAnalyst, EconomicInput


WATCHLIST = os.getenv("WATCHLIST", "NGD.TO,AEM.TO,ABX.TO,K.TO,GDX").split(",")
OUTDIR = Path("data/forecast") / f"dt={datetime.utcnow().strftime('%Y%m%d')}"


def main() -> int:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    out: Dict[str, Any] = {"asof": datetime.utcnow().isoformat()+"Z", "tickers": []}
    agent = EconomicAnalyst()

    for t in [x.strip().upper() for x in WATCHLIST if x.strip()]:
        try:
            snap = build_snapshot(regions=["US","INTL"], window="last_week", ticker=t, limit=180)
            feats = (snap or {}).get("features") or {}
            news = (snap or {}).get("news") or []
            ein = EconomicInput(
                question=f"Prévision 1 mois pour {t}: direction probable, drivers, risques, et probabilité.",
                features=feats,
                news=news,
                attachments=None,
                locale="fr-FR",
                meta={"ticker": t, "horizon": "1m"},
            )
            res = agent.analyze_ensemble(ein, top_n=3, force_power=True, adjudicate=True)
            out["tickers"].append({"ticker": t, "ensemble": res})
        except Exception as e:
            out["tickers"].append({"ticker": t, "error": str(e)})

    (OUTDIR/"llm_agents.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"ok": True, "out": str(OUTDIR/"llm_agents.json")}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

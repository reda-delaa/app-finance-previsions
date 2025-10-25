"""
G4F Model Watcher — keeps a local working list of high‑quality g4f models.

- Selects SOTA verified models (using src.runners.sanity_runner_ia_chat if network allows)
- Tests a handful with short prompts via g4f, measures latency
- Writes: data/llm/models/working.json

Use:
  python -m src.agents.g4f_model_watcher --refresh --limit 8
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


WORKING_PATH = Path("data/llm/models/working.json")


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class ModelProbe:
    model: str
    ok: bool
    provider: Optional[str]
    latency_s: Optional[float]
    pass_rate: Optional[float] = None
    tested_at: str = _now_iso()


def _ensure_dir(p: Path):
    p.parent.mkdir(parents=True, exist_ok=True)


def _load_working() -> Dict[str, Any]:
    try:
        return json.loads(WORKING_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {"asof": _now_iso(), "models": []}


def _save_working(objs: List[ModelProbe]) -> Path:
    payload = {
        "asof": _now_iso(),
        "models": [asdict(x) for x in objs],
    }
    _ensure_dir(WORKING_PATH)
    WORKING_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return WORKING_PATH


def _static_candidates() -> List[str]:
    # fallback: use static power list from econ_llm_agent
    try:
        from analytics.econ_llm_agent import POWER_NOAUTH_MODELS
        return POWER_NOAUTH_MODELS[:]
    except Exception:
        return []


def _verified_candidates(limit: int = 12, refresh: bool = True) -> List[Dict[str, Any]]:
    # import lazy to avoid hard dependency when network is blocked
    try:
        from runners.sanity_runner_ia_chat import select_verified_models
        return select_verified_models(caps_need=("text",), min_pass=0.30, only_sota=True, limit=limit, refresh=refresh)
    except Exception:
        # degrade gracefully
        return [{"model": m, "pass_rate": None, "hint": None} for m in _static_candidates()[:limit]]


def _probe_model(model_name: str, system: Optional[str] = None, prompt: Optional[str] = None,
                 providers_per_model: int = 4, tries_per_model: int = 2, timeout: int = 45) -> ModelProbe:
    system = system or "Tu es un analyste macro‑financier factuel et concis."
    prompt = prompt or "Donne 3 risques macro majeurs à surveiller cette semaine (puces courtes)."
    try:
        from runners.sanity_runner_ia_chat import ask_with_specific_model
        res = ask_with_specific_model(model_name, prompt=prompt, system=system,
                                      providers_per_model=providers_per_model,
                                      tries_per_model=tries_per_model,
                                      timeout=timeout)
        return ModelProbe(model=model_name, ok=bool(res.get("ok")), provider=res.get("provider"), latency_s=res.get("latency_s"))
    except Exception:
        return ModelProbe(model=model_name, ok=False, provider=None, latency_s=None)


def refresh(limit: int = 8, refresh_verified: bool = True) -> Path:
    cand = _verified_candidates(limit=limit, refresh=refresh_verified)
    out: List[ModelProbe] = []
    for c in cand:
        m = c.get("model") if isinstance(c, dict) else str(c)
        pr = ModelProbe(model=m, ok=False, provider=None, latency_s=None, pass_rate=c.get("pass_rate") if isinstance(c, dict) else None)
        try:
            probe = _probe_model(m)
            probe.pass_rate = pr.pass_rate
            out.append(probe)
        except Exception:
            out.append(pr)
        # small pacing to avoid hammering providers
        time.sleep(0.5)
    return _save_working(out)


def load_working_models(max_age_hours: int = 24) -> List[str]:
    obj = _load_working()
    try:
        asof = obj.get("asof")
        if asof:
            dt = datetime.fromisoformat(asof.replace("Z","+00:00"))
            age_h = (datetime.now(timezone.utc) - dt).total_seconds()/3600.0
            if age_h > max_age_hours:
                return []
    except Exception:
        pass
    rows = obj.get("models") or []
    # Sort by ok desc, pass_rate desc, latency asc
    rows.sort(key=lambda r: (not bool(r.get("ok")), -(r.get("pass_rate") or 0.0), (r.get("latency_s") or 9999.0)))
    return [r.get("model") for r in rows if r.get("ok")]


def main(argv: Optional[List[str]] = None) -> int:
    import argparse
    p = argparse.ArgumentParser(description="G4F Model Watcher")
    p.add_argument("--refresh", action="store_true", help="Refresh working models and write JSON")
    p.add_argument("--limit", type=int, default=int(os.getenv("G4F_TEST_LIMIT","8")))
    p.add_argument("--no-refresh-verified", action="store_true", help="Skip refreshing verified list, use cache")
    args = p.parse_args(argv)
    if args.refresh:
        path = refresh(limit=args.limit, refresh_verified=(not args.no_refresh_verified))
        print(json.dumps({"ok": True, "path": str(path)}, ensure_ascii=False))
        return 0
    # else show current
    print(json.dumps({"ok": True, "models": load_working_models() }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


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
    source: Optional[str] = None  # 'verified' | 'official' | None
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


def _official_candidates(limit: int = 50) -> List[Dict[str, Any]]:
    """Return a best-effort list of 'official' models.

    Priority:
    1) Local file data/llm/official/models.txt (lines: provider|model or model)
    2) g4f library introspection (heuristic)
    """
    out: List[Dict[str, Any]] = []
    # 1) Local file seed
    txt = Path('data/llm/official/models.txt')
    if txt.exists():
        try:
            for line in txt.read_text(encoding='utf-8').splitlines():
                s = line.strip()
                if not s or s.startswith('#'):
                    continue
                if '|' in s:
                    prov, model = s.split('|', 1)
                    out.append({"model": model.strip(), "hint": prov.strip(), "pass_rate": None})
                else:
                    out.append({"model": s, "hint": None, "pass_rate": None})
        except Exception:
            pass
    # 2) g4f introspection (best-effort)
    if not out:
        try:
            import g4f
            # Heuristic: scan attributes that look like model name lists
            cand = []
            for name in dir(g4f):
                if name.lower() in ("models", "model"):  # common pattern
                    try:
                        val = getattr(g4f, name)
                        if isinstance(val, (list, tuple)):
                            cand.extend(str(x) for x in val if isinstance(x, (str,)))
                        elif isinstance(val, dict):
                            cand.extend(str(k) for k in val.keys())
                    except Exception:
                        continue
            cand = [c for c in cand if c and isinstance(c, str)]
            # unique preserve
            seen = set(); ordered = []
            for c in cand:
                if c not in seen:
                    seen.add(c); ordered.append(c)
            for m in ordered:
                out.append({"model": m, "hint": None, "pass_rate": None})
        except Exception:
            pass
    # clip
    uniq = []
    seen = set()
    for it in out:
        m = it.get('model')
        if m and m not in seen:
            seen.add(m); uniq.append(it)
    return uniq[:limit]


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
    import os
    source = os.getenv('G4F_SOURCE', 'both').lower().strip()
    cand: List[Dict[str, Any]] = []
    if source in ('verified', 'both'):
        for it in _verified_candidates(limit=limit, refresh=refresh_verified):
            it = dict(it); it['__source'] = 'verified'; cand.append(it)
    if source in ('official', 'both'):
        for it in _official_candidates(limit=limit):
            it = dict(it); it['__source'] = 'official'; cand.append(it)
    # Deduplicate by model preserving order
    seen = set(); merged: List[Dict[str, Any]] = []
    for it in cand:
        m = it.get('model')
        if m and m not in seen:
            seen.add(m); merged.append(it)
    if not merged:
        merged = [{"model": m, "pass_rate": None, "hint": None, "__source": None} for m in _static_candidates()[:limit]]
    out: List[ModelProbe] = []
    for c in merged:
        m = c.get("model") if isinstance(c, dict) else str(c)
        pr = ModelProbe(model=m, ok=False, provider=None, latency_s=None, pass_rate=c.get("pass_rate") if isinstance(c, dict) else None)
        try:
            probe = _probe_model(m)
            probe.source = c.get('__source')
            probe.pass_rate = pr.pass_rate
            out.append(probe)
        except Exception:
            pr.source = c.get('__source')
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


def merge_from_working_txt(txt_path: Path) -> Path:
    """Merge provider|model|media_type lines into working.json, marking them ok.

    Lines format: provider|model|media_type
    Unknown latency/pass_rate will be left as None.
    """
    if isinstance(txt_path, str):
        txt_path = Path(txt_path)
    models: List[ModelProbe] = []
    try:
        with txt_path.open('r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line or '|' not in line:
                    continue
                parts = line.split('|')
                if len(parts) >= 2:
                    provider, model = parts[0].strip(), parts[1].strip()
                    models.append(ModelProbe(model=model, ok=True, provider=provider, latency_s=None, pass_rate=None))
    except Exception:
        pass
    if not models:
        return WORKING_PATH
    # Load existing and union by model name (prefer existing with latency)
    current = _load_working()
    cur_map: Dict[str, Dict[str, Any]] = {m.get('model'): m for m in (current.get('models') or [])}
    for pr in models:
        if pr.model not in cur_map:
            cur_map[pr.model] = asdict(pr)
        else:
            # ensure ok stays True
            cur_map[pr.model]['ok'] = True
            if not cur_map[pr.model].get('provider'):
                cur_map[pr.model]['provider'] = pr.provider
    merged = [ModelProbe(**{**x, 'tested_at': x.get('tested_at') or _now_iso()}) for x in cur_map.values()]
    # Keep deterministic order: ok first, then name
    merged.sort(key=lambda r: (not r.ok, (r.model or '').lower()))
    return _save_working(merged)


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

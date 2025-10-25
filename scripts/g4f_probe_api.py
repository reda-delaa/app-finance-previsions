"""
G4F API probe: fetch providers and models from a local g4f.api server, test simple chat responses
and update the working models list used by the app (data/llm/models/working.json).

Usage:
  PYTHONPATH=src python3 scripts/g4f_probe_api.py --base http://127.0.0.1:8081 --limit 40 --update-working

Optional: start the g4f API server within this script by passing --start-server.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Dict, List, Tuple

import requests


def start_g4f_api_server(port: int = 8081, api_key: str | None = None):
    try:
        import threading
        import g4f.api
    except Exception as e:
        raise RuntimeError("g4f.api not installed. pip install -U g4f") from e

    def run():
        try:
            if api_key:
                g4f.api.AppConfig.set_config(g4f_api_key=api_key)
            g4f.api.run_api(port=port, debug=False)
        except Exception as e:
            print(f"[g4f.api] error: {e}")

    th = threading.Thread(target=run, daemon=True)
    th.start()
    time.sleep(5)
    return th


def fetch_providers(base: str, headers: Dict[str, str]) -> List[Dict]:
    r = requests.get(f"{base}/v1/providers", headers=headers, timeout=20)
    r.raise_for_status()
    return r.json()


def fetch_models_for_provider(base: str, provider_id: str, headers: Dict[str, str]) -> List[str]:
    r = requests.get(f"{base}/api/{provider_id}/models", headers=headers, timeout=30)
    r.raise_for_status()
    data = r.json()
    if isinstance(data, dict) and 'data' in data:
        return [m.get('id') for m in data['data'] if m.get('id')]
    return []


def sota_filter(models: List[str]) -> List[str]:
    try:
        from src.runners.sanity_runner_ia_chat import SOTA_PAT
        import re
        pat = SOTA_PAT
    except Exception:
        import re
        pat = re.compile(r"(deepseek|qwen3|glm-4|llama-3\.3|gpt-oss)", re.I)
    return [m for m in models if pat.search(m or "")]


def test_chat_stream(base: str, provider: str, model: str, headers: Dict[str, str]) -> bool:
    payload = {
        "model": model,
        "provider": provider,
        "messages": [{"role": "user", "content": "Say 'Yes' if you are working."}],
        "stream": True,
        "max_tokens": 32,
    }
    try:
        with requests.post(f"{base}/v1/chat/completions", headers=headers, json=payload, stream=True, timeout=60) as r:
            if r.status_code != 200:
                return False
            for line in r.iter_lines(decode_unicode=True):
                if not line:
                    continue
                if line.startswith("data: "):
                    data = line[6:]
                    if data == "[DONE]":
                        break
                    try:
                        js = json.loads(data)
                        if "error" in js:
                            return False
                        ch = (js.get("choices") or [{}])[0]
                        delta = (ch.get("delta") or {})
                        if isinstance(delta.get("content"), str) and delta.get("content"):
                            return True
                    except Exception:
                        continue
    except Exception:
        return False
    return False


def main(argv: List[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Probe g4f.api providers/models and update working list")
    p.add_argument("--base", default=os.getenv("G4F_API_BASE", "http://127.0.0.1:8081"))
    p.add_argument("--api-key", default=os.getenv("G4F_API_KEY"))
    p.add_argument("--limit", type=int, default=int(os.getenv("G4F_PROBE_LIMIT", "40")))
    p.add_argument("--start-server", action="store_true")
    p.add_argument("--update-working", action="store_true")
    args = p.parse_args(argv)

    if args.start_server:
        start_g4f_api_server(port=int(args.base.split(":")[-1]), api_key=args.api_key)

    headers = {"Content-Type": "application/json"}
    if args.api_key:
        headers["Authorization"] = f"Bearer {args.api_key}"

    providers = fetch_providers(args.base, headers)
    pairs: List[Tuple[str, str]] = []
    for pr in providers:
        pid = pr.get('id')
        if not pid:
            continue
        models = fetch_models_for_provider(args.base, pid, headers)
        models = sota_filter(models)
        for m in models:
            pairs.append((pid, m))
    # Truncate
    pairs = pairs[: args.limit]

    results_txt = []
    working_dir = Path("data/llm/probe")
    working_dir.mkdir(parents=True, exist_ok=True)
    res_json: List[Dict] = []
    for provider, model in pairs:
        ok = test_chat_stream(args.base, provider, model, headers)
        res_json.append({"provider": provider, "model": model, "ok": ok})
        if ok:
            results_txt.append(f"{provider}|{model}|text")
        time.sleep(0.1)

    (working_dir / "results.json").write_text(json.dumps(res_json, ensure_ascii=False, indent=2), encoding="utf-8")
    (working_dir / "working_results.txt").write_text("\n".join(results_txt), encoding="utf-8")

    if args.update_working:
        try:
            from src.agents.g4f_model_watcher import merge_from_working_txt
            out = merge_from_working_txt(working_dir / "working_results.txt")
            print(json.dumps({"ok": True, "updated": str(out)}, ensure_ascii=False))
        except Exception as e:
            print(json.dumps({"ok": False, "error": str(e)}, ensure_ascii=False))
            return 1
    else:
        print(json.dumps({"ok": True, "pairs": len(pairs), "working": len(results_txt)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


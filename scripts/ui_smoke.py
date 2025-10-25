from __future__ import annotations

"""
Lightweight UI smoke test against a running Streamlit app on localhost:8501.

Requires: pip install playwright; python -m playwright install chromium

Outputs screenshots under artifacts/smoke/ui and a JSON report at
data/reports/dt=YYYYMMDD/ui_smoke_report.json
"""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List


APP_BASE = os.getenv("UI_BASE", "http://localhost:8501").rstrip("/")
PAGES = [
    "Dashboard",
    "Agents_Status",
    "LLM_Scoreboard",
    "Earnings",
    "Risk",
    "Recession",
    "Signals",
    "Portfolio",
    "Alerts",
]


def _today_dt() -> str:
    return datetime.utcnow().strftime("%Y%m%d")


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def run_playwright() -> Dict[str, Any]:
    from playwright.sync_api import sync_playwright

    out: Dict[str, Any] = {"asof": datetime.utcnow().isoformat() + "Z", "results": []}
    shots_dir = Path("artifacts/smoke/ui")
    ensure_dir(shots_dir)
    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=True)
        context = browser.new_context(viewport={"width": 1440, "height": 900})
        page = context.new_page()
        for name in PAGES:
            url = f"{APP_BASE}/{name}"
            status = "ok"
            errors: List[str] = []
            warns: List[str] = []
            try:
                page.goto(url, wait_until="domcontentloaded", timeout=30000)
                # Give streamlit a moment to render
                page.wait_for_timeout(1500)
                txt = page.content()
                # Simple heuristics for error detection
                for kw in ["Traceback", "TypeError", "ValueError", "Exception:"]:
                    if kw in txt:
                        errors.append(kw)
                for kw in [
                    "Aucun",
                    "Aucune",
                    "not found",
                    "No data",
                    "pas de données",
                ]:
                    if kw in txt:
                        warns.append(kw)
                # Screenshot
                shot = shots_dir / f"{name}.png"
                page.screenshot(path=str(shot), full_page=True)
            except Exception as e:
                status = "error"
                errors.append(str(e))
            out["results"].append({
                "page": name,
                "url": url,
                "status": status,
                "errors": errors,
                "warnings": warns,
            })
        browser.close()
    # Write report
    repdir = Path("data/reports") / f"dt={_today_dt()}"
    ensure_dir(repdir)
    p = repdir / "ui_smoke_report.json"
    p.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    return out


def main() -> int:
    try:
        rep = run_playwright()
        print(json.dumps({"ok": True, "pages": len(rep.get("results", []))}))
        return 0
    except ModuleNotFoundError as e:
        print(
            json.dumps(
                {
                    "ok": False,
                    "error": "playwright not installed",
                    "hint": "pip install playwright && python -m playwright install chromium",
                }
            )
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())


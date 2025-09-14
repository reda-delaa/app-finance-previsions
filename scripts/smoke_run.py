#!/usr/bin/env python3
"""Simple smoke-run harness: runs a short list of module CLIs, captures stdout/stderr,
and writes a JSON report under artifacts/smoke/.

Run from repository root. It sets PYTHONPATH=$PWD for subprocesses.
"""
import os
import sys
import json
import shlex
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts" / "smoke"
ART.mkdir(parents=True, exist_ok=True)

RUNS = [
    {
        "name": "finviz",
        "cmd": [sys.executable, "-m", "src.ingestion.finviz_client", "--ticker", "AAPL", "--no_cache"],
        "out": ART / "finviz_aapl.stdout.txt",
        "err": ART / "finviz_aapl.stderr.txt",
    },
    {
        "name": "financials_snapshot",
        "cmd": [sys.executable, "-m", "src.ingestion.financials_ownership_client", "snapshot", "--ticker", "AAPL", "--no_cache"],
        "out": ART / "financials_snapshot_aapl.stdout.txt",
        "err": ART / "financials_snapshot_aapl.stderr.txt",
    },
    {
        "name": "market_intel",
        "cmd": [sys.executable, "-m", "src.analytics.market_intel", "run", "--ticker", "AAPL"],
        "out": ART / "market_intel_aapl.stdout.txt",
        "err": ART / "market_intel_aapl.stderr.txt",
    },
]


def run_one(spec):
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT)
    start = time.time()
    p = subprocess.run(spec["cmd"], env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    duration = time.time() - start
    spec["out"].write_bytes(p.stdout)
    spec["err"].write_bytes(p.stderr)
    return {
        "name": spec["name"],
        "cmd": " ".join(shlex.quote(x) for x in spec["cmd"]),
        "rc": p.returncode,
        "duration_s": round(duration, 3),
        "stdout_bytes": len(p.stdout),
        "stderr_bytes": len(p.stderr),
    }


def main():
    results = []
    for r in RUNS:
        print(f"Running {r['name']}...", flush=True)
        res = run_one(r)
        print(f" -> {r['name']} rc={res['rc']} (out {res['stdout_bytes']}b err {res['stderr_bytes']}b)")
        results.append(res)

    report = {"timestamp": time.time(), "results": results}
    (ART / "report.json").write_text(json.dumps(report, indent=2))
    print("Smoke run complete. Report written to artifacts/smoke/report.json")


if __name__ == "__main__":
    main()
